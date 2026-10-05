"""Main orchestration pipeline for faithfulness evaluation."""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelClient
from src.config import MODEL_CONFIGS

from .config import FaithfulnessConfig
from .dataset import DatasetLoader, Document
from .claim_decomposition import ClaimDecomposer, Claim, validate_claims_atomic
from .retrieval import PassageRetriever, RetrievalAugmentedEvaluator
from .faithfulness_judge import FaithfulnessJudge
from .jury_aggregation import FaithfulnessJury, calculate_inter_judge_agreement
from .position_bias import FaithfulnessBiasChecker, analyze_bias_results
from .human_validation import HumanValidator
from .analysis import ComparativeAnalyzer

class FaithfulnessPipeline:
    """Main pipeline for faithfulness evaluation."""
    
    def __init__(self, config: FaithfulnessConfig):
        self.config = config
        self.model_client = ModelClient()
        
        # Initialize components
        self.dataset_loader = DatasetLoader(
            dataset_name=config.dataset_name,
            num_docs=config.num_documents,
            min_words=config.min_word_count
        )
        
        self.claim_decomposer = ClaimDecomposer(
            self.model_client,
            config.claim_decomposition_model
        )
        
        self.retriever = PassageRetriever(
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            top_k=config.top_k_passages
        )
        
        self.jury = FaithfulnessJury(
            self.model_client,
            jury_size=config.jury_size,
            parallel=True
        )
        
        self.bias_checker = FaithfulnessBiasChecker(self.model_client)
        self.human_validator = HumanValidator(config.output_dir)
        self.analyzer = ComparativeAnalyzer()
        
        # Storage for results
        self.documents: List[Document] = []
        self.summaries: Dict[str, Dict[str, str]] = {}  # {doc_id: {model_id: summary}}
        self.claims: Dict[str, List[Claim]] = {}  # {doc_id_model: [claims]}
        self.jury_verdicts: Dict[str, any] = {}  # {claim_id: verdict}
        self.bias_results: Dict[str, any] = {}
    
    def load_existing_results(self):
        """Load existing results from previous runs."""
        import os
        import json

        # Load documents
        doc_path = os.path.join(self.config.output_dir, "documents.json")
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as f:
                docs_data = json.load(f)
            self.documents = [
                Document(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    text=doc["text"],
                    summary=doc.get("summary"),
                    word_count=doc["word_count"]
                )
                for doc in docs_data
            ]
            print(f"Loaded {len(self.documents)} existing documents")

        # Load jury verdicts
        verdict_path = os.path.join(self.config.output_dir, "jury_verdicts.json")
        if os.path.exists(verdict_path):
            with open(verdict_path, 'r') as f:
                verdicts_data = json.load(f)

            from .jury_aggregation import JuryVerdict
            for claim_id, verdict_data in verdicts_data.items():
                self.jury_verdicts[claim_id] = JuryVerdict(
                    majority_label=verdict_data["majority_label"],
                    label_distribution=verdict_data["label_distribution"],
                    confidence=verdict_data["confidence"],
                    individual_verdicts=[],  # Not stored in simplified format
                    agreement_score=verdict_data.get("agreement_score", 1.0),
                    jury_size=verdict_data["jury_size"]
                )
            print(f"Loaded {len(self.jury_verdicts)} existing jury verdicts")

        # Load claims if needed
        if not self.claims:
            claims_path = os.path.join(self.config.output_dir, "claims.json")
            if os.path.exists(claims_path):
                with open(claims_path, 'r') as f:
                    claims_data = json.load(f)

                # Convert back to Claim objects
                for key, claims_list in claims_data.items():
                    self.claims[key] = [
                        Claim(
                            claim_id=claim["claim_id"],
                            claim_text=claim["claim_text"],
                            source_sentence=claim.get("source_sentence")
                        )
                        for claim in claims_list
                    ]
                print(f"Loaded {len(self.claims)} existing claim sets")
    
    def run_full_pipeline(self, generate_summaries: bool = True,
                         run_bias_check: bool = False) -> Dict:
        """Run the complete faithfulness evaluation pipeline."""
        print("="*60)
        print("FAITHFULNESS EVALUATION PIPELINE")
        print("="*60)
        
        results = {
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Load dataset
        print("\n[Stage 1/7] Loading dataset...")
        self.documents = self.dataset_loader.load_dataset()
        self.dataset_loader.save_documents(
            os.path.join(self.config.output_dir, "documents.json")
        )
        results["stages"]["dataset"] = {
            "num_documents": len(self.documents),
            "avg_word_count": sum(d.word_count for d in self.documents) / len(self.documents)
        }
        
        # Stage 2: Generate summaries (if requested)
        if generate_summaries:
            print("\n[Stage 2/7] Generating summaries...")
            self._generate_summaries()
            results["stages"]["summaries"] = {
                "total_summaries": len(self.summaries) * len(self.config.summary_models)
            }
        else:
            print("\n[Stage 2/7] Skipping summary generation (use existing)")
        
        # Stage 3: Decompose claims
        print("\n[Stage 3/7] Decomposing claims...")
        self._decompose_claims()
        total_claims = sum(len(claims) for claims in self.claims.values())
        results["stages"]["claim_decomposition"] = {
            "total_claims": total_claims,
            "avg_claims_per_summary": total_claims / len(self.claims) if self.claims else 0
        }
        
        # Stage 4: Retrieve passages and classify
        print("\n[Stage 4/7] Retrieving passages and classifying claims...")
        self._classify_claims()
        results["stages"]["classification"] = {
            "total_classifications": len(self.jury_verdicts)
        }
        
        # Stage 5: Calculate inter-judge agreement
        print("\n[Stage 5/7] Calculating inter-judge agreement...")
        agreement_metrics = calculate_inter_judge_agreement(
            list(self.jury_verdicts.values())
        )
        results["stages"]["inter_judge_agreement"] = agreement_metrics
        print(f"  Percent agreement: {agreement_metrics['percent_agreement']:.2%}")
        print(f"  Fleiss' kappa: {agreement_metrics['fleiss_kappa']:.3f}")
        
        # Stage 6: Position bias check (if requested)
        if run_bias_check:
            print("\n[Stage 6/7] Running position bias checks...")
            self._run_bias_checks()
            bias_analysis = analyze_bias_results(self.bias_results)
            results["stages"]["position_bias"] = bias_analysis
            print(f"  Bias detected in {bias_analysis['bias_rate']:.1%} of checks")
        
        # Stage 7: Save results
        print("\n[Stage 7/7] Saving results...")
        self._save_results(results)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return results
    
    def _generate_summaries(self):
        """Generate summaries for all documents using all models."""
        for doc in self.documents:
            self.summaries[doc.doc_id] = {}
            
            for model_id in self.config.summary_models:
                model_config = MODEL_CONFIGS[model_id]
                
                print(f"  Generating summary for {doc.doc_id} with {model_config.name}...")
                
                # Generate summary using the model
                summary_prompt = f"Summarize the following document in 3-5 sentences:\n\n{doc.text}"
                
                response = self.model_client.call_model(
                    model_id=model_id,
                    provider=model_config.provider,
                    prompt=summary_prompt,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                self.summaries[doc.doc_id][model_id] = response.content
    
    def _decompose_claims(self):
        """Decompose all summaries into atomic claims."""
        for doc_id, model_summaries in self.summaries.items():
            for model_id, summary in model_summaries.items():
                key = f"{doc_id}_{model_id}"
                
                print(f"  Decomposing claims for {key}...")
                claims = self.claim_decomposer.decompose_summary(summary, key)
                
                # Validate claims
                issues = validate_claims_atomic(claims)
                if any(issues.values()):
                    print(f"    Warning: Found claim decomposition issues: {issues}")
                
                self.claims[key] = claims
    
    def _classify_claims(self):
        """Classify all claims using jury evaluation."""
        retrieval_evaluator = RetrievalAugmentedEvaluator(self.retriever)
        
        for key, claims in self.claims.items():
            doc_id = key.split('_')[0]
            doc = next((d for d in self.documents if d.doc_id == doc_id), None)
            
            if not doc:
                print(f"  Warning: Document not found for {key}")
                continue
            
            # Index document for retrieval
            chunks, embeddings = retrieval_evaluator.process_document(doc.text, doc_id)
            
            print(f"  Classifying {len(claims)} claims for {key}...")
            
            for claim in claims:
                # Retrieve passages
                passages = self.retriever.retrieve_passages(claim, chunks, embeddings, doc_id)
                
                # Classify with jury
                verdict = self.jury.evaluate_claim(claim, passages, claim.claim_id)
                self.jury_verdicts[claim.claim_id] = verdict
    
    def _run_bias_checks(self):
        """Run position bias checks on a sample of claims."""
        # Sample a subset of claims for bias checking
        sample_size = min(10, len(self.jury_verdicts))
        sample_claim_ids = list(self.jury_verdicts.keys())[:sample_size]
        
        retrieval_evaluator = RetrievalAugmentedEvaluator(self.retriever)
        
        for claim_id in sample_claim_ids:
            # Get claim and document
            claim = None
            doc_id = claim_id.split('_')[0]
            
            for claims in self.claims.values():
                for c in claims:
                    if c.claim_id == claim_id:
                        claim = c
                        break
                if claim:
                    break
            
            if not claim:
                continue
            
            doc = next((d for d in self.documents if d.doc_id == doc_id), None)
            if not doc:
                continue
            
            # Retrieve passages
            chunks, embeddings = retrieval_evaluator.process_document(doc.text, doc_id)
            passages = self.retriever.retrieve_passages(claim, chunks, embeddings, doc_id)
            
            # Run bias check
            result = self.bias_checker.check_claim_source_bias(claim, passages)
            self.bias_results[claim_id] = result
    
    def _save_results(self, results: Dict):
        """Save all results to files."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save main results
        with open(os.path.join(self.config.output_dir, "pipeline_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save claims for human validation
        claims_serializable = {}
        for key, claims in self.claims.items():
            claims_serializable[key] = [
                {
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "source_sentence": claim.source_sentence
                }
                for claim in claims
            ]
        
        with open(os.path.join(self.config.output_dir, "claims.json"), 'w') as f:
            json.dump(claims_serializable, f, indent=2)
        
        # Save jury verdicts
        jury_verdicts_serializable = {}
        for claim_id, verdict in self.jury_verdicts.items():
            if isinstance(verdict, dict):
                jury_verdicts_serializable[claim_id] = verdict
            else:
                jury_verdicts_serializable[claim_id] = {
                    "majority_label": verdict.majority_label,
                    "label_distribution": verdict.label_distribution,
                    "confidence": verdict.confidence,
                    "agreement_score": verdict.agreement_score,
                    "jury_size": verdict.jury_size
                }
        
        with open(os.path.join(self.config.output_dir, "jury_verdicts.json"), 'w') as f:
            json.dump(jury_verdicts_serializable, f, indent=2)
        
        # Save bias results
        with open(os.path.join(self.config.output_dir, "bias_results.json"), 'w') as f:
            bias_serializable = {}
            for claim_id, result in self.bias_results.items():
                bias_serializable[claim_id] = {
                    "claim_first_label": result.claim_first_label,
                    "source_first_label": result.source_first_label,
                    "bias_detected": result.bias_detected,
                    "label_changed": result.label_changed,
                    "confidence_delta": result.confidence_delta
                }
            json.dump(bias_serializable, f, indent=2)
        
        print(f"  Results saved to {self.config.output_dir}")
    
    def run_human_validation(self, annotation_template_path: str,
                          annotations_path: str = None) -> Dict:
        """Run human validation pipeline."""
        print("\n[HUMAN VALIDATION]")
        
        # Load existing results if available
        if not self.documents or not self.jury_verdicts:
            self.load_existing_results()
        
        # Create annotation template
        all_claims = []
        for claims in self.claims.values():
            all_claims.extend(claims)
        
        if not all_claims:
            print("No claims found. Please run the evaluation first to generate claims.")
            return {"template_created": False, "error": "no_claims"}
        
        print(f"Creating annotation template with {len(all_claims)} claims...")
        
        # Create simplified verdict dict for template generation
        simplified_verdicts = {}
        for claim_id, verdict in self.jury_verdicts.items():
            if isinstance(verdict, dict):
                simplified_verdicts[claim_id] = verdict
            else:
                # It's a JuryVerdict object
                simplified_verdicts[claim_id] = {
                    "majority_label": verdict.majority_label,
                    "confidence": verdict.confidence,
                    "individual_verdicts": [
                        {
                            "label": v.label,
                            "justification": v.justification
                        } for v in verdict.individual_verdicts
                    ] if verdict.individual_verdicts else []
                }
        
        self.human_validator.create_annotation_template(
            all_claims,
            simplified_verdicts,
            annotation_template_path
        )
        
        print(f"\nAnnotation template created: {annotation_template_path}")
        print(f"Total claims to annotate: {len(all_claims)}")
        
        if annotations_path:
            print(f"\nProcessing annotations from: {annotations_path}")
            result = self.process_human_annotations(annotations_path, annotation_template_path.replace("template", "report"))
            return result
        
        return {"template_created": True, "template_path": annotation_template_path, "total_claims": len(all_claims)}
    
    def process_human_annotations(self, annotations_path: str, 
                                 report_path: str) -> Dict:
        """Process human annotations and calculate agreement."""
        print("\n[PROCESSING HUMAN ANNOTATIONS]")
        
        # Load annotations
        annotations = self.human_validator.load_annotations(annotations_path)
        
        # Calculate agreement
        agreement_result = self.human_validator.calculate_agreement(
            annotations,
            self.jury_verdicts
        )
        
        print(f"Cohen's kappa: {agreement_result.cohen_kappa:.3f}")
        print(f"Percent agreement: {agreement_result.percent_agreement:.2%}")
        print(f"Agreement category: {agreement_result.agreement_category}")
        
        # Categorize disagreements
        disagreement_categories = self.human_validator.categorize_disagreements(
            agreement_result.disagreements
        )
        
        print(f"\nDisagreement breakdown:")
        for category, claim_ids in disagreement_categories.items():
            print(f"  {category}: {len(claim_ids)} claims")
        
        # Generate report
        self.human_validator.generate_validation_report(agreement_result, report_path)
        
        return {
            "cohen_kappa": agreement_result.cohen_kappa,
            "percent_agreement": agreement_result.percent_agreement,
            "agreement_category": agreement_result.agreement_category,
            "disagreement_categories": disagreement_categories,
            "total_disagreements": len(agreement_result.disagreements)
        }
