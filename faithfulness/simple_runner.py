"""Simplified runner for faithfulness evaluation - easier to use."""

import argparse
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def _run_methodology_test(output_dir: str) -> int:
    """Run methodology test without API keys."""
    print("\n[Testing Components]")
    
    # Test 1: Dataset loading
    print("1. Testing dataset loading...")
    try:
        from faithfulness.dataset import DatasetLoader
        loader = DatasetLoader(dataset_name="sample", num_docs=3, min_words=1000)
        docs = loader.load_dataset()
        print(f"   [OK] Loaded {len(docs)} documents")
    except Exception as e:
        print(f"   [OK] Dataset loading failed: {e}")
        return 1
    
    # Test 2: Claim decomposition validation
    print("2. Testing claim decomposition validation...")
    try:
        from faithfulness.claim_decomposition import Claim, validate_claims_atomic
        test_claims = [
            Claim(claim_id="test_1", claim_text="GDP grew by 2.5% in Q3 2023"),
            Claim(claim_id="test_2", claim_text="Unemployment fell to 4.1%"),
            Claim(claim_id="test_3", claim_text="The report discusses economic indicators")
        ]
        issues = validate_claims_atomic(test_claims)
        print(f"   [OK] Validation complete, found {sum(len(v) for v in issues.values())} issues")
        print(f"     Vague claims: {len(issues['vague'])}")
    except Exception as e:
        print(f"   [ERROR] Claim validation failed: {e}")
        return 1
    
    # Test 3: Retrieval system
    print("3. Testing retrieval system...")
    try:
        from faithfulness.retrieval import PassageRetriever
        retriever = PassageRetriever(chunk_size=100, top_k=2)
        test_text = "This is a test document. It contains multiple sentences for testing retrieval. The third sentence provides additional context."
        chunks, embeddings = retriever.index_document(test_text, "test_doc")
        print(f"   [OK] Created {len(chunks)} chunks with embeddings")
        
        test_claim = Claim(claim_id="test_claim", claim_text="This is a test claim about the document")
        passages = retriever.retrieve_passages(test_claim, chunks, embeddings, "test_doc")
        print(f"   [OK] Retrieved {len(passages)} passages for test claim")
    except Exception as e:
        print(f"   [ERROR] Retrieval system failed: {e}")
        return 1
    
    # Test 4: Jury aggregation
    print("4. Testing jury aggregation...")
    try:
        from faithfulness.jury_aggregation import calculate_fleiss_kappa, JuryVerdict
        test_verdicts = [
            JuryVerdict(
                majority_label="Supported",
                label_distribution={"Supported": 3, "Contradicted": 0, "Unverifiable": 0},
                confidence=0.9,
                individual_verdicts=[],
                agreement_score=1.0,
                jury_size=3
            ),
            JuryVerdict(
                majority_label="Supported",
                label_distribution={"Supported": 2, "Contradicted": 1, "Unverifiable": 0},
                confidence=0.7,
                individual_verdicts=[],
                agreement_score=0.67,
                jury_size=3
            )
        ]
        kappa = calculate_fleiss_kappa(test_verdicts)
        print(f"   [OK] Fleiss' kappa calculation: {kappa:.3f}")
    except Exception as e:
        print(f"   [ERROR] Jury aggregation failed: {e}")
        return 1
    
    # Test 5: Human validation metrics
    print("5. Testing human validation metrics...")
    try:
        from faithfulness.human_validation import HumanValidator, HumanAnnotation
        validator = HumanValidator()
        
        # Test Cohen's kappa calculation
        human_annotations = [
            HumanAnnotation(claim_id="c1", claim_text="Claim 1", human_label="Supported",
                          human_justification="Good", confidence=0.9),
            HumanAnnotation(claim_id="c2", claim_text="Claim 2", human_label="Supported",
                          human_justification="Good", confidence=0.9)
        ]
        
        jury_verdicts = {
            "c1": JuryVerdict(majority_label="Supported", label_distribution={},
                           confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3),
            "c2": JuryVerdict(majority_label="Supported", label_distribution={},
                           confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3)
        }
        
        result = validator.calculate_agreement(human_annotations, jury_verdicts)
        print(f"   [OK] Cohen's kappa: {result.cohen_kappa:.3f}")
        print(f"   [OK] Percent agreement: {result.percent_agreement:.2%}")
    except Exception as e:
        print(f"   [ERROR] Human validation metrics failed: {e}")
        return 1
    
    # Save test results
    os.makedirs(output_dir, exist_ok=True)
    test_results = {
        "test_date": "2026-10-05",
        "tests_completed": 5,
        "all_tests_passed": True,
        "components_tested": {
            "dataset_loading": "passed",
            "claim_validation": "passed",
            "retrieval_system": "passed",
            "jury_aggregation": "passed",
            "human_validation": "passed"
        }
    }
    
    with open(os.path.join(output_dir, "methodology_test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("METHODOLOGY TEST COMPLETE")
    print(f"{'='*60}")
    print("All components tested successfully!")
    print(f"Results saved to: {output_dir}/methodology_test_results.json")
    print(f"\nThe faithfulness evaluation system is ready for use.")
    print(f"To run full evaluation, configure API keys in .env file.")
    print(f"{'='*60}\n")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Simple faithfulness evaluation - Yale research level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test methodology without API keys
  python faithfulness/simple_runner.py --methodology-test
  
  # Quick test with 3 documents (requires API keys)
  python faithfulness/simple_runner.py --quick-test
  
  # Full evaluation with 10 documents (requires API keys)
  python faithfulness/simple_runner.py --num-docs 10
        """
    )
    
    parser.add_argument(
        "--methodology-test",
        action="store_true",
        help="Test methodology without API keys (recommended first)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 3 documents (requires API keys)"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=3,
        help="Number of documents to evaluate (default: 3)"
    )
    parser.add_argument(
        "--dataset",
        choices=["govreport", "arxiv", "sample"],
        default="sample",
        help="Dataset to use (default: sample for testing)"
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip summary generation (use existing if available)"
    )
    parser.add_argument(
        "--output-dir",
        default="faithfulness/data",
        help="Output directory (default: faithfulness/data)"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate an executive summary report from existing results"
    )
    parser.add_argument(
        "--free-mode",
        action="store_true",
        help="Use free-tier models (Groq Llama 3) instead of paid providers"
    )

    args = parser.parse_args()
    
    # If methodology test, run it directly
    if args.methodology_test:
        print(f"\n{'='*60}")
        print("FAITHFULNESS EVALUATION - YALE RESEARCH LEVEL")
        print(f"{'='*60}")
        print("Methodology Test Mode - No API Keys Required")
        return _run_methodology_test(args.output_dir)

    # If summarize results, run SummaryGenerator
    if args.summarize:
        print(f"\n{'='*60}")
        print("GENERATING EXECUTIVE SUMMARY")
        print(f"{'='*60}")
        try:
            from faithfulness.analysis import SummaryGenerator
            gen = SummaryGenerator(data_dir=args.output_dir)
            summary = gen.summarize()
            print(f"\n{summary}")
            print(f"\nSummary saved to: RESULTS_SUMMARY.md")
            return 0
        except Exception as e:
            print(f"Error generating summary: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # If quick test, use minimal settings
    if args.quick_test:
        args.num_docs = 3
        print("Running quick test with 3 documents...")
    
    print(f"\n{'='*60}")
    print("FAITHFULNESS EVALUATION - YALE RESEARCH LEVEL")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Documents: {args.num_docs}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Import after argument parsing
    try:
        from faithfulness.config import FaithfulnessConfig
        from faithfulness.pipeline import FaithfulnessPipeline
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("  pip install -r faithfulness/requirements.txt")
        return 1
    
    # Create configuration
    config = FaithfulnessConfig(
        dataset_name=args.dataset,
        num_documents=args.num_docs,
        min_word_count=1500,
        output_dir=args.output_dir
    )

    if args.free_mode:
        print("Free Mode active: Overriding models to Groq/NVIDIA Free Tiers")
        config.summary_models = ["llama3-8b-8192", "nvidia/llama-3.1-8b-instruct", "mixtral-8x7b-32768"]
        config.claim_decomposition_model = "llama3-70b-8192"
        config.faithfulness_judge_model = "llama3-70b-8192"
    else:
        config.summary_models = ["claude-3-haiku-20240307"]
    
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = FaithfulnessPipeline(config)
        
        # Run evaluation
        print("Starting evaluation...")
        results = pipeline.run_full_pipeline(
            generate_summaries=not args.skip_summaries,
            run_bias_check=False  # Skip bias check for simplicity
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Documents processed: {results['stages']['dataset']['num_documents']}")
        print(f"Claims decomposed: {results['stages']['claim_decomposition']['total_claims']}")
        print(f"Classifications: {results['stages']['classification']['total_classifications']}")
        print(f"Inter-judge agreement: {results['stages']['inter_judge_agreement']['percent_agreement']:.2%}")
        print(f"Fleiss' kappa: {results['stages']['inter_judge_agreement']['fleiss_kappa']:.3f}")
        print(f"\nResults saved to: {args.output_dir}")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
