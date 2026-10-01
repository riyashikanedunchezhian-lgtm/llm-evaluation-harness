"""Human validation pipeline for faithfulness evaluation."""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faithfulness.claim_decomposition import Claim
from faithfulness.jury_aggregation import JuryVerdict
from faithfulness.config import FAITHFULNESS_LABELS

@dataclass
class HumanAnnotation:
    """Human annotation for a claim."""
    claim_id: str
    claim_text: str
    human_label: str  # Supported, Contradicted, Unverifiable
    human_justification: str
    confidence: float
    annotated_by: str = "human"
    timestamp: str = ""

@dataclass
class AgreementResult:
    """Result of agreement calculation."""
    cohen_kappa: float
    percent_agreement: float
    agreement_matrix: Dict[str, Dict[str, int]]  # Confusion matrix
    disagreements: List[Dict[str, str]]  # Details of disagreements
    total_claims: int
    agreement_category: str  # "substantial", "moderate", "slight", "poor"

class HumanValidator:
    """Manage human validation and agreement calculation."""
    
    def __init__(self, output_dir: str = "faithfulness/data"):
        self.output_dir = output_dir
        self.annotations: List[HumanAnnotation] = []
    
    def create_annotation_template(self, claims: List[Claim], 
                                  jury_verdicts: Dict[str, JuryVerdict],
                                  output_path: str) -> None:
        """Create a template file for human annotation."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        template_data = []
        for claim in claims:
            verdict = jury_verdicts.get(claim.claim_id)
            
            template_data.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "automated_label": verdict.majority_label if verdict else "Unknown",
                "automated_confidence": verdict.confidence if verdict else 0.0,
                "automated_justification": verdict.individual_verdicts[0].justification if verdict and verdict.individual_verdicts else "",
                "human_label": "",  # To be filled by human
                "human_justification": "",  # To be filled by human
                "human_confidence": 0.0,  # To be filled by human
                "disagreement_reason": ""  # To be filled if disagrees with automated
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2)
        
        print(f"Created annotation template with {len(template_data)} claims: {output_path}")
    
    def load_annotations(self, input_path: str) -> List[HumanAnnotation]:
        """Load human annotations from file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = [
            HumanAnnotation(
                claim_id=item["claim_id"],
                claim_text=item["claim_text"],
                human_label=item["human_label"],
                human_justification=item["human_justification"],
                confidence=item.get("human_confidence", 0.0),
                annotated_by="human",
                timestamp=item.get("timestamp", "")
            )
            for item in data
            if item.get("human_label")  # Only load completed annotations
        ]
        
        print(f"Loaded {len(self.annotations)} human annotations from {input_path}")
        return self.annotations
    
    def calculate_agreement(self, human_annotations: List[HumanAnnotation],
                          jury_verdicts: Dict[str, JuryVerdict]) -> AgreementResult:
        """Calculate Cohen's kappa between human and automated annotations."""
        
        # Build paired annotations
        paired = []
        for annotation in human_annotations:
            verdict = jury_verdicts.get(annotation.claim_id)
            if verdict:
                paired.append({
                    "claim_id": annotation.claim_id,
                    "human_label": annotation.human_label,
                    "auto_label": verdict.majority_label,
                    "human_justification": annotation.human_justification,
                    "auto_justification": verdict.individual_verdicts[0].justification if verdict.individual_verdicts else ""
                })
        
        if not paired:
            return AgreementResult(
                cohen_kappa=0.0,
                percent_agreement=0.0,
                agreement_matrix={},
                disagreements=[],
                total_claims=0,
                agreement_category="no_data"
            )
        
        # Calculate percent agreement
        agreements = sum(1 for p in paired if p["human_label"] == p["auto_label"])
        percent_agreement = agreements / len(paired)
        
        # Build confusion matrix
        agreement_matrix = {}
        for human_label in FAITHFULNESS_LABELS:
            agreement_matrix[human_label] = {}
            for auto_label in FAITHFULNESS_LABELS:
                agreement_matrix[human_label][auto_label] = 0
        
        for p in paired:
            agreement_matrix[p["human_label"]][p["auto_label"]] += 1
        
        # Calculate Cohen's kappa
        cohen_kappa = self._calculate_cohen_kappa(paired)
        
        # Identify disagreements
        disagreements = []
        for p in paired:
            if p["human_label"] != p["auto_label"]:
                disagreements.append({
                    "claim_id": p["claim_id"],
                    "human_label": p["human_label"],
                    "auto_label": p["auto_label"],
                    "human_justification": p["human_justification"],
                    "auto_justification": p["auto_justification"]
                })
        
        # Categorize agreement level
        agreement_category = self._categorize_agreement(cohen_kappa)
        
        return AgreementResult(
            cohen_kappa=cohen_kappa,
            percent_agreement=percent_agreement,
            agreement_matrix=agreement_matrix,
            disagreements=disagreements,
            total_claims=len(paired),
            agreement_category=agreement_category
        )
    
    def _calculate_cohen_kappa(self, paired_annotations: List[Dict]) -> float:
        """Calculate Cohen's kappa for two raters."""
        if not paired_annotations:
            return 0.0
        
        n = len(paired_annotations)  # Total number of items
        k = len(FAITHFULNESS_LABELS)  # Number of categories
        
        # Build confusion matrix
        confusion_matrix = {}
        for human_label in FAITHFULNESS_LABELS:
            confusion_matrix[human_label] = {}
            for auto_label in FAITHFULNESS_LABELS:
                confusion_matrix[human_label][auto_label] = 0
        
        for p in paired_annotations:
            confusion_matrix[p["human_label"]][p["auto_label"]] += 1
        
        # Calculate observed agreement (Po)
        observed_agreement = sum(
            confusion_matrix[label][label] 
            for label in FAITHFULNESS_LABELS
        ) / n
        
        # Calculate expected agreement (Pe)
        # First, calculate marginal probabilities
        human_marginals = {}
        auto_marginals = {}
        
        for label in FAITHFULNESS_LABELS:
            human_marginals[label] = sum(confusion_matrix[label].values()) / n
            auto_marginals[label] = sum(confusion_matrix[h_label][label] for h_label in FAITHFULNESS_LABELS) / n
        
        # Expected agreement
        expected_agreement = sum(
            human_marginals[label] * auto_marginals[label]
            for label in FAITHFULNESS_LABELS
        )
        
        # Calculate kappa
        if expected_agreement == 1:
            return 1.0  # Perfect agreement
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        return kappa
    
    def _categorize_agreement(self, kappa: float) -> str:
        """Categorize agreement level based on kappa score."""
        if kappa >= 0.81:
            return "almost_perfect"
        elif kappa >= 0.61:
            return "substantial"
        elif kappa >= 0.41:
            return "moderate"
        elif kappa >= 0.21:
            return "slight"
        else:
            return "poor"
    
    def categorize_disagreements(self, disagreements: List[Dict]) -> Dict[str, List[str]]:
        """Categorize the reasons for disagreements."""
        categories = {
            "ambiguous_claim": [],
            "retrieval_failure": [],
            "judge_strictness": [],
            "judge_leniency": [],
            "human_error": [],
            "other": []
        }
        
        for disagreement in disagreements:
            reason = disagreement.get("disagreement_reason", "").lower()
            claim_id = disagreement["claim_id"]
            
            if "ambiguous" in reason or "unclear" in reason:
                categories["ambiguous_claim"].append(claim_id)
            elif "retrieval" in reason or "passage" in reason or "missing" in reason:
                categories["retrieval_failure"].append(claim_id)
            elif "strict" in reason or "harsh" in reason:
                categories["judge_strictness"].append(claim_id)
            elif "lenient" in reason or "generous" in reason:
                categories["judge_leniency"].append(claim_id)
            elif "human" in reason and "error" in reason:
                categories["human_error"].append(claim_id)
            else:
                categories["other"].append(claim_id)
        
        return categories
    
    def generate_validation_report(self, agreement_result: AgreementResult,
                                   output_path: str) -> None:
        """Generate a human validation report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            "summary": {
                "total_claims": agreement_result.total_claims,
                "cohen_kappa": agreement_result.cohen_kappa,
                "percent_agreement": agreement_result.percent_agreement,
                "agreement_category": agreement_result.agreement_category
            },
            "agreement_matrix": agreement_result.agreement_matrix,
            "disagreement_count": len(agreement_result.disagreements),
            "disagreements": agreement_result.disagreements
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Generated validation report: {output_path}")
