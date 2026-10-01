"""Position bias checking for faithfulness evaluation."""

from typing import List, Dict
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faithfulness.faithfulness_judge import FaithfulnessJudge, FaithfulnessVerdict
from faithfulness.claim_decomposition import Claim
from faithfulness.retrieval import RetrievedPassage

@dataclass
class BiasCheckResult:
    """Result of position bias check."""
    claim_first_label: str
    source_first_label: str
    bias_detected: bool
    label_changed: bool
    confidence_delta: float
    justification_delta: str

class FaithfulnessBiasChecker:
    """Check for position bias in faithfulness evaluation."""
    
    def __init__(self, model_client):
        self.model_client = model_client
        self.judge = FaithfulnessJudge(model_client)
    
    def check_claim_source_bias(
        self,
        claim: Claim,
        retrieved_passages: List[RetrievedPassage]
    ) -> BiasCheckResult:
        """Check if order (claim-then-source vs source-then-claim) affects verdict."""
        
        # Evaluate with claim first, then source
        claim_first_verdict = self._evaluate_claim_first(claim, retrieved_passages)
        
        # Evaluate with source first, then claim
        source_first_verdict = self._evaluate_source_first(claim, retrieved_passages)
        
        # Check for bias
        label_changed = claim_first_verdict.label != source_first_verdict.label
        confidence_delta = abs(claim_first_verdict.confidence - source_first_verdict.confidence)
        
        # Bias detected if label changed or significant confidence change
        bias_detected = label_changed or confidence_delta > 0.2
        
        justification_delta = self._compare_justifications(
            claim_first_verdict.justification,
            source_first_verdict.justification
        )
        
        return BiasCheckResult(
            claim_first_label=claim_first_verdict.label,
            source_first_label=source_first_verdict.label,
            bias_detected=bias_detected,
            label_changed=label_changed,
            confidence_delta=confidence_delta,
            justification_delta=justification_delta
        )
    
    def _evaluate_claim_first(self, claim: Claim, 
                            retrieved_passages: List[RetrievedPassage]) -> FaithfulnessVerdict:
        """Evaluate with claim presented first (standard order)."""
        # This is the standard evaluation order
        return self.judge.classify_claim(claim, retrieved_passages, judge_id=0)
    
    def _evaluate_source_first(self, claim: Claim, 
                             retrieved_passages: List[RetrievedPassage]) -> FaithfulnessVerdict:
        """Evaluate with source presented first (swapped order)."""
        # Build prompt with source first
        system_prompt = self.judge._build_system_prompt()
        user_prompt = self._build_source_first_prompt(claim, retrieved_passages)
        
        response = self.model_client.call_model(
            model_id=self.judge.model_id,
            provider=self.judge.model_config.provider,
            prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
            system_prompt=system_prompt
        )
        
        return self.judge._parse_verdict(response.content, judge_id=1)
    
    def _build_source_first_prompt(self, claim: Claim, 
                                  retrieved_passages: List[RetrievedPassage]) -> str:
        """Build prompt with source passages first, then claim."""
        user_prompt = """**Source Passages:**
"""
        
        for i, passage in enumerate(retrieved_passages, 1):
            user_prompt += f"""
Passage {i} (relevance: {passage.relevance_score:.3f}):
{passage.passage}
"""
        
        user_prompt += f"""
**Claim to Evaluate:**
{claim.claim_text}

Based on the source passages above, classify the claim as Supported, Contradicted, or Unverifiable.
Provide your classification as a JSON object with label, justification, and confidence."""
        
        return user_prompt
    
    def _compare_justifications(self, just1: str, just2: str) -> str:
        """Compare two justifications and describe the difference."""
        if just1 == just2:
            return "Justifications are identical"
        
        # Simple comparison based on length difference
        len_diff = abs(len(just1) - len(just2))
        if len_diff < 50:
            return "Justifications are very similar"
        elif len_diff < 150:
            return "Justifications differ moderately in detail"
        else:
            return "Justifications differ significantly in detail"

def batch_bias_check(claims: List, retrieved_passages_dict: Dict, 
                    bias_checker: FaithfulnessBiasChecker) -> Dict[str, BiasCheckResult]:
    """Run bias checks on multiple claims."""
    results = {}
    
    for claim in claims:
        passages = retrieved_passages_dict.get(claim.claim_id, [])
        if passages:
            result = bias_checker.check_claim_source_bias(claim, passages)
            results[claim.claim_id] = result
    
    return results

def analyze_bias_results(bias_results: Dict[str, BiasCheckResult]) -> Dict[str, any]:
    """Analyze bias check results across multiple claims."""
    if not bias_results:
        return {"total_checks": 0, "bias_detected_count": 0, "bias_rate": 0.0}
    
    total_checks = len(bias_results)
    bias_detected_count = sum(1 for r in bias_results.values() if r.bias_detected)
    label_changed_count = sum(1 for r in bias_results.values() if r.label_changed)
    
    # Average confidence delta
    confidence_deltas = [r.confidence_delta for r in bias_results.values()]
    avg_confidence_delta = sum(confidence_deltas) / len(confidence_deltas) if confidence_deltas else 0.0
    
    return {
        "total_checks": total_checks,
        "bias_detected_count": bias_detected_count,
        "bias_rate": bias_detected_count / total_checks if total_checks > 0 else 0.0,
        "label_changed_count": label_changed_count,
        "label_change_rate": label_changed_count / total_checks if total_checks > 0 else 0.0,
        "avg_confidence_delta": avg_confidence_delta
    }
