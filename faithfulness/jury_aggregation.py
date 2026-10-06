"""Jury aggregation for faithfulness evaluation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faithfulness.faithfulness_judge import FaithfulnessJudge, FaithfulnessVerdict
from faithfulness.config import FAITHFULNESS_LABELS

@dataclass
class JuryVerdict:
    """Aggregated verdict from jury of judges."""
    majority_label: str
    label_distribution: Dict[str, int]  # Count of each label
    confidence: float  # Based on agreement level
    individual_verdicts: List[FaithfulnessVerdict]
    agreement_score: float  # Percent agreement
    jury_size: int

class FaithfulnessJury:
    """Jury-style evaluation for faithfulness classification."""
    
    def __init__(self, model_client, judge_model: str = "claude-3-5-sonnet-20241022", jury_size: int = 3, parallel: bool = True):
        self.model_client = model_client
        self.judge = FaithfulnessJudge(model_client, model_id=judge_model)
        self.jury_size = jury_size
        self.parallel = parallel
    
    def evaluate_claim(self, claim, retrieved_passages: List, 
                      claim_id: str) -> JuryVerdict:
        """Evaluate a claim using a jury of judges."""
        if self.parallel:
            return self._evaluate_parallel(claim, retrieved_passages, claim_id)
        else:
            return self._evaluate_sequential(claim, retrieved_passages, claim_id)
    
    def _evaluate_sequential(self, claim, retrieved_passages: List, 
                           claim_id: str) -> JuryVerdict:
        """Evaluate sequentially."""
        verdicts = []
        
        for i in range(self.jury_size):
            verdict = self.judge.classify_claim(
                claim=claim,
                retrieved_passages=retrieved_passages,
                judge_id=i
            )
            verdicts.append(verdict)
        
        return self._aggregate_verdicts(verdicts)
    
    def _evaluate_parallel(self, claim, retrieved_passages: List, 
                         claim_id: str) -> JuryVerdict:
        """Evaluate in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        verdicts = []
        
        with ThreadPoolExecutor(max_workers=self.jury_size) as executor:
            future_to_judge_id = {
                executor.submit(
                    self.judge.classify_claim,
                    claim=claim,
                    retrieved_passages=retrieved_passages,
                    judge_id=i
                ): i for i in range(self.jury_size)
            }
            
            for future in as_completed(future_to_judge_id):
                judge_id = future_to_judge_id[future]
                try:
                    verdict = future.result()
                    verdicts.append(verdict)
                except Exception as e:
                    print(f"Judge {judge_id} generated an exception: {e}")
                    # Create fallback verdict
                    fallback = self.judge._create_fallback_verdict("", judge_id, str(e))
                    verdicts.append(fallback)
        
        # Sort by judge_id for consistency
        verdicts.sort(key=lambda v: v.judge_id)
        
        return self._aggregate_verdicts(verdicts)
    
    def _aggregate_verdicts(self, verdicts: List[FaithfulnessVerdict]) -> JuryVerdict:
        """Aggregate multiple verdicts using majority vote."""
        # Count label votes
        label_counts = Counter([v.label for v in verdicts])
        
        # Get majority label
        majority_label = label_counts.most_common(1)[0][0]
        
        # Calculate agreement score (percent agreement with majority)
        agreement_count = label_counts[majority_label]
        agreement_score = agreement_count / len(verdicts)
        
        # Confidence based on agreement level
        if agreement_score >= 0.8:
            confidence = 0.9
        elif agreement_score >= 0.6:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return JuryVerdict(
            majority_label=majority_label,
            label_distribution=dict(label_counts),
            confidence=confidence,
            individual_verdicts=verdicts,
            agreement_score=agreement_score,
            jury_size=len(verdicts)
        )

def calculate_inter_judge_agreement(verdicts: List[JuryVerdict]) -> Dict[str, float]:
    """Calculate inter-judge agreement metrics across all jury evaluations."""
    if not verdicts:
        return {"percent_agreement": 0.0, "fleiss_kappa": 0.0}
    
    # Calculate average percent agreement
    agreement_scores = [v.agreement_score for v in verdicts]
    avg_agreement = sum(agreement_scores) / len(agreement_scores)
    
    # Calculate Fleiss' kappa
    fleiss_kappa = calculate_fleiss_kappa(verdicts)
    
    return {
        "percent_agreement": avg_agreement,
        "fleiss_kappa": fleiss_kappa,
        "num_evaluations": len(verdicts)
    }

def calculate_fleiss_kappa(verdicts: List[JuryVerdict]) -> float:
    """Calculate Fleiss' kappa for multiple raters on categorical data."""
    if not verdicts:
        return 0.0
    
    # Build the rating matrix
    # rows = items (claims), columns = categories (labels)
    # entries = number of raters who assigned that category
    
    jury_size = verdicts[0].jury_size
    num_items = len(verdicts)
    num_categories = len(FAITHFULNESS_LABELS)
    
    # Initialize rating matrix
    ratings = [[0] * num_categories for _ in range(num_items)]
    
    # Fill in ratings
    for i, verdict in enumerate(verdicts):
        for j, label in enumerate(FAITHFULNESS_LABELS):
            ratings[i][j] = verdict.label_distribution.get(label, 0)
    
    # Calculate Fleiss' kappa
    # Step 1: Calculate proportion of assignments for each category
    n = jury_size  # number of raters per item
    N = num_items  # number of items
    k = num_categories  # number of categories
    
    # Total number of ratings
    total_ratings = N * n
    
    # Proportion of all ratings for each category
    category_proportions = []
    for j in range(k):
        category_total = sum(ratings[i][j] for i in range(N))
        category_proportions.append(category_total / total_ratings)
    
    # Step 2: Calculate observed agreement (P̄)
    # For each item, calculate the proportion of agreeing pairs
    item_agreements = []
    for i in range(N):
        # For this item, sum of squared ratings divided by n(n-1)
        sum_squared = sum(count ** 2 for count in ratings[i])
        item_agreement = (sum_squared - n) / (n * (n - 1))
        item_agreements.append(item_agreement)
    
    P_bar = sum(item_agreements) / N
    
    # Step 3: Calculate expected agreement (P̄e)
    P_e = sum(p ** 2 for p in category_proportions)
    
    # Step 4: Calculate kappa
    if P_e == 1:
        return 1.0  # Perfect agreement
    
    kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa
