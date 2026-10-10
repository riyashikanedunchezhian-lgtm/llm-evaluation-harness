#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faithfulness.jury_aggregation import JuryVerdict, calculate_fleiss_kappa
from faithfulness.config import FAITHFULNESS_LABELS

def test_true_random_distribution():
    """Test Fleiss' kappa with true random distribution."""
    # Simulate random assignments where agreement is at chance level
    # We'll create a case where observed agreement equals expected agreement

    # For true randomness with n=3, k=3, we want P_bar ≈ P_e
    # Let's construct a case that should give kappa ≈ 0

    # Example: 2 items, 3 raters each
    # Item 1: 2 raters say "Supported", 1 says "Contradicted"
    # Item 2: 2 raters say "Contradicted", 1 says "Supported"
    verdicts = [
        JuryVerdict(
            majority_label="Supported",
            label_distribution={"Supported": 2, "Contradicted": 1, "Unverifiable": 0},
            confidence=0.67,
            individual_verdicts=[],
            agreement_score=0.67,
            jury_size=3
        ),
        JuryVerdict(
            majority_label="Contradicted",
            label_distribution={"Supported": 1, "Contradicted": 2, "Unverifiable": 0},
            confidence=0.67,
            individual_verdicts=[],
            agreement_score=0.67,
            jury_size=3
        )
    ]

    print("Verdicts:")
    for i, v in enumerate(verdicts):
        print(f"  Verdict {i}: {v.label_distribution}")

    kappa = calculate_fleiss_kappa(verdicts)
    print(f"Calculated kappa: {kappa}")

    # Manual calculation
    jury_size = verdicts[0].jury_size  # n = 3
    num_items = len(verdicts)          # N = 2
    num_categories = len(FAITHFULNESS_LABELS)  # k = 3

    # Build rating matrix
    ratings = [[0] * num_categories for _ in range(num_items)]
    for i, verdict in enumerate(verdicts):
        for j, label in enumerate(FAITHFULNESS_LABELS):
            ratings[i][j] = verdict.label_distribution.get(label, 0)

    print(f"\nRating matrix:")
    for i, row in enumerate(ratings):
        print(f"  Item {i}: {row} (labels: {FAITHFULNESS_LABELS})")

    # Calculate Fleiss' kappa manually
    n = jury_size
    N = num_items
    k = num_categories
    total_ratings = N * n

    print(f"\nParameters: n={n}, N={N}, k={k}, total_ratings={total_ratings}")

    # Category proportions
    category_proportions = []
    for j in range(k):
        category_total = sum(ratings[i][j] for i in range(N))
        proportion = category_total / total_ratings
        category_proportions.append(proportion)
        print(f"  Category {FAITHFULNESS_LABELS[j]}: {category_total}/{total_ratings} = {proportion}")

    # Observed agreement
    item_agreements = []
    for i in range(N):
        sum_squared = sum(count ** 2 for count in ratings[i])
        item_agreement = (sum_squared - n) / (n * (n - 1))
        item_agreements.append(item_agreement)
        print(f"  Item {i}: sum_squared={sum_squared}, agreement=({sum_squared}-{n})/({n}*{n-1}) = {item_agreement}")

    P_bar = sum(item_agreements) / N
    print(f"  P_bar = {P_bar}")

    # Expected agreement
    P_e = sum(p ** 2 for p in category_proportions)
    print(f"  P_e = {P_e}")

    # Kappa
    if P_e == 1:
        kappa_result = 1.0
    else:
        kappa_result = (P_bar - P_e) / (1 - P_e)
    print(f"  kappa = ({P_bar} - {P_e}) / (1 - {P_e}) = {kappa_result}")

if __name__ == "__main__":
    test_true_random_distribution()