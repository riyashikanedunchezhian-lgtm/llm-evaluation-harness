#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faithfulness.jury_aggregation import JuryVerdict, calculate_fleiss_kappa
from faithfulness.config import FAITHFULNESS_LABELS

def debug_fleiss_kappa():
    """Debug the Fleiss' kappa calculation."""
    # No agreement case (uniform distribution across 3 categories)
    verdicts = [
        JuryVerdict(
            majority_label="Supported",
            label_distribution={"Supported": 1, "Contradicted": 1, "Unverifiable": 1},
            confidence=0.5,
            individual_verdicts=[],
            agreement_score=0.33,
            jury_size=3
        ),
        JuryVerdict(
            majority_label="Contradicted",
            label_distribution={"Supported": 1, "Contradicted": 1, "Unverifiable": 1},
            confidence=0.5,
            individual_verdicts=[],
            agreement_score=0.33,
            jury_size=3
        )
    ]

    print("Verdicts:")
    for i, v in enumerate(verdicts):
        print(f"  Verdict {i}: {v.label_distribution}")

    print(f"FAITHFULNESS_LABELS: {FAITHFULNESS_LABELS}")

    kappa = calculate_fleiss_kappa(verdicts)
    print(f"Calculated kappa: {kappa}")

    # Let's manually trace through the calculation
    jury_size = verdicts[0].jury_size
    num_items = len(verdicts)
    num_categories = len(FAITHFULNESS_LABELS)

    print(f"\nParameters:")
    print(f"  jury_size (n): {jury_size}")
    print(f"  num_items (N): {num_items}")
    print(f"  num_categories (k): {num_categories}")

    # Build the rating matrix
    ratings = [[0] * num_categories for _ in range(num_items)]

    print(f"\nRating matrix:")
    for i, verdict in enumerate(verdicts):
        for j, label in enumerate(FAITHFULNESS_LABELS):
            ratings[i][j] = verdict.label_distribution.get(label, 0)
            print(f"  ratings[{i}][{j}] ({label}): {ratings[i][j]}")

    # Calculate Fleiss' kappa
    # Step 1: Calculate proportion of assignments for each category
    n = jury_size  # number of raters per item
    N = num_items  # number of items
    k = num_categories  # number of categories

    print(f"\nStep 1: Category proportions")
    # Total number of ratings
    total_ratings = N * n
    print(f"  Total ratings: {total_ratings}")

    # Proportion of all ratings for each category
    category_proportions = []
    for j in range(k):
        category_total = sum(ratings[i][j] for i in range(N))
        proportion = category_total / total_ratings
        category_proportions.append(proportion)
        print(f"  Category {FAITHFULNESS_LABELS[j]}: {category_total}/{total_ratings} = {proportion}")

    # Step 2: Calculate observed agreement (P̄)
    print(f"\nStep 2: Observed agreement")
    # For each item, calculate the proportion of agreeing pairs
    item_agreements = []
    for i in range(N):
        # For this item, sum of squared ratings divided by n(n-1)
        sum_squared = sum(count ** 2 for count in ratings[i])
        item_agreement = (sum_squared - n) / (n * (n - 1))
        item_agreements.append(item_agreement)
        print(f"  Item {i}: sum_squared={sum_squared}, item_agreement=({sum_squared} - {n}) / ({n} * {n - 1}) = {item_agreement}")

    P_bar = sum(item_agreements) / N
    print(f"  P_bar = ({' + '.join([str(a) for a in item_agreements])}) / {N} = {P_bar}")

    # Step 3: Calculate expected agreement (P_e)
    print(f"\nStep 3: Expected agreement")
    P_e = sum(p ** 2 for p in category_proportions)
    print(f"  P_e = {' + '.join([f'{p}^2' for p in category_proportions])} = {P_e}")

    # Step 4: Calculate kappa
    print(f"\nStep 4: Kappa calculation")
    if P_e == 1:
        kappa_result = 1.0  # Perfect agreement
        print(f"  P_e == 1, so kappa = 1.0")
    else:
        kappa_result = (P_bar - P_e) / (1 - P_e)
        print(f"  kappa = (P_bar - P_e) / (1 - P_e) = ({P_bar} - {P_e}) / (1 - {P_e}) = {kappa_result}")

    print(f"\nResult: {kappa_result}")

if __name__ == "__main__":
    debug_fleiss_kappa()