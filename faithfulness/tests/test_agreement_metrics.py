"""Tests for agreement metrics (Cohen's kappa)."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from faithfulness.human_validation import HumanValidator, HumanAnnotation

def test_cohen_kappa_perfect_agreement():
    """Test Cohen's kappa calculation with perfect agreement."""
    validator = HumanValidator()
    
    # Perfect agreement: human and automated always agree
    human_annotations = [
        HumanAnnotation(claim_id="c1", claim_text="Claim 1", human_label="Supported", 
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c2", claim_text="Claim 2", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c3", claim_text="Claim 3", human_label="Contradicted",
                      human_justification="Bad", confidence=0.8)
    ]
    
    # Mock jury verdicts that match human annotations
    from faithfulness.jury_aggregation import JuryVerdict
    jury_verdicts = {
        "c1": JuryVerdict(majority_label="Supported", label_distribution={}, 
                         confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3),
        "c2": JuryVerdict(majority_label="Supported", label_distribution={},
                         confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3),
        "c3": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.8, individual_verdicts=[], agreement_score=1.0, jury_size=3)
    }
    
    result = validator.calculate_agreement(human_annotations, jury_verdicts)
    
    # Should have perfect agreement
    assert result.cohen_kappa == 1.0, f"Expected kappa=1.0, got {result.cohen_kappa}"
    assert result.percent_agreement == 1.0, f"Expected 100% agreement, got {result.percent_agreement}"
    assert result.agreement_category == "almost_perfect"

def test_cohen_kappa_no_agreement():
    """Test Cohen's kappa calculation with no agreement."""
    validator = HumanValidator()
    
    # No agreement: human and automated always disagree
    human_annotations = [
        HumanAnnotation(claim_id="c1", claim_text="Claim 1", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c2", claim_text="Claim 2", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c3", claim_text="Claim 3", human_label="Supported",
                      human_justification="Good", confidence=0.9)
    ]
    
    # Mock jury verdicts that disagree
    from faithfulness.jury_aggregation import JuryVerdict
    jury_verdicts = {
        "c1": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.7, individual_verdicts=[], agreement_score=0.67, jury_size=3),
        "c2": JuryVerdict(majority_label="Unverifiable", label_distribution={},
                         confidence=0.5, individual_verdicts=[], agreement_score=0.33, jury_size=3),
        "c3": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.7, individual_verdicts=[], agreement_score=0.67, jury_size=3)
    }
    
    result = validator.calculate_agreement(human_annotations, jury_verdicts)
    
    # Should have low agreement
    assert result.cohen_kappa < 0.4, f"Expected low kappa, got {result.cohen_kappa}"
    assert result.percent_agreement == 0.0, f"Expected 0% agreement, got {result.percent_agreement}"

def test_cohen_kappa_partial_agreement():
    """Test Cohen's kappa calculation with partial agreement."""
    validator = HumanValidator()

    # Partial agreement: some matches, some disagreements
    human_annotations = [
        HumanAnnotation(claim_id="c1", claim_text="Claim 1", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c2", claim_text="Claim 2", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c3", claim_text="Claim 3", human_label="Contradicted",
                      human_justification="Bad", confidence=0.8),
        HumanAnnotation(claim_id="c4", claim_text="Claim 4", human_label="Unverifiable",
                      human_justification="Unknown", confidence=0.5)
    ]

    # Mock jury verdicts with partial agreement
    from faithfulness.jury_aggregation import JuryVerdict
    jury_verdicts = {
        "c1": JuryVerdict(majority_label="Supported", label_distribution={},
                         confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3),
        "c2": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.7, individual_verdicts=[], agreement_score=0.67, jury_size=3),
        "c3": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.8, individual_verdicts=[], agreement_score=1.0, jury_size=3),
        "c4": JuryVerdict(majority_label="Unverifiable", label_distribution={},
                         confidence=0.5, individual_verdicts=[], agreement_score=1.0, jury_size=3)
    }

    result = validator.calculate_agreement(human_annotations, jury_verdicts)

    # Should have partial agreement (3 out of 4 = 75%)
    assert result.percent_agreement == 0.75, f"Expected 75% agreement, got {result.percent_agreement}"
    # Kappa should be moderate
    assert 0.0 < result.cohen_kappa < 0.8, f"Expected moderate kappa, got {result.cohen_kappa}"

def test_cohen_kappa_hand_computed():
    """Test Cohen's kappa against a hand-computed example."""
    validator = HumanValidator()
    
    # Hand-computed example:
    # Human: [S, S, C, C]
    # Auto:  [S, C, S, C]
    # Agreement matrix:
    #           S   C   U
    #        S  1   1   0
    #        C  1   1   0
    #        U  0   0   0
    
    human_annotations = [
        HumanAnnotation(claim_id="c1", claim_text="Claim 1", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c2", claim_text="Claim 2", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c3", claim_text="Claim 3", human_label="Contradicted",
                      human_justification="Bad", confidence=0.8),
        HumanAnnotation(claim_id="c4", claim_text="Claim 4", human_label="Contradicted",
                      human_justification="Bad", confidence=0.8)
    ]
    
    from faithfulness.jury_aggregation import JuryVerdict
    jury_verdicts = {
        "c1": JuryVerdict(majority_label="Supported", label_distribution={},
                         confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3),
        "c2": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.7, individual_verdicts=[], agreement_score=0.67, jury_size=3),
        "c3": JuryVerdict(majority_label="Supported", label_distribution={},
                         confidence=0.9, individual_verdicts=[], agreement_score=0.67, jury_size=3),
        "c4": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.8, individual_verdicts=[], agreement_score=1.0, jury_size=3)
    }
    
    result = validator.calculate_agreement(human_annotations, jury_verdicts)
    
    # Hand calculation:
    # Po (observed agreement) = 2/4 = 0.5
    # Human marginals: S=0.5, C=0.5, U=0.0
    # Auto marginals: S=0.5, C=0.5, U=0.0
    # Pe (expected agreement) = 0.5*0.5 + 0.5*0.5 + 0.0*0.0 = 0.5
    # Kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0
    
    assert abs(result.cohen_kappa - 0.0) < 0.01, f"Expected kappa ~0.0, got {result.cohen_kappa}"
    assert result.percent_agreement == 0.5

def test_agreement_categorization():
    """Test agreement categorization based on kappa."""
    validator = HumanValidator()
    
    # Test different kappa ranges
    test_cases = [
        (0.9, "almost_perfect"),
        (0.7, "substantial"),
        (0.5, "moderate"),
        (0.3, "slight"),
        (0.1, "poor")
    ]
    
    for kappa, expected_category in test_cases:
        category = validator._categorize_agreement(kappa)
        assert category == expected_category, f"Expected {expected_category} for kappa={kappa}, got {category}"

def test_disagreement_identification():
    """Test that disagreements are correctly identified."""
    validator = HumanValidator()
    
    human_annotations = [
        HumanAnnotation(claim_id="c1", claim_text="Claim 1", human_label="Supported",
                      human_justification="Good", confidence=0.9),
        HumanAnnotation(claim_id="c2", claim_text="Claim 2", human_label="Supported",
                      human_justification="Good", confidence=0.9)
    ]
    
    from faithfulness.jury_aggregation import JuryVerdict
    jury_verdicts = {
        "c1": JuryVerdict(majority_label="Supported", label_distribution={},
                         confidence=0.9, individual_verdicts=[], agreement_score=1.0, jury_size=3),
        "c2": JuryVerdict(majority_label="Contradicted", label_distribution={},
                         confidence=0.7, individual_verdicts=[], agreement_score=0.67, jury_size=3)
    }
    
    result = validator.calculate_agreement(human_annotations, jury_verdicts)
    
    # Should identify 1 disagreement
    assert len(result.disagreements) == 1, f"Expected 1 disagreement, got {len(result.disagreements)}"
    assert result.disagreements[0]["claim_id"] == "c2"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
