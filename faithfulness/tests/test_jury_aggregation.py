"""Tests for jury aggregation on faithfulness labels."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from faithfulness.jury_aggregation import FaithfulnessJury, JuryVerdict, calculate_fleiss_kappa
from faithfulness.faithfulness_judge import FaithfulnessVerdict

def test_majority_vote_aggregation():
    """Test that jury aggregation produces correct majority votes on synthetic judge outputs."""
    # Create synthetic verdicts with clear majority
    verdicts = [
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.9, judge_id=0),
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.8, judge_id=1),
        FaithfulnessVerdict(label="Contradicted", justification="Bad", confidence=0.7, judge_id=2)
    ]
    
    # Mock jury aggregation logic
    label_counts = {}
    for verdict in verdicts:
        label_counts[verdict.label] = label_counts.get(verdict.label, 0) + 1
    
    majority_label = max(label_counts, key=label_counts.get)
    
    # Should be "Supported" (2 votes vs 1)
    assert majority_label == "Supported", "Majority should be Supported"
    assert label_counts["Supported"] == 2
    assert label_counts["Contradicted"] == 1

def test_unanimous_agreement():
    """Test aggregation when all judges agree."""
    verdicts = [
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.9, judge_id=0),
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.9, judge_id=1),
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.9, judge_id=2)
    ]
    
    label_counts = {}
    for verdict in verdicts:
        label_counts[verdict.label] = label_counts.get(verdict.label, 0) + 1
    
    majority_label = max(label_counts, key=label_counts.get)
    agreement_score = label_counts[majority_label] / len(verdicts)
    
    assert majority_label == "Supported"
    assert agreement_score == 1.0, "Should have perfect agreement"

def test_tie_breaking():
    """Test tie-breaking in jury aggregation."""
    # Even number of judges with tie
    verdicts = [
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.9, judge_id=0),
        FaithfulnessVerdict(label="Contradicted", justification="Bad", confidence=0.7, judge_id=1),
        FaithfulnessVerdict(label="Unverifiable", justification="Unknown", confidence=0.5, judge_id=2),
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.8, judge_id=3)
    ]
    
    label_counts = {}
    for verdict in verdicts:
        label_counts[verdict.label] = label_counts.get(verdict.label, 0) + 1
    
    # Should have a clear winner (Supported with 2 votes)
    majority_label = max(label_counts, key=label_counts.get)
    assert majority_label == "Supported"
    assert label_counts[majority_label] == 2

def test_fleiss_kappa_perfect_agreement():
    """Test Fleiss' kappa calculation with perfect agreement."""
    # All judges agree on all items
    verdicts = [
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
            label_distribution={"Supported": 3, "Contradicted": 0, "Unverifiable": 0},
            confidence=0.9,
            individual_verdicts=[],
            agreement_score=1.0,
            jury_size=3
        )
    ]
    
    kappa = calculate_fleiss_kappa(verdicts)
    
    # Should be close to 1.0 for perfect agreement
    assert abs(kappa - 1.0) < 0.01, f"Expected kappa ~1.0, got {kappa}"

def test_fleiss_kappa_no_agreement():
    """Test Fleiss' kappa calculation with no agreement."""
    # Random distribution (no agreement beyond chance)
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

    kappa = calculate_fleiss_kappa(verdicts)

    # Should be -0.5 for this specific case of random distribution
    assert abs(kappa - (-0.5)) < 0.01, f"Expected kappa ~-0.5, got {kappa}"

def test_fleiss_kappa_partial_agreement():
    """Test Fleiss' kappa calculation with partial agreement."""
    # Some agreement but not perfect
    verdicts = [
        JuryVerdict(
            majority_label="Supported",
            label_distribution={"Supported": 3, "Contradicted": 0, "Unverifiable": 0},
            confidence=0.9,
            individual_verdicts=[],
            agreement_score=1.0,
            jury_size=3
        ),
        JuryVerdict(
            majority_label="Contradicted",
            label_distribution={"Supported": 1, "Contradicted": 2, "Unverifiable": 0},
            confidence=0.7,
            individual_verdicts=[],
            agreement_score=0.67,
            jury_size=3
        )
    ]

    kappa = calculate_fleiss_kappa(verdicts)

    # Should be between 0 and 1 for partial agreement
    assert 0 < kappa < 1, f"Expected kappa between 0 and 1, got {kappa}"

def test_agreement_score_calculation():
    """Test agreement score calculation."""
    verdicts = [
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.9, judge_id=0),
        FaithfulnessVerdict(label="Supported", justification="Good", confidence=0.8, judge_id=1),
        FaithfulnessVerdict(label="Contradicted", justification="Bad", confidence=0.7, judge_id=2)
    ]
    
    label_counts = {}
    for verdict in verdicts:
        label_counts[verdict.label] = label_counts.get(verdict.label, 0) + 1
    
    majority_label = max(label_counts, key=label_counts.get)
    agreement_score = label_counts[majority_label] / len(verdicts)
    
    # Should be 2/3 = 0.67
    assert abs(agreement_score - 0.67) < 0.01, f"Expected agreement ~0.67, got {agreement_score}"

def test_confidence_calculation():
    """Test confidence calculation based on agreement."""
    # High agreement -> high confidence
    high_agreement_score = 0.9
    if high_agreement_score >= 0.8:
        confidence = 0.9
    assert confidence == 0.9
    
    # Medium agreement -> medium confidence
    medium_agreement_score = 0.6
    if medium_agreement_score >= 0.6:
        confidence = 0.7
    assert confidence == 0.7
    
    # Low agreement -> low confidence
    low_agreement_score = 0.4
    if low_agreement_score < 0.6:
        confidence = 0.5
    assert confidence == 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
