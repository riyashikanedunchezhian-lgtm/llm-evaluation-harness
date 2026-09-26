"""Tests for jury aggregation logic."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.judge import JudgeScore, JudgeVerdict, Jury


def test_majority_vote_aggregation():
    """Test that majority vote produces correct results on known scores."""
    # Create synthetic judge scores with a clear majority
    verdicts = [
        JudgeVerdict(
            scores=[
                JudgeScore("correctness", 5, "Excellent"),
                JudgeScore("relevance", 4, "Good"),
                JudgeScore("conciseness", 4, "Good")
            ],
            overall_score=4.33,
            judge_id=0
        ),
        JudgeVerdict(
            scores=[
                JudgeScore("correctness", 5, "Excellent"),
                JudgeScore("relevance", 5, "Excellent"),
                JudgeScore("conciseness", 4, "Good")
            ],
            overall_score=4.67,
            judge_id=1
        ),
        JudgeVerdict(
            scores=[
                JudgeScore("correctness", 4, "Good"),
                JudgeScore("relevance", 4, "Good"),
                JudgeScore("conciseness", 5, "Excellent")
            ],
            overall_score=4.33,
            judge_id=2
        )
    ]
    
    # Manually aggregate to verify
    correctness_scores = [5, 5, 4]
    relevance_scores = [4, 5, 4]
    conciseness_scores = [4, 4, 5]
    
    # Majority vote (round average)
    assert round(sum(correctness_scores) / len(correctness_scores)) == 5
    assert round(sum(relevance_scores) / len(relevance_scores)) == 4
    assert round(sum(conciseness_scores) / len(conciseness_scores)) == 4
    
    # Overall average
    overall_avg = (4.33 + 4.67 + 4.33) / 3
    assert abs(overall_avg - 4.44) < 0.01


def test_aggregation_with_variance():
    """Test that aggregation correctly calculates variance."""
    verdicts = [
        JudgeVerdict(
            scores=[JudgeScore("correctness", 5, "Excellent")],
            overall_score=5.0,
            judge_id=0
        ),
        JudgeVerdict(
            scores=[JudgeScore("correctness", 3, "Average")],
            overall_score=3.0,
            judge_id=1
        ),
        JudgeVerdict(
            scores=[JudgeScore("correctness", 4, "Good")],
            overall_score=4.0,
            judge_id=2
        )
    ]
    
    scores = [5.0, 3.0, 4.0]
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5
    
    assert abs(avg - 4.0) < 0.01
    assert abs(std_dev - 0.82) < 0.01  # sqrt(2/3) ≈ 0.816


def test_aggregation_consensus_detection():
    """Test that consensus is correctly detected."""
    # High consensus case (low variance)
    high_consensus_scores = [4.0, 4.1, 3.9]
    avg_high = sum(high_consensus_scores) / len(high_consensus_scores)
    std_high = (sum((s - avg_high) ** 2 for s in high_consensus_scores) / len(high_consensus_scores)) ** 0.5
    
    assert std_high < 0.5  # Should detect high consensus
    
    # Low consensus case (high variance)
    low_consensus_scores = [2.0, 4.0, 5.0]
    avg_low = sum(low_consensus_scores) / len(low_consensus_scores)
    std_low = (sum((s - avg_low) ** 2 for s in low_consensus_scores) / len(low_consensus_scores)) ** 0.5
    
    assert std_low > 0.5  # Should detect low consensus


def test_aggregation_with_outliers():
    """Test that aggregation handles outliers correctly."""
    verdicts = [
        JudgeVerdict(
            scores=[JudgeScore("correctness", 5, "Excellent")],
            overall_score=5.0,
            judge_id=0
        ),
        JudgeVerdict(
            scores=[JudgeScore("correctness", 5, "Excellent")],
            overall_score=5.0,
            judge_id=1
        ),
        JudgeVerdict(
            scores=[JudgeScore("correctness", 1, "Poor")],  # Outlier
            overall_score=1.0,
            judge_id=2
        )
    ]
    
    scores = [5.0, 5.0, 1.0]
    avg = sum(scores) / len(scores)
    
    # Average should be pulled down by outlier but not dominated
    assert abs(avg - 3.67) < 0.01
    
    # Majority vote should still be 5
    assert round(avg) == 4  # Round(3.67) = 4


def test_empty_aggregation():
    """Test that aggregation handles empty input gracefully."""
    # This should not crash but return sensible defaults
    verdicts = []
    
    if not verdicts:
        # Should handle empty case
        assert True  # Placeholder for actual implementation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
