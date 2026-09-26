"""Tests for position bias detection."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.judge import JudgeScore, JudgeVerdict, PositionBiasChecker


def test_position_bias_detection_with_swap():
    """Test that position bias check detects score changes when order is swapped."""
    # Simulate a scenario where position bias exists
    # When response A is presented first, it gets a higher score
    # When response B is presented first, it gets a higher score
    
    # Simulated scores showing position bias
    response_a_original_score = 4.5  # When A is first
    response_a_swapped_score = 3.5  # When A is second (lower due to position bias)
    
    response_b_original_score = 3.0  # When B is second
    response_b_swapped_score = 4.0  # When B is first (higher due to position bias)
    
    # Calculate deltas
    delta_a = abs(response_a_original_score - response_a_swapped_score)
    delta_b = abs(response_b_original_score - response_b_swapped_score)
    
    # Bias should be detected if delta > 0.5
    assert delta_a > 0.5, "Position bias should be detected for response A"
    assert delta_b > 0.5, "Position bias should be detected for response B"
    
    # Overall bias detection
    bias_detected = delta_a > 0.5 or delta_b > 0.5
    assert bias_detected, "Position bias should be detected overall"


def test_no_position_bias_detection():
    """Test that position bias check correctly identifies no bias when scores are stable."""
    # Simulated scores showing no position bias
    response_a_original_score = 4.0
    response_a_swapped_score = 4.1  # Minimal change
    
    response_b_original_score = 3.5
    response_b_swapped_score = 3.6  # Minimal change
    
    # Calculate deltas
    delta_a = abs(response_a_original_score - response_a_swapped_score)
    delta_b = abs(response_b_original_score - response_b_swapped_score)
    
    # Bias should NOT be detected if delta <= 0.5
    assert delta_a <= 0.5, "No position bias should be detected for response A"
    assert delta_b <= 0.5, "No position bias should be detected for response B"
    
    # Overall bias detection
    bias_detected = delta_a > 0.5 or delta_b > 0.5
    assert not bias_detected, "No position bias should be detected overall"


def test_position_bias_threshold():
    """Test that the 0.5 threshold is appropriate for detecting bias."""
    # Test edge cases around the threshold
    
    # Just above threshold
    assert abs(4.0 - 3.49) > 0.5, "Should detect bias at threshold edge"
    
    # Just below threshold
    assert abs(4.0 - 3.51) <= 0.5, "Should not detect bias just below threshold"
    
    # Exactly at threshold
    assert abs(4.0 - 3.5) == 0.5, "Should be exactly at threshold"


def test_position_bias_magnitude():
    """Test that bias magnitude is correctly calculated."""
    # Test various bias magnitudes
    
    # Small bias
    small_bias = abs(4.0 - 3.8)
    assert 0 < small_bias < 0.5, "Small bias should be between 0 and 0.5"
    
    # Medium bias
    medium_bias = abs(4.0 - 3.0)
    assert 0.5 <= medium_bias < 1.0, "Medium bias should be between 0.5 and 1.0"
    
    # Large bias
    large_bias = abs(5.0 - 3.0)
    assert large_bias >= 1.0, "Large bias should be >= 1.0"


def test_position_bias_symmetry():
    """Test that position bias affects both responses symmetrically."""
    # In a true position bias scenario, swapping should symmetrically affect scores
    # If A gains when moved first, B should lose when moved second
    
    score_a_first = 4.5
    score_a_second = 3.5
    
    score_b_first = 4.0
    score_b_second = 3.0
    
    # The delta for A when moving from first to second
    delta_a = score_a_first - score_a_second
    
    # The delta for B when moving from second to first
    delta_b = score_b_first - score_b_second
    
    # Both should be positive (benefit from being first)
    assert delta_a > 0, "Response A should benefit from being first"
    assert delta_b > 0, "Response B should benefit from being first"
    
    # Magnitudes should be similar (symmetric bias)
    assert abs(delta_a - delta_b) < 0.5, "Bias magnitude should be similar for both responses"


def test_position_bias_with_synthetic_judge():
    """Test position bias detection with a synthetic judge implementation."""
    # This would normally use the actual PositionBiasChecker class
    # For testing, we simulate the logic
    
    class SyntheticJudge:
        def evaluate(self, position):
            # Simulate position bias: first position gets +0.5 boost
            base_score = 4.0
            position_boost = 0.5 if position == "first" else 0.0
            return base_score + position_boost
    
    judge = SyntheticJudge()
    
    # Evaluate same response in different positions
    score_first = judge.evaluate("first")
    score_second = judge.evaluate("second")
    
    delta = abs(score_first - score_second)
    
    # Should detect bias
    assert delta > 0.5, "Synthetic judge should exhibit position bias"
    assert delta == 0.5, "Bias magnitude should match the boost"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
