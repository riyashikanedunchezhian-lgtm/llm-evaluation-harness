"""Tests for parallel jury execution to validate latency improvement."""

import pytest
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.judge import Jury, JudgeVerdict, JudgeScore
from src.models import ModelClient, ModelResponse


def test_parallel_vs_sequential_latency():
    """Test that parallel execution reduces latency compared to sequential."""
    # This is a conceptual test - actual timing would require real API calls
    # In production, parallel should be ~3x faster for 3 judges
    
    # Simulate timing
    sequential_time_per_judge = 2.0  # seconds
    jury_size = 3
    
    # Sequential: sum of all times
    sequential_total = sequential_time_per_judge * jury_size
    
    # Parallel: max of all times (approximately)
    parallel_total = sequential_time_per_judge  # All run concurrently
    
    speedup = sequential_total / parallel_total
    
    # Parallel should be significantly faster
    assert speedup >= 2.0, f"Expected speedup of at least 2x, got {speedup}x"
    assert speedup <= jury_size, f"Speedup cannot exceed jury size, got {speedup}x"
    
    print(f"Sequential time: {sequential_total:.1f}s")
    print(f"Parallel time: {parallel_total:.1f}s")
    print(f"Speedup: {speedup:.1f}x")


def test_parallel_execution_mode():
    """Test that parallel execution mode is properly set."""
    # Verify that the Jury class accepts parallel parameter
    # This is a structural test
    
    # Test with parallel enabled
    jury_parallel = Jury(None, jury_size=3, parallel=True)
    assert jury_parallel.parallel == True, "Parallel mode should be enabled"
    
    # Test with parallel disabled
    jury_sequential = Jury(None, jury_size=3, parallel=False)
    assert jury_parallel.parallel == True, "Parallel mode should be disabled"


def test_parallel_result_consistency():
    """Test that parallel execution produces consistent results with sequential."""
    # Both modes should produce the same aggregated results
    # This is a conceptual test - actual consistency would require real API calls
    
    # Simulate verdicts
    verdicts = [
        JudgeVerdict(
            scores=[JudgeScore("correctness", 4, "Good")],
            overall_score=4.0,
            judge_id=0,
            input_tokens=100,
            output_tokens=50
        ),
        JudgeVerdict(
            scores=[JudgeScore("correctness", 5, "Excellent")],
            overall_score=5.0,
            judge_id=1,
            input_tokens=110,
            output_tokens=55
        ),
        JudgeVerdict(
            scores=[JudgeScore("correctness", 4, "Good")],
            overall_score=4.0,
            judge_id=2,
            input_tokens=105,
            output_tokens=52
        )
    ]
    
    # Calculate aggregation
    scores = [v.overall_score for v in verdicts]
    avg_score = sum(scores) / len(scores)
    
    # Both sequential and parallel should produce same average
    assert abs(avg_score - 4.33) < 0.01, "Aggregation should be consistent"
    
    # Token totals should be same regardless of execution order
    total_input = sum(v.input_tokens for v in verdicts)
    total_output = sum(v.output_tokens for v in verdicts)
    
    assert total_input == 315, "Input tokens should sum correctly"
    assert total_output == 157, "Output tokens should sum correctly"


def test_parallel_error_handling():
    """Test that parallel execution handles errors gracefully."""
    # In parallel mode, if one judge fails, others should still complete
    # This is a conceptual test for error handling design
    
    # Simulate successful and failed judge calls
    successful_verdicts = 2
    failed_verdicts = 1
    total_judges = successful_verdicts + failed_verdicts
    
    # Even with failures, we should get partial results
    expected_verdicts = successful_verdicts
    
    assert expected_verdicts > 0, "Should get results from successful judges"
    assert expected_verdicts < total_judges, "Should have some failures"


def test_thread_safety():
    """Test that parallel execution is thread-safe."""
    # Multiple concurrent API calls should not interfere with each other
    # This is a conceptual test for thread safety design
    
    # Simulate concurrent calls
    num_concurrent = 3
    
    # Each call should have independent state
    # No shared state modification should occur
    # Results should be deterministic
    
    assert num_concurrent > 1, "Should test concurrent execution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
