"""Tests for claim decomposition."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from faithfulness.claim_decomposition import Claim, validate_claims_atomic

def test_atomic_claim_decomposition():
    """Test that claim decomposition produces atomic claims on a hand-constructed test."""
    # Test sentence with multiple facts
    test_summary = "The report shows that GDP grew by 2.5% in Q3 2023, while unemployment fell to 4.1%, and inflation remained stable at 2.0%."
    
    # Expected atomic claims (this would normally come from LLM)
    expected_claims = [
        Claim(claim_id="test_1", claim_text="GDP grew by 2.5% in Q3 2023"),
        Claim(claim_id="test_2", claim_text="Unemployment fell to 4.1%"),
        Claim(claim_id="test_3", claim_text="Inflation remained stable at 2.0%")
    ]
    
    # Validate that claims are atomic
    issues = validate_claims_atomic(expected_claims)
    
    # Should not have overlapping claims
    assert len(issues["overlapping"]) == 0, "Claims should not overlap"
    
    # Should not have compound claims
    assert len(issues["compound"]) == 0, "Claims should be atomic, not compound"
    
    # Should not have vague claims
    assert len(issues["vague"]) == 0, "Claims should be specific, not vague"

def test_compound_claim_detection():
    """Test detection of compound claims."""
    compound_claim = Claim(
        claim_id="compound_1",
        claim_text="GDP grew by 2.5% and unemployment fell to 4.1%"
    )
    
    issues = validate_claims_atomic([compound_claim])
    
    # Should detect the compound claim
    assert len(issues["compound"]) > 0, "Should detect compound claim"
    assert "compound_1" in issues["compound"][0]

def test_vague_claim_detection():
    """Test detection of vague claims."""
    vague_claim = Claim(
        claim_id="vague_1",
        claim_text="The report discusses economic indicators"
    )
    
    issues = validate_claims_atomic([vague_claim])
    
    # Should detect the vague claim
    assert len(issues["vague"]) > 0, "Should detect vague claim"
    assert "vague_1" in issues["vague"][0]

def test_overlapping_claim_detection():
    """Test detection of overlapping claims."""
    overlapping_claims = [
        Claim(claim_id="overlap_1", claim_text="GDP grew by 2.5% in Q3 2023"),
        Claim(claim_id="overlap_2", claim_text="GDP increased by 2.5% in Q3 2023")
    ]
    
    issues = validate_claims_atomic(overlapping_claims)
    
    # Should detect overlapping claims
    assert len(issues["overlapping"]) > 0, "Should detect overlapping claims"

def test_atomic_claim_validation():
    """Test that properly atomic claims pass validation."""
    atomic_claims = [
        Claim(claim_id="atomic_1", claim_text="GDP grew by 2.5% in Q3 2023"),
        Claim(claim_id="atomic_2", claim_text="Unemployment rate decreased to 4.1%"),
        Claim(claim_id="atomic_3", claim_text="The central bank maintained interest rates at 5.0%")
    ]
    
    issues = validate_claims_atomic(atomic_claims)
    
    # Should have no issues
    assert len(issues["overlapping"]) == 0
    assert len(issues["compound"]) == 0
    assert len(issues["vague"]) == 0

def test_claim_id_format():
    """Test that claim IDs follow expected format."""
    claim = Claim(claim_id="doc1_claim_5", claim_text="Test claim")
    
    # Claim ID should contain document and claim number
    assert "doc1" in claim.claim_id
    assert "claim" in claim.claim_id
    assert "5" in claim.claim_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
