"""Claim decomposition for faithfulness evaluation."""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelClient
from src.config import MODEL_CONFIGS
from faithfulness.config import CLAIM_SCHEMA

@dataclass
class Claim:
    """Represents an atomic factual claim."""
    claim_id: str
    claim_text: str
    source_sentence: Optional[str] = None  # Original sentence if applicable
    
class ClaimDecomposer:
    """Decompose summaries into atomic factual claims."""
    
    def __init__(self, model_client: ModelClient, model_id: str = "claude-3-5-sonnet-20241022"):
        self.model_client = model_client
        self.model_id = model_id
        self.model_config = MODEL_CONFIGS[model_id]
    
    def decompose_summary(self, summary: str, doc_id: str) -> List[Claim]:
        """Decompose a summary into atomic claims."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(summary)
        
        response = self.model_client.call_model(
            model_id=self.model_id,
            provider=self.model_config.provider,
            prompt=user_prompt,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for consistent decomposition
            system_prompt=system_prompt
        )
        
        claims = self._parse_claims(response.content, doc_id)
        return claims
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for claim decomposition."""
        return """You are an expert at breaking down text into atomic factual claims. Your task is to decompose a summary into individual, non-overlapping factual claims.

Rules for claim decomposition:
1. Each claim should represent a single factual assertion
2. Claims should be atomic - don't combine multiple facts into one claim
3. Claims should be self-contained and understandable in isolation
4. Include specific details (numbers, names, dates) in the claim
5. Avoid vague claims like "the report discusses X" - be specific about what is stated
6. Maintain the original meaning - don't interpret or add information
7. Each sentence may yield 1-3 claims depending on complexity

Output format: Provide a JSON object with a "claims" array, where each claim has:
- "claim_id": A unique identifier (e.g., "claim_1", "claim_2")
- "claim": The atomic claim text

Example:
Input: "The report shows that GDP grew by 2.5% in Q3 2023, while unemployment fell to 4.1%."
Output:
{
  "claims": [
    {"claim_id": "claim_1", "claim": "GDP grew by 2.5% in Q3 2023"},
    {"claim_id": "claim_2", "claim": "Unemployment fell to 4.1%"}
  ]
}"""
    
    def _build_user_prompt(self, summary: str) -> str:
        """Build user prompt for claim decomposition."""
        return f"""Decompose the following summary into atomic factual claims:

Summary:
{summary}

Provide your response as a JSON object with a "claims" array."""
    
    def _parse_claims(self, response_text: str, doc_id: str) -> List[Claim]:
        """Parse the LLM response into Claim objects."""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                # Fallback: parse line by line
                return self._parse_fallback_claims(response_text, doc_id)
            
            claims = []
            for i, claim_data in enumerate(data.get("claims", [])):
                claim = Claim(
                    claim_id=f"{doc_id}_claim_{i+1}",
                    claim_text=claim_data.get("claim", "").strip()
                )
                if claim.claim_text:  # Only add non-empty claims
                    claims.append(claim)
            
            return claims
        
        except Exception as e:
            print(f"Error parsing claims: {e}")
            return self._parse_fallback_claims(response_text, doc_id)
    
    def _parse_fallback_claims(self, response_text: str, doc_id: str) -> List[Claim]:
        """Fallback parsing if JSON extraction fails."""
        # Try to extract claims using regex
        claims = []
        
        # Look for numbered or bulleted claims
        patterns = [
            r'(?:claim[_\s]*\d+[:\s]+)(.*?)(?=\n|$)',
            r'(?:\d+[\.\)]\s+)(.*?)(?=\n|$)',
            r'(?:[-*]\s+)(.*?)(?=\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.MULTILINE)
            if matches:
                for i, match in enumerate(matches):
                    claim_text = match.strip()
                    if claim_text and len(claim_text) > 10:  # Filter out very short matches
                        claims.append(Claim(
                            claim_id=f"{doc_id}_claim_{i+1}",
                            claim_text=claim_text
                        ))
                if claims:
                    break
        
        # If still no claims, split by sentences
        if not claims:
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]
            for i, sentence in enumerate(sentences):
                if len(sentence) > 20:  # Filter out very short sentences
                    claims.append(Claim(
                        claim_id=f"{doc_id}_claim_{i+1}",
                        claim_text=sentence + "."
                    ))
        
        return claims

def validate_claims_atomic(claims: List[Claim]) -> Dict[str, List[str]]:
    """Validate that claims are atomic (non-overlapping, single facts)."""
    issues = {
        "overlapping": [],
        "compound": [],
        "vague": []
    }
    
    # Check for overlapping claims (similar content)
    for i, claim1 in enumerate(claims):
        for j, claim2 in enumerate(claims[i+1:], i+1):
            # Simple similarity check
            words1 = set(claim1.claim_text.lower().split())
            words2 = set(claim2.claim_text.lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                if overlap > 0.7:  # High overlap threshold
                    issues["overlapping"].append(f"{claim1.claim_id} and {claim2.claim_id}")
    
    # Check for compound claims (multiple facts)
    compound_indicators = ["and", "while", "but", "whereas", "although", "also"]
    for claim in claims:
        words = claim.claim_text.lower().split()
        # Look for conjunctions that might indicate compound claims
        for indicator in compound_indicators:
            if indicator in words:
                # Check if it's actually connecting separate facts
                if words.count(indicator) >= 1:
                    issues["compound"].append(claim.claim_id)
                    break
    
    # Check for vague claims
    vague_patterns = [
        r"discusses?", r"mentions?", r"talks about", 
        r"covers?", r"addresses?", r"related to"
    ]
    for claim in claims:
        for pattern in vague_patterns:
            if re.search(pattern, claim.claim_text, re.IGNORECASE):
                issues["vague"].append(claim.claim_id)
                break
    
    return issues
