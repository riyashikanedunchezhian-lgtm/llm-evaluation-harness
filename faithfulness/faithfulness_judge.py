"""Faithfulness classification for claims."""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelClient
from src.config import MODEL_CONFIGS
from faithfulness.config import FAITHFULNESS_LABELS, FAITHFULNESS_SCHEMA
from faithfulness.claim_decomposition import Claim
from faithfulness.retrieval import RetrievedPassage

@dataclass
class FaithfulnessVerdict:
    """Verdict from a single faithfulness judge."""
    label: str  # Supported, Contradicted, Unverifiable
    justification: str
    confidence: float
    judge_id: int

class FaithfulnessJudge:
    """Judge for classifying claim faithfulness."""
    
    def __init__(self, model_client: ModelClient, model_id: str = "claude-3-5-sonnet-20241022"):
        self.model_client = model_client
        self.model_id = model_id
        self.model_config = MODEL_CONFIGS[model_id]
    
    def classify_claim(self, claim: Claim, retrieved_passages: List[RetrievedPassage], 
                      judge_id: int = 0) -> FaithfulnessVerdict:
        """Classify a claim as Supported, Contradicted, or Unverifiable."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(claim, retrieved_passages)
        
        response = self.model_client.call_model(
            model_id=self.model_id,
            provider=self.model_config.provider,
            prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,  # Lower temperature for consistent classification
            system_prompt=system_prompt
        )
        
        verdict = self._parse_verdict(response.content, judge_id)
        return verdict
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for faithfulness classification."""
        labels_desc = "\n".join([
            f"- {label}: {self._get_label_description(label)}"
            for label in FAITHFULNESS_LABELS
        ])
        
        return f"""You are an expert fact-checker. Your task is to evaluate whether a claim from a summary is supported by the source document.

Classification labels:
{labels_desc}

Evaluation criteria:
1. **Supported**: The claim is directly stated or clearly implied by the source text. Numbers, names, and details match the source.
2. **Contradicted**: The claim directly contradicts the source text. Different values, opposite claims, or mutually exclusive statements.
3. **Unverifiable**: The source text does not contain enough information to verify the claim. This is different from contradiction - the claim might be true, but cannot be verified from the given passages.

Output format: Provide a JSON object with:
- "label": One of {FAITHFULNESS_LABELS}
- "justification": Detailed explanation of your reasoning (2-3 sentences)
- "confidence": Your confidence level (0.0 to 1.0)

Be precise and conservative. If in doubt, choose "Unverifiable" rather than guessing."""
    
    def _get_label_description(self, label: str) -> str:
        """Get description for a faithfulness label."""
        descriptions = {
            "Supported": "The claim is directly supported by evidence in the source passages",
            "Contradicted": "The claim directly contradicts information in the source passages",
            "Unverifiable": "The source passages do not contain sufficient information to verify the claim"
        }
        return descriptions.get(label, label)
    
    def _build_user_prompt(self, claim: Claim, retrieved_passages: List[RetrievedPassage]) -> str:
        """Build user prompt for faithfulness classification."""
        user_prompt = f"""Evaluate the following claim:

**Claim:**
{claim.claim_text}

**Source Passages:**
"""
        
        for i, passage in enumerate(retrieved_passages, 1):
            user_prompt += f"""
Passage {i} (relevance: {passage.relevance_score:.3f}):
{passage.passage}
"""
        
        user_prompt += "\nProvide your classification as a JSON object with label, justification, and confidence."
        return user_prompt
    
    def _parse_verdict(self, response_text: str, judge_id: int) -> FaithfulnessVerdict:
        """Parse the judge's response into a structured verdict."""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                # Fallback parsing
                return self._create_fallback_verdict(response_text, judge_id)
            
            label = data.get("label", "Unverifiable")
            # Normalize label
            if label not in FAITHFULNESS_LABELS:
                label = "Unverifiable"
            
            return FaithfulnessVerdict(
                label=label,
                justification=data.get("justification", "No justification provided"),
                confidence=float(data.get("confidence", 0.5)),
                judge_id=judge_id
            )
        
        except Exception as e:
            return self._create_fallback_verdict(response_text, judge_id, str(e))
    
    def _create_fallback_verdict(self, response_text: str, judge_id: int, 
                                error: str = "") -> FaithfulnessVerdict:
        """Create a fallback verdict on parsing error."""
        # Try to extract label from text
        label = "Unverifiable"
        for valid_label in FAITHFULNESS_LABELS:
            if valid_label.lower() in response_text.lower():
                label = valid_label
                break
        
        return FaithfulnessVerdict(
            label=label,
            justification=f"Parsing error: {error}. Original response: {response_text[:200]}",
            confidence=0.3,
            judge_id=judge_id
        )
