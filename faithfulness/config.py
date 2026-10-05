"""Configuration for faithfulness evaluation."""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class FaithfulnessConfig:
    """Configuration for faithfulness evaluation."""
    
    # Dataset settings
    dataset_name: str = "govreport"  # or "arxiv"
    num_documents: int = 30
    min_word_count: int = 2000
    
    # Summary generation models
    summary_models: List[str] = None  # Will be set from main config
    
    # Claim decomposition
    claim_decomposition_model: str = "claude-3-5-sonnet-20241022"
    
    # Retrieval settings
    embedding_model: str = "tfidf"  # Using sklearn TF-IDF to avoid torch dependency
    chunk_size: int = 200  # words per chunk
    chunk_overlap: int = 50
    top_k_passages: int = 3
    
    # Faithfulness classification
    faithfulness_judge_model: str = "claude-3-5-sonnet-20241022"
    jury_size: int = 3
    
    # Agreement metrics
    agreement_threshold: float = 0.65  # Minimum acceptable agreement
    
    # Output settings
    output_dir: str = "faithfulness/data"
    
    def __post_init__(self):
        if self.summary_models is None:
            self.summary_models = [
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
                "gpt-4o-mini"
            ]

# Faithfulness labels
FAITHFULNESS_LABELS = ["Supported", "Contradicted", "Unverifiable"]

# Claim decomposition schema
CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "claim_id": {"type": "string"}
                },
                "required": ["claim", "claim_id"]
            }
        }
    },
    "required": ["claims"]
}

# Faithfulness classification schema
FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": FAITHFULNESS_LABELS
        },
        "justification": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["label", "justification", "confidence"]
}
