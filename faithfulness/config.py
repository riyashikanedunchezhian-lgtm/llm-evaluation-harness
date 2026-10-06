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
    # Defaulting to Groq Llama 3 for free-tier accessibility
    summary_models: List[str] = None

    # Claim decomposition
    # Using Llama 3 70B via Groq for high-reasoning decomposition (Free Tier)
    claim_decomposition_model: str = "llama3-70b-8192"

    # Retrieval settings
    embedding_model: str = "dense"  # Using Sentence-Transformers (Local/Free)
    chunk_size: int = 200  # words per chunk
    chunk_overlap: int = 50
    top_k_passages: int = 3

    # Faithfulness classification
    # Using Llama 3 70B via Groq for rigorous judgment (Free Tier)
    faithfulness_judge_model: str = "llama3-70b-8192"
    jury_size: int = 3

    # Agreement metrics
    agreement_threshold: float = 0.65  # Minimum acceptable agreement

    # Output settings
    output_dir: str = "faithfulness/data"

    def __post_init__(self):
        if self.summary_models is None:
            # Default to a mix of free-tier models for comparison
            self.summary_models = [
                "llama3-8b-8192",     # Groq (Free/Fast)
                "nvidia/llama-3.1-8b-instruct", # NVIDIA NIM (Free Trial)
                "mixtral-8x7b-32768", # Groq (Free/Strong)
                "claude-3-haiku-20240307" # Optional Paid
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
