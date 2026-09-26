"""Configuration for LLM evaluation harness."""

import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str  # 'anthropic' or 'openai'
    model_id: str
    input_price_per_1k: float  # USD per 1K input tokens
    output_price_per_1k: float  # USD per 1K output tokens
    max_tokens: int = 4096
    temperature: float = 0.7

# Model configurations with pricing (as of 2024)
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "claude-3-5-sonnet-20241022": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        input_price_per_1k=3.0,
        output_price_per_1k=15.0,
        max_tokens=4096,
        temperature=0.7
    ),
    "claude-3-haiku-20240307": ModelConfig(
        name="Claude 3 Haiku",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        input_price_per_1k=0.25,
        output_price_per_1k=1.25,
        max_tokens=4096,
        temperature=0.7
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        input_price_per_1k=5.0,
        output_price_per_1k=15.0,
        max_tokens=4096,
        temperature=0.7
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        input_price_per_1k=0.15,
        output_price_per_1k=0.60,
        max_tokens=4096,
        temperature=0.7
    ),
}

# Judge model configuration (typically use a stronger model for judging)
JUDGE_MODEL = "claude-3-5-sonnet-20241022"

# Evaluation rubric dimensions
RUBRIC_DIMENSIONS = [
    "correctness",
    "relevance",
    "conciseness",
    "clarity",
    "safety"
]

# Jury evaluation settings
JURY_SIZE = 3  # Number of judge calls per evaluation
