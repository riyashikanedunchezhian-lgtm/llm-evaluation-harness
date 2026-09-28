"""Configuration for LLM evaluation harness."""

from dataclasses import dataclass
from typing import Dict

__all__ = ["ModelConfig", "MODEL_CONFIGS", "JUDGE_MODEL", "RUBRIC_DIMENSIONS", "JURY_SIZE", "PARALLEL_JUDGE"]

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str  # 'anthropic', 'openai', 'google', 'cohere', or 'local'
    model_id: str
    input_price_per_1k: float  # USD per 1K input tokens
    output_price_per_1k: float  # USD per 1K output tokens
    max_tokens: int = 4096
    temperature: float = 0.7
    api_base: str = None  # For local models or custom endpoints

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
    "gemini-1.5-pro": ModelConfig(
        name="Gemini 1.5 Pro",
        provider="google",
        model_id="gemini-1.5-pro",
        input_price_per_1k=3.5,
        output_price_per_1k=10.5,
        max_tokens=8192,
        temperature=0.7
    ),
    "gemini-1.5-flash": ModelConfig(
        name="Gemini 1.5 Flash",
        provider="google",
        model_id="gemini-1.5-flash",
        input_price_per_1k=0.075,
        output_price_per_1k=0.30,
        max_tokens=8192,
        temperature=0.7
    ),
    "command-r-plus": ModelConfig(
        name="Command R+",
        provider="cohere",
        model_id="command-r-plus",
        input_price_per_1k=3.0,
        output_price_per_1k=15.0,
        max_tokens=4096,
        temperature=0.7
    ),
    "command-r": ModelConfig(
        name="Command R",
        provider="cohere",
        model_id="command-r",
        input_price_per_1k=0.50,
        output_price_per_1k=1.50,
        max_tokens=4096,
        temperature=0.7
    ),
    "local-llama-3-8b": ModelConfig(
        name="Local Llama 3 8B",
        provider="local",
        model_id="llama-3-8b",
        input_price_per_1k=0.0,
        output_price_per_1k=0.0,
        max_tokens=4096,
        temperature=0.7,
        api_base="http://localhost:8000/v1"
    ),
    "local-mistral-7b": ModelConfig(
        name="Local Mistral 7B",
        provider="local",
        model_id="mistral-7b",
        input_price_per_1k=0.0,
        output_price_per_1k=0.0,
        max_tokens=4096,
        temperature=0.7,
        api_base="http://localhost:8000/v1"
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
PARALLEL_JUDGE = True  # Enable parallel execution for reduced latency
