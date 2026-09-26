"""LLM Evaluation Harness."""

__version__ = "0.1.0"

from .config import MODEL_CONFIGS, JUDGE_MODEL, RUBRIC_DIMENSIONS, JURY_SIZE
from .models import ModelClient, ModelResponse
from .judge import Judge, Jury, PositionBiasChecker, JudgeScore, JudgeVerdict
from .harness import EvaluationHarness, EvaluationResult

__all__ = [
    "MODEL_CONFIGS",
    "JUDGE_MODEL", 
    "RUBRIC_DIMENSIONS",
    "JURY_SIZE",
    "ModelClient",
    "ModelResponse",
    "Judge",
    "Jury",
    "PositionBiasChecker",
    "JudgeScore",
    "JudgeVerdict",
    "EvaluationHarness",
    "EvaluationResult"
]
