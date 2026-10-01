"""Comparative analysis for faithfulness vs fluency."""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faithfulness.jury_aggregation import JuryVerdict
from faithfulness.config import FAITHFULNESS_LABELS

@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_name: str
    faithfulness_rate: float  # % of claims supported
    contradiction_rate: float  # % of claims contradicted
    unverifiable_rate: float  # % of claims unverifiable
    avg_jury_agreement: float
    avg_confidence: float
    total_claims: int

@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    correlation_type: str  # "pearson" or "spearman"
    correlation_coefficient: float
    p_value: float
    significant: bool  # p < 0.05
    interpretation: str

class ComparativeAnalyzer:
    """Analyze faithfulness vs fluency correlations."""
    
    def __init__(self, original_results_path: str = "data/results.json"):
        self.original_results_path = original_results_path
        self.original_results = None
        self.faithfulness_results = None
    
    def load_original_results(self) -> None:
        """Load original evaluation harness results."""
        if os.path.exists(self.original_results_path):
            with open(self.original_results_path, 'r', encoding='utf-8') as f:
                self.original_results = json.load(f)
            print(f"Loaded {len(self.original_results)} original evaluation results")
        else:
            print(f"Warning: Original results not found at {self.original_results_path}")
    
    def load_faithfulness_results(self, faithfulness_path: str) -> None:
        """Load faithfulness evaluation results."""
        with open(faithfulness_path, 'r', encoding='utf-8') as f:
            self.faithfulness_results = json.load(f)
        print(f"Loaded faithfulness results from {faithfulness_path}")
    
    def calculate_model_faithfulness(self, jury_verdicts: Dict[str, JuryVerdict]) -> ModelPerformance:
        """Calculate faithfulness metrics for a model."""
        if not jury_verdicts:
            return ModelPerformance(
                model_name="unknown",
                faithfulness_rate=0.0,
                contradiction_rate=0.0,
                unverifiable_rate=0.0,
                avg_jury_agreement=0.0,
                avg_confidence=0.0,
                total_claims=0
            )
        
        total_claims = len(jury_verdicts)
        
        # Count labels
        label_counts = {}
        for verdict in jury_verdicts.values():
            label = verdict.majority_label
            label_counts[label] = label_counts.get(label, 0) + 1
        
        supported = label_counts.get("Supported", 0)
        contradicted = label_counts.get("Contradicted", 0)
        unverifiable = label_counts.get("Unverifiable", 0)
        
        # Calculate rates
        faithfulness_rate = (supported / total_claims) * 100 if total_claims > 0 else 0.0
        contradiction_rate = (contradicted / total_claims) * 100 if total_claims > 0 else 0.0
        unverifiable_rate = (unverifiable / total_claims) * 100 if total_claims > 0 else 0.0
        
        # Calculate average jury agreement and confidence
        avg_agreement = sum(v.agreement_score for v in jury_verdicts.values()) / total_claims
        avg_confidence = sum(v.confidence for v in jury_verdicts.values()) / total_claims
        
        return ModelPerformance(
            model_name="model",  # Will be set by caller
            faithfulness_rate=faithfulness_rate,
            contradiction_rate=contradiction_rate,
            unverifiable_rate=unverifiable_rate,
            avg_jury_agreement=avg_agreement,
            avg_confidence=avg_confidence,
            total_claims=total_claims
        )
    
    def correlate_faithfulness_fluency(self) -> Optional[CorrelationResult]:
        """Correlate faithfulness scores with fluency scores from original evaluation."""
        if not self.original_results or not self.faithfulness_results:
            print("Warning: Missing original or faithfulness results for correlation")
            return None
        
        # Build dataset: for each model-document pair, get both scores
        data_points = []
        
        # This assumes faithfulness results are structured with model/document info
        # Adjust based on actual structure
        
        # Placeholder implementation - needs actual data structure
        # Extract (fluency_score, faithfulness_score) pairs
        
        if not data_points:
            print("Warning: No matching data points found for correlation")
            return None
        
        fluency_scores = [dp["fluency"] for dp in data_points]
        faithfulness_scores = [dp["faithfulness"] for dp in data_points]
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = pearsonr(fluency_scores, faithfulness_scores)
        
        # Calculate Spearman correlation (rank-based)
        spearman_corr, spearman_p = spearmanr(fluency_scores, faithfulness_scores)
        
        # Interpret results
        interpretation = self._interpret_correlation(pearson_corr, pearson_p)
        
        return CorrelationResult(
            correlation_type="pearson",
            correlation_coefficient=pearson_corr,
            p_value=pearson_p,
            significant=pearson_p < 0.05,
            interpretation=interpretation
        )
    
    def _interpret_correlation(self, coefficient: float, p_value: float) -> str:
        """Interpret correlation coefficient and p-value."""
        # Interpret strength
        if abs(coefficient) >= 0.7:
            strength = "strong"
        elif abs(coefficient) >= 0.5:
            strength = "moderate"
        elif abs(coefficient) >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        # Interpret direction
        if coefficient > 0:
            direction = "positive"
        elif coefficient < 0:
            direction = "negative"
        else:
            direction = "no"
        
        # Interpret significance
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.1:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        if direction == "no":
            return f"No correlation detected (not significant)"
        else:
            return f"{strength} {direction} correlation ({significance})"
    
    def generate_comparison_report(self, output_path: str) -> None:
        """Generate comprehensive comparison report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            "analysis_summary": {
                "has_original_results": self.original_results is not None,
                "has_faithfulness_results": self.faithfulness_results is not None,
                "correlation_available": False
            },
            "model_comparisons": {},
            "correlation_analysis": {}
        }
        
        # Add correlation if available
        correlation = self.correlate_faithfulness_fluency()
        if correlation:
            report["analysis_summary"]["correlation_available"] = True
            report["correlation_analysis"] = {
                "correlation_type": correlation.correlation_type,
                "coefficient": correlation.correlation_coefficient,
                "p_value": correlation.p_value,
                "significant": correlation.significant,
                "interpretation": correlation.interpretation
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Generated comparison report: {output_path}")
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame for visualization."""
        data = []
        
        # Add model performance data
        # This would be populated with actual results
        
        return pd.DataFrame(data)

def generate_model_comparison_table(model_performances: Dict[str, ModelPerformance]) -> str:
    """Generate a formatted comparison table."""
    table = []
    table.append("| Model | Faithfulness Rate | Contradiction Rate | Unverifiable Rate | Avg Jury Agreement | Total Claims |")
    table.append("|-------|-------------------|-------------------|-------------------|-------------------|-------------|")
    
    for model_name, perf in model_performances.items():
        table.append(
            f"| {model_name} | {perf.faithfulness_rate:.1f}% | "
            f"{perf.contradiction_rate:.1f}% | {perf.unverifiable_rate:.1f}% | "
            f"{perf.avg_jury_agreement:.2f} | {perf.total_claims} |"
        )
    
    return "\n".join(table)
