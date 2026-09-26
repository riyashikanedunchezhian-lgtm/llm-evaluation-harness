"""Main evaluation harness for LLM evaluation."""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from .models import ModelClient, ModelResponse
from .judge import Jury, PositionBiasChecker
from .config import MODEL_CONFIGS, JUDGE_MODEL, JURY_SIZE, PARALLEL_JUDGE

@dataclass
class EvaluationResult:
    """Result of evaluating a single prompt with a single model."""
    test_id: str
    category: str
    prompt: str
    model_id: str
    model_name: str
    response: str
    reference_answer: Optional[str]
    evaluation_criteria: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    judge_results: Dict[str, Any]
    overall_score: float
    timestamp: str
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    execution_mode: str = "sequential"

class EvaluationHarness:
    """Main harness for running LLM evaluations."""
    
    def __init__(self, model_ids: List[str], test_set_path: str = "prompts/test_set.json", parallel_judge: bool = PARALLEL_JUDGE):
        self.model_client = ModelClient()
        self.models = [MODEL_CONFIGS[mid] for mid in model_ids]
        self.jury = Jury(self.model_client, parallel=parallel_judge)
        self.bias_checker = PositionBiasChecker(self.model_client)
        self.test_set = self._load_test_set(test_set_path)
        self.results: List[EvaluationResult] = []
        self.parallel_judge = parallel_judge
    
    def _load_test_set(self, path: str) -> Dict[str, List[Dict]]:
        """Load test set from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_config
    ) -> float:
        """Calculate cost in USD for a single model call."""
        input_cost = (input_tokens / 1000) * model_config.input_price_per_1k
        output_cost = (output_tokens / 1000) * model_config.output_price_per_1k
        return input_cost + output_cost
    
    def _calculate_total_cost(
        self,
        candidate_input_tokens: int,
        candidate_output_tokens: int,
        candidate_config,
        judge_input_tokens: int,
        judge_output_tokens: int,
        judge_config
    ) -> float:
        """Calculate total cost including candidate model and all judge calls."""
        # Candidate model cost
        candidate_cost = self._calculate_cost(
            candidate_input_tokens,
            candidate_output_tokens,
            candidate_config
        )
        
        # Judge model cost (multiplied by jury size)
        judge_cost_per_call = self._calculate_cost(
            judge_input_tokens,
            judge_output_tokens,
            judge_config
        )
        total_judge_cost = judge_cost_per_call * JURY_SIZE
        
        return candidate_cost + total_judge_cost
    
    def _flatten_test_set(self) -> List[Dict]:
        """Flatten test set into list of individual test cases."""
        test_cases = []
        for category, tests in self.test_set.items():
            for test in tests:
                test_cases.append({
                    "category": category,
                    **test
                })
        return test_cases
    
    def run_evaluation(
        self,
        categories: Optional[List[str]] = None,
        enable_bias_check: bool = False
    ) -> List[EvaluationResult]:
        """Run full evaluation across all test cases and models."""
        test_cases = self._flatten_test_set()
        
        if categories:
            test_cases = [tc for tc in test_cases if tc["category"] in categories]
        
        total_tests = len(test_cases) * len(self.models)
        current = 0
        
        for test_case in test_cases:
            for model_config in self.models:
                current += 1
                print(f"[{current}/{total_tests}] Evaluating {test_case['id']} with {model_config.name}...")
                
                result = self._evaluate_single(test_case, model_config)
                self.results.append(result)
        
        if enable_bias_check:
            print("\nRunning position bias checks...")
            self._run_bias_checks()
        
        return self.results
    
    def _evaluate_single(
        self,
        test_case: Dict,
        model_config
    ) -> EvaluationResult:
        """Evaluate a single test case with a single model."""
        # Get model response
        model_response = self.model_client.call_model(
            model_id=model_config.model_id,
            provider=model_config.provider,
            prompt=test_case["prompt"],
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature
        )
        
        # Judge the response
        judge_results = self.jury.evaluate(
            prompt=test_case["prompt"],
            response=model_response.content,
            reference_answer=test_case.get("reference_answer"),
            evaluation_criteria=test_case.get("evaluation_criteria")
        )
        
        # Calculate total cost (candidate + judges)
        judge_config = MODEL_CONFIGS[JUDGE_MODEL]
        cost = self._calculate_total_cost(
            candidate_input_tokens=model_response.input_tokens,
            candidate_output_tokens=model_response.output_tokens,
            candidate_config=model_config,
            judge_input_tokens=judge_results["total_judge_input_tokens"],
            judge_output_tokens=judge_results["total_judge_output_tokens"],
            judge_config=judge_config
        )
        
        # Extract overall score
        overall_score = judge_results["aggregated"]["overall_average"]
        
        return EvaluationResult(
            test_id=test_case["id"],
            category=test_case["category"],
            prompt=test_case["prompt"],
            model_id=model_config.model_id,
            model_name=model_config.name,
            response=model_response.content,
            reference_answer=test_case.get("reference_answer"),
            evaluation_criteria=test_case.get("evaluation_criteria"),
            input_tokens=model_response.input_tokens,
            output_tokens=model_response.output_tokens,
            latency_ms=model_response.latency_ms,
            cost_usd=cost,
            judge_results=judge_results,
            overall_score=overall_score,
            timestamp=datetime.now().isoformat(),
            judge_input_tokens=judge_results["total_judge_input_tokens"],
            judge_output_tokens=judge_results["total_judge_output_tokens"],
            execution_mode=judge_results.get("execution_mode", "sequential")
        )
    
    def _run_bias_checks(self):
        """Run position bias checks on a sample of test cases."""
        # Sample a few test cases for bias checking
        test_cases = self._flatten_test_set()[:3]  # Check first 3
        
        for test_case in test_cases:
            # Get responses from two different models
            if len(self.models) >= 2:
                response_a = self.model_client.call_model(
                    model_id=self.models[0].model_id,
                    provider=self.models[0].provider,
                    prompt=test_case["prompt"]
                ).content
                
                response_b = self.model_client.call_model(
                    model_id=self.models[1].model_id,
                    provider=self.models[1].provider,
                    prompt=test_case["prompt"]
                ).content
                
                bias_result = self.bias_checker.check_position_bias(
                    prompt=test_case["prompt"],
                    response_a=response_a,
                    response_b=response_b,
                    reference_answer=test_case.get("reference_answer"),
                    evaluation_criteria=test_case.get("evaluation_criteria")
                )
                
                print(f"Bias check for {test_case['id']}: {'BIAS DETECTED' if bias_result['bias_detected'] else 'No bias detected'}")
                print(f"  Score delta A: {bias_result['score_delta_a']:.2f}")
                print(f"  Score delta B: {bias_result['score_delta_b']:.2f}")
    
    def save_results(self, output_path: str = "data/results.json"):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_dict = [asdict(r) for r in self.results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get summary of results as a pandas DataFrame."""
        data = []
        for r in self.results:
            data.append({
                "test_id": r.test_id,
                "category": r.category,
                "model": r.model_name,
                "overall_score": r.overall_score,
                "latency_ms": r.latency_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.input_tokens + r.output_tokens,
                "judge_input_tokens": r.judge_input_tokens,
                "judge_output_tokens": r.judge_output_tokens,
                "total_judge_tokens": r.judge_input_tokens + r.judge_output_tokens,
                "cost_usd": r.cost_usd,
                "consensus": r.judge_results["aggregated"]["consensus"],
                "execution_mode": r.execution_mode
            })
        
        return pd.DataFrame(data)
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics by model and category."""
        df = self.get_summary_dataframe()
        
        metrics = {}
        
        # By model
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            metrics[model] = {
                "avg_score": model_df["overall_score"].mean(),
                "avg_latency_ms": model_df["latency_ms"].mean(),
                "avg_cost_usd": model_df["cost_usd"].mean(),
                "total_cost_usd": model_df["cost_usd"].sum(),
                "total_tokens": model_df["total_tokens"].sum(),
                "num_tests": len(model_df)
            }
        
        # By category
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            metrics[f"category_{category}"] = {
                "avg_score": cat_df["overall_score"].mean(),
                "avg_latency_ms": cat_df["latency_ms"].mean(),
                "avg_cost_usd": cat_df["cost_usd"].mean(),
                "num_tests": len(cat_df)
            }
        
        # By model and category
        for model in df["model"].unique():
            for category in df["category"].unique():
                key = f"{model}_{category}"
                subset = df[(df["model"] == model) & (df["category"] == category)]
                if len(subset) > 0:
                    metrics[key] = {
                        "avg_score": subset["overall_score"].mean(),
                        "avg_latency_ms": subset["latency_ms"].mean(),
                        "avg_cost_usd": subset["cost_usd"].mean(),
                        "num_tests": len(subset)
                    }
        
        return metrics
