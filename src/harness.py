"""Main evaluation harness for LLM evaluation."""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from .models import ModelClient, ModelResponse
from .judge import Jury, PositionBiasChecker
from .database import Database
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

    def __init__(self, model_ids: List[str], test_set_path: str = "prompts/test_set.json", parallel_judge: bool = PARALLEL_JUDGE, progress_callback=None, database: Optional[Database] = None, db_path: str = "data/evaluations.db", dry_run: bool = False):
        self.model_client = ModelClient()
        self.models = [MODEL_CONFIGS[mid] for mid in model_ids]
        self.jury = Jury(self.model_client, parallel=parallel_judge)
        self.bias_checker = PositionBiasChecker(self.model_client)
        self.test_set = self._load_test_set(test_set_path)
        self.results: List[EvaluationResult] = []
        self.parallel_judge = parallel_judge
        self.progress_callback = progress_callback
        self.database = database or Database(db_path)
        self.dry_run = dry_run
    
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
        enable_bias_check: bool = False,
        max_workers: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Run full evaluation across all test cases and models.

        Args:
            categories: Optional list of categories to filter test cases
            enable_bias_check: Whether to run position bias checks
            max_workers: Maximum number of worker threads for parallel execution.
                        If None, defaults to number of test cases.
        """
        test_cases = self._flatten_test_set()

        if categories:
            test_cases = [tc for tc in test_cases if tc["category"] in categories]

        if not test_cases:
            print("No test cases to evaluate.")
            return []

        # Clear previous results
        self.results = []

        # Calculate total tests for progress tracking
        total_tests = len(test_cases) * len(self.models)
        completed_tests = 0
        completed_tests_lock = threading.Lock()

        def update_progress(test_id: str, model_name: str, category: str):
            nonlocal completed_tests
            with completed_tests_lock:
                completed_tests += 1
                current = completed_tests
                progress_info = {
                    "current": current,
                    "total": total_tests,
                    "test_id": test_id,
                    "model": model_name,
                    "category": category,
                    "progress_percent": (current / total_tests) * 100
                }

                if self.progress_callback:
                    self.progress_callback(progress_info)

                print(f"[{current}/{total_tests}] Evaluating {test_id} with {model_name}...")

        def evaluate_test_case(test_case: Dict) -> List[EvaluationResult]:
            """Evaluate a single test case with all models."""
            test_results = []
            for model_config in self.models:
                result = self._evaluate_single(test_case, model_config)
                test_results.append(result)

                # Update progress
                update_progress(test_case['id'], model_config.name, test_case['category'])

            return test_results

        # Use ThreadPoolExecutor to parallelize outer loop (test cases)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Determine number of workers
        if max_workers is None:
            max_workers = min(len(test_cases), 4)  # Cap at 4 to avoid overloading

        print(f"Starting evaluation with {max_workers} workers for {len(test_cases)} test cases...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test cases for evaluation
            future_to_test = {
                executor.submit(evaluate_test_case, test_case): test_case
                for test_case in test_cases
            }

            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    test_results = future.result()
                    self.results.extend(test_results)
                except Exception as e:
                    print(f"Error evaluating test case {test_case['id']}: {str(e)}")

        # Sort results by test_id and model_id for consistent ordering
        self.results.sort(key=lambda r: (r.test_id, r.model_id))

        if enable_bias_check:
            print("\nRunning position bias checks...")
            if self.progress_callback:
                self.progress_callback({"status": "bias_check", "message": "Running position bias checks..."})
            self._run_bias_checks()

        if self.progress_callback:
            self.progress_callback({"status": "complete", "message": "Evaluation complete!"})

        # Save aggregated metrics to database
        if self.results:  # Only save if we have results
            aggregated_metrics = self.get_aggregated_metrics()
            test_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save each metric type
            for metric_key, metric_value in aggregated_metrics.items():
                # Determine metric type based on key prefix
                if metric_key.startswith('category_'):
                    metric_type = 'category'
                    clean_key = metric_key.replace('category_', '')
                elif '_' in metric_key and not metric_key.startswith('category_'):
                    # Format: model_category
                    parts = metric_key.split('_', 1)
                    if len(parts) == 2:
                        metric_type = 'model_category'
                        clean_key = metric_key
                    else:
                        metric_type = 'model'
                        clean_key = metric_key
                else:
                    metric_type = 'model'
                    clean_key = metric_key

                self.database.save_aggregated_metrics(
                    metric_type=metric_type,
                    metric_key=clean_key,
                    metric_value=metric_value,
                    test_run_id=test_run_id
                )

        return self.results
    
    def _evaluate_single(
        self,
        test_case: Dict,
        model_config
    ) -> EvaluationResult:
        """Evaluate a single test case with a single model."""
        # Handle dry-run mode - skip actual API calls
        if self.dry_run:
            import time
            import uuid
            # Return mock response
            mock_response = ModelResponse(
                content=f"[DRY RUN] Mock response for {test_case['id']} with {model_config.name}",
                input_tokens=100,
                output_tokens=200,
                latency_ms=10.0,
                model_id=model_config.model_id
            )

            # Mock judge results
            mock_judge_results = {
                "aggregated": {
                    "overall_average": 3.5,
                    "consensus": True
                },
                "total_judge_input_tokens": 500,
                "total_judge_output_tokens": 100,
                "execution_mode": "sequential"
            }

            # Calculate cost using mock tokens
            judge_config = MODEL_CONFIGS[JUDGE_MODEL]
            cost = self._calculate_total_cost(
                candidate_input_tokens=mock_response.input_tokens,
                candidate_output_tokens=mock_response.output_tokens,
                candidate_config=model_config,
                judge_input_tokens=mock_judge_results["total_judge_input_tokens"],
                judge_output_tokens=mock_judge_results["total_judge_output_tokens"],
                judge_config=judge_config
            )

            result = EvaluationResult(
                test_id=test_case["id"],
                category=test_case["category"],
                prompt=test_case["prompt"],
                model_id=model_config.model_id,
                model_name=model_config.name,
                response=mock_response.content,
                reference_answer=test_case.get("reference_answer"),
                evaluation_criteria=test_case.get("evaluation_criteria"),
                input_tokens=mock_response.input_tokens,
                output_tokens=mock_response.output_tokens,
                latency_ms=mock_response.latency_ms,
                cost_usd=cost,
                judge_results=mock_judge_results,
                overall_score=mock_judge_results["aggregated"]["overall_average"],
                timestamp=datetime.now().isoformat(),
                judge_input_tokens=mock_judge_results["total_judge_input_tokens"],
                judge_output_tokens=mock_judge_results["total_judge_output_tokens"],
                execution_mode=mock_judge_results.get("execution_mode", "sequential")
            )

            # Persist to database
            self.database.save_evaluation(result)

            return result

        # Get model response
        model_response = self.model_client.call_model(
            model_id=model_config.model_id,
            provider=model_config.provider,
            prompt=test_case["prompt"],
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            api_base=model_config.api_base
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
        
        result = EvaluationResult(
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

        # Persist to database
        self.database.save_evaluation(result)

        return result
    
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
                    prompt=test_case["prompt"],
                    api_base=self.models[0].api_base
                ).content

                response_b = self.model_client.call_model(
                    model_id=self.models[1].model_id,
                    provider=self.models[1].provider,
                    prompt=test_case["prompt"],
                    api_base=self.models[1].api_base
                ).content

                bias_result = self.bias_checker.check_position_bias(
                    prompt=test_case["prompt"],
                    response_a=response_a,
                    response_b=response_b,
                    reference_answer=test_case.get("reference_answer"),
                    evaluation_criteria=test_case.get("evaluation_criteria")
                )

                # Persist bias check to database
                self.database.save_bias_check(bias_result)

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
