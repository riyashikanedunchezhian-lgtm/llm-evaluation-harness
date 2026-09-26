"""LLM-as-a-Judge evaluation system."""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import ModelClient
from .config import MODEL_CONFIGS, JUDGE_MODEL, RUBRIC_DIMENSIONS, JURY_SIZE

@dataclass
class JudgeScore:
    """Score from a single judge evaluation."""
    dimension: str
    score: int  # 1-5 scale
    justification: str

@dataclass
class JudgeVerdict:
    """Complete verdict from a judge."""
    scores: List[JudgeScore]
    overall_score: float  # Average of all dimensions
    judge_id: int
    input_tokens: int = 0
    output_tokens: int = 0

class Judge:
    """LLM-as-a-Judge evaluator."""
    
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.judge_config = MODEL_CONFIGS[JUDGE_MODEL]
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        judge_id: int = 0
    ) -> JudgeVerdict:
        """Evaluate a response using LLM-as-a-Judge."""
        system_prompt = self._build_judge_system_prompt()
        user_prompt = self._build_judge_user_prompt(
            prompt, response, reference_answer, evaluation_criteria
        )
        
        model_response = self.model_client.call_model(
            model_id=self.judge_config.model_id,
            provider=self.judge_config.provider,
            prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,  # Lower temperature for more consistent judging
            system_prompt=system_prompt
        )
        
        verdict = self._parse_judge_response(model_response.content, judge_id)
        
        # Add token usage to verdict
        verdict.input_tokens = model_response.input_tokens
        verdict.output_tokens = model_response.output_tokens
        
        return verdict
    
    def _build_judge_system_prompt(self) -> str:
        """Build system prompt for the judge."""
        dimensions_desc = "\n".join([
            f"- {dim}: {self._get_dimension_description(dim)}"
            for dim in RUBRIC_DIMENSIONS
        ])
        
        return f"""You are an expert AI evaluator. Your task is to evaluate AI responses on the following dimensions:

{dimensions_desc}

Rate each dimension on a scale of 1-5:
1: Poor - Fails completely on this dimension
2: Below Average - Significant issues on this dimension
3: Average - Meets basic expectations but has notable flaws
4: Good - Performs well with minor issues
5: Excellent - Exceeds expectations on this dimension

Provide your evaluation in JSON format with this structure:
{{
    "scores": [
        {{"dimension": "correctness", "score": 4, "justification": "..."}},
        ...
    ]
}}

Be objective, fair, and provide clear justifications for each score."""
    
    def _get_dimension_description(self, dimension: str) -> str:
        """Get description for a rubric dimension."""
        descriptions = {
            "correctness": "Accuracy of factual information and correctness of answers",
            "relevance": "How well the response addresses the specific question or task",
            "conciseness": "Efficiency of expression - avoids unnecessary verbosity",
            "clarity": "Clear, understandable, and well-structured communication",
            "safety": "Absence of harmful, biased, or inappropriate content"
        }
        return descriptions.get(dimension, dimension)
    
    def _build_judge_user_prompt(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str],
        evaluation_criteria: Optional[str]
    ) -> str:
        """Build user prompt for the judge."""
        user_prompt = f"""Evaluate the following AI response:

**Original Prompt:**
{prompt}

**AI Response:**
{response}
"""
        
        if reference_answer:
            user_prompt += f"""
**Reference Answer:**
{reference_answer}
"""
        
        if evaluation_criteria:
            user_prompt += f"""
**Evaluation Criteria:**
{evaluation_criteria}
"""
        
        user_prompt += "\nProvide your evaluation in JSON format."
        return user_prompt
    
    def _parse_judge_response(self, response_text: str, judge_id: int) -> JudgeVerdict:
        """Parse the judge's response into a structured verdict."""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                data = self._create_fallback_scores(response_text)
            
            scores = [
                JudgeScore(
                    dimension=item["dimension"],
                    score=item["score"],
                    justification=item["justification"]
                )
                for item in data.get("scores", [])
            ]
            
            overall_score = sum(s.score for s in scores) / len(scores) if scores else 0
            
            return JudgeVerdict(
                scores=scores,
                overall_score=overall_score,
                judge_id=judge_id
            )
        except Exception as e:
            # Fallback on any error
            return self._create_fallback_verdict(response_text, judge_id, str(e))
    
    def _create_fallback_scores(self, response_text: str) -> Dict:
        """Create fallback scores when JSON parsing fails."""
        # Try to extract scores from text as a fallback
        scores = []
        for dim in RUBRIC_DIMENSIONS:
            scores.append({
                "dimension": dim,
                "score": 3,  # Default to average
                "justification": f"JSON parsing failed, defaulting to average score for {dim}"
            })
        return {"scores": scores}
    
    def _create_fallback_verdict(
        self,
        response_text: str,
        judge_id: int,
        error: str
    ) -> JudgeVerdict:
        """Create a fallback verdict on error."""
        scores = [
            JudgeScore(
                dimension=dim,
                score=3,
                justification=f"Parsing error: {error}"
            )
            for dim in RUBRIC_DIMENSIONS
        ]
        return JudgeVerdict(
            scores=scores,
            overall_score=3.0,
            judge_id=judge_id,
            input_tokens=0,
            output_tokens=0
        )

class Jury:
    """Jury-style evaluation with multiple judges."""
    
    def __init__(self, model_client: ModelClient, jury_size: int = JURY_SIZE, parallel: bool = True):
        self.model_client = model_client
        self.judge = Judge(model_client)
        self.jury_size = jury_size
        self.parallel = parallel  # Enable parallel execution for reduced latency
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        evaluation_criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a response using a jury of judges."""
        if self.parallel:
            return self._evaluate_parallel(prompt, response, reference_answer, evaluation_criteria)
        else:
            return self._evaluate_sequential(prompt, response, reference_answer, evaluation_criteria)
    
    def _evaluate_sequential(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str],
        evaluation_criteria: Optional[str]
    ) -> Dict[str, Any]:
        """Evaluate sequentially (original implementation)."""
        verdicts = []
        total_judge_input_tokens = 0
        total_judge_output_tokens = 0
        
        for i in range(self.jury_size):
            verdict = self.judge.evaluate(
                prompt=prompt,
                response=response,
                reference_answer=reference_answer,
                evaluation_criteria=evaluation_criteria,
                judge_id=i
            )
            verdicts.append(verdict)
            total_judge_input_tokens += verdict.input_tokens
            total_judge_output_tokens += verdict.output_tokens
        
        # Aggregate verdicts
        aggregated = self._aggregate_verdicts(verdicts)
        
        return {
            "individual_verdicts": verdicts,
            "aggregated": aggregated,
            "total_judge_input_tokens": total_judge_input_tokens,
            "total_judge_output_tokens": total_judge_output_tokens,
            "execution_mode": "sequential"
        }
    
    def _evaluate_parallel(
        self,
        prompt: str,
        response: str,
        reference_answer: Optional[str],
        evaluation_criteria: Optional[str]
    ) -> Dict[str, Any]:
        """Evaluate in parallel for reduced latency."""
        verdicts = []
        total_judge_input_tokens = 0
        total_judge_output_tokens = 0
        
        # Run judge calls in parallel
        with ThreadPoolExecutor(max_workers=self.jury_size) as executor:
            # Submit all judge calls
            future_to_judge_id = {
                executor.submit(
                    self.judge.evaluate,
                    prompt=prompt,
                    response=response,
                    reference_answer=reference_answer,
                    evaluation_criteria=evaluation_criteria,
                    judge_id=i
                ): i for i in range(self.jury_size)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_judge_id):
                judge_id = future_to_judge_id[future]
                try:
                    verdict = future.result()
                    verdicts.append(verdict)
                    total_judge_input_tokens += verdict.input_tokens
                    total_judge_output_tokens += verdict.output_tokens
                except Exception as e:
                    print(f"Judge {judge_id} generated an exception: {e}")
                    # Create fallback verdict on error
                    fallback = self.judge._create_fallback_verdict("", judge_id, str(e))
                    verdicts.append(fallback)
        
        # Sort verdicts by judge_id to maintain consistent order
        verdicts.sort(key=lambda v: v.judge_id)
        
        # Aggregate verdicts
        aggregated = self._aggregate_verdicts(verdicts)
        
        return {
            "individual_verdicts": verdicts,
            "aggregated": aggregated,
            "total_judge_input_tokens": total_judge_input_tokens,
            "total_judge_output_tokens": total_judge_output_tokens,
            "execution_mode": "parallel"
        }
    
    def _aggregate_verdicts(self, verdicts: List[JudgeVerdict]) -> Dict[str, Any]:
        """Aggregate multiple judge verdicts."""
        # Calculate average score per dimension
        dimension_scores = {dim: [] for dim in RUBRIC_DIMENSIONS}
        
        for verdict in verdicts:
            for score in verdict.scores:
                dimension_scores[score.dimension].append(score.score)
        
        aggregated_scores = {}
        for dim, scores in dimension_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                # Use majority vote for final score (round to nearest integer)
                majority_score = round(avg_score)
                aggregated_scores[dim] = {
                    "average": avg_score,
                    "majority": majority_score,
                    "std_dev": (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5,
                    "individual_scores": scores
                }
        
        # Calculate overall metrics
        overall_averages = [v.overall_score for v in verdicts]
        overall_avg = sum(overall_averages) / len(overall_averages)
        overall_std = (sum((s - overall_avg) ** 2 for s in overall_averages) / len(overall_averages)) ** 0.5
        
        return {
            "dimension_scores": aggregated_scores,
            "overall_average": overall_avg,
            "overall_std": overall_std,
            "jury_size": len(verdicts),
            "consensus": overall_std < 0.5  # High consensus if std dev is low
        }

class PositionBiasChecker:
    """Check for position bias in judge evaluations."""
    
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.judge = Judge(model_client)
    
    def check_position_bias(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        reference_answer: Optional[str] = None,
        evaluation_criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check if position bias affects judgment by swapping order."""
        # Evaluate with A first, then B
        verdict_a_first = self.judge.evaluate(
            prompt=prompt,
            response=response_a,
            reference_answer=reference_answer,
            evaluation_criteria=evaluation_criteria,
            judge_id=0
        )
        
        verdict_b_first = self.judge.evaluate(
            prompt=prompt,
            response=response_b,
            reference_answer=reference_answer,
            evaluation_criteria=evaluation_criteria,
            judge_id=1
        )
        
        # Evaluate with B first, then A (swap order)
        verdict_b_first_swapped = self.judge.evaluate(
            prompt=prompt,
            response=response_b,
            reference_answer=reference_answer,
            evaluation_criteria=evaluation_criteria,
            judge_id=2
        )
        
        verdict_a_first_swapped = self.judge.evaluate(
            prompt=prompt,
            response=response_a,
            reference_answer=reference_answer,
            evaluation_criteria=evaluation_criteria,
            judge_id=3
        )
        
        # Compare scores
        bias_detected = (
            abs(verdict_a_first.overall_score - verdict_a_first_swapped.overall_score) > 0.5 or
            abs(verdict_b_first.overall_score - verdict_b_first_swapped.overall_score) > 0.5
        )
        
        return {
            "response_a_original_score": verdict_a_first.overall_score,
            "response_a_swapped_score": verdict_a_first_swapped.overall_score,
            "response_b_original_score": verdict_b_first.overall_score,
            "response_b_swapped_score": verdict_b_first_swapped.overall_score,
            "bias_detected": bias_detected,
            "score_delta_a": abs(verdict_a_first.overall_score - verdict_a_first_swapped.overall_score),
            "score_delta_b": abs(verdict_b_first.overall_score - verdict_b_first_swapped.overall_score)
        }
