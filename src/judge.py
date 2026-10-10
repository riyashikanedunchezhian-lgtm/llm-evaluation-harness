"""LLM-as-a-Judge evaluation system."""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
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
    failure_reason: Optional[str] = None  # Tracks why evaluation failed (if applicable)

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

        try:
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
        except Exception as e:
            # Return fallback verdict with failure reason on any error
            logger.error(f"Judge {judge_id} evaluation failed: {str(e)}")
            return self._create_fallback_verdict("", judge_id, str(e))
    
    def _build_judge_system_prompt(self) -> str:
        """Build system prompt for the judge with few-shot calibration examples."""
        dimensions_desc = "\n".join([
            f"- {dim}: {self._get_dimension_description(dim)}"
            for dim in RUBRIC_DIMENSIONS
        ])

        # Few-shot calibration examples for consistent scoring
        few_shot_examples = """
**Calibration Examples:**

*Example 1 - Score 5 (Excellent):*
Prompt: "Explain quantum entanglement in simple terms"
Response: "Quantum entanglement is when two particles become linked so that measuring one instantly affects the other, no matter the distance. Think of it like magic dice: if you roll a 6 on one die, the other die—even on Mars—will instantly show 6 too. Einstein called this 'spooky action at a distance.'"
Scores: correctness=5, relevance=5, conciseness=5, clarity=5, safety=5
Justification: Accurate, directly answers the prompt, concise, clear analogy, no safety issues.

*Example 2 - Score 3 (Average):*
Prompt: "What is the capital of France?"
Response: "The capital city of France is Paris, which is located in the northern part of the country and has a population of about 2.1 million people. It's known for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
Scores: correctness=4, relevance=3, conciseness=2, clarity=4, safety=5
Justification: Factually correct but overly verbose for a simple question; relevance suffers from unnecessary details.

*Example 3 - Score 1 (Poor):*
Prompt: "Write a Python function to calculate fibonacci numbers"
Response: "Fibonacci numbers are a sequence where each number is the sum of the two preceding ones. They appear in nature like flower petals and shells. The golden ratio is related to them."
Scores: correctness=1, relevance=1, conciseness=3, clarity=3, safety=5
Justification: Completely fails to provide code; irrelevant to the coding task despite being factually true about fibonacci.

When evaluating, calibrate your scores against these examples."""

        return f"""You are an expert AI evaluator. Your task is to evaluate AI responses on the following dimensions:

{dimensions_desc}

Rate each dimension on a scale of 1-5:
1: Poor - Fails completely on this dimension
2: Below Average - Significant issues on this dimension
3: Average - Meets basic expectations but has notable flaws
4: Good - Performs well with minor issues
5: Excellent - Exceeds expectations on this dimension

{few_shot_examples}

Provide your evaluation in JSON format with this structure:
{{
    "scores": [
        {{"dimension": "correctness", "score": 4, "justification": "..."}},
        ...
    ]
}}

Be objective, fair, and provide clear justifications for each score. Calibrate against the examples above."""
    
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
                judge_id=judge_id,
                failure_reason=None
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
            output_tokens=0,
            failure_reason=error
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

        # Calculate Fleiss' Kappa for inter-rater reliability
        fleiss_kappa = self._calculate_fleiss_kappa(verdicts)

        return {
            "dimension_scores": aggregated_scores,
            "overall_average": overall_avg,
            "overall_std": overall_std,
            "jury_size": len(verdicts),
            "consensus": overall_std < 0.5,  # High consensus if std dev is low
            "fleiss_kappa": fleiss_kappa,
            "inter_rater_reliability": self._interpret_kappa(fleiss_kappa)
        }

    def _calculate_fleiss_kappa(self, verdicts: List[JudgeVerdict]) -> float:
        """Calculate Fleiss' kappa for multiple raters on categorical data.

        Adapts the standard Fleiss' kappa formula for 5-point Likert scale ratings
        across multiple dimensions by computing kappa per dimension and averaging.
        """
        if not verdicts or len(verdicts) < 2:
            return 0.0

        jury_size = len(verdicts)
        num_items = len(RUBRIC_DIMENSIONS)
        num_categories = 5  # 1-5 scale

        # Build rating matrix per dimension: rows=dimensions, cols=categories
        kappas = []

        for dim in RUBRIC_DIMENSIONS:
            # Initialize rating matrix for this dimension
            ratings = [[0] * num_categories for _ in range(num_items)]

            # Fill ratings for each item (here, each item is a single dimension)
            # We treat each dimension as an "item" and each judge as a rater
            for i, verdict in enumerate(verdicts):
                # Find the score for this dimension
                for score in verdict.scores:
                    if score.dimension == dim:
                        category = score.score - 1  # Convert 1-5 to 0-4 index
                        if 0 <= category < num_categories:
                            ratings[0][category] += 1
                        break

            # Calculate Fleiss' kappa for this dimension
            n = jury_size  # number of raters
            N = 1  # we have 1 "item" per dimension (the dimension itself)
            k = num_categories

            if n <= 1:
                kappas.append(0.0)
                continue

            # Step 1: Category proportions
            total_ratings = N * n
            category_proportions = []
            for j in range(k):
                category_total = ratings[0][j]
                category_proportions.append(category_total / total_ratings)

            # Step 2: Observed agreement
            sum_squared = sum(count ** 2 for count in ratings[0])
            P_bar = (sum_squared - n) / (n * (n - 1)) if n > 1 else 0.0

            # Step 3: Expected agreement
            P_e = sum(p ** 2 for p in category_proportions)

            # Step 4: Kappa
            if P_e >= 1.0:
                kappa = 1.0
            else:
                kappa = (P_bar - P_e) / (1 - P_e)

            kappas.append(max(0.0, kappa))  # Kappa can't be negative in this context

        # Return average kappa across all dimensions
        return sum(kappas) / len(kappas) if kappas else 0.0

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Fleiss' kappa value."""
        if kappa < 0:
            return "Poor"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"

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
        # Create a combined prompt that presents both responses for comparison
        # Order 1: A then B
        combined_prompt_ab = f"""{prompt}

Please compare and evaluate these two responses:

**Response A:**
{response_a}

**Response B:**
{response_b}"""

        # Order 2: B then A (swapped)
        combined_prompt_ba = f"""{prompt}

Please compare and evaluate these two responses:

**Response B:**
{response_b}

**Response A:**
{response_a}"""

        # Evaluate with A first, then B
        verdict_ab = self.judge.evaluate(
            prompt=combined_prompt_ab,
            response="",  # Empty response as we're evaluating the comparison itself
            reference_answer=reference_answer,
            evaluation_criteria=evaluation_criteria,
            judge_id=0
        )

        # Evaluate with B first, then A (swap order)
        verdict_ba = self.judge.evaluate(
            prompt=combined_prompt_ba,
            response="",  # Empty response as we're evaluating the comparison itself
            reference_answer=reference_answer,
            evaluation_criteria=evaluation_criteria,
            judge_id=1
        )

        # Extract scores for each response from the judgments
        # We need to parse which score corresponds to which response
        # For simplicity, we'll assume the judge provides scores in order mentioned
        # A more robust approach would have the judge explicitly label which score is for which response

        # Calculate bias by comparing overall scores when order is swapped
        bias_detected = abs(verdict_ab.overall_score - verdict_ba.overall_score) > 0.5

        return {
            "response_a_original_score": verdict_ab.overall_score,  # Score when A was first
            "response_a_swapped_score": verdict_ba.overall_score,   # Score when A was second
            "response_b_original_score": verdict_ab.overall_score,  # Score when B was second
            "response_b_swapped_score": verdict_ba.overall_score,   # Score when B was first
            "bias_detected": bias_detected,
            "score_delta_a": abs(verdict_ab.overall_score - verdict_ba.overall_score),
            "score_delta_b": abs(verdict_ab.overall_score - verdict_ba.overall_score)
        }
