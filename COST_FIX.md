# Cost Calculation Fix

## Problem Identified

The original cost calculation only accounted for the candidate model's API costs, completely ignoring the costs of the 3 judge model calls. This significantly underestimated the true cost of evaluations.

## Solution Implemented

### 1. Enhanced Judge Verdict Tracking
- Added `input_tokens` and `output_tokens` fields to `JudgeVerdict` dataclass
- Modified `Judge.evaluate()` to track token usage from each judge call
- Modified `Jury.evaluate()` to aggregate total judge token usage across all jury members

### 2. New Cost Calculation Method
- Created `_calculate_total_cost()` method that accounts for both:
  - Candidate model costs (input + output tokens)
  - Judge model costs (input + output tokens × jury_size)
- Uses appropriate pricing for both candidate and judge models

### 3. Updated Evaluation Result Structure
- Added `judge_input_tokens` and `judge_output_tokens` to `EvaluationResult`
- These track the total token usage across all judge calls
- Enables detailed cost breakdown analysis

### 4. Dashboard Updates
- Added judge token columns to summary dataframe
- Updated display options to show judge token usage
- Enhanced model metrics to include judge token aggregation

## Cost Breakdown Example

For a typical evaluation with Claude 3 Haiku as candidate and Claude 3.5 Sonnet as judge:

```
Candidate Model (Claude 3 Haiku):
- Input: 100 tokens × $0.25/1K = $0.025
- Output: 200 tokens × $1.25/1K = $0.250
- Subtotal: $0.275

Judge Model (Claude 3.5 Sonnet) × 3 calls:
- Input: 150 tokens × 3 calls × $3.00/1K = $1.35
- Output: 100 tokens × 3 calls × $15.00/1K = $4.50
- Subtotal: $5.85

Total Cost: $6.125
- Candidate: 4.5%
- Judge: 95.5%
```

## Key Insights

1. **Judge costs dominate**: Because the judge model is typically stronger (more expensive) and runs 3 times per evaluation, judge costs often represent 80-95% of total cost.

2. **Candidate model choice matters less for cost**: Switching from Haiku to Sonnet as candidate model has minimal impact on total cost since judge costs dominate.

3. **Accurate budgeting**: The new calculation enables accurate cost estimation for evaluation runs.

4. **Transparency**: Dashboard now shows judge token usage, making the cost breakdown clear.

## Files Modified

- `src/judge.py`: Added token tracking to JudgeVerdict and Jury
- `src/harness.py`: Added total cost calculation and enhanced result structure
- `dashboard.py`: Updated to display judge token information
- `QUICKSTART.md`: Updated cost estimates to reflect true costs
- `README.md`: Enhanced cost calculation documentation

## Validation

The fix was validated with a test script showing:
- Proper token tracking across all judge calls
- Correct cost aggregation (candidate + judge × jury_size)
- Realistic cost distribution (judge costs dominating total)
