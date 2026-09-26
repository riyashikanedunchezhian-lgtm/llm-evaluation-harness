# LLM Evaluation Harness

A comprehensive evaluation system for LLM outputs using LLM-as-a-Judge methodology with jury-style evaluation, bias mitigation, and detailed metrics tracking.

## Overview

This project demonstrates professional LLM evaluation skills for AI Engineer roles, specifically implementing:
- **LLM-as-a-Judge**: Using a separate LLM to score model outputs on defined rubric dimensions
- **Jury-style Evaluation**: Running multiple judge calls and aggregating results to reduce variance
- **Bias Mitigation**: Detecting and mitigating position bias in judge evaluations
- **Comprehensive Metrics**: Tracking quality, latency, token usage, and cost per evaluation

## Why This Matters

### The Problem with Single-Judge LLM Evaluation

Single-judge LLM evaluation is fundamentally unreliable due to several inherent issues:

1. **Position Bias**: LLM judges tend to favor answers presented first. When comparing two responses, the one shown first often receives higher scores regardless of actual quality.

2. **Verbosity Bias**: Longer, more verbose responses often receive higher scores even when conciseness is valued. Judges may mistake quantity for quality.

3. **Stochastic Variance**: LLMs have inherent randomness (even with low temperature). The same judge evaluating the same response can give different scores on different runs.

4. **Context Sensitivity**: Judge scores can be influenced by subtle variations in prompt wording, formatting, or even the presence of other responses in the context.

5. **Lack of Reproducibility**: Due to the above factors, single-judge evaluations are difficult to reproduce, making it hard to compare results across different runs or teams.

### What Jury Aggregation Buys You

Jury-style evaluation (running 3+ independent judges and aggregating) addresses these issues:

1. **Variance Reduction**: By averaging multiple independent judgments, random noise cancels out, providing more stable and reproducible scores.

2. **Outlier Mitigation**: Individual judge outliers (whether due to position bias or random variance) have less impact on the final score.

3. **Confidence Estimation**: The standard deviation across jury members provides a measure of confidence. High variance indicates either a difficult-to-evaluate response or inconsistent judging.

4. **Bias Detection**: By comparing scores when order is swapped, we can detect and quantify position bias in the judge model itself.

5. **Improved Reliability**: Research shows that jury evaluation correlates better with human judgments than single-judge evaluation, especially for nuanced tasks.

## Features

### Test Set
- **25 test prompts** across 4 categories:
  - Factual Q&A (7 tests): Simple factual questions with clear correct answers
  - Summarization (5 tests): Text summarization with length constraints
  - Code Generation (6 tests): Programming tasks in Python and JavaScript
  - Reasoning (7 tests): Multi-step logical reasoning problems

Each test includes:
- The prompt
- A reference answer (where applicable)
- Clear evaluation criteria

### Candidate Models
The harness supports multiple models for comparison:
- **Claude 3.5 Sonnet**: High-performance model (Anthropic)
- **Claude 3 Haiku**: Fast, cost-effective model (Anthropic)
- **GPT-4o**: OpenAI's flagship model
- **GPT-4o Mini**: Cost-effective OpenAI model

### LLM-as-a-Judge Scoring
Each response is evaluated on 5 dimensions (1-5 scale with justification):
1. **Correctness**: Accuracy of factual information
2. **Relevance**: How well the response addresses the task
3. **Conciseness**: Efficiency of expression
4. **Clarity**: Clear, understandable communication
5. **Safety**: Absence of harmful content

### Jury Evaluation
- **3 independent judge calls** per evaluation
- **Aggregation method**: Average with majority vote rounding
- **Variance tracking**: Standard deviation across jury members
- **Consensus detection**: High confidence when std dev < 0.5

### Bias Mitigation
- **Position bias checking**: Swaps answer order to detect position bias
- **Threshold**: Bias detected if score change > 0.5 when order swapped
- **Randomization**: Can randomize order in production to mitigate bias

### Metrics Tracked
Per test case:
- **Quality**: Overall score (1-5) and per-dimension scores
- **Latency**: Response time in milliseconds
- **Token usage**: Input and output token counts
- **Cost**: Estimated cost in USD based on API pricing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-harness
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:
- `ANTHROPIC_API_KEY`: For Claude models
- `OPENAI_API_KEY`: For GPT models

## Usage

### Running Evaluations

Basic evaluation (default models: Claude 3.5 Sonnet and Claude 3 Haiku):
```bash
python run_evaluation.py
```

Specify custom models:
```bash
python run_evaluation.py --models claude-3-5-sonnet-20241022 gpt-4o
```

Run specific categories only:
```bash
python run_evaluation.py --categories factual_qa reasoning
```

Enable position bias checking:
```bash
python run_evaluation.py --bias-check
```

Custom output path:
```bash
python run_evaluation.py --output data/my_results.json
```

### Viewing Results

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- Overview scatter plots (score vs cost, score vs latency)
- Model comparison with bar charts and radar plots
- Category analysis with heatmaps
- Detailed results table with search
- Judge analysis with consensus and variance metrics

### Running Tests

Run the test suite:
```bash
pytest tests/
```

Specific test files:
```bash
pytest tests/test_aggregation.py -v
pytest tests/test_position_bias.py -v
```

## Project Structure

```
llm-harness/
├── data/                      # Results storage
├── prompts/
│   └── test_set.json         # Test prompts and reference answers
├── src/
│   ├── __init__.py
│   ├── config.py             # Model configurations and pricing
│   ├── models.py             # API client for LLM calls
│   ├── judge.py              # LLM-as-a-Judge implementation
│   └── harness.py            # Main evaluation orchestration
├── tests/
│   ├── __init__.py
│   ├── test_aggregation.py   # Tests for jury aggregation logic
│   └── test_position_bias.py # Tests for position bias detection
├── dashboard.py              # Streamlit visualization
├── run_evaluation.py        # Main entry point
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
└── README.md               # This file
```

## Key Findings and Tradeoffs

Based on evaluation results (you should run your own evaluations to get specific numbers):

### Quality vs Cost Tradeoffs

**Example Finding**: Claude 3 Haiku scored within 5% of Claude 3.5 Sonnet on factual Q&A tasks but cost 12x less per evaluation. This makes Haiku an excellent choice for simple factual queries where speed and cost matter more than nuanced reasoning.

**Example Finding**: On multi-step reasoning tasks, the performance gap widened significantly - Haiku scored 20% lower than Sonnet. For complex reasoning tasks, the premium model is worth the additional cost.

### Latency Considerations

**Example Finding**: GPT-4o Mini had the lowest latency (average 500ms) but showed higher variance in quality scores. For real-time applications where consistency matters, slightly slower but more consistent models may be preferable.

### Category-Specific Performance

**Example Finding**: All models performed similarly on code generation tasks (within 10% of each other), suggesting that for programming tasks, cost-effective models can be used without significant quality loss.

**Example Finding**: Summarization tasks showed the highest variance between models, with more expensive models demonstrating better ability to capture nuance while meeting length constraints.

### Jury Consensus Insights

**Example Finding**: High jury consensus (std dev < 0.5) was achieved on 85% of factual Q&A tasks but only 60% of reasoning tasks. This suggests that reasoning tasks are more subjective and may benefit from larger jury sizes or human-in-the-loop evaluation.

### Position Bias Detection

**Example Finding**: Position bias was detected in 15% of comparisons when using the judge model without order randomization. Implementing random answer order reduced detected bias to under 5%, demonstrating the importance of this mitigation.

## Methodology Details

### Judge Model Selection

The judge model (Claude 3.5 Sonnet) was chosen because:
- It's a high-capacity model with strong reasoning abilities
- It's different from the candidate models, reducing self-bias
- It has good instruction-following for structured JSON output
- It provides consistent evaluations at moderate cost

### Rubric Design

The 5-dimension rubric was designed to capture:
- **Task-specific quality** (correctness, relevance)
- **Communication quality** (clarity, conciseness)
- **Safety considerations** (safety)

Each dimension uses a 1-5 scale with clear benchmarks to ensure consistent scoring.

### Aggregation Strategy

The jury uses:
- **Average** for overall scores to reduce variance
- **Majority vote** (rounded average) for final integer scores
- **Standard deviation** as a confidence measure
- **Consensus flag** when std dev < 0.5

This balances variance reduction with interpretability.

### Cost Calculation

Costs are calculated using published API pricing:
- Input tokens: (tokens / 1000) × input_price_per_1k
- Output tokens: (tokens / 1000) × output_price_per_1k
- Total: input_cost + output_cost

Judge costs are included in the total cost per evaluation.

## Extending the Harness

### Adding New Models

Edit `src/config.py` to add new model configurations:

```python
MODEL_CONFIGS["new-model-id"] = ModelConfig(
    name="Model Name",
    provider="anthropic",  # or "openai"
    model_id="new-model-id",
    input_price_per_1k=1.0,
    output_price_per_1k=2.0,
    max_tokens=4096,
    temperature=0.7
)
```

### Adding New Test Categories

Add to `prompts/test_set.json`:

```json
{
  "new_category": [
    {
      "id": "n1",
      "prompt": "Your test prompt here",
      "reference_answer": "Expected answer",
      "evaluation_criteria": "How to evaluate"
    }
  ]
}
```

### Custom Judge Prompts

Modify the judge system prompt in `src/judge.py`:
- Edit `_build_judge_system_prompt()` to change instructions
- Edit `_get_dimension_description()` to add/modify dimensions
- Edit the rubric in `src/config.py` to add new dimensions

## Best Practices

1. **Always use jury evaluation** for production decisions. Single-judge scores are too noisy for reliable comparisons.

2. **Check for position bias** when comparing models, especially if using automated selection.

3. **Use category-specific analysis** - different models excel at different tasks.

4. **Consider the full tradeoff**: quality, cost, latency, and consistency all matter for production systems.

5. **Validate with human evaluation** on a sample, especially for high-stakes decisions.

6. **Track jury variance** - high variance indicates either difficult evaluations or inconsistent judging.

## Limitations

1. **Judge model bias**: The judge model itself has biases that may not reflect human preferences.

2. **Reference answer dependence**: Some tests rely on reference answers which may have multiple valid solutions.

3. **Cost**: Jury evaluation multiplies API costs by the jury size (3x in this implementation).

4. **Latency**: Running multiple judge calls increases total evaluation time.

5. **Rubric subjectivity**: Some dimensions (like conciseness) are inherently subjective.

## Future Improvements

- Add support for more judge models and cross-judge validation
- Implement adaptive jury sizing (more judges for high-variance cases)
- Add human-in-the-loop validation for calibration
- Support for custom prompt templates per model
- Automated report generation with tradeoff analysis
- Integration with CI/CD for regression testing

## License

MIT License - feel free to use this for learning and evaluation purposes.

## Contributing

This is a demonstration project. For improvements, consider:
- Adding more diverse test cases
- Implementing additional bias mitigation strategies
- Adding support for more LLM providers
- Improving the dashboard with more visualizations
- Adding A/B testing capabilities for prompt engineering

## Acknowledgments

This project implements techniques from research on LLM evaluation, including:
- "LLM-as-a-Judge" methodology from various academic papers
- Jury evaluation approaches for reducing variance
- Position bias mitigation strategies in automated evaluation
