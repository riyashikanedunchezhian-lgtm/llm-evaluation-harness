# Quick Start Guide

This guide will help you get the LLM evaluation harness running quickly.

## Prerequisites

- Python 3.8 or higher
- API keys for Anthropic and/or OpenAI
- pip (Python package manager)

## Setup (5 minutes)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure API keys:**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

3. **Run a quick evaluation:**
```bash
python run_evaluation.py --models claude-3-haiku-20240307 --categories factual_qa
```

This will run 7 factual Q&A tests with Claude 3 Haiku (fast and inexpensive).

4. **View results:**
```bash
streamlit run dashboard.py
```

## Common Use Cases

### Compare two models
```bash
python run_evaluation.py --models claude-3-haiku-20240307 gpt-4o-mini
```

### Test specific categories
```bash
python run_evaluation.py --categories code_generation reasoning
```

### Full evaluation with bias checking
```bash
python run_evaluation.py --bias-check
```

### Run tests to verify implementation
```bash
pytest tests/ -v
```

## Understanding the Output

After running evaluation, you'll see:
- Progress indicators for each test
- Summary statistics by model and category
- Total cost and token usage
- Results saved to `data/results.json`

The dashboard provides:
- Visual comparisons between models
- Score vs cost/latency scatter plots
- Category-specific performance
- Detailed per-test results

## Next Steps

1. **Customize the test set**: Edit `prompts/test_set.json` to add your own test cases
2. **Add more models**: Edit `src/config.py` to add new model configurations
3. **Adjust judge rubric**: Modify `src/judge.py` to change evaluation criteria
4. **Analyze results**: Use the dashboard to explore tradeoffs between models

## Troubleshooting

**API key errors**: Ensure your `.env` file is in the project root and contains valid keys.

**Out of memory**: Reduce the number of models or categories being evaluated.

**Slow evaluation**: Start with just one model and one category to test.

**Dashboard not loading**: Ensure you've run an evaluation first to generate `data/results.json`.

## Cost Estimation

Rough costs for a full evaluation (25 tests × 2 models):
- Claude 3 Haiku: ~$0.10-0.20
- Claude 3.5 Sonnet: ~$1.00-2.00
- GPT-4o Mini: ~$0.05-0.10
- GPT-4o: ~$1.50-3.00

Costs include both candidate model calls and judge model calls (3x multiplier).
