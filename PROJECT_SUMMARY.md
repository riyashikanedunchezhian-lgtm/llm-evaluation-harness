# LLM Evaluation Harness - Project Summary

## What Was Built

A comprehensive LLM evaluation system demonstrating professional AI engineering skills with:

### Core Components

1. **Test Set** (`prompts/test_set.json`)
   - 25 test prompts across 4 categories
   - Factual Q&A (7 tests), Summarization (5 tests), Code Generation (6 tests), Reasoning (7 tests)
   - Each with reference answers and evaluation criteria

2. **Model Interface** (`src/models.py`)
   - Unified client for Anthropic and OpenAI APIs
   - Automatic metric tracking (latency, tokens, cost)
   - Support for custom system prompts

3. **LLM-as-a-Judge System** (`src/judge.py`)
   - 5-dimension rubric (correctness, relevance, conciseness, clarity, safety)
   - 1-5 scale with detailed justifications
   - JSON-structured output for parsing

4. **Jury Evaluation** (`src/judge.py`)
   - 3 independent judge calls per evaluation
   - Aggregation with average and majority vote
   - Variance tracking and consensus detection
   - Reduces single-judge noise and bias

5. **Position Bias Checker** (`src/judge.py`)
   - Detects position bias by swapping answer order
   - Quantifies bias magnitude
   - Threshold-based detection (>0.5 score change)

6. **Evaluation Harness** (`src/harness.py`)
   - Orchestrates full evaluation pipeline
   - Runs tests across multiple models
   - Comprehensive metrics per test case
   - Results export to JSON

7. **Configuration** (`src/config.py`)
   - Model configurations with pricing
   - Judge model selection
   - Rubric dimensions and jury size settings

8. **Command-line Interface** (`run_evaluation.py`)
   - Flexible model selection
   - Category filtering
   - Bias checking option
   - Custom output paths

9. **Visualization Dashboard** (`dashboard.py`)
   - Streamlit-based interactive UI
   - Score vs cost/latency scatter plots
   - Model comparison with bar charts and radar plots
   - Category analysis with heatmaps
   - Detailed results table with search
   - Judge analysis with consensus metrics

10. **Test Suite** (`tests/`)
    - Aggregation logic tests (majority vote, variance, consensus)
    - Position bias detection tests
    - Synthetic judge validation

11. **Documentation**
    - Comprehensive README with methodology
    - Quick start guide
    - API key configuration template

## Key Features Implemented

✅ **LLM-as-a-Judge**: Multi-dimensional scoring with justifications
✅ **Jury Evaluation**: 3-judge aggregation with variance tracking
✅ **Bias Mitigation**: Position bias detection and analysis
✅ **Metrics Tracking**: Quality, latency, tokens, cost per evaluation
✅ **Multiple Models**: Support for Claude and GPT models
✅ **Interactive Dashboard**: Streamlit visualization
✅ **Test Coverage**: Unit tests for core logic
✅ **Comprehensive Docs**: README explaining methodology and tradeoffs
✅ **Parallel Execution**: ~3x latency reduction through concurrent judge calls

## Project Structure

```
llm-harness/
├── data/                      # Results storage
├── prompts/
│   └── test_set.json         # 25 test cases across 4 categories
├── src/
│   ├── __init__.py
│   ├── config.py             # Model configs with pricing
│   ├── models.py             # API client with metrics
│   ├── judge.py              # LLM-as-a-Judge + jury + bias check
│   └── harness.py            # Main orchestration
├── tests/
│   ├── test_aggregation.py   # Jury aggregation tests
│   └── test_position_bias.py # Bias detection tests
├── dashboard.py              # Streamlit visualization
├── run_evaluation.py        # CLI entry point
├── requirements.txt         # Dependencies
├── .env.example             # API key template
├── README.md               # Full documentation
└── QUICKSTART.md           # Quick start guide
```

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run evaluation:**
   ```bash
   python run_evaluation.py
   ```

4. **View results:**
   ```bash
   streamlit run dashboard.py
   ```

5. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

## What This Demonstrates

This project showcases:
- **Professional LLM evaluation methodology** beyond simple API calls
- **Understanding of LLM limitations** (bias, variance, reproducibility)
- **Systematic approach** to model comparison
- **Full-stack implementation** from data collection to visualization
- **Testing mindset** with unit tests for critical logic
- **Documentation skills** explaining complex methodology clearly
- **Cost-awareness** with detailed pricing and cost tracking
- **Bias awareness** with active mitigation strategies

## Ready for Interview

This project is ready to demonstrate for AI Engineer Intern roles, specifically:
- LLM-as-a-Judge implementation
- Jury-style evaluation benefits
- Bias detection and mitigation
- Systematic model comparison
- Quality vs cost vs latency tradeoff analysis
