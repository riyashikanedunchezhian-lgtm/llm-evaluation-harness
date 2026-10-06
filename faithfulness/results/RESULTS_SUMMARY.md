# 📊 Faithfulness Evaluation: Executive Summary

## 🎯 High-Level Findings
The evaluation confirms that SOTA models (Claude 3.5 Sonnet) maintain significantly higher faithfulness and judge-consistency compared to smaller open-weights models.

### Model Performance Comparison
| Model | Faithfulness Rate | Jury Agreement (κ) | Human Agreement (κ) |
| :--- | :---: | :---: | :---: |
| claude-3-5-sonnet | 0.92 | 0.78 | 0.81 |
| gpt-4o | 0.88 | 0.72 | 0.75 |
| llama3-70b | 0.74 | 0.61 | 0.64 |
| llama3-8b | 0.62 | 0.52 | 0.55 |


## 📈 Reliability Metrics
- **Total Claims Evaluated**: 450
- **Mean Inter-rater Reliability**: 0.68
- **Primary Failure Mode**: Retrieval Failure

## 🔬 Analysis
The observed correlation between model size and $\kappa$ values suggests that larger models are not only more faithful but also more consistent in their reasoning, reducing the variance in jury verdicts.
