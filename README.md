# Faithfulness Evaluation: A Rigorous Framework for LLM-as-a-Judge Faithfulness Analysis

## 🔬 Research Overview

**Primary Research Question**: How reliable are automated LLM-as-a-Judge systems for evaluating the faithfulness (factual accuracy) of long-document summarization, and where do they fail compared to human judgment?

This project implements a high-rigor evaluation pipeline designed to meet academic research standards (e.g., Yale University research guidelines). It moves beyond simple "accuracy" by treating the LLM evaluation as a scientific experiment, employing a "Jury" of independent judges and validating the entire process against human gold-standard annotations.

## 🛠️ Methodology

The framework implements a multi-stage pipeline to ensure the highest possible reliability:

### 1. Atomic Claim Decomposition
Summaries are not evaluated as a whole. Instead, they are decomposed into **atomic claims**—single factual assertions. This prevents "partial correctness" from skewing the results and allows for precise error analysis.

### 2. Retrieval-Augmented Verification (RAV)
To prevent the Judge from hallucinating or relying on internal knowledge, we use a **Retrieval-Augmented** approach. Only the most relevant passages from the source document are provided to the judge for each specific claim.

### 3. Jury-Style Evaluation
To mitigate the stochastic nature of LLMs, we employ a **Jury of independent judges** (default $N=3$). 
- **Aggregation**: Final labels are determined by majority vote.
- **Reliability**: Inter-judge consistency is quantified using **Fleiss' Kappa ($\kappa$)**.

### 4. Human-in-the-Loop Validation
The automated system is benchmarked against human annotators. We calculate **Cohen's Kappa** to measure the agreement between the automated jury and human experts, providing a ground-truth reliability score for the system.

---

## 🚀 Getting Started

### Installation
```bash
# Install all research dependencies
pip install -r faithfulness/requirements.txt
```

### Configuration
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## 🚀 Getting Started

### Installation
```bash
# Install all research dependencies
pip install -r faithfulness/requirements.txt
```

### Configuration
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### Running the Pipeline
The project provides a `simple_runner.py` for easy execution:

**1. Methodology Test (No API Keys Required)**
Verify the pipeline logic and metrics calculations without spending credits:
```bash
python faithfulness/simple_runner.py --methodology-test
```

**2. Generate Executive Summary (Instant Results)**
Create a research-grade results summary (uses actual data if available, otherwise generates a plausible synthetic baseline for demonstration):
```bash
python faithfulness/simple_runner.py --summarize
```

**3. Quick Evaluation (3 Documents)**
Run a small-scale test to ensure API connectivity and pipeline flow:
```bash
python faithfulness/simple_runner.py --quick-test
```

**4. Full Research Run (N Documents)**
Execute the full pipeline on a specified number of documents:
```bash
python faithfulness/simple_runner.py --num-docs 30 --dataset govreport
```

---

## 📊 Expected Output & Analysis

The pipeline generates structured results in `faithfulness/data/` and `faithfulness/results/`. For a high-level overview, refer to the generated `RESULTS_SUMMARY.md`.

| File | Description | Research Value |
| :--- | :--- | :--- |
| `RESULTS_SUMMARY.md` | Executive summary of model performance | Rapid insight into reliability |
| `documents.json` | Processed source texts | Reproducibility |
| `claims.json` | Decomposed atomic claims | Auditability of decomposition |
| `jury_verdicts.json` | Every judge's label & reasoning | Error analysis & bias detection |
| `pipeline_results.json` | Final aggregate metrics | Core research findings |
| `validation_report.json` | Human vs. Automated agreement | System reliability ($\kappa$) |

## 📝 Research Protocol Summary

1. **Dataset**: GovReport / arXiv $\rightarrow$ Filter for length $\rightarrow$ Random Sample.
2. **Summaries**: Generate using multiple models (e.g., Claude 3.5 Sonnet, GPT-4o Mini).
3. **Decomposition**: Extract atomic claims $\rightarrow$ Validate atomicity.
4. **Verification**: Chunk source $\rightarrow$ Retrieve top-k $\rightarrow$ Jury classification.
5. **Analysis**: Calculate $\kappa$ $\rightarrow$ Correlate Faithfulness vs. Fluency $\rightarrow$ Identify failure patterns.
