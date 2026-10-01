# Faithfulness Evaluation for Long-Document Summarization

## Research Question

**Does an automated LLM-judge faithfulness pipeline agree with human judgment, and where does it fail?**

This project investigates the reliability of automated faithfulness evaluation for long-document summarization using a rigorous methodology that combines LLM-as-a-Judge with jury aggregation, retrieval-augmented verification, and human validation.

## Overview

This is a research-oriented extension of the LLM Evaluation Harness, specifically focused on evaluating the faithfulness (factual accuracy) of long-document summarization. Unlike generic summarization systems, this project emphasizes the evaluation methodology itself, implementing a complete pipeline for:

1. **Atomic claim decomposition** from summaries
2. **Retrieval-augmented verification** using dense passage retrieval
3. **Jury-style classification** (Supported/Contradicted/Unverifiable) with 3 judges
4. **Inter-judge agreement metrics** (Fleiss' kappa, percent agreement)
5. **Position bias detection** adapted for faithfulness evaluation
6. **Human validation** with Cohen's kappa agreement calculation
7. **Comparative analysis** of faithfulness vs fluency correlations

## Methodology

### Dataset
- **Primary**: GovReport dataset (government reports with 2,000+ words)
- **Alternative**: arXiv scientific papers subset
- **Scope**: 30 documents per evaluation
- **Stress test**: Minimum 2,000 words per document to test context handling

### Models Evaluated
- Claude 3 Haiku (fast, cost-effective)
- Claude 3.5 Sonnet (high-performance)
- GPT-4o Mini (cost-effective OpenAI model)

### Pipeline Architecture

```
Document → Summary Generation → Claim Decomposition → Passage Retrieval → 
Jury Classification → Aggregation → Bias Check → Human Validation → Analysis
```

### Key Innovations

#### 1. Atomic Claim Decomposition
- LLM-based decomposition of summaries into single-fact claims
- Validation for non-overlapping, atomic claims
- Fallback parsing for robust JSON extraction

#### 2. Retrieval-Augmented Verification
- Dense retrieval using sentence-transformers (all-MiniLM-L6-v2)
- Document chunking with overlap for context preservation
- Top-k passage retrieval per claim

#### 3. Jury-Style Classification
- 3 independent judges per claim classification
- Labels: Supported, Contradicted, Unverifiable
- Majority vote aggregation with confidence scoring
- Required justification text for explainability

#### 4. Inter-Judge Agreement Metrics
- **Fleiss' kappa**: Multi-rater agreement for categorical data
- **Percent agreement**: Simple agreement rate
- **Confidence estimation**: Based on jury consensus level

#### 5. Position Bias Detection
- Tests whether claim-then-source vs source-then-claim order affects verdicts
- Threshold-based detection (>0.2 confidence delta)
- Analysis of bias patterns across evaluations

#### 6. Human Validation Pipeline
- Template generation for blind human annotation
- Cohen's kappa calculation for human-automated agreement
- Disagreement categorization (ambiguous claims, retrieval failures, judge strictness)
- Honest reporting even when agreement is mediocre

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
pip install -r faithfulness/requirements.txt
```

### Run Full Pipeline
```bash
# Basic evaluation (30 documents, GovReport dataset)
python run_faithfulness.py

# Custom configuration
python run_faithfulness.py --dataset arxiv --num-docs 50 --min-words 2500

# Use specific models
python run_faithfulness.py --models claude-3-haiku-20240307 gpt-4o-mini

# Skip summary generation (use existing)
python run_faithfulness.py --skip-summaries

# Include position bias checks
python run_faithfulness.py --bias-check
```

### Human Validation
```bash
# Create annotation template
python run_faithfulness.py --create-annotations

# Annotate the template file manually, then process:
python run_faithfulness.py --process-annotations faithfulness/data/completed_annotations.json
```

### Run Tests
```bash
# Faithfulness-specific tests
pytest faithfulness/tests/ -v

# All tests
pytest tests/ faithfulness/tests/ -v
```

## Research Findings

### Human-Automated Agreement

**Human Validation Results** (to be populated after running evaluation):

- **Cohen's kappa**: [X.XXX] (to be measured)
- **Percent agreement**: [XX.X%] (to be measured)
- **Agreement category**: [substantial/moderate/slight/poor] (to be measured)

**Interpretation**: The automated system achieves [substantial/moderate] agreement with human judgment, indicating [strong/acceptable] reliability for faithfulness evaluation.

### Disagreement Analysis

**Categorized Failure Patterns** (to be populated after human validation):

- **Ambiguous claims**: [X] claims - Claims where the correct classification is subjective or context-dependent
- **Retrieval failures**: [X] claims - Retrieval system missed the relevant passage, leading to incorrect "Unverifiable" labels
- **Judge over-strictness**: [X] claims - Automated judge was too strict compared to human judgment
- **Judge leniency**: [X] claims - Automated judge was too lenient compared to human judgment
- **Human error**: [X] claims - Human annotator made an error (self-identified)

**Key Insight**: The most common disagreement pattern is [pattern], suggesting that [improvement opportunity].

### Model Comparison

**Faithfulness Rates by Model** (to be populated after evaluation):

| Model | Faithfulness Rate | Contradiction Rate | Unverifiable Rate | Avg Jury Agreement |
|-------|-------------------|-------------------|-------------------|-------------------|
| Claude 3 Haiku | [XX.X%] | [XX.X%] | [XX.X%] | [X.XX] |
| Claude 3.5 Sonnet | [XX.X%] | [XX.X%] | [XX.X%] | [X.XX] |
| GPT-4o Mini | [XX.X%] | [XX.X%] | [XX.X%] | [X.XX] |

**Finding**: [Model X] achieved the highest faithfulness rate at [XX.X%], while [Model Y] had the lowest at [XX.X%]. The difference suggests [implication].

### Faithfulness vs Fluency Correlation

**Correlation Analysis** (to be populated after evaluation):

- **Pearson correlation**: [X.XXX] (p=[X.XXX])
- **Spearman correlation**: [X.XXX] (p=[X.XXX])
- **Significance**: [significant/not significant]
- **Interpretation**: [strong positive correlation/strong negative correlation/no significant correlation]

**Research Context**: Previous research has found that more fluent summaries are sometimes less faithful (more confident hallucination). Our analysis [confirms/contradicts] this finding, showing [description of pattern].

## Limitations and Threats to Validity

### 1. Retrieval Quality as a Confound

**Issue**: A claim marked "Unverifiable" might actually be true - the retrieval system may have simply failed to find the relevant passage. This is a fundamental limitation: our faithfulness evaluation is measuring "verifiability given retrieved passages," not absolute truth.

**Impact**: This systematically biases results toward "Unverifiable" labels, especially for:
- Claims mentioned in less central parts of documents
- Claims requiring synthesis across multiple passages
- Claims with paraphrased wording different from source

**Mitigation**: 
- Use multiple retrieval methods (dense + sparse)
- Increase top-k passages per claim
- Manual inspection of "Unverifiable" claims
- Report unverifiable rate as a quality metric

### 2. Judge Model Bias

**Issue**: The judge model (Claude 3.5 Sonnet) has its own biases that may not reflect human preferences, particularly for:
- Edge cases and ambiguous claims
- Domain-specific knowledge
- Recent events not in training data

**Impact**: Systematic bias in one direction (e.g., overly strict) could affect all models similarly, masking true differences.

**Mitigation**:
- Use multiple judge models and compare
- Human validation of judge calibration
- Transparent reporting of judge characteristics

### 3. Claim Decomposition Quality

**Issue**: The quality of claim decomposition affects all downstream evaluation. Poor decomposition can lead to:
- Overly specific claims that are hard to verify
- Overly general claims that lose important details
- Missed claims entirely

**Impact**: Inconsistent claim granularity across summaries makes comparison difficult.

**Mitigation**:
- Manual validation of decomposition quality
- Standardized decomposition prompts
- Inter-rater reliability for decomposition

### 4. Dataset Representativeness

**Issue**: GovReport documents may not represent all summarization use cases. They are:
- Government reports (formal, structured)
- Policy-focused (specific domain)
- English language only

**Impact**: Findings may not generalize to:
- Scientific papers (different structure)
- News articles (different style)
- Creative content (different objectives)

**Mitigation**:
- Test on multiple datasets
- Report dataset characteristics
- Avoid overgeneralizing findings

### 5. Human Annotation Limitations

**Issue**: Human validation itself has limitations:
- Subjectivity in faithfulness judgment
- Annotator fatigue (30-50 claims is substantial)
- Blind annotation may still have unconscious biases

**Impact**: Human judgment is not a perfect ground truth, making Cohen's kappa an estimate rather than absolute measure.

**Mitigation**:
- Multiple human annotators where feasible
- Clear annotation guidelines
- Report annotator characteristics

## Technical Details

### File Structure
```
faithfulness/
├── config.py                  # Configuration and schemas
├── dataset.py                 # Dataset loading (GovReport/arXiv)
├── claim_decomposition.py     # Atomic claim extraction
├── retrieval.py               # Dense passage retrieval
├── faithfulness_judge.py      # Faithfulness classification
├── jury_aggregation.py        # Jury logic + Fleiss' kappa
├── position_bias.py           # Position bias detection
├── human_validation.py        # Human validation + Cohen's kappa
├── analysis.py                # Comparative analysis
├── pipeline.py                # Main orchestration
├── tests/
│   ├── test_claim_decomposition.py
│   ├── test_jury_aggregation.py
│   └── test_agreement_metrics.py
└── data/                      # Results storage
```

### Key Algorithms

**Fleiss' Kappa Calculation**:
```
κ = (P̄ - P̄e) / (1 - P̄e)

Where:
- P̄ = Average observed agreement across all items
- P̄e = Expected agreement by chance
- κ = 1.0 (perfect agreement), κ = 0.0 (chance agreement), κ < 0 (worse than chance)
```

**Cohen's Kappa Calculation**:
```
κ = (Po - Pe) / (1 - Pe)

Where:
- Po = Observed agreement proportion
- Pe = Expected agreement proportion
- Same interpretation as Fleiss' kappa
```

## Comparison with Related Work

This project differs from generic summarization systems in several key ways:

1. **Methodology Focus**: The deliverable is the evaluation methodology, not the summarizer itself
2. **Rigorous Validation**: Includes human validation with Cohen's kappa (uncommon in automated systems)
3. **Bias Awareness**: Explicit position bias detection and reporting
4. **Inter-Judge Metrics**: Fleiss' kappa for jury consistency (methodological detail reviewers expect)
5. **Honest Reporting**: Commits to reporting agreement even if mediocre

## Future Work

Potential improvements for research rigor:

1. **Multiple Retrieval Methods**: Combine dense (sentence-transformers) with sparse (BM25) retrieval
2. **Cross-Judge Validation**: Use different judge models and compare results
3. **Larger Human Validation**: Scale to 100+ human annotations with multiple annotators
4. **Dataset Expansion**: Test on scientific papers, news articles, legal documents
5. **Fine-grained Labels**: Add "Partially Supported" for nuanced cases
6. **Error Analysis**: Deeper analysis of systematic errors by model and category

## Citation

If you use this methodology in your research, please cite:

```
Faithfulness Evaluation for Long-Document Summarization
LLM-as-a-Judge with Jury Aggregation and Human Validation
[Your Name], [Year]
GitHub: [repository URL]
```

## License

MIT License - research and educational use encouraged.

## Acknowledgments

This methodology builds on research in:
- LLM-as-a-Judge evaluation
- Retrieval-augmented generation
- Inter-rater reliability metrics
- Faithfulness evaluation for summarization
