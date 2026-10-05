# Faithfulness Evaluation: Yale Research-Level Implementation

## Research Overview

**Primary Research Question**: How reliable are automated LLM-as-a-Judge systems for evaluating the faithfulness (factual accuracy) of long-document summarization, and where do they fail compared to human judgment?

## Yale Research Standards

This implementation meets Yale-level research standards through:

### 1. Methodological Rigor
- **Inter-rater reliability**: Fleiss' kappa for jury consistency (κ > 0.6 target)
- **Human validation**: Cohen's kappa for human-automated agreement (κ > 0.6 target)
- **Bias mitigation**: Position bias detection and correction
- **Statistical significance**: P-value calculations for all correlations
- **Transparent reporting**: Honest reporting of all results, including failures

### 2. Dataset Standards
- **Public datasets**: GovReport (government reports) and arXiv (scientific papers)
- **Sample size**: N=30 documents (standard for initial research)
- **Document length**: 1,500+ words (stress test for context handling)
- **Representative sampling**: Random selection from larger datasets

### 3. Evaluation Standards
- **Atomic claims**: Each claim represents a single factual assertion
- **Retrieval-augmented**: Dense passage retrieval for verification
- **Jury evaluation**: 3 independent judges per claim
- **Multi-dimensional**: Supported/Contradicted/Unverifiable classification
- **Justification required**: All classifications include reasoning

### 4. Reproducibility Standards
- **Version control**: Complete implementation tracking
- **Dependency management**: Explicit requirements files
- **Configuration management**: All parameters saved
- **Structured output**: JSON results with full metadata
- **Test coverage**: Unit tests for critical algorithms

## Quick Start (Simplified)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r faithfulness/requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Run Evaluation (Simplified)
```bash
# Quick test with 3 documents (recommended for first run)
python faithfulness/simple_runner.py --quick-test

# Standard evaluation with 10 documents
python faithfulness/simple_runner.py --num-docs 10

# Use arXiv dataset
python faithfulness/simple_runner.py --dataset arxiv
```

### Human Validation
```bash
# Create annotation template
python faithfulness/simple_runner.py --create-annotations

# Process completed annotations
python faithfulness/simple_runner.py --process-annotations <path>
```

## Research Protocol

### Phase 1: Dataset Preparation
1. Load 30 documents from GovReport (or arXiv)
2. Filter for 1,500+ word documents
3. Save to standardized JSON format
4. Verify document quality and length

### Phase 2: Summary Generation
1. Generate summaries using 3 models:
   - Claude 3 Haiku (cost-effective)
   - Claude 3.5 Sonnet (high-performance)
   - GPT-4o Mini (OpenAI baseline)
2. Save all summaries with metadata
3. Ensure consistent summary length (3-5 sentences)

### Phase 3: Claim Decomposition
1. Decompose each summary into atomic claims
2. Validate claims for atomicity (no compound claims)
3. Filter out vague or overlapping claims
4. Aim for 5-10 claims per summary

### Phase 4: Retrieval-Augmented Verification
1. Chunk documents into 200-word passages with 50-word overlap
2. Generate embeddings using sentence-transformers
3. Retrieve top-3 passages per claim
4. Calculate relevance scores

### Phase 5: Jury Classification
1. Classify each claim using 3 independent judges
2. Labels: Supported, Contradicted, Unverifiable
3. Require justification and confidence for each
4. Aggregate using majority vote

### Phase 6: Quality Metrics
1. Calculate inter-judge agreement (Fleiss' kappa)
2. Report percent agreement and confidence intervals
3. Identify high-variance classifications
4. Flag claims requiring human review

### Phase 7: Human Validation
1. Sample 30-50 claims across all models/documents
2. Blind human annotation (hide automated labels)
3. Calculate Cohen's kappa (human vs automated)
4. Categorize disagreements systematically

### Phase 8: Analysis
1. Calculate per-model faithfulness rates
2. Analyze faithfulness vs fluency correlations
3. Identify systematic failure patterns
4. Generate comprehensive report

## Expected Results Framework

### Model Performance
| Model | Faithfulness Rate | Contradiction Rate | Unverifiable Rate |
|-------|-------------------|-------------------|-------------------|
| Claude 3 Haiku | [XX.X%] | [XX.X%] | [XX.X%] |
| Claude 3.5 Sonnet | [XX.X%] | [XX.X%] | [XX.X%] |
| GPT-4o Mini | [XX.X%] | [XX.X%] | [XX.X%] |

### Agreement Metrics
- **Inter-judge agreement**: Fleiss' κ = [X.XX] (target: > 0.6)
- **Human-automated agreement**: Cohen's κ = [X.XX] (target: > 0.6)
- **Percent agreement**: [XX.X%]

### Correlation Analysis
- **Faithfulness vs fluency**: r = [X.XX], p = [X.XX]
- **Significance**: [significant/not significant]
- **Interpretation**: [description]

## Failure Analysis Framework

### Expected Disagreement Categories
1. **Ambiguous claims** (target: < 20%): Claims where correct classification is subjective
2. **Retrieval failures** (target: < 30%): Retrieval missed relevant passage
3. **Judge over-strictness** (target: < 25%): Automated judge too strict
4. **Judge leniency** (target: < 15%): Automated judge too lenient
5. **Human error** (target: < 10%): Human annotator mistakes

### Threats to Validity
1. **Retrieval quality**: "Unverifiable" may mean retrieval failure, not false claim
2. **Judge model bias**: Claude 3.5 Sonnet may have systematic biases
3. **Claim granularity**: Inconsistent decomposition affects comparison
4. **Dataset representativeness**: GovReport may not generalize to all domains
5. **Human annotation limits**: Subjectivity in faithfulness judgment

## Timeline (Recommended)

- **Week 1**: Dataset loading and summary generation
- **Week 2**: Claim decomposition and retrieval setup
- **Week 3**: Jury classification and aggregation
- **Week 4**: Human validation and agreement calculation
- **Week 5**: Analysis and report writing

## Validation Checklist

Before submission, ensure:

- [ ] Dataset loaded correctly with N=30 documents
- [ ] All summaries generated successfully
- [ ] Claims decomposed with atomicity validation
- [ ] Retrieval system functioning (dense or fallback)
- [ ] Jury classification completed for all claims
- [ ] Inter-judge agreement calculated (Fleiss' κ)
- [ ] Human annotations completed (30-50 claims)
- [ ] Cohen's κ calculated (human vs automated)
- [ ] Disagreements categorized systematically
- [ ] Per-model faithfulness rates calculated
- [ ] Correlation analysis completed
- [ ] All limitations documented honestly
- [ ] Results saved in standardized format
- [ ] README updated with actual findings

## Yale-Specific Reporting

### Abstract Format
"We evaluate the reliability of automated LLM-as-a-Judge systems for faithfulness evaluation in long-document summarization. Using [N] documents from [dataset], we generate summaries with [M] models and decompose them into [X] atomic claims. Each claim is verified using retrieval-augmented jury evaluation with 3 judges. We achieve inter-judge agreement of κ = [X.XX] and human-automated agreement of κ = [X.XX]. Our analysis reveals that [key finding]. The most common disagreement pattern is [pattern], suggesting that [implication]. These findings inform the reliability of automated faithfulness evaluation for research and practical applications."

### Key Contributions
1. **Methodological**: Validated jury-style faithfulness evaluation with human benchmarks
2. **Empirical**: Quantified human-automated agreement on faithfulness classification
3. **Analytical**: Identified systematic failure patterns in automated evaluation
4. **Practical**: Provided recommendations for automated faithfulness system design

## Standardized Output Format

### File Structure
```
faithfulness/
├── data/
│   ├── documents.json              # Source documents
│   ├── summaries.json              # Generated summaries
│   ├── claims.json                 # Decomposed claims
│   ├── jury_verdicts.json          # Jury classifications
│   ├── human_annotations.json      # Human labels
│   ├── validation_report.json      # Agreement analysis
│   └── final_report.json          # Complete results
├── results/
│   ├── model_comparison.csv        # Per-model metrics
│   ├── correlation_analysis.csv    # Statistical tests
│   └── failure_analysis.csv        # Disagreement breakdown
└── paper/
    ├── figures/                    # Visualizations
    ├── tables/                     # Formatted tables
    └── manuscript.tex             # LaTeX paper
```

### JSON Schema
All JSON files follow standardized schemas with required fields for reproducibility.

## Citation Format

```bibtex
@article{faithfulness_evaluation_2024,
  title={Automated Faithfulness Evaluation for Long-Document Summarization: 
         A Human-Validated Study},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2024},
  note={Yale University Research Internship Application}
}
```

## Contact & Support

For questions about methodology or implementation:
- Review the complete documentation in FAITHFULNESS_README.md
- Check the technical implementation in FAITHFULNESS_IMPLEMENTATION.md
- Run the simplified tests using faithfulness/simple_runner.py --quick-test

## Acknowledgments

This methodology builds on established research in:
- LLM-as-a-Judge evaluation (Clark et al., 2021)
- Retrieval-augmented generation (Lewis et al., 2020)
- Inter-rater reliability metrics (Fleiss, 1971; Cohen, 1960)
- Faithfulness evaluation (Pagnoni et al., 2021; Gao et al., 2023)
