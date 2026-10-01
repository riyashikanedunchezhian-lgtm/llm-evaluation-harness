"""Main entry point for faithfulness evaluation."""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'faithfulness'))

from faithfulness.config import FaithfulnessConfig
from faithfulness.pipeline import FaithfulnessPipeline

def main():
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation for long-document summarization")
    parser.add_argument(
        "--dataset",
        choices=["govreport", "arxiv"],
        default="govreport",
        help="Dataset to use"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=30,
        help="Number of documents to evaluate"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2000,
        help="Minimum word count for documents"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022", "gpt-4o-mini"],
        help="Models to use for summarization"
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip summary generation (use existing)"
    )
    parser.add_argument(
        "--bias-check",
        action="store_true",
        help="Run position bias checks"
    )
    parser.add_argument(
        "--output-dir",
        default="faithfulness/data",
        help="Output directory for results"
    )
    parser.add_argument(
        "--create-annotations",
        action="store_true",
        help="Create human annotation template"
    )
    parser.add_argument(
        "--annotation-template",
        default="faithfulness/data/annotation_template.json",
        help="Path for annotation template"
    )
    parser.add_argument(
        "--process-annotations",
        type=str,
        help="Path to completed human annotations"
    )
    parser.add_argument(
        "--validation-report",
        default="faithfulness/data/validation_report.json",
        help="Path for validation report"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = FaithfulnessConfig(
        dataset_name=args.dataset,
        num_documents=args.num_docs,
        min_word_count=args.min_words,
        summary_models=args.models,
        output_dir=args.output_dir
    )
    
    # Initialize pipeline
    pipeline = FaithfulnessPipeline(config)
    
    if args.create_annotations:
        # Create annotation template for human validation
        print("Creating annotation template...")
        result = pipeline.run_human_validation(
            args.annotation_template,
            args.process_annotations if args.process_annotations else None
        )
        return
    
    if args.process_annotations:
        # Process human annotations
        print("Processing human annotations...")
        result = pipeline.process_human_annotations(
            args.process_annotations,
            args.validation_report
        )
        print(f"\nHuman Validation Results:")
        print(f"  Cohen's kappa: {result['cohen_kappa']:.3f}")
        print(f"  Percent agreement: {result['percent_agreement']:.2%}")
        print(f"  Agreement category: {result['agreement_category']}")
        print(f"  Total disagreements: {result['total_disagreements']}")
        return
    
    # Run full pipeline
    print("Starting faithfulness evaluation pipeline...")
    results = pipeline.run_full_pipeline(
        generate_summaries=not args.skip_summaries,
        run_bias_check=args.bias_check
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\nDataset: {args.dataset}")
    print(f"Documents processed: {results['stages']['dataset']['num_documents']}")
    print(f"Average word count: {results['stages']['dataset']['avg_word_count']:.0f}")
    
    if 'summaries' in results['stages']:
        print(f"Summaries generated: {results['stages']['summaries']['total_summaries']}")
    
    print(f"Claims decomposed: {results['stages']['claim_decomposition']['total_claims']}")
    print(f"Classifications performed: {results['stages']['classification']['total_classifications']}")
    
    print(f"\nInter-Judge Agreement:")
    print(f"  Percent agreement: {results['stages']['inter_judge_agreement']['percent_agreement']:.2%}")
    print(f"  Fleiss' kappa: {results['stages']['inter_judge_agreement']['fleiss_kappa']:.3f}")
    
    if 'position_bias' in results['stages']:
        print(f"\nPosition Bias Check:")
        print(f"  Bias detected in: {results['stages']['position_bias']['bias_rate']:.1%} of checks")
        print(f"  Label changes: {results['stages']['position_bias']['label_change_rate']:.1%}")
    
    print("\n" + "="*60)
    print("Results saved to:", args.output_dir)
    print("="*60)

if __name__ == "__main__":
    main()
