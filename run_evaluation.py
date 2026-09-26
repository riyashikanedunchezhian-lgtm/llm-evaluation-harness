"""Main entry point for running LLM evaluation."""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.harness import EvaluationHarness
from src.config import MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation harness")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        help="Model IDs to evaluate (space-separated)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["factual_qa", "summarization", "code_generation", "reasoning"],
        help="Categories to evaluate (default: all)"
    )
    parser.add_argument(
        "--bias-check",
        action="store_true",
        help="Enable position bias checking"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential judge execution instead of parallel (slower but more predictable)"
    )
    parser.add_argument(
        "--output",
        default="data/results.json",
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    # Validate models
    for model_id in args.models:
        if model_id not in MODEL_CONFIGS:
            print(f"Error: Unknown model '{model_id}'")
            print(f"Available models: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
    
    print(f"Starting evaluation with models: {args.models}")
    if args.categories:
        print(f"Categories: {args.categories}")
    else:
        print("Categories: all")
    
    execution_mode = "sequential" if args.sequential else "parallel"
    print(f"Judge execution mode: {execution_mode}")
    
    # Initialize harness
    harness = EvaluationHarness(
        model_ids=args.models,
        test_set_path="prompts/test_set.json",
        parallel_judge=not args.sequential
    )
    
    # Run evaluation
    results = harness.run_evaluation(
        categories=args.categories,
        enable_bias_check=args.bias_check
    )
    
    # Save results
    harness.save_results(args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    df = harness.get_summary_dataframe()
    metrics = harness.get_aggregated_metrics()
    
    print("\nBy Model:")
    for model in df["model"].unique():
        model_metrics = metrics[model]
        print(f"\n{model}:")
        print(f"  Average Score: {model_metrics['avg_score']:.2f}/5.0")
        print(f"  Average Latency: {model_metrics['avg_latency_ms']:.0f}ms")
        print(f"  Average Cost: ${model_metrics['avg_cost_usd']:.4f}")
        print(f"  Total Cost: ${model_metrics['total_cost_usd']:.4f}")
        print(f"  Total Tokens: {model_metrics['total_tokens']}")
    
    # Show execution mode breakdown
    if "execution_mode" in df.columns:
        print(f"\nExecution Mode Distribution:")
        mode_counts = df["execution_mode"].value_counts()
        for mode, count in mode_counts.items():
            print(f"  {mode}: {count} evaluations")
    
    print("\nBy Category:")
    for category in df["category"].unique():
        cat_metrics = metrics[f"category_{category}"]
        print(f"\n{category}:")
        print(f"  Average Score: {cat_metrics['avg_score']:.2f}/5.0")
        print(f"  Average Latency: {cat_metrics['avg_latency_ms']:.0f}ms")
        print(f"  Average Cost: ${cat_metrics['avg_cost_usd']:.4f}")
    
    print("\n" + "="*60)
    print(f"Total evaluations: {len(results)}")
    print(f"Results saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
