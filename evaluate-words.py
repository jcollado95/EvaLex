#!/usr/bin/env python3
"""
Evaluate word predictions against original words.

This script is the third and final step in the EvaLex evaluation pipeline.
It checks if the LLM was able to predict the original words from the definitions.

Usage:
    python evaluate-words.py config.yaml
"""

import sys
import os
import argparse

import pandas as pd

# Add parent directory to path for evalex import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalex.config import EvaLexConfig
from evalex.pipeline import WordEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate word predictions")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    config = EvaLexConfig.from_yaml(args.config)
    
    print(f"Evaluating words for {config.model_name}")
    
    # Load generations from previous steps
    generations_path = config.get_generations_path() / config.get_output_filename()
    
    if not generations_path.exists():
        print(f"Error: Generations file not found: {generations_path}")
        print("Please run generate-definitions.py and generate-words.py first.")
        sys.exit(1)
    
    generations_df = pd.read_csv(generations_path, sep="\t")
    print(f"Loaded {len(generations_df)} generations from {generations_path}")
    
    # Create evaluator
    evaluator = WordEvaluator(num_return_sequences=config.num_return_sequences)
    
    # Evaluate
    print("Evaluating predictions...")
    results_df, metrics = evaluator.evaluate(generations_df)
    
    # Print metrics
    print(f"\n{'='*50}")
    print(f"LEXICAL COMPETENCE RESULTS")
    print(f"{'='*50}")
    print(f"Model: {config.model_name}")
    print(f"Words evaluated: {metrics['total_count']}")
    print(f"Words known: {metrics['known_count']}")
    print(f"Accuracy: {metrics['accuracy_percentage']}")
    print(f"{'='*50}\n")
    
    # Save results
    output_dir = config.get_results_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / config.get_output_filename()
    results_df.to_csv(output_file, sep="\t", index=False)
    
    print(f"Saved results to {output_file}")
    
    return metrics


if __name__ == "__main__":
    main()