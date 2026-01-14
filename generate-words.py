#!/usr/bin/env python3
"""
Generate candidate words from definitions.

This script is the second step in the EvaLex evaluation pipeline.
It takes the generated definitions and asks the LLM to predict the original words.

Usage:
    python generate-words.py config.yaml
    python generate-words.py config.yaml --backend openai
"""

import sys
import os
import argparse

import pandas as pd
from transformers import set_seed

# Add parent directory to path for evalex import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalex.config import EvaLexConfig
from evalex.models import create_backend
from evalex.prompts import PromptManager
from evalex.pipeline import WordGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate words from definitions")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--backend", choices=["local", "openai"], default=None,
                        help="Override backend type from config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load config
    config = EvaLexConfig.from_yaml(args.config)
    
    # Override backend if specified
    if args.backend:
        config.backend = args.backend
    
    print(f"Generating words with {config.model_name} (backend: {config.backend})")
    
    # Load definitions from previous step
    model_name = config.get_sanitized_model_name()
    definitions_path = config.get_generations_path() / config.get_output_filename()
    
    if not definitions_path.exists():
        print(f"Error: Definitions file not found: {definitions_path}")
        print("Please run generate-definitions.py first.")
        sys.exit(1)
    
    definitions = pd.read_csv(definitions_path, sep="\t", keep_default_na=False, na_values=[])
    print(f"Loaded {len(definitions)} definitions from {definitions_path}")
    
    # Create components
    backend = create_backend(config)
    prompt_manager = PromptManager(config.prompts_file if config.prompts_file else None)
    
    # Create generator
    generator = WordGenerator(
        backend=backend,
        prompt_manager=prompt_manager,
        config=config,
    )
    
    # Generate words
    print("Generating candidate words...")
    words_df = generator.generate(definitions)
    
    # Clean up
    backend.cleanup()
    
    # Save results (overwrite the definitions file with the complete data)
    output_dir = config.get_generations_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / config.get_output_filename()
    words_df.to_csv(output_file, sep="\t", index=False)
    
    print(f"Saved {len(words_df)} word predictions to {output_file}")


if __name__ == "__main__":
    main()
