#!/usr/bin/env python3
"""
Generate definitions from a list of words.

This script is the first step in the EvaLex evaluation pipeline.
It generates multiple definitions for each word using an LLM.

Usage:
    python generate-definitions.py config.yaml
    python generate-definitions.py config.yaml --backend openai
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
from evalex.pipeline import DefinitionGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate definitions from words")
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
    
    print(f"Generating definitions with {config.model_name} (backend: {config.backend})")
    
    # Load words
    words = pd.read_csv(config.words_file, sep="\t")
    words = words[:config.num_words]
    print(f"Loaded {len(words)} words from {config.words_file}")
    
    # Initialize Stanza for lemmatization (optional, for local processing)
    nlp = None
    if config.stanza_dir:
        try:
            import stanza
            nlp = stanza.Pipeline(
                lang='es',
                dir=config.stanza_dir,
                processors='tokenize,mwt,pos,lemma',
                download_method=None
            )
        except Exception as e:
            print(f"Warning: Could not load Stanza: {e}")
    else:
        # Try default Stanza path
        try:
            import stanza
            default_path = '/mnt/beegfs/jcollado/stanza_resources/'
            if os.path.exists(default_path):
                nlp = stanza.Pipeline(
                    lang='es',
                    dir=default_path,
                    processors='tokenize,mwt,pos,lemma',
                    download_method=None
                )
        except Exception as e:
            print(f"Warning: Could not load Stanza: {e}")
    
    # Create components
    backend = create_backend(config)
    prompt_manager = PromptManager(config.prompts_file if config.prompts_file else None)
    
    # Create generator
    generator = DefinitionGenerator(
        backend=backend,
        prompt_manager=prompt_manager,
        config=config,
        nlp=nlp,
    )
    
    # Generate definitions
    print("Generating definitions...")
    definitions_df = generator.generate(words)
    
    # Clean up
    backend.cleanup()
    
    # Save results
    output_dir = config.get_generations_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / config.get_output_filename()
    definitions_df.to_csv(output_file, sep="\t", index=False)
    
    print(f"Saved {len(definitions_df)} definitions to {output_file}")


if __name__ == "__main__":
    main()
