"""
Configuration management for EvaLex.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class EvaLexConfig:
    """Configuration for EvaLex evaluation pipeline."""
    
    # Model settings
    model_name: str = ""
    model_path: Optional[str] = None  # Full path for local models
    eos_token: str = "<|eot_id|>"
    
    # Backend settings ('local' or 'openai')
    backend: str = "local"
    
    # OpenAI API settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    
    # Data settings
    words_file: str = ""
    prompts_file: str = ""
    num_words: int = 100
    categories: bool = False
    
    # Generation settings
    batch_size_def: int = 10
    batch_size_words: int = 10
    num_return_sequences: int = 5
    stop_strings: Optional[str] = "}"
    
    # Output settings
    output_dir: str = "generations"
    results_dir: str = "results"
    iteration: str = "v1"
    
    # Stanza settings for lemmatization
    stanza_dir: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "EvaLexConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Map old config keys to new names
        key_mapping = {
            "words": "words_file",
            "prompts": "prompts_file",
            "iter": "iteration",
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)
        
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            "model_name": self.model_name,
            "eos_token": self.eos_token,
            "backend": self.backend,
            "words": self.words_file,
            "prompts": self.prompts_file,
            "num_words": self.num_words,
            "categories": self.categories,
            "batch_size_def": self.batch_size_def,
            "batch_size_words": self.batch_size_words,
            "stop_strings": self.stop_strings,
            "iter": self.iteration,
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_model_id(self, base_path: str = "/mnt/beegfs/sinai-data") -> str:
        """Get the full model path for local models."""
        if self.model_path:
            return self.model_path
        return f"{base_path}/{self.model_name}"
    
    def get_sanitized_model_name(self) -> str:
        """Get a filesystem-safe model name."""
        return self.model_name.replace("/", "_")
    
    def get_generations_path(self) -> Path:
        """Get the path for generations output."""
        return Path(self.output_dir) / f"v{self.iteration}"
    
    def get_results_path(self) -> Path:
        """Get the path for results output."""
        return Path(self.results_dir) / f"v{self.iteration}"
    
    def get_output_filename(self) -> str:
        """Generate output filename based on model and words file."""
        words_basename = Path(self.words_file).stem
        model_name = self.get_sanitized_model_name()
        return f"{model_name}_{words_basename}.tsv"


# Default available models for the Gradio interface
AVAILABLE_MODELS = {
    "Local Models": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "google/gemma-7b-it",
        "01-ai/Yi-6B-Chat",
        "BSC-LT/salamandra-7b-instruct",
        "bertin-project/Gromenauer-7B-Instruct",
    ],
    "OpenAI API Models": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
}

# Default word lists for ranking
DEFAULT_WORD_LISTS = [
    "data/CREA_10k/3-CREA_10k_pos_filter.tsv",
    "data/CREA_10k/4-CREA_10k_lemmas.tsv",
]
