"""
Model backends for EvaLex.

Supports both local models (via transformers) and remote models (via OpenAI API).
"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[List[Dict[str, str]]],
        num_return_sequences: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = True,
        stop_strings: Optional[str] = None,
    ) -> List[str]:
        """Generate text completions for a batch of message sequences."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources when done."""
        pass


class LocalModelBackend(ModelBackend):
    """Backend for local models using transformers."""
    
    def __init__(
        self,
        model_id: str,
        eos_token: str = "<|eot_id|>",
        device_map: str = "auto",
        compile_model: bool = True,
    ):
        """
        Initialize the local model backend.
        
        Args:
            model_id: HuggingFace model ID or local path
            eos_token: End of sequence token
            device_map: Device mapping strategy
            compile_model: Whether to compile the model with torch.compile
        """
        self.model_id = model_id
        self.eos_token = eos_token
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        
        # Optionally compile for speed
        if compile_model:
            self.model = torch.compile(self.model)
        
        # Enable mixed precision
        torch.amp.autocast("cuda", enabled=True)
    
    def generate(
        self,
        messages: List[List[Dict[str, str]]],
        num_return_sequences: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = True,
        stop_strings: Optional[str] = None,
    ) -> List[str]:
        """Generate text completions for a batch of message sequences."""
        
        # Build generation config
        eos_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(self.eos_token),
        ]
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "num_return_sequences": num_return_sequences,
            "eos_token_id": eos_token_ids,
        }
        
        if stop_strings:
            gen_kwargs["stop_strings"] = stop_strings
        
        generation_config = GenerationConfig(**{k: v for k, v in gen_kwargs.items() if v is not None})
        
        # Tokenize input
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to("cuda")
        
        input_length = model_inputs.shape[1]
        
        # Generate
        outputs = self.model.generate(
            model_inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
        )
        
        # Decode outputs
        return self.tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )
    
    def cleanup(self) -> None:
        """Clean up GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class OpenAIModelBackend(ModelBackend):
    """Backend for OpenAI-compatible API models."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI API backend.
        
        Args:
            model_name: Name of the model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for the API (defaults to OPENAI_BASE_URL env var)
        """
        import os
        from openai import OpenAI
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def generate(
        self,
        messages: List[List[Dict[str, str]]],
        num_return_sequences: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = True,
        stop_strings: Optional[str] = None,
    ) -> List[str]:
        """Generate text completions for a batch of message sequences."""
        results = []
        
        for message_list in messages:
            for _ in range(num_return_sequences):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=message_list,
                        max_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 0,
                        stop=[stop_strings] if stop_strings else None,
                    )
                    results.append(response.choices[0].message.content or "")
                except Exception as e:
                    print(f"OpenAI API error: {e}")
                    results.append("")
        
        return results
    
    def cleanup(self) -> None:
        """No cleanup needed for API backend."""
        pass


def create_backend(
    config: "EvaLexConfig",
    base_model_path: str = "/mnt/beegfs/sinai-data",
) -> ModelBackend:
    """
    Factory function to create the appropriate backend.
    
    Args:
        config: EvaLex configuration
        base_model_path: Base path for local models
        
    Returns:
        ModelBackend instance
    """
    if config.backend == "openai":
        return OpenAIModelBackend(
            model_name=config.model_name,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
    else:
        model_id = config.get_model_id(base_model_path)
        return LocalModelBackend(
            model_id=model_id,
            eos_token=config.eos_token,
        )
