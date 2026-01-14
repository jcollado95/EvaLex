"""
EvaLex - Lexical Competence Evaluation for Language Models

This package provides tools to evaluate the lexical competence of LLMs
by testing their ability to generate and recognize vocabulary.
"""

__version__ = "1.0.0"

from evalex.config import EvaLexConfig
from evalex.models import create_backend, LocalModelBackend, OpenAIModelBackend
from evalex.pipeline import EvaLexPipeline, DefinitionGenerator, WordGenerator, WordEvaluator

__all__ = [
    "EvaLexConfig",
    "create_backend",
    "LocalModelBackend", 
    "OpenAIModelBackend",
    "EvaLexPipeline",
    "DefinitionGenerator",
    "WordGenerator",
    "WordEvaluator",
]
