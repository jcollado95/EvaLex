"""
Prompt management for EvaLex.
"""

import json
import copy
from typing import Dict, List, Optional
from pathlib import Path


# Default prompts (based on llama/v9.json)
DEFAULT_DEFINITION_PROMPT = [
    {
        "role": "system",
        "content": (
            "Eres un experto en vocabulario español. Tu tarea es dar una definición para una palabra dada.\n"
            "Instrucciones:\n"
            "- La definición debe ser objetiva y clara.\n"
            "- Usa entre 5 y 25 palabras.\n"
            "- La definición generada no debe incluir la palabra original ni derivados directos o familia léxica.\n"
            "IMPORTANTE:\n"
            "La respuesta debe ser un **objeto JSON**:\n"
            "Formato de respuesta:\n"
            "{\n"
            '  "palabra": "(palabra original)",\n'
            '  "definición": "(definición sin usar la palabra ni su familia léxica)"\n'
            "}"
        ),
    },
    {"role": "user", "content": "Palabra: {word}"},
]

DEFAULT_DEFINITION_PROMPT_WITH_CATEGORY = [
    {
        "role": "system",
        "content": (
            "Eres un experto en vocabulario español. Tu tarea es dar una definición para una palabra dada.\n"
            "Instrucciones:\n"
            "- La definición debe ser objetiva y clara.\n"
            "- Usa entre 5 y 25 palabras.\n"
            "- La definición generada no debe incluir la palabra original ni derivados directos o familia léxica.\n"
            "IMPORTANTE:\n"
            "La respuesta debe ser un **objeto JSON**:\n"
            "Formato de respuesta:\n"
            "{\n"
            '  "palabra": "(palabra original)",\n'
            '  "definición": "(definición sin usar la palabra ni su familia léxica)"\n'
            "}"
        ),
    },
    {"role": "user", "content": "Palabra: {word} (categoría: {cat})"},
]

DEFAULT_WORDS_PROMPT = [
    {
        "role": "system",
        "content": (
            "Eres un experto en vocabulario español. Tu tarea consiste en encontrar una lista de posibles palabras en español que pueden definirse con una definición dada.\n\n"
            "INSTRUCCIONES:\n"
            "- Las palabras posibles no están en la definición.\n"
            "- Genera todas las palabras que encajen con la definición.\n"
            "- Responde con una lista vacía si no encuentras ninguna palabra adecuada para la definición dada.\n"
            "- La respuesta debe ser un **objeto JSON**.\n"
            "Formato de respuesta:\n"
            "{\n"
            '  "palabras": ["(palabra 1)", "(palabra 2)", "(palabra n)"]\n'
            "}"
        ),
    },
    {
        "role": "user",
        "content": "Definición: instrumento de escritura que utiliza tinta y tiene una punta",
    },
    {
        "role": "assistant",
        "content": '{\n  "palabras": ["bolígrafo", "pluma", "estilográfica"]\n}',
    },
    {"role": "user", "content": "Definición: {definition}"},
]

DEFAULT_WORDS_PROMPT_WITH_CATEGORY = [
    {
        "role": "system",
        "content": (
            "Eres un experto en vocabulario español. Tu tarea consiste en encontrar una lista de posibles palabras en español que pueden definirse con una definición dada.\n\n"
            "INSTRUCCIONES:\n"
            "- Las palabras posibles no están en la definición.\n"
            "- Genera todas las palabras que encajen con la definición.\n"
            "- Responde con una lista vacía si no encuentras ninguna palabra adecuada para la definición dada.\n"
            "- La respuesta debe ser un **objeto JSON**.\n"
            "Formato de respuesta:\n"
            "{\n"
            '  "palabras": ["(palabra 1)", "(palabra 2)", "(palabra n)"]\n'
            "}"
        ),
    },
    {
        "role": "user",
        "content": "Definición: instrumento de escritura que utiliza tinta y tiene una punta (categoría: sustantivo)",
    },
    {
        "role": "assistant",
        "content": '{\n  "palabras": ["bolígrafo", "pluma", "estilográfica"]\n}',
    },
    {"role": "user", "content": "Definición: {definition} (categoría: {cat})"},
]


class PromptManager:
    """Manages prompts for definition and word generation."""
    
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_file: Path to JSON file with custom prompts
        """
        self.def_prompt = DEFAULT_DEFINITION_PROMPT
        self.def_prompt_with_cat = DEFAULT_DEFINITION_PROMPT_WITH_CATEGORY
        self.words_prompt = DEFAULT_WORDS_PROMPT
        self.words_prompt_with_cat = DEFAULT_WORDS_PROMPT_WITH_CATEGORY
        
        if prompts_file:
            self.load_prompts(prompts_file)
    
    def load_prompts(self, path: str) -> None:
        """Load prompts from a JSON file."""
        with open(path, "r") as f:
            prompts = json.load(f)
        
        if "def" in prompts:
            self.def_prompt = prompts["def"]
            self.def_prompt_with_cat = prompts["def"]
        
        if "words" in prompts:
            self.words_prompt = prompts["words"]
            self.words_prompt_with_cat = prompts["words"]
    
    def create_definition_prompt(
        self,
        word: str,
        category: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Create a prompt for generating a definition.
        
        Args:
            word: The word to define
            category: Optional category/POS tag
            
        Returns:
            List of message dicts for the chat template
        """
        if category:
            prompt = copy.deepcopy(self.def_prompt_with_cat)
        else:
            prompt = copy.deepcopy(self.def_prompt)
        
        for message in prompt:
            if message["role"] == "user":
                message["content"] = message["content"].replace("{word}", word)
                if category and "{cat}" in message["content"]:
                    message["content"] = message["content"].replace("{cat}", category)
        
        return prompt
    
    def create_words_prompt(
        self,
        definition: str,
        category: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Create a prompt for generating candidate words from a definition.
        
        Args:
            definition: The definition to find words for
            category: Optional category/POS tag
            
        Returns:
            List of message dicts for the chat template
        """
        if category:
            prompt = copy.deepcopy(self.words_prompt_with_cat)
        else:
            prompt = copy.deepcopy(self.words_prompt)
        
        for message in prompt:
            if message["role"] == "user" and "{definition}" in message["content"]:
                message["content"] = message["content"].replace("{definition}", definition)
                if category and "{cat}" in message["content"]:
                    message["content"] = message["content"].replace("{cat}", category)
        
        return prompt
    
    def create_definition_dataset(
        self,
        words_df: "pd.DataFrame",
        use_categories: bool = False,
    ) -> List[List[Dict[str, str]]]:
        """
        Create a dataset of definition prompts from a DataFrame of words.
        
        Args:
            words_df: DataFrame with 'word' column (and optionally 'category')
            use_categories: Whether to include categories in prompts
            
        Returns:
            List of prompt message lists
        """
        prompts = []
        for row in words_df.itertuples(index=False):
            category = getattr(row, "category", None) if use_categories else None
            prompts.append(self.create_definition_prompt(row.word, category))
        return prompts
    
    def create_words_dataset(
        self,
        definitions_df: "pd.DataFrame",
        use_categories: bool = False,
    ) -> List[List[Dict[str, str]]]:
        """
        Create a dataset of word generation prompts from a DataFrame of definitions.
        
        Args:
            definitions_df: DataFrame with 'definition' column (and optionally 'category')
            use_categories: Whether to include categories in prompts
            
        Returns:
            List of prompt message lists
        """
        prompts = []
        for row in definitions_df.itertuples(index=False):
            category = getattr(row, "category", None) if use_categories else None
            prompts.append(self.create_words_prompt(row.definition, category))
        return prompts
