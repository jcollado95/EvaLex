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

DEFAULT_DEFINITION_PROMPT_EN = [
    {
        "role": "system",
        "content": (
            "You are an expert in English vocabulary. Your task is to provide a definition for a given word.\n"
            "Instructions:\n"
            "- The definition must be objective and clear.\n"
            "- Use between 5 and 25 words.\n"
            "- The generated definition must not include the original word or any direct derivatives or lexical family members.\n"
            "IMPORTANT:\n"
            "The response must be a **JSON object**:\n"
            "Response format:\n"
            "{\n"
            '  "word": "(original word)",\n'
            '  "definition": "(definition without using the word or its lexical family)"\n'
            "}"
        ),
    },
    {"role": "user", "content": "Word: {word}"},
]

DEFAULT_WORDS_PROMPT_EN = [
    {
        "role": "system",
        "content": (
            "You are an expert in English vocabulary. Your task is to find a list of possible English words that can be defined by a given definition.\n\n"
            "INSTRUCTIONS:\n"
            "- The possible words must not appear in the definition.\n"
            "- Generate all words that fit the definition.\n"
            "- Respond with an empty list if no suitable words are found for the given definition.\n"
            "- The response must be a **JSON object**.\n"
            "Response format:\n"
            "{\n"
            '  "words": ["(word 1)", "(word 2)", "(word n)"]\n'
            "}"
        ),
    },
    {
        "role": "user",
        "content": "Definition: writing instrument that uses ink and has a tip",
    },
    {
        "role": "assistant",
        "content": '{\n  "words": ["pen", "fountain pen", "ballpoint"]\n}',
    },
    {"role": "user", "content": "Definition: {definition}"},
]

DEFAULT_DEFINITION_PROMPT_FR = [
    {
        "role": "system",
        "content": (
            "Tu es un expert du vocabulaire français. Ta tâche consiste à donner une définition pour un mot donné.\n"
            "Instructions :\n"
            "- La définition doit être objective et claire.\n"
            "- Utilise entre 5 et 25 mots.\n"
            "- La définition générée ne doit pas inclure le mot d'origine ni ses dérivés directs ou sa famille lexicale.\n"
            "IMPORTANT :\n"
            "La réponse doit être un **objet JSON** :\n"
            "Format de réponse :\n"
            "{\n"
            '  "mot": "(mot d\'origine)",\n'
            '  "définition": "(définition sans utiliser le mot ni sa famille lexicale)"\n'
            "}"
        ),
    },
    {"role": "user", "content": "Mot : {word}"},
]

DEFAULT_WORDS_PROMPT_FR = [
    {
        "role": "system",
        "content": (
            "Tu es un expert du vocabulaire français. Ta tâche consiste à trouver une liste de mots français possibles qui correspondent à une définition donnée.\n\n"
            "INSTRUCTIONS :\n"
            "- Les mots possibles n'apparaissent pas dans la définition.\n"
            "- Génère tous les mots qui correspondent à la définition.\n"
            "- Réponds avec une liste vide si tu ne trouves aucun mot adapté à la définition donnée.\n"
            "- La réponse doit être un **objet JSON**.\n"
            "Format de réponse :\n"
            "{\n"
            '  "mots": ["(mot 1)", "(mot 2)", "(mot n)"]\n'
            "}"
        ),
    },
    {
        "role": "user",
        "content": "Définition : instrument d'écriture qui utilise de l'encre et possède une pointe",
    },
    {
        "role": "assistant",
        "content": '{\n  "mots": ["stylo", "plume", "stylo-plume"]\n}',
    },
    {"role": "user", "content": "Définition : {definition}"},
]

# Supported prompt languages (code -> display name)
LANGUAGES = {"es": "Español", "en": "English", "fr": "Français"}

# Default prompts and expected JSON response keys per language
DEFAULT_PROMPTS = {
    "es": {
        "def": DEFAULT_DEFINITION_PROMPT,
        "words": DEFAULT_WORDS_PROMPT,
        "def_key": "definición",
        "words_key": "palabras",
    },
    "en": {
        "def": DEFAULT_DEFINITION_PROMPT_EN,
        "words": DEFAULT_WORDS_PROMPT_EN,
        "def_key": "definition",
        "words_key": "words",
    },
    "fr": {
        "def": DEFAULT_DEFINITION_PROMPT_FR,
        "words": DEFAULT_WORDS_PROMPT_FR,
        "def_key": "définition",
        "words_key": "mots",
    },
}


class PromptManager:
    """Manages prompts for definition and word generation."""
    
    def __init__(
        self,
        prompts_file: Optional[str] = None,
        language: str = "es",
        def_prompt: Optional[List[Dict[str, str]]] = None,
        words_prompt: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialize the prompt manager.

        Args:
            prompts_file: Path to JSON file with custom prompts
            language: Language of the default prompts (see LANGUAGES)
            def_prompt: Optional custom definition prompt (overrides the default)
            words_prompt: Optional custom words prompt (overrides the default)
        """
        defaults = DEFAULT_PROMPTS[language]
        self.def_key = defaults["def_key"]
        self.words_key = defaults["words_key"]

        if language == "es":
            self.def_prompt = DEFAULT_DEFINITION_PROMPT
            self.def_prompt_with_cat = DEFAULT_DEFINITION_PROMPT_WITH_CATEGORY
            self.words_prompt = DEFAULT_WORDS_PROMPT
            self.words_prompt_with_cat = DEFAULT_WORDS_PROMPT_WITH_CATEGORY
        else:
            self.def_prompt = self.def_prompt_with_cat = defaults["def"]
            self.words_prompt = self.words_prompt_with_cat = defaults["words"]

        if def_prompt:
            self.def_prompt = self.def_prompt_with_cat = def_prompt
        if words_prompt:
            self.words_prompt = self.words_prompt_with_cat = words_prompt

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
