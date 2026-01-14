"""
Utility functions for EvaLex.
"""

import re
import json
import unicodedata
from typing import Dict, List, Optional, Any

from nltk.tokenize import word_tokenize


def clean_text(text: str) -> str:
    """
    Remove accents and special characters from text.
    Keeps only alphanumeric characters and spaces.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Normalize and remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Remove special characters (keep only letters, numbers and spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text


def extract_json_definition(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON containing a definition from text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        json_data = json.loads(text)
        if isinstance(json_data, dict) and "definición" in json_data:
            return json_data
        else:
            print(f"JSON mal formado o sin 'definición': {text}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e} - Texto: {text}")
        return None


def extract_json_words(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON containing words from text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        json_data = json.loads(text)
        if isinstance(json_data, dict) and "palabras" in json_data:
            return json_data
        else:
            print(f"JSON mal formado o sin 'palabras': {text}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e} - Texto: {text}")
        return None


def anonymize_word_by_lemma(nlp, text: str, base_lemma: str) -> str:
    """
    Replaces all inflected forms of a word (by lemma) with the token _.

    Args:
        nlp: A Stanza pipeline.
        text: Input text.
        base_lemma: Lemma of the word to replace.

    Returns:
        Modified text with matching words replaced by "_".
    """
    base_lemma_norm = clean_text(base_lemma.lower())
    doc = nlp(str(text))
    output = []
    index = 0

    for sentence in doc.sentences:
        for token in sentence.tokens:
            word = token.text
            start = token.start_char
            end = token.end_char

            # Use lemma of the first word in the token (as Stanza supports multi-word tokens)
            lemma = token.words[0].lemma if token.words else ''
            lemma_norm = clean_text(lemma.lower()) if lemma else ''

            # Add untouched text between previous token and current
            output.append(text[index:start])

            # Replace word if normalized lemma matches
            if lemma_norm == base_lemma_norm:
                output.append("_")
            else:
                output.append(word)

            index = end

    output.append(text[index:])  # Add any remaining text
    return ''.join(output)


def process_definition_output(
    definition: str,
    word: str,
    nlp=None,
) -> str:
    """
    Process a raw definition output from the model.
    
    Args:
        definition: Raw definition text (may be JSON)
        word: The original word being defined
        nlp: Optional Stanza pipeline for lemma-based anonymization
        
    Returns:
        Processed and anonymized definition
    """
    # Try to extract JSON
    definition_data = extract_json_definition(definition)
    
    if definition_data and "definición" in definition_data:
        definition_text = definition_data["definición"]
    else:
        # If not valid JSON, try to clean up the raw text
        definition_text = definition
        definition_text = definition_text.split(":")[-1]  # Ignore JSON key if exists
        definition_text = re.sub(r'[\{\}\[\]\"]', '', definition_text).strip()
    
    if not definition_text:
        return "Término no conocido."
    
    # Anonymize the word
    try:
        if nlp:
            anonymized = anonymize_word_by_lemma(nlp, definition_text, word)
            anonymized = re.sub(rf'\b{re.escape(word)}\b', '_', anonymized, flags=re.IGNORECASE)
        else:
            anonymized = re.sub(rf'\b{re.escape(word)}\b', '_', definition_text, flags=re.IGNORECASE)
    except Exception as e:
        print(f"Error in lemmatization: {e}")
        anonymized = re.sub(rf'\b{re.escape(word)}\b', '_', definition_text, flags=re.IGNORECASE)
    
    # Clean up multiple paragraphs
    anonymized = anonymized.split("\n\n")[0]
    
    return anonymized


def process_words_output(
    predicted_words: str,
) -> str:
    """
    Process raw word predictions from the model.
    
    Args:
        predicted_words: Raw model output (may be JSON)
        
    Returns:
        Space-separated list of predicted words
    """
    predicted_words = predicted_words.lower()
    
    # Try to extract JSON
    words_data = extract_json_words(predicted_words)
    
    if words_data and "palabras" in words_data:
        words_list = words_data["palabras"]
        return " ".join(map(str, words_list))
    else:
        # Clean up raw text
        words_text = predicted_words
        words_text = words_text.split(":")[-1]  # Ignore JSON key if exists
        # Remove JSON characters
        for char in ['"', "'", '{', '}', '[', ']']:
            words_text = words_text.replace(char, '')
        words_text = words_text.strip()
        
        if not words_text:
            return "Definición no conocida."
        
        return words_text


def evaluate_word_known(word: str, predicted_words: str) -> bool:
    """
    Check if a word appears in the predicted words.
    
    Args:
        word: The target word
        predicted_words: Space-separated predicted words
        
    Returns:
        True if the word is found in predictions
    """
    word_lower = word.lower()
    tokens = word_tokenize(predicted_words.lower())
    return word_lower in tokens


def calculate_lexical_competence(
    words: List[str],
    group_known: List[bool],
) -> Dict[str, Any]:
    """
    Calculate lexical competence metrics.
    
    Args:
        words: List of words evaluated
        group_known: List of booleans indicating if each word is known
        
    Returns:
        Dict with metrics (known_count, total_count, accuracy)
    """
    known_count = sum(group_known)
    total_count = len(words)
    accuracy = known_count / total_count if total_count > 0 else 0.0
    
    return {
        "known_count": known_count,
        "total_count": total_count,
        "accuracy": accuracy,
        "accuracy_percentage": f"{accuracy * 100:.2f}%",
    }
