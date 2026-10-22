"""
Utility functions for EvaLex.
"""

import io
import re
import json
import contextlib
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


def strip_code_fences(text: str) -> str:
    """
    Remove a surrounding Markdown code fence (e.g. ```json ... ```) from text.
    
    Args:
        text: Model output that may be wrapped in a code fence
        
    Returns:
        Text without the code fence
    """
    # The closing fence is optional, as generation may stop before it
    match = re.match(r'^\s*```[a-zA-Z]*\s*\n?(.*?)(?:\n?\s*```)?\s*$', text, flags=re.DOTALL)
    return match.group(1) if match else text


def extract_json_definition(text: str, key: str = "definición") -> Optional[Dict[str, Any]]:
    """
    Extract JSON containing a definition from text.
    
    Args:
        text: Text containing JSON
        key: JSON key holding the definition
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        json_data = json.loads(strip_code_fences(text))
        if isinstance(json_data, dict) and key in json_data:
            return json_data
        else:
            print(f"JSON mal formado o sin '{key}': {text}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e} - Texto: {text}")
        return None


def extract_json_words(text: str, key: str = "palabras") -> Optional[Dict[str, Any]]:
    """
    Extract JSON containing words from text.
    
    Args:
        text: Text containing JSON
        key: JSON key holding the list of words
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        json_data = json.loads(strip_code_fences(text))
        if isinstance(json_data, dict) and key in json_data:
            return json_data
        else:
            print(f"JSON mal formado o sin '{key}': {text}")
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


def mask_word_forms(nlp, text: str, word: str) -> str:
    """
    Replaces every form of a word with the token _, ignoring case and accents.

    A token is masked if its surface form matches the word or the word's lemma,
    or (with a Stanza pipeline) if its lemma does. Multi-word targets are only
    masked as an exact phrase.

    Args:
        nlp: Optional Stanza pipeline for the language of the text.
        text: Input text.
        word: The word to mask.

    Returns:
        Modified text with matching words replaced by "_".
    """
    text = re.sub(rf'\b{re.escape(word)}\b', '_', text, flags=re.IGNORECASE)

    targets = {clean_text(word.lower())}
    if nlp:
        target_words = [w for s in nlp(word).sentences for w in s.words]
        if len(target_words) == 1 and target_words[0].lemma:
            targets.add(clean_text(target_words[0].lemma.lower()))
    targets.discard("")

    def matches(form: str, lemmas: List[str]) -> bool:
        candidates = [form] + lemmas
        return any(clean_text(c.lower()) in targets for c in candidates if c)

    if not nlp:
        return re.sub(r'\w+', lambda m: "_" if matches(m.group(0), []) else m.group(0), text)

    output = []
    index = 0
    for sentence in nlp(text).sentences:
        for token in sentence.tokens:
            output.append(text[index:token.start_char])
            lemmas = [w.lemma for w in token.words if w.lemma]
            output.append("_" if token.text != "_" and matches(token.text, lemmas) else token.text)
            index = token.end_char
    output.append(text[index:])
    return ''.join(output)


def process_definition_output(
    definition: str,
    word: str,
    nlp=None,
    key: str = "definición",
    strict_masking: bool = False,
) -> str:
    """
    Process a raw definition output from the model.

    Args:
        definition: Raw definition text (may be JSON)
        word: The original word being defined
        nlp: Optional Stanza pipeline for lemma-based anonymization
        key: JSON key holding the definition
        strict_masking: Mask every form of the word with mask_word_forms instead of
            the lemma + exact match used in the paper

    Returns:
        Processed and anonymized definition
    """
    # Try to extract JSON
    definition_data = extract_json_definition(definition, key)
    
    if definition_data and key in definition_data:
        definition_text = str(definition_data[key])
    else:
        # If not valid JSON, try to clean up the raw text
        definition_text = strip_code_fences(definition)
        definition_text = definition_text.split(":")[-1]  # Ignore JSON key if exists
        definition_text = re.sub(r'[\{\}\[\]\"]', '', definition_text).strip()
    
    if not definition_text:
        return "Término no conocido."
    
    # Anonymize the word
    try:
        if strict_masking:
            anonymized = mask_word_forms(nlp, definition_text, word)
        elif nlp:
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
    key: str = "palabras",
) -> str:
    """
    Process raw word predictions from the model.
    
    Args:
        predicted_words: Raw model output (may be JSON)
        key: JSON key holding the list of words
        
    Returns:
        Space-separated list of predicted words
    """
    predicted_words = predicted_words.lower()
    
    # Try to extract JSON
    words_data = extract_json_words(predicted_words, key)
    
    if words_data and key in words_data:
        words_list = words_data[key]
        return " ".join(map(str, words_list))
    else:
        # Clean up raw text
        words_text = strip_code_fences(predicted_words)
        words_text = words_text.split(":")[-1]  # Ignore JSON key if exists
        # Remove JSON characters
        for char in ['"', "'", '{', '}', '[', ']']:
            words_text = words_text.replace(char, '')
        words_text = words_text.strip()
        
        if not words_text:
            return "Definición no conocida."
        
        return words_text


def parse_candidate_list(
    predicted_words: str,
    key: str = "palabras",
) -> List[str]:
    """
    Parse raw word predictions into a list of candidates (keeping multi-word terms).
    
    Args:
        predicted_words: Raw model output (may be JSON)
        key: JSON key holding the list of words
        
    Returns:
        List of candidate words
    """
    # Parsing errors are already reported when processing the same output for evaluation
    with contextlib.redirect_stdout(io.StringIO()):
        words_data = extract_json_words(predicted_words.lower(), key)
        if words_data and isinstance(words_data.get(key), list):
            candidates = [str(w) for w in words_data[key]]
        else:
            # Not valid JSON: split the cleaned-up text on list separators
            text = process_words_output(predicted_words, key)
            if text == "Definición no conocida.":
                return []
            candidates = re.split(r'[,;\n]+', text)
    
    candidates = [c.strip(" .\t") for c in candidates]
    return [c for c in candidates if c]


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
