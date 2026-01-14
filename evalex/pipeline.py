"""
Pipeline components for EvaLex lexical competence evaluation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from evalex.config import EvaLexConfig
from evalex.models import ModelBackend, create_backend
from evalex.prompts import PromptManager
from evalex.utils import (
    process_definition_output,
    process_words_output,
    evaluate_word_known,
    calculate_lexical_competence,
)


class DefinitionGenerator:
    """Generates definitions for a list of words."""
    
    def __init__(
        self,
        backend: ModelBackend,
        prompt_manager: PromptManager,
        config: EvaLexConfig,
        nlp=None,
    ):
        """
        Initialize the definition generator.
        
        Args:
            backend: Model backend for generation
            prompt_manager: Prompt manager for creating prompts
            config: EvaLex configuration
            nlp: Optional Stanza pipeline for lemmatization
        """
        self.backend = backend
        self.prompt_manager = prompt_manager
        self.config = config
        self.nlp = nlp
    
    def generate(
        self,
        words_df: pd.DataFrame,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Generate definitions for all words in the DataFrame.
        
        Args:
            words_df: DataFrame with 'word' column (and optionally 'category')
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with words and their generated definitions
        """
        # Create prompts
        prompts = self.prompt_manager.create_definition_dataset(
            words_df,
            use_categories=self.config.categories,
        )
        
        definitions = []
        batch_size = self.config.batch_size_def
        num_sequences = self.config.num_return_sequences
        
        # Generate in batches
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=total_batches):
            batch = prompts[i:i+batch_size]
            
            outputs = self.backend.generate(
                batch,
                num_return_sequences=num_sequences,
                max_new_tokens=128,
                temperature=1.0,
                do_sample=True,
                stop_strings=self.config.stop_strings,
            )
            definitions.extend(outputs)
            
            if progress_callback:
                progress = min((i + batch_size) / len(prompts), 1.0)
                progress_callback(progress, f"Generating definitions: {i+batch_size}/{len(prompts)}")
        
        # Process definitions
        outputs = {}
        for idx, definition in enumerate(definitions):
            word_idx = idx // num_sequences
            word = words_df.iloc[word_idx]["word"].lower()
            
            processed_def = process_definition_output(definition, word, self.nlp)
            
            outputs[idx] = [word, processed_def]
        
        columns = ["word", "definition"]
        return pd.DataFrame.from_dict(outputs, orient="index", columns=columns)


class WordGenerator:
    """Generates candidate words from definitions."""
    
    def __init__(
        self,
        backend: ModelBackend,
        prompt_manager: PromptManager,
        config: EvaLexConfig,
    ):
        """
        Initialize the word generator.
        
        Args:
            backend: Model backend for generation
            prompt_manager: Prompt manager for creating prompts
            config: EvaLex configuration
        """
        self.backend = backend
        self.prompt_manager = prompt_manager
        self.config = config
    
    def generate(
        self,
        definitions_df: pd.DataFrame,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Generate candidate words for all definitions in the DataFrame.
        
        Args:
            definitions_df: DataFrame with 'word' and 'definition' columns
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with definitions and predicted words
        """
        # Create prompts
        prompts = self.prompt_manager.create_words_dataset(
            definitions_df,
            use_categories=self.config.categories,
        )
        
        words = []
        batch_size = self.config.batch_size_words
        
        # Generate in batches
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=total_batches):
            batch = prompts[i:i+batch_size]
            
            outputs = self.backend.generate(
                batch,
                num_return_sequences=1,
                max_new_tokens=128,
                temperature=0,  # Deterministic for word prediction
                do_sample=False,
                stop_strings=self.config.stop_strings,
            )
            words.extend(outputs)
            
            if progress_callback:
                progress = min((i + batch_size) / len(prompts), 1.0)
                progress_callback(progress, f"Generating words: {i+batch_size}/{len(prompts)}")
        
        # Process words
        outputs = {}
        for idx, predicted in enumerate(words):
            word = definitions_df.iloc[idx]["word"].lower()
            definition = definitions_df.iloc[idx]["definition"].lower()
            
            processed_words = process_words_output(predicted)
            
            outputs[idx] = [word, definition, processed_words]
        
        columns = ["word", "definition", "predicted_words"]
        return pd.DataFrame.from_dict(outputs, orient="index", columns=columns)


class WordEvaluator:
    """Evaluates if words are known based on predictions."""
    
    def __init__(self, num_return_sequences: int = 5):
        """
        Initialize the evaluator.
        
        Args:
            num_return_sequences: Number of definitions generated per word
        """
        self.num_return_sequences = num_return_sequences
    
    def evaluate_single(self, word: str, predicted_words: str) -> bool:
        """
        Evaluate if a single word is known.
        
        Args:
            word: Target word
            predicted_words: Predicted words string
            
        Returns:
            True if word is found in predictions
        """
        return evaluate_word_known(word, predicted_words)
    
    def evaluate(
        self,
        generations_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate all words in the generations DataFrame.
        
        Args:
            generations_df: DataFrame with 'word' and 'predicted_words' columns
            
        Returns:
            Tuple of (results DataFrame, metrics dict)
        """
        df = generations_df.copy()
        df = df.fillna("")
        
        # Evaluate each row
        df['known'] = df.apply(
            lambda row: evaluate_word_known(row['word'], row['predicted_words']),
            axis=1
        )
        
        # Group by word (every num_return_sequences rows)
        group_known = df.groupby(df.index // self.num_return_sequences)['known'].transform('any')
        unique_words = df.groupby(df.index // self.num_return_sequences)['word'].first()
        
        result_df = pd.DataFrame({
            'word': unique_words,
            'group_known': group_known[::self.num_return_sequences].values
        })
        
        # Calculate metrics
        metrics = calculate_lexical_competence(
            result_df['word'].tolist(),
            result_df['group_known'].tolist(),
        )
        
        return result_df, metrics


class EvaLexPipeline:
    """
    Complete EvaLex evaluation pipeline.
    
    Orchestrates the full flow: words -> definitions -> candidate words -> evaluation
    """
    
    def __init__(
        self,
        config: EvaLexConfig,
        nlp=None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: EvaLex configuration
            nlp: Optional Stanza pipeline for lemmatization
        """
        self.config = config
        self.nlp = nlp
        self.backend = None
        self.prompt_manager = None
    
    def setup(self) -> None:
        """Set up the pipeline components."""
        # Create backend
        self.backend = create_backend(self.config)
        
        # Create prompt manager
        self.prompt_manager = PromptManager(
            self.config.prompts_file if self.config.prompts_file else None
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.backend:
            self.backend.cleanup()
    
    def run(
        self,
        words_df: pd.DataFrame,
        progress_callback: Optional[callable] = None,
        save_intermediate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            words_df: DataFrame with 'word' column
            progress_callback: Optional callback for progress updates
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (results DataFrame, metrics dict)
        """
        if not self.backend:
            self.setup()
        
        # Step 1: Generate definitions
        if progress_callback:
            progress_callback(0.0, "Generating definitions...")
        
        def_generator = DefinitionGenerator(
            self.backend,
            self.prompt_manager,
            self.config,
            self.nlp,
        )
        
        definitions_df = def_generator.generate(words_df, progress_callback)
        
        if save_intermediate:
            self._save_intermediate(definitions_df, "definitions")
        
        # Step 2: Generate candidate words
        if progress_callback:
            progress_callback(0.33, "Generating candidate words...")
        
        word_generator = WordGenerator(
            self.backend,
            self.prompt_manager,
            self.config,
        )
        
        generations_df = word_generator.generate(definitions_df, progress_callback)
        
        if save_intermediate:
            self._save_intermediate(generations_df, "generations")
        
        # Step 3: Evaluate
        if progress_callback:
            progress_callback(0.66, "Evaluating results...")
        
        evaluator = WordEvaluator(self.config.num_return_sequences)
        results_df, metrics = evaluator.evaluate(generations_df)
        
        if save_intermediate:
            self._save_results(results_df)
        
        if progress_callback:
            progress_callback(1.0, "Evaluation complete!")
        
        return results_df, metrics
    
    def _save_intermediate(self, df: pd.DataFrame, stage: str) -> None:
        """Save intermediate results."""
        output_dir = self.config.get_generations_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self.config.get_output_filename()
        filepath = output_dir / filename
        
        df.to_csv(filepath, sep="\t", index=False)
    
    def _save_results(self, df: pd.DataFrame) -> None:
        """Save final results."""
        output_dir = self.config.get_results_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self.config.get_output_filename()
        filepath = output_dir / filename
        
        df.to_csv(filepath, sep="\t", index=False)


def load_results_for_ranking(
    results_dir: str = "results",
    word_list_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all results files for the ranking display.
    
    Args:
        results_dir: Directory containing results
        word_list_filter: Optional filter for specific word list
        
    Returns:
        DataFrame with model rankings
    """
    results_path = Path(results_dir)
    all_results = []
    
    # Find all result files
    for version_dir in results_path.iterdir():
        if not version_dir.is_dir():
            continue
        
        for result_file in version_dir.glob("*.tsv"):
            # Parse filename to get model and word list
            filename = result_file.stem
            parts = filename.split("_")
            
            # Model name is everything except the last part (word list)
            if len(parts) >= 2:
                # Find where the word list part starts
                word_list_parts = []
                model_parts = []
                found_word_list = False
                
                for i, part in enumerate(parts):
                    # Check if this part is numeric or starts with a digit
                    if part[0].isdigit() or part in ["CREA", "DEA", "palabras"]:
                        found_word_list = True
                    
                    if found_word_list:
                        word_list_parts.append(part)
                    else:
                        model_parts.append(part)
                
                model_name = "_".join(model_parts) if model_parts else filename
                word_list = "_".join(word_list_parts) if word_list_parts else "unknown"
                
                # Apply filter if specified
                if word_list_filter and word_list_filter not in word_list:
                    continue
                
                try:
                    df = pd.read_csv(result_file, sep="\t")
                    
                    if "group_known" in df.columns:
                        known_count = df["group_known"].sum()
                        total_count = len(df)
                        accuracy = known_count / total_count if total_count > 0 else 0
                        
                        all_results.append({
                            "model": model_name.replace("_", "/"),
                            "word_list": word_list,
                            "known_words": int(known_count),
                            "total_words": total_count,
                            "accuracy": accuracy,
                            "accuracy_pct": f"{accuracy * 100:.2f}%",
                            "version": version_dir.name,
                        })
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
    
    if not all_results:
        return pd.DataFrame(columns=["model", "word_list", "known_words", "total_words", "accuracy", "accuracy_pct", "version"])
    
    ranking_df = pd.DataFrame(all_results)
    ranking_df = ranking_df.sort_values(["word_list", "accuracy"], ascending=[True, False])
    
    return ranking_df
