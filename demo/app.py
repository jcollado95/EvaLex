"""
EvaLex Gradio Interface

An interactive demo for evaluating lexical competence in language models.
"""

import html
import os
import re
import time
import traceback
from typing import List, Optional

import gradio as gr
import nltk
import pandas as pd
import stanza
from dotenv import load_dotenv
from stanza.pipeline.core import DownloadMethod

from evalex.config import EvaLexConfig
from evalex.models import OpenAIModelBackend
from evalex.pipeline import DefinitionGenerator, WordGenerator, WordEvaluator
from evalex.prompts import DEFAULT_PROMPTS, LANGUAGES, PromptManager
from evalex.utils import evaluate_word_known
from usage import log_query


load_dotenv()

# Inference server (OpenAI-compatible API), configured through the environment / .env
API_BASE_URL = os.getenv("VLLM_BASE_URL")
API_KEY = os.getenv("VLLM_API_KEY")

# Models offered in the demo (display name -> model ID on the inference server)
MODELS = {
    "Llama 3.1 8B Instruct": "/mnt/beegfs/sinai-data/meta-llama/Llama-3.1-8B-Instruct",
    "Qwen 3.5 9B": "Qwen3.5-9B",
    "Gemma 4 31B": "gemma-4-31B-it",
    "Gemma 4 E4B": "gemma-4-E4B-it",
}

# Sent with every request: disables the reasoning mode of hybrid models (e.g. Qwen),
# which would otherwise spend the whole token budget before answering
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

FLAGS = {"es": "🇪🇸", "en": "🇬🇧", "fr": "🇫🇷"}

MAX_WORDS = 20
DEFAULT_NUM_DEFINITIONS = 5
DEFAULT_MAX_TOKENS_WORDS = 128
MAX_CONCURRENCY = 8

EXAMPLE_WORDS = {
    "es": ["casa", "ardilla", "conspirar", "kepis", "lipotimia"],
    "en": ["house", "squirrel", "conspire", "kepi", "syncope"],
    "fr": ["maison", "écureuil", "comploter", "képi", "lipothymie"],
}

GITHUB_URL = "https://github.com/jcollado95/EvaLex"


class EvaLexTheme(gr.themes.Base):
    """Indigo accent on slate neutrals, flat bordered cards, designed for light and dark."""

    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
            radius_size=gr.themes.sizes.radius_lg,
            spacing_size=gr.themes.sizes.spacing_md,
            text_size=gr.themes.sizes.text_md,
            # The flag font only covers flag code points (see CSS), so other text falls through to Inter
            font=["Twemoji Country Flags", gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        )
        self.set(
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_950",
            background_fill_primary="white",
            background_fill_primary_dark="*neutral_900",
            background_fill_secondary="*neutral_50",
            background_fill_secondary_dark="*neutral_950",
            block_background_fill="transparent",
            block_background_fill_dark="transparent",
            block_border_width="0px",
            block_shadow="none",
            block_shadow_dark="none",
            block_label_text_weight="600",
            block_label_text_color="*neutral_700",
            block_label_text_color_dark="*neutral_300",
            block_info_text_color="*neutral_500",
            block_info_text_color_dark="*neutral_400",
            input_background_fill="*neutral_50",
            input_background_fill_dark="*neutral_950",
            input_border_color="*neutral_200",
            input_border_color_dark="*neutral_700",
            input_border_color_focus="*primary_400",
            input_border_color_focus_dark="*primary_500",
            input_shadow="none",
            input_shadow_dark="none",
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_hover="*primary_700",
            button_primary_background_fill_dark="*primary_500",
            button_primary_background_fill_hover_dark="*primary_400",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            button_primary_border_color="transparent",
            button_primary_border_color_dark="transparent",
            button_secondary_background_fill="*neutral_100",
            button_secondary_background_fill_hover="*neutral_200",
            button_secondary_background_fill_dark="*neutral_800",
            button_secondary_background_fill_hover_dark="*neutral_700",
            button_secondary_border_color="transparent",
            button_secondary_border_color_dark="transparent",
            slider_color="*primary_500",
            slider_color_dark="*primary_400",
        )


CSS = """
/* Flag emoji are not drawn by Windows browsers; this font (limited to flag code points) fills the gap */
@font-face {
    font-family: "Twemoji Country Flags";
    unicode-range: U+1F1E6-1F1FF, U+1F3F4, U+E0062-E007F;
    src: url("https://cdn.jsdelivr.net/npm/country-flag-emoji-polyfill@0.1.8/dist/TwemojiCountryFlags.woff2") format("woff2");
    font-display: swap;
}

/* Defined on the container (not :root) so var() resolves against the dark theme values too */
.gradio-container {
    --evalex-ok: #059669;
    --evalex-ok-soft: #ecfdf5;
    --evalex-ko: #e11d48;
    --evalex-ko-soft: #fff1f2;
    --evalex-card: var(--background-fill-primary);
    --evalex-border: var(--neutral-200);
}
.dark .gradio-container {
    --evalex-ok: #34d399;
    --evalex-ok-soft: rgba(52, 211, 153, 0.12);
    --evalex-ko: #fb7185;
    --evalex-ko-soft: rgba(251, 113, 133, 0.12);
    --evalex-border: var(--neutral-800);
}

.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* Cards */
.ev-card {
    background: var(--evalex-card) !important;
    border: 1px solid var(--evalex-border) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
.ev-card .form, .ev-card .block, .ev-card .tabitem {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
.ev-card .form { gap: 14px !important; }
.ev-card .block { padding-left: 0 !important; padding-right: 0 !important; }
.ev-card-title h3 { margin: 0 !important; font-size: 1.05rem !important; }
.ev-card-title p { margin: 2px 0 0 !important; color: var(--body-text-color-subdued); font-size: var(--text-sm); }

/* Hero */
.ev-hero { padding: 28px 4px 8px; }
.ev-hero-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.ev-hero h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; line-height: 1.1; }
.ev-hero h1 span { color: var(--primary-600); }
.dark .ev-hero h1 span { color: var(--primary-400); }
.ev-tagline { margin: 8px 0 0; font-size: 1.05rem; color: var(--body-text-color-subdued); max-width: 640px; }
.ev-links { display: flex; gap: 8px; }
.ev-links a {
    text-decoration: none !important; font-size: var(--text-sm); font-weight: 600;
    padding: 6px 14px; border-radius: 999px; border: 1px solid var(--evalex-border);
    color: var(--body-text-color) !important; background: var(--evalex-card);
}
.ev-links a:hover { border-color: var(--primary-400); }
.ev-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }
.ev-step {
    display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px;
    background: var(--evalex-card); border: 1px solid var(--evalex-border); border-radius: 14px;
}
.ev-step-num {
    flex: 0 0 28px; height: 28px; border-radius: 999px; display: grid; place-items: center;
    font-weight: 700; font-size: 0.85rem; background: var(--primary-100); color: var(--primary-700);
}
.dark .ev-step-num { background: var(--primary-900); color: var(--primary-200); }
.ev-step b { display: block; font-size: 0.95rem; }
.ev-step p { margin: 2px 0 0; font-size: var(--text-sm); color: var(--body-text-color-subdued); line-height: 1.4; }
@media (max-width: 760px) {
    .ev-steps { grid-template-columns: 1fr; }
    .ev-hero h1 { font-size: 2rem; }
    .ev-lang label { padding: 6px 10px !important; white-space: nowrap !important; }
}

/* Language segmented control */
.ev-lang .wrap { display: inline-flex !important; flex-wrap: nowrap !important; gap: 4px !important; padding: 4px !important;
    background: var(--background-fill-secondary); border: 1px solid var(--evalex-border); border-radius: 999px; }
.ev-lang label { border: none !important; border-radius: 999px !important; background: transparent !important;
    box-shadow: none !important; padding: 6px 14px !important; cursor: pointer; }
.ev-lang label.selected { background: var(--evalex-card) !important; box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important; }
.ev-lang input[type="radio"] { display: none !important; }
.ev-lang label span { white-space: nowrap !important; }

/* Word input + chips */
.ev-word-input .input-container { align-items: stretch !important; gap: 8px !important; }
.ev-word-input .submit-button {
    height: auto !important; min-width: 72px !important; padding: 0 16px !important;
    border-radius: var(--input-radius) !important; font-weight: 600 !important;
    background: var(--primary-50) !important; color: var(--primary-700) !important;
    border: 1px solid var(--primary-200) !important;
}
.ev-word-input .submit-button:hover { background: var(--primary-100) !important; }
.dark .ev-word-input .submit-button { background: rgba(99, 102, 241, 0.15) !important; color: var(--primary-200) !important;
    border-color: rgba(129, 140, 248, 0.35) !important; }
.dark .ev-word-input .submit-button:hover { background: rgba(99, 102, 241, 0.25) !important; }
.ev-words-title { margin: 4px 0 0 !important; font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.ev-chips { flex-wrap: wrap !important; gap: 8px !important; }
.ev-chips button {
    flex: 0 0 auto !important; border-radius: 999px !important; padding: 4px 12px !important;
    background: var(--primary-50) !important; color: var(--primary-700) !important;
    border: 1px solid var(--primary-200) !important; font-weight: 500 !important;
}
.dark .ev-chips button { background: rgba(99, 102, 241, 0.15) !important; color: var(--primary-200) !important;
    border-color: rgba(129, 140, 248, 0.35) !important; }
.ev-chips button:hover { border-color: var(--evalex-ko) !important; color: var(--evalex-ko) !important; }
.ev-text-btns { gap: 4px !important; }
.ev-text-btns button { flex: 0 0 auto !important; background: transparent !important; border: none !important;
    box-shadow: none !important; color: var(--primary-600) !important; padding: 2px 8px !important; }
.dark .ev-text-btns button { color: var(--primary-300) !important; }
.ev-text-btns button:hover { text-decoration: underline; }
.ev-evaluate button, button.ev-evaluate { font-weight: 700 !important; font-size: 1rem !important; }

/* Prompts */
.prompt-box textarea { font-family: var(--font-mono) !important; font-size: 0.8rem !important; line-height: 1.5 !important; }

/* Section titles */
.ev-section h2 { font-size: 1.3rem !important; margin: 16px 0 0 !important; }

/* Results */
.ev-empty {
    border: 1px dashed var(--evalex-border); border-radius: 16px; padding: 36px 20px; text-align: center;
    color: var(--body-text-color-subdued); background: var(--evalex-card);
}
.ev-empty svg { display: block; margin: 0 auto 8px; width: 32px; height: 32px; color: var(--primary-400); }
.ev-score {
    display: flex; align-items: center; gap: 24px; padding: 20px 24px; margin-bottom: 16px;
    background: var(--evalex-card); border: 1px solid var(--evalex-border); border-radius: 16px;
}
.ev-score-value { font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; color: var(--primary-600); }
.dark .ev-score-value { color: var(--primary-400); }
.ev-score-body { flex: 1; min-width: 0; }
.ev-score-title { font-size: 1.1rem; font-weight: 700; }
.ev-bar { height: 8px; border-radius: 999px; background: var(--background-fill-secondary);
    border: 1px solid var(--evalex-border); margin: 10px 0 8px; overflow: hidden; }
.ev-bar span { display: block; height: 100%; border-radius: 999px; background: var(--primary-500); }
.ev-meta { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.ev-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; align-items: start; }
.ev-word { background: var(--evalex-card); border: 1px solid var(--evalex-border); border-radius: 14px;
    padding: 14px 16px; border-left: 4px solid var(--evalex-ko); }
.ev-word.ok { border-left-color: var(--evalex-ok); }
.ev-word-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.ev-word-name { font-size: 1.15rem; font-weight: 700; }
.ev-badge { font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    color: var(--evalex-ko); background: var(--evalex-ko-soft); }
.ev-badge.ok { color: var(--evalex-ok); background: var(--evalex-ok-soft); }
.ev-dots { display: flex; align-items: center; gap: 5px; margin-top: 8px; font-size: var(--text-sm);
    color: var(--body-text-color-subdued); }
.ev-dot { width: 10px; height: 10px; border-radius: 999px; background: var(--background-fill-secondary);
    border: 1px solid var(--evalex-border); }
.ev-dot.hit { background: var(--evalex-ok); border-color: var(--evalex-ok); }
.ev-dots-label { margin-left: 6px; }
.ev-word details { margin-top: 10px; }
.ev-word summary { cursor: pointer; font-size: var(--text-sm); font-weight: 600; color: var(--primary-600); }
.dark .ev-word summary { color: var(--primary-300); }
.ev-defs { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-direction: column; gap: 10px; }
.ev-defs li { padding: 10px 12px; border-radius: 10px; background: var(--background-fill-secondary); }
.ev-def { margin: 0 0 8px; font-size: var(--text-sm); line-height: 1.45; }
.ev-def-match { float: right; margin-left: 8px; font-size: 0.72rem; font-weight: 700;
    padding: 1px 8px; border-radius: 999px; color: var(--evalex-ko); background: var(--evalex-ko-soft); }
.ev-def-match.hit { color: var(--evalex-ok); background: var(--evalex-ok-soft); }
.ev-mask { display: inline-block; min-width: 2.2em; border-bottom: 2px solid var(--primary-400);
    margin: 0 2px; height: 0.9em; vertical-align: baseline; }
.ev-cands { display: flex; flex-wrap: wrap; gap: 4px; }
.ev-cand { font-size: 0.78rem; padding: 2px 8px; border-radius: 999px; background: var(--evalex-card);
    border: 1px solid var(--evalex-border); }
.ev-cand.hit { color: var(--evalex-ok); border-color: var(--evalex-ok); background: var(--evalex-ok-soft); font-weight: 700; }
.ev-cand.none { font-style: italic; color: var(--body-text-color-subdued); }

/* Reference */
.ev-bibtex textarea { font-family: var(--font-mono) !important; font-size: 0.78rem !important; line-height: 1.5 !important; }
.ev-contact p { margin: 0 0 6px !important; }
"""

HERO = f"""
<div class="ev-hero">
  <div class="ev-hero-top">
    <div>
      <h1>Eva<span>Lex</span></h1>
      <p class="ev-tagline">Which language model fits your needs?</p>
    </div>
    <nav class="ev-links">
      <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
      <a href="#reference">Cite</a>
    </nav>
  </div>
  <div class="ev-steps">
    <div class="ev-step"><span class="ev-step-num">1</span><div><b>Define</b>
      <p>The model writes several definitions of a given word.</p></div></div>
    <div class="ev-step"><span class="ev-step-num">2</span><div><b>Retrieve</b>
      <p>In a fresh session, it names the words that fit each definition, with the original word hidden.</p></div></div>
    <div class="ev-step"><span class="ev-step-num">3</span><div><b>Match</b>
      <p>The word is <i>known</i> if it appears among the candidates of at least one definition.</p></div></div>
  </div>
</div>
"""

EMPTY_RESULTS = """
<div class="ev-empty">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
  Add some words above and press <b>Evaluate</b> to see which ones the model knows.
</div>
"""

BIBTEX = """@inproceedings{collado-montanez-etal-2026-evalex,
    title = "{E}va{L}ex: Automatic Evaluation of Lexical Competence in Language Models",
    author = "Collado-Monta{\\~n}ez, Jaime and
      Ure{\\~n}a-L{\\'o}pez, L. Alfonso and
      Montejo-R{\\'a}ez, Arturo",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    publisher = "Association for Computational Linguistics",
}"""

CONTACT = f"""
### Contact
**Jaime Collado-Montañez**, **L. Alfonso Ureña-López** and **Arturo Montejo-Ráez**

University of Jaén, Spain

Questions about the paper or the demo: [jcollado@ujaen.es](mailto:jcollado@ujaen.es)

Bugs and suggestions: [GitHub issues]({GITHUB_URL}/issues)
"""

# Stanza pipelines per language, used to mask every form of the target word in definitions
LEMMATIZERS = {}


def load_lemmatizers() -> None:
    """Load (downloading on first use) a Stanza lemmatizer for each language."""
    for code in LANGUAGES:
        try:
            LEMMATIZERS[code] = stanza.Pipeline(
                code,
                processors="tokenize,mwt,pos,lemma",
                download_method=DownloadMethod.REUSE_RESOURCES,
                logging_level="WARN",
            )
        except Exception:
            # Masking falls back to case- and accent-insensitive matching of the word itself
            traceback.print_exc()
            print(f"Warning: no lemmatizer for '{code}'; inflected forms will not be masked.")


def prompt_fields(language: str) -> tuple:
    """Return the editable prompt fields for a language's default prompts."""
    def_prompt = DEFAULT_PROMPTS[language]["def"]
    words_prompt = DEFAULT_PROMPTS[language]["words"]
    return (
        def_prompt[0]["content"],
        def_prompt[1]["content"],
        words_prompt[0]["content"],
        words_prompt[1]["content"],
        words_prompt[2]["content"],
        words_prompt[3]["content"],
    )


def build_prompts(
    def_system: str,
    def_user: str,
    words_system: str,
    words_example_user: str,
    words_example_answer: str,
    words_user: str,
) -> tuple:
    """Build the chat prompts for both steps from the edited fields."""
    def_prompt = [
        {"role": "system", "content": def_system},
        {"role": "user", "content": def_user},
    ]
    words_prompt = [{"role": "system", "content": words_system}]
    if words_example_user.strip() and words_example_answer.strip():
        words_prompt += [
            {"role": "user", "content": words_example_user},
            {"role": "assistant", "content": words_example_answer},
        ]
    words_prompt.append({"role": "user", "content": words_user})
    return def_prompt, words_prompt


# A single word: letters/digits, optionally joined by hyphens or apostrophes ("_" is the mask token)
WORD_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")


def add_word(new_word: str, words: List[str]) -> tuple:
    """
    Add a word to the list after validating it.

    Returns:
        Tuple of (textbox value, updated word list)
    """
    word = new_word.strip().lower().replace("’", "'")
    if not word:
        return new_word, words
    if any(c.isspace() for c in word):
        gr.Warning(
            f"“{word}” contains spaces. EvaLex evaluates single words only: "
            "add each word separately."
        )
        return new_word, words
    if not WORD_PATTERN.fullmatch(word):
        gr.Warning(f"“{word}” is not a valid word. Use letters only (hyphens and apostrophes are allowed).")
        return new_word, words
    if word in words:
        gr.Warning(f"“{word}” is already in the list.")
        return "", words
    if len(words) >= MAX_WORDS:
        gr.Warning(f"You can evaluate up to {MAX_WORDS} words at a time.")
        return new_word, words
    return "", words + [word]


def validate(words: list, def_user: str, words_user: str) -> Optional[str]:
    """Return an error message if the request is invalid, else None."""
    if not words:
        return "Add at least one word to evaluate."
    if len(words) > MAX_WORDS:
        return f"Please enter at most {MAX_WORDS} words (got {len(words)})."
    if "{word}" not in def_user:
        return "The Step 1 user message must contain the {word} placeholder."
    if "{definition}" not in words_user:
        return "The Step 2 user message must contain the {definition} placeholder."
    return None


def render_definition(definition: str) -> str:
    """Escape a definition and show the masked word ("_") as a blank."""
    escaped = html.escape(definition)
    return re.sub(r'(?<!\w)_(?!\w)', '<span class="ev-mask" title="hidden word"></span>', escaped)


def render_candidates(word: str, candidates: List[str]) -> str:
    """Render candidates as chips, highlighting the ones that match the target word."""
    if not candidates:
        return '<span class="ev-cand none">no candidates</span>'
    return "".join(
        f'<span class="ev-cand{" hit" if evaluate_word_known(word, c) else ""}">{html.escape(c)}</span>'
        for c in candidates
    )


def render_results(
    model_label: str,
    language: str,
    num_definitions: int,
    metrics: dict,
    results_df: pd.DataFrame,
    generations_df: pd.DataFrame,
) -> str:
    """Render the score summary and one card per word."""
    pct = metrics["accuracy"] * 100
    parts = [f"""
<div class="ev-score">
  <div class="ev-score-value">{pct:.0f}%</div>
  <div class="ev-score-body">
    <div class="ev-score-title">{metrics['known_count']} of {metrics['total_count']} words known</div>
    <div class="ev-bar"><span style="width: {pct:.1f}%"></span></div>
    <div class="ev-meta">{html.escape(model_label)} · {FLAGS[language]} {LANGUAGES[language]} · {num_definitions} definitions per word</div>
  </div>
</div>
<div class="ev-grid">"""]

    for word, known in zip(results_df["word"], results_df["group_known"]):
        rows = generations_df[generations_df["word"] == word]
        hits = int(rows["match"].sum())
        dots = "".join(f'<span class="ev-dot{" hit" if m else ""}"></span>' for m in rows["match"])
        definitions = "".join(
            f'<li><p class="ev-def"><span class="ev-def-match{" hit" if row.match else ""}">'
            f'{"match" if row.match else "no match"}</span>'
            f'{render_definition(row.definition)}</p>'
            f'<div class="ev-cands">{render_candidates(word, row.candidates)}</div></li>'
            for row in rows.itertuples(index=False)
        )
        parts.append(f"""
  <div class="ev-word{" ok" if known else ""}">
    <div class="ev-word-head">
      <span class="ev-word-name">{html.escape(word)}</span>
      <span class="ev-badge{" ok" if known else ""}">{"Known" if known else "Unknown"}</span>
    </div>
    <div class="ev-dots">{dots}<span class="ev-dots-label">recovered from {hits} of {len(rows)} definitions</span></div>
    <details>
      <summary>Definitions and candidates</summary>
      <ol class="ev-defs">{definitions}</ol>
    </details>
  </div>""")

    parts.append("</div>")
    return "".join(parts)


def on_language_change(language: str) -> tuple:
    """Load the default prompts and example words for the selected language."""
    gr.Info(f"Prompts reset to the {LANGUAGES[language]} defaults.")
    return (
        *prompt_fields(language),
        gr.update(placeholder=f"e.g. {EXAMPLE_WORDS[language][1]}"),
    )


def evaluate(
    words: List[str],
    language: str,
    model_label: str,
    num_definitions: float,
    max_tokens_words: float,
    def_system: str,
    def_user: str,
    words_system: str,
    words_example_user: str,
    words_example_answer: str,
    words_user: str,
    request: gr.Request = None,
    progress=gr.Progress(),
) -> str:
    """
    Run EvaLex on the given words. Every request is logged for usage statistics.

    Returns:
        HTML with the score summary and one card per word
    """
    start = time.time()
    num_definitions = int(num_definitions)
    max_tokens_words = int(max_tokens_words)
    fields = (def_system, def_user, words_system, words_example_user, words_example_answer, words_user)
    words = list(words or [])
    def_prompt, words_prompt = build_prompts(*fields)

    record = {
        "session": request.session_hash if request else None,
        "language": language,
        "model": model_label,
        "num_definitions": num_definitions,
        "max_tokens_words": max_tokens_words,
        "num_words": len(words),
        "words": words,
        "prompts_edited": fields != prompt_fields(language),
        "prompts": {"definition": def_prompt, "words": words_prompt},
    }

    if not API_BASE_URL or not API_KEY:
        error = "The evaluation service is not configured."
        log_query({**record, "status": "error", "message": error})
        raise gr.Error(error)

    error = validate(words, def_user, words_user)
    if error:
        log_query({**record, "status": "rejected", "message": error})
        raise gr.Error(error)

    prompt_manager = PromptManager(
        language=language,
        def_prompt=def_prompt,
        words_prompt=words_prompt,
    )
    config = EvaLexConfig(
        model_name=MODELS[model_label],
        backend="openai",
        num_return_sequences=num_definitions,
        max_new_tokens_words=max_tokens_words,
        # Let models finish their JSON so it can be parsed
        stop_strings=None,
        strict_masking=True,
        batch_size_def=4,
        batch_size_words=4 * num_definitions,
    )
    backend = OpenAIModelBackend(
        model_name=MODELS[model_label],
        api_key=API_KEY,
        base_url=API_BASE_URL,
        max_concurrency=MAX_CONCURRENCY,
        extra_body=EXTRA_BODY,
        raise_errors=True,
    )
    words_df = pd.DataFrame({"word": words})

    try:
        progress(0.0, desc="Generating definitions...")
        definitions_df = DefinitionGenerator(
            backend, prompt_manager, config, nlp=LEMMATIZERS.get(language),
        ).generate(
            words_df,
            lambda pct, _: progress(0.5 * pct, desc="Generating definitions..."),
        )
        progress(0.5, desc="Retrieving words from definitions...")
        generations_df = WordGenerator(
            backend, prompt_manager, config, keep_candidate_lists=True,
        ).generate(
            definitions_df,
            lambda pct, _: progress(0.5 + 0.5 * pct, desc="Retrieving words from definitions..."),
        )
    except Exception as e:
        traceback.print_exc()
        log_query({
            **record,
            "status": "error",
            "message": f"{type(e).__name__}: {e}"[:1000],
            "duration_s": round(time.time() - start, 2),
        })
        raise gr.Error("The evaluation service is not available right now. Please try again later.")

    evaluator = WordEvaluator(num_definitions)
    results_df, metrics = evaluator.evaluate(generations_df)

    generations_df["match"] = generations_df.apply(
        lambda row: evaluator.evaluate_single(row["word"], row["predicted_words"]),
        axis=1,
    )

    log_query({
        **record,
        "status": "ok",
        "known_count": int(metrics["known_count"]),
        "duration_s": round(time.time() - start, 2),
        "results": {w: bool(k) for w, k in zip(results_df["word"], results_df["group_known"])},
        "details": [
            {
                "word": row.word,
                "definition": row.definition,
                "candidates": row.candidates,
                "match": bool(row.match),
            }
            for row in generations_df.itertuples(index=False)
        ],
    })

    return render_results(model_label, language, num_definitions, metrics, results_df, generations_df)


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""
    default_language = "es"
    defaults = prompt_fields(default_language)

    with gr.Blocks(title="EvaLex · Lexical Competence Evaluation") as app:
        gr.HTML(HERO, padding=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_classes="ev-card"):
                gr.Markdown("### Evaluate\nPick a language and a model, then add the words to test.",
                            elem_classes="ev-card-title")
                language = gr.Radio(
                    choices=[(f"{FLAGS[code]} {name}", code) for code, name in LANGUAGES.items()],
                    value=default_language,
                    label="Language",
                    elem_classes="ev-lang",
                )
                model = gr.Dropdown(
                    choices=list(MODELS),
                    value=next(iter(MODELS)),
                    label="Model",
                )
                word_list = gr.State([])
                words_input = gr.Textbox(
                    label="Word",
                    placeholder=f"e.g. {EXAMPLE_WORDS[default_language][1]}",
                    max_lines=1,
                    submit_btn="Add",
                    elem_classes="ev-word-input",
                )

                @gr.render(inputs=word_list)
                def render_word_list(words):
                    gr.Markdown(
                        f"**{len(words)}/{MAX_WORDS} words**"
                        + (" · add one word at a time (press Enter)" if not words else " · click a word to remove it"),
                        elem_classes="ev-words-title",
                    )
                    if not words:
                        return
                    with gr.Row(elem_classes="ev-chips"):
                        for word in words:
                            chip = gr.Button(f"{word}  ✕", size="sm", min_width=0)
                            chip.click(
                                fn=lambda current, w=word: [x for x in current if x != w],
                                inputs=word_list,
                                outputs=word_list,
                            )

                with gr.Row(elem_classes="ev-text-btns"):
                    example_btn = gr.Button("Use example words", size="sm", min_width=0)
                    clear_btn = gr.Button("Clear all", size="sm", min_width=0)
                evaluate_btn = gr.Button("Evaluate", variant="primary", size="lg", elem_classes="ev-evaluate")

            with gr.Column(scale=7, min_width=320, elem_classes="ev-card"):
                gr.Markdown(
                    "### Configuration\nThe exact prompts sent to the model. Edit them to try your own "
                    "variations: `{word}` and `{definition}` are filled in for each request.",
                    elem_classes="ev-card-title",
                )
                with gr.Tabs():
                    with gr.Tab("1 · Define"):
                        def_system = gr.Textbox(
                            value=defaults[0], label="System message", lines=10,
                            elem_classes="prompt-box",
                        )
                        def_user = gr.Textbox(
                            value=defaults[1], label="User message", info="Must contain {word}.",
                            lines=1, elem_classes="prompt-box",
                        )
                        num_definitions = gr.Slider(
                            minimum=1, maximum=10, step=1, value=DEFAULT_NUM_DEFINITIONS,
                            label="Definitions per word",
                        )
                    with gr.Tab("2 · Retrieve"):
                        words_system = gr.Textbox(
                            value=defaults[2], label="System message", lines=10,
                            elem_classes="prompt-box",
                        )
                        with gr.Row():
                            words_example_user = gr.Textbox(
                                value=defaults[3], label="Example: user message", lines=2,
                                elem_classes="prompt-box",
                            )
                            words_example_answer = gr.Textbox(
                                value=defaults[4], label="Example: model answer", lines=2,
                                elem_classes="prompt-box",
                            )
                        words_user = gr.Textbox(
                            value=defaults[5], label="User message", info="Must contain {definition}.",
                            lines=1, elem_classes="prompt-box",
                        )
                        max_tokens_words = gr.Slider(
                            minimum=16, maximum=512, step=16, value=DEFAULT_MAX_TOKENS_WORDS,
                            label="Max tokens per candidate list",
                        )
                with gr.Row(elem_classes="ev-text-btns"):
                    reset_btn = gr.Button("Reset prompts to defaults", size="sm", min_width=0)

        gr.Markdown("## Results", elem_classes="ev-section")
        results = gr.HTML(EMPTY_RESULTS, padding=False)

        gr.Markdown("## Reference", elem_classes="ev-section", elem_id="reference")
        with gr.Row(equal_height=False, elem_classes="ev-card ev-reference"):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("### Cite\nIf you use EvaLex in your research, please cite our paper:",
                            elem_classes="ev-card-title")
                gr.Textbox(
                    value=BIBTEX, show_label=False, interactive=False, lines=BIBTEX.count("\n") + 1,
                    buttons=["copy"], elem_classes="ev-bibtex",
                )
            with gr.Column(scale=2, min_width=260):
                gr.Markdown(CONTACT, elem_classes="ev-contact")

        prompt_inputs = [
            def_system, def_user, words_system, words_example_user, words_example_answer, words_user,
        ]

        language.change(
            fn=on_language_change,
            inputs=[language],
            outputs=[*prompt_inputs, words_input],
        )
        reset_btn.click(fn=prompt_fields, inputs=[language], outputs=prompt_inputs)
        words_input.submit(fn=add_word, inputs=[words_input, word_list], outputs=[words_input, word_list])
        example_btn.click(
            fn=lambda lang: list(EXAMPLE_WORDS[lang]),
            inputs=[language],
            outputs=[word_list],
        )
        clear_btn.click(fn=lambda: [], outputs=[word_list])
        evaluate_btn.click(
            fn=evaluate,
            inputs=[word_list, language, model, num_definitions, max_tokens_words, *prompt_inputs],
            outputs=[results],
        )

    return app


def main():
    """Run the Gradio application."""
    # Tokenizer data used to match the target word among the candidates
    nltk.download("punkt_tab", quiet=True)
    load_lemmatizers()

    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=EvaLexTheme(),
        css=CSS,
        footer_links=["gradio", "settings"],
    )


if __name__ == "__main__":
    main()
