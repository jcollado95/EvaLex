# EvaLex

EvaLex is a benchmark designed to automatically evaluate Lexical Competence in Large Language Models (LLMs). It aims to aid researchers in benchmarking and improving the lexical competence of both small and large language models.

## Features

- **Modular Pipeline**: Generate definitions → Generate candidate words → Evaluate results
- **Multiple Backends**: Support for local models (via transformers) and remote models (via OpenAI API)
- **Interactive Demo**: Gradio web interface to evaluate words in Spanish, English or French with editable prompts

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

Clone the repository:

```bash
git clone https://github.com/jcollado95/EvaLex.git
cd EvaLex
```

Install EvaLex and its dependencies:

```bash
pip install -e .            # core pipeline (CLI)
pip install -e ".[openai]"  # + OpenAI-compatible API backend
pip install -r demo/requirements.txt  # + Gradio web demo
```

Alternatively, `pip install -r requirements.txt` installs the environment used for the paper experiments (without the demo).

### Reproducing the paper

The exact code, data, generations and results reported in the EMNLP 2026 paper are frozen under the `v1.0-emnlp2026` tag:

```bash
git checkout v1.0-emnlp2026
```

The paper results are in `generations/` and `results/`. To regenerate them, run the command-line pipeline (Option 2 below) with the configs in `config/`.

### Usage

#### Option 1: Web Interface (Recommended)

The demo sends its requests to an OpenAI-compatible inference server. Copy `.env.example` to `.env` and fill in the server URL and API key:

```bash
cp .env.example .env
pip install -r demo/requirements.txt
python demo/app.py
```

On first start, the demo downloads the Stanza lemmatizers for Spanish, English and French. It uses them to hide every form of the target word in the generated definitions. Then open `http://localhost:7860` in your browser. Pick a language (Spanish, English or French) and a model, enter up to 20 words and press **Evaluate**. You can also change how many definitions the model writes per word and the token limit for each list of candidates. The prompts for both steps are shown next to the setup, and you can edit them. The results show which words the model knows, plus the definitions and candidate words behind each result.

Every evaluation request (settings, prompts, words and results) is logged to a local SQLite database at `demo/usage/usage.sqlite3`, which is gitignored. Set `EVALEX_USAGE_DB` to use another path. No IP addresses are stored, only an anonymous session id. To print usage statistics:

```bash
python demo/usage.py
```

#### Option 2: Command Line Pipeline

To evaluate models via command line, run the scripts in order:

1. **Generate definitions** from a list of words:

```bash
python generate-definitions.py config/config_Llama-3.1-8B-Instruct.yaml
```

2. **Generate candidate terms** from the definitions:

```bash
python generate-words.py config/config_Llama-3.1-8B-Instruct.yaml
```

3. **Evaluate** the resulting terms by matching them to the original words:

```bash
python evaluate-words.py config/config_Llama-3.1-8B-Instruct.yaml
```

#### Using OpenAI API

For remote models, set your API credentials:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or custom endpoint
```

Then use the `--backend openai` flag:

```bash
python generate-definitions.py config/your_config.yaml --backend openai
```

## Configuration

Create a YAML config file with the following options:

```yaml
model_name: meta-llama/Llama-3.1-8B-Instruct
eos_token: <|eot_id|>
words: data/CREA_10k/3-CREA_10k_pos_filter.tsv
iter: v1
prompts: prompts/llama/v9.json
num_words: 100
categories: False
batch_size_def: 10
batch_size_words: 10
stop_strings: "}"
```

## Project Structure

```
EvaLex/
├── evalex/                  # Core module
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── models.py           # Model backends (local/OpenAI)
│   ├── prompts.py          # Prompt templates
│   ├── utils.py            # Utility functions
│   └── pipeline.py         # Evaluation pipeline
├── demo/                   # Gradio web demo
│   ├── app.py
│   └── requirements.txt    # Demo dependencies (installs the local checkout)
├── pyproject.toml          # Package definition and optional extras
├── generate-definitions.py # CLI: Step 1
├── generate-words.py       # CLI: Step 2
├── evaluate-words.py       # CLI: Step 3
├── config/                 # Model configurations
├── data/                   # Word lists
├── prompts/                # Prompt templates
├── generations/            # Generated outputs
└── results/                # Evaluation results
```

## Citation

If you use EvaLex in your research, please cite:

```bibtex
@inproceedings{collado-montanez-etal-2026-evalex,
    title = "{E}va{L}ex: Automatic Evaluation of Lexical Competence in Language Models",
    author = "Collado-Monta{\~n}ez, Jaime and
      Ure{\~n}a-L{\'o}pez, L. Alfonso and
      Montejo-R{\'a}ez, Arturo",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    publisher = "Association for Computational Linguistics",
}
```

## Contact

For questions or support, please open an issue or contact the maintainer (jcollado@ujaen.es).
