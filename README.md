# EvaLex

EvaLex is a benchmark designed to automatically evaluate Lexical Competence in Large Language Models (LLMs). It aims to aid researchers in benchmarking and improving the lexical competence of both small and large language models.

## Features

- **Modular Pipeline**: Generate definitions → Generate candidate words → Evaluate results
- **Multiple Backends**: Support for local models (via transformers) and remote models (via OpenAI API)
- **Gradio Interface**: User-friendly web interface for evaluation and model comparison
- **Ranking Leaderboard**: Compare performance across different models

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

Clone the repository:

```bash
git clone https://github.com/jcollado95/EvaLex.git
cd EvaLex
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open `http://localhost:7860` in your browser. The interface provides:
- **Evaluate Model**: Enter words, select a model, and get lexical competence scores
- **Ranking**: Compare results across different models for default word lists

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
├── app.py                  # Gradio web interface
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

WIP

## Contact

For questions or support, please open an issue or contact the maintainer (jcollado@ujaen.es).