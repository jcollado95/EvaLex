# EvaLex

EvaLex is a benchmark designed to automatically evaluate Lexical Competence in Large Language Models (LLMs). It aims to aid researchers in benchmarking and improving the lexical competence of both small and large language models.

## Getting Started

### Prerequisites

- Python 3.x

### Installation

Clone the repository:

```bash
git clone https://github.com/jcollado95/EvaLex.git
cd EvaLex
```

Install required dependencies (you may want to activate your own virtual environment before this step):

```bash
pip install -r requirements.txt
```


### Usage

To evaluate you own models, please run the scripts in the following order:

1. Generate definitions from a list of words:

```bash
python generate-definitions.py
```

2. Generate candidate terms from the already generated definitions:

```bash
python generate-words.py
```

3. Evaluate the resulting terms by matching them to the original words:

```bash
python evaluate-words.py
```

## Citation

WIP

## Contact

For questions or support, please open an issue or contact the maintainer (jcollado@ujaen.es).