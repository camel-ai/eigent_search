# EigentSearch-Q+

Eigent search agent enhanced with query processing toolkit.

## Development Logs

- [X] Support search benchmarks on `SimpleQA`.
- [X] Support three predefined agent types: (1) `eigent_search`: default eigent search agent, (2) `eigent_search_plus`: eigent search agent with enhanced query processing toolkit, (3) `search_only`: search agent with google search tool only.

## Get Started

1. Clone the repository:
```bash
git clone https://github.com/camel-ai/librarian.git
```

2. Install dependencies:
```bash
cd librarian
uv venv --python 3.10 --prompt "(deep_research) " .venv
source .venv/bin/activate
uv sync
```

3. Set up environment variables (required for some agent types):
```bash
# For deep_search agent, you need OpenAI and Google API credentials:
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export SEARCH_ENGINE_ID="your-search-engine-id"
```

4. Run the evaluation script:
```bash
# run first five SimpleQA examples, starting from the first question
# Available agent types: search_only, eigent_search, eigent_search_plus, 

python scripts/simpleqa_eval.py -a eigent_search_plus -n 5 -s 0
```

> Run `scripts/simpleqa_eval_wsl2.py` should you are on WSL2 platform.

Please see input parameters inside the script.

5. View results in the `results` directory.

## License

This project is licensed under the Apache License 2.0.