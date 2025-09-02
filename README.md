# CAMEL-AI DeepResearch

## Development Logs

- [X] Base module design.
- [ ] Create evaluation guidelines.
- [X] Establish base evaluation at `evaluation`; evaluators for SimpleQA and BrowseComp are both ready. 
- [X] Create simple single agent baselines.

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
# Available agent types: simple_research, direct_answer, chain_of_thought, 
# knowledge_then_reasoning, deep_search
python scripts/simpleqa_eval.py -a deep_search -n 5 -s 1
```

> Run `scripts/simpleqa_eval_wsl2.py` should you are on WSL2 platform.

Please see input parameters inside the script.

5. View results in the `results` directory.

## License

This project is licensed under the Apache License 2.0.