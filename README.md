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

3. Run the evaluation script:
```bash
# run first five SimpleQA examples
python scripts/simpleqa_eval.py -a research -n 5 
```

4. View results in the `results` directory.

## License

This project is licensed under the Apache License 2.0.