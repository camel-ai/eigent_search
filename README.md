# The Librarian Paradigm: Structured Knowledge and Grounded Reasoning

## Introduction

Large language models (LLMs) often hallucinate, and it’s difficult to pinpoint the root cause — is it faulty knowledge or flawed reasoning? Given that knowledge is ever-changing while reasoning patterns remain relatively stable, why not **separate the two stages before generating an answer**? Much like a *librarian*, we could introduce a knowledge agent responsible for locating and organising relevant information. Reasoning would then operate solely on this curated knowledge, ensuring greater transparency and control over the final output.

## Road Map

![Road Map](docs/images/roadmap.png)


> Important code reference by OpenAI: https://github.com/openai/simple-evals

## Get Started

1. Clone the repository:
```bash
git clone https://github.com/camel-ai/librarian.git
```

2. Install dependencies:
```bash
cd librarian
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

3. Run the evaluation script:
```bash
python scripts/simpleqa_eval.py -a simple_librarian -n 500
```

4. View results in the `results` directory.

## License

This project is licensed under the Apache License 2.0.