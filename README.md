# Eigent Search (Q+)

Benchmarking [Eigent](https://github.com/eigent-ai/eigent)'s search agent and enhancing it with query processing toolkit (a suite of structured thinking tools).

## Supported Preset Agent Types

| Agent Type         | Description                                                                      |
|--------------------|----------------------------------------------------------------------------------|
| base               | Base chat agent with no tools  |
| search_only        | Search-only agent using Google Search tool                                       |
| eigent_search      | Default eigent search agent with search, browse, note-taking, and terminal tools |
| eigent_search_q+   | Eigent search agent with enhanced query processing toolkit                       |

## Supported Benchmarks

| Benchmark Name         | HuggingFace Data Path                  |
|------------------------|----------------------------------------|
| SimpleQA               | basicv8vc/SimpleQA                     |
| SimpleQA-Verified      | google/simpleqa-verified               |
| BrowseComp             | smolagents/browse_comp                 |
| WebWalker              | callanwu/WebWalkerQA                   |
| Musique                | dgslibisey/MuSiQue                     |
| Frames                 | google/frames-benchmark                |


## Get Started

1. Clone the repository:
```bash
git clone https://github.com/camel-ai/eigent_search_q_plus.git
```

2. Install dependencies:
```bash
cd eigent_search_q_plus
uv sync
source .venv/bin/activate
```

3. Set up environment variables:

Either export:

```bash
export OPENAI_API_KEY="your-openai-api-key"  # if you're using OPENAI backend models
export GOOGLE_API_KEY="your-google-api-key"  # for google search tool
export SEARCH_ENGINE_ID="your-search-engine-id"  # for google search tool
```

or save them in `.env`.

If you are using Azure, you need the following env variables:

```bash
export AZURE_OPENAI_BASE_URL="your-azure-url"
export AZURE_OPENAI_API_KEY="your-azure-api-key"
```

If you are using Minimax M2.5, you need the following env variables:

```bash
export MINIMAX_API_KEY="your-minimax-api-key"
export MINIMAX_BASE_URL="your-minimax-base-url"
```



4. Run the evaluation script (on the first five questions):
```bash
python scripts/simpleqa_eval.py -a eigent_search_q+ -n 5 
```

Please refer to the input parameters defined in the script. The benchmarking pipeline supports multiple error-handling mechanisms; if errors occur (e.g., failed problem IDs or other interruptions), use the `--resume-from` parameter to resume the process.

5. View results in the `results` directory.

## License

This project is licensed under the Apache License 2.0.
