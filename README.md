# EigentSearch-Q+

Eigent search agent enhanced with query processing toolkit.

## Supported Preset Agent Types

| Agent Type         | Description                                                                      |
|--------------------|----------------------------------------------------------------------------------|
| eigent_search_q+   | Eigent search agent with enhanced query processing toolkit                       |
| eigent_search      | Default eigent search agent with search, browse, note-taking, and terminal tools |
| search_only        | Search-only agent using Google Search tool                                       |

## Supported Benchmarks

| Benchmark Name         | HuggingFace Data Path                  |
|------------------------|----------------------------------------|
| SimpleQA               | basicv8vc/SimpleQA                     |
| SimpleQA-Verified      | google/simpleqa-verified               |
| BrowseComp             | smolagents/browse_comp                 |
| WebWalker              | callanwu/WebWalkerQA                   |
| Musique                | dgslibisey/MuSiQue                     |
| Frames                 | ...                                    |
| WideSearch             | ...                                    |


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
export AZURE_API_VERSION="your-azure-api-version"
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_DEPLOYMENT_NAME="your-azure-dev-name"
```



4. Run the evaluation script (on the first five questions):
```bash
python scripts/simpleqa_eval.py -a eigent_search_q+ -n 5 
```

Please see input parameters inside the script.

5. View results in the `results` directory.

## License

This project is licensed under the Apache License 2.0.