# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Literal, Type

from camel.logger import get_logger
from pydantic import BaseModel
from tqdm.auto import tqdm

from eigent_search.config import (
    BackendModelConfig,
    LLMasJudgeConfig,
    SearchAgentType,
    SearchConfig,
)
from eigent_search.evaluation.base import BaseEvaluator
from eigent_search.orchestrator import SearchOrchestrator

logger = get_logger(__name__)

AGENT_TYPES = {
    "eigent_search": SearchAgentType.EIGENT_SEARCH,
    "eigent_search_q+": SearchAgentType.EIGENT_SEARCH_Q_PLUS,
    "search_only": SearchAgentType.SEARCH_ONLY,
}


MODEL_CONFIGS = {
    # Azure models
    "azure-gpt-5-mini": BackendModelConfig.AZURE_GPT_5_MINI,
    "azure-gpt-4.1": BackendModelConfig.AZURE_GPT_4_1,
    "azure-gpt-4.1-mini": BackendModelConfig.AZURE_GPT_4_1_MINI,
    # OpenAI models
    "gpt-5-mini": BackendModelConfig.GPT_5_MINI,
    "gpt-4.1": BackendModelConfig.GPT_4_1,
    "gpt-4.1-mini": BackendModelConfig.GPT_4_1_MINI,
    "gpt-4o": BackendModelConfig.GPT_4O,
    "gpt-4o-mini": BackendModelConfig.GPT_4O_MINI,
    # Ollama models
    "gpt-oss": BackendModelConfig.GPT_OSS,
}


def set_up_search_and_judge_config(
    working_directory: Path,
    agent_type: Literal[AGENT_TYPES.keys()],
    model_name: Literal[MODEL_CONFIGS.keys()],
    response_format: Type[BaseModel] | None = None,
) -> dict:
    """Set up the search config and judge config."""

    config = {
        "search_config": SearchConfig(
            working_directory=working_directory,
            **MODEL_CONFIGS[model_name].value,
            agent_type=AGENT_TYPES[agent_type],
            response_format=response_format,
        ),
        "judge_config": LLMasJudgeConfig(
            **MODEL_CONFIGS["azure-gpt-4.1"].value,  # We use gpt-4.1 as the judge model
        ),
    }

    saved_config = {
        "search_config": json.loads(config["search_config"].model_dump_json(indent=2)),
        "judge_config": json.loads(config["judge_config"].model_dump_json(indent=2)),
    }
    with open(working_directory / "config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    return config


def run_search_and_evaluate(
    test_sample: dict,
    search_config: SearchConfig,
    judge_config: LLMasJudgeConfig,
    evaluator_class: Type[BaseEvaluator],
) -> dict:
    """Run the search and evaluation for a single test sample."""

    # run the search
    search_orchestrator = SearchOrchestrator(search_config)
    search_result = search_orchestrator.run_agent(
        search_orchestrator.create_search_request(
            input_query=test_sample["problem"], query_id=test_sample["id"]
        )
    )
    if hasattr(search_result, "error"):
        return {
            "input_sample": test_sample,
            "search_result": {
                "error": search_result.error,
            },
            "eval_result": None,
        }

    # run the evaluation
    judge_agent = judge_config.create_agent()
    evaluator = evaluator_class(judge_agent)
    eval_request = evaluator.create_request(
        query=test_sample["problem"],
        reference_answer=test_sample["answer"],
        model_answer=search_result.formatted_response,
    )
    eval_result = evaluator.evaluate(eval_request)
    return {
        "input_sample": test_sample,
        "search_result": {
            "response": search_result.formatted_response,
            "tool_trajectory": search_result.tool_trajectory.model_dump(),
            "token_usage": search_result.token_usage,
        },
        "eval_result": {
            "score": eval_result.score,
            "metrics": eval_result.metrics,
        },
    }


def run_search_and_evaluate_multithreaded(
    test_samples: list[dict],
    working_directory: Path,
    benchmark_name: str,
    agent_type: Literal[AGENT_TYPES.keys()],
    model_name: Literal[MODEL_CONFIGS.keys()],
    evaluator_class: Type[BaseEvaluator],
    response_format: Type[BaseModel] | None = None,
    num_workers: int = 10,
    existing_results: list[dict] = [],
) -> list[dict]:
    """Run the search and evaluation for a list of test samples in parallel."""

    process_bar = tqdm(total=len(test_samples), desc=f"{benchmark_name} evaluation")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for test_sample in test_samples:
            sample_wise_working_directory = working_directory / str(test_sample["id"])
            sample_wise_working_directory.mkdir(exist_ok=True)
            config = set_up_search_and_judge_config(
                working_directory=sample_wise_working_directory,
                agent_type=agent_type,
                model_name=model_name,
                response_format=response_format,
            )
            search_config = config["search_config"]
            judge_config = config["judge_config"]

            futures.append(
                executor.submit(
                    run_search_and_evaluate,
                    test_sample,
                    search_config,
                    judge_config,
                    evaluator_class,
                )
            )
        results = []

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 5 == 0 or i == len(test_samples) - 1:
                with open(working_directory / "results.jsonl", "w") as f:
                    for result in existing_results + results:
                        f.write(json.dumps(result) + "\n")
                logger.info(
                    f"Progress: {i + 1}/{len(test_samples)} ({(i + 1) / len(test_samples) * 100:.1f}%) - Results saved to {working_directory / 'results.jsonl'} ..."
                )
            process_bar.update(1)

        return results
