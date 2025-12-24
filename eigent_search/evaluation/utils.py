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

import json
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Type

from camel.logger import get_logger

from eigent_search.config import (
    BackendModelConfig,
    LLMasJudgeConfig,
    SearchAgentType,
    SearchConfig,
)
from eigent_search.evaluation.base import BaseEvaluator
from eigent_search.orchestrator import SearchOrchestrator
from pydantic import BaseModel
from tqdm.auto import tqdm

logger = get_logger(__name__)

AGENT_TYPES = {
    "eigent_search": SearchAgentType.EIGENT_SEARCH,
    "eigent_search_q+": SearchAgentType.EIGENT_SEARCH_Q_PLUS,
    "search_only": SearchAgentType.SEARCH_ONLY,
    "baseline": SearchAgentType.BASELINE,
}
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
    effective_response_format = None if agent_type == "baseline" else response_format

    config = {
        "search_config": SearchConfig(
            working_directory=working_directory,
            **MODEL_CONFIGS[model_name].value,
            agent_type=AGENT_TYPES[agent_type],
            response_format=effective_response_format,
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
    max_eval_retries: int = 3,
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

    # run the evaluation with retry logic for transient API errors
    judge_agent = judge_config.create_agent()
    evaluator = evaluator_class(judge_agent)
    eval_request = evaluator.create_request(
        query=test_sample["problem"],
        reference_answer=test_sample["answer"],
        model_answer=search_result.formatted_response,
    )
    #TODO: Rewrite with tenacity in a more elegant way
    eval_result = None
    last_error = None
    for attempt in range(max_eval_retries):
        try:
            eval_result = evaluator.evaluate(eval_request)
            break
        except Exception as e:
            last_error = e
            logger.warning(
                f"[{test_sample['id']}] Evaluation attempt {attempt + 1}/{max_eval_retries} failed: {type(e).__name__}: {e}"
            )
            if attempt < max_eval_retries - 1:
                import time

                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
                # Create a fresh judge agent for retry
                judge_agent = judge_config.create_agent()
                evaluator = evaluator_class(judge_agent)

    if eval_result is None:
        logger.error(
            f"[{test_sample['id']}] Evaluation failed after {max_eval_retries} attempts: {last_error}"
        )
        return {
            "input_sample": test_sample,
            "search_result": {
                "response": search_result.formatted_response,
                "tool_trajectory": search_result.tool_trajectory.model_dump(),
                "token_usage": search_result.token_usage,
            },
            "eval_result": {
                "error": str(last_error),
                "score": None,
                "metrics": None,
            },
        }

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
                        try:
                            f.write(json.dumps(result) + "\n")
                        except TypeError as e:
                            # Catch serialization error and output which sample caused it
                            sample_id = result.get("input_sample", {}).get(
                                "id", "UNKNOWN"
                            )
                            logger.error(
                                f"\n{'='*80}\n"
                                f"JSON Serialization Error!\n"
                                f"Sample ID: {sample_id}\n"
                                f"Error: {e}\n"
                                f"Problematic keys in result:\n"
                            )

                            # Recursive function to find non-serializable objects
                            def find_non_serializable(obj, path=""):
                                """Recursively find non-serializable objects in nested structures."""
                                try:
                                    json.dumps(obj)
                                    return []  # This object is serializable
                                except (TypeError, ValueError):
                                    # This object is not serializable
                                    issues = []

                                    if isinstance(obj, dict):
                                        for key, value in obj.items():
                                            current_path = (
                                                f"{path}.{key}" if path else key
                                            )
                                            try:
                                                json.dumps(value)
                                                # This value is serializable, skip
                                            except (TypeError, ValueError):
                                                # Recursively check this value
                                                sub_issues = find_non_serializable(
                                                    value, current_path
                                                )
                                                if sub_issues:
                                                    issues.extend(sub_issues)
                                                else:
                                                    # This is a leaf non-serializable value
                                                    issues.append(
                                                        {
                                                            "path": current_path,
                                                            "type": type(
                                                                value
                                                            ).__name__,
                                                            "value": str(value)[
                                                                :100
                                                            ],  # First 100 chars
                                                        }
                                                    )
                                    elif isinstance(obj, (list, tuple)):
                                        for idx, item in enumerate(obj):
                                            current_path = f"{path}[{idx}]"
                                            try:
                                                json.dumps(item)
                                            except (TypeError, ValueError):
                                                sub_issues = find_non_serializable(
                                                    item, current_path
                                                )
                                                if sub_issues:
                                                    issues.extend(sub_issues)
                                                else:
                                                    issues.append(
                                                        {
                                                            "path": current_path,
                                                            "type": type(item).__name__,
                                                            "value": str(item)[:100],
                                                        }
                                                    )
                                    else:
                                        # Leaf non-serializable object
                                        issues.append(
                                            {
                                                "path": path or "(root)",
                                                "type": type(obj).__name__,
                                                "value": str(obj)[:100],
                                            }
                                        )

                                    return issues

                            # Find all non-serializable objects
                            issues = find_non_serializable(result)

                            if issues:
                                logger.error(
                                    f"\n🔍 Found {len(issues)} non-serializable object(s):\n"
                                )
                                for idx, issue in enumerate(issues, 1):
                                    logger.error(f"  [{idx}] Path: {issue['path']}")
                                    logger.error(f"      Type: {issue['type']}")
                                    logger.error(f"      Value: {issue['value']}")
                                    logger.error("")

                            logger.error(f"{'='*80}\n")
                            logger.error(
                                f"⚠️  Skipping sample {sample_id} due to serialization error. Other results will still be saved.\n"
                            )
                            # Don't raise - continue processing other results
                            continue
                logger.info(
                    f"Progress: {i + 1}/{len(test_samples)} ({(i + 1) / len(test_samples) * 100:.1f}%) - Results saved to {working_directory / 'results.jsonl'} ..."
                )
            process_bar.update(1)

        return results
