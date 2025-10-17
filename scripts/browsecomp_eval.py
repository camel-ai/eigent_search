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
import base64
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from camel.logger import get_logger, set_log_file, set_log_level
import click
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from eigent_search import (
    BackendModelConfig,
    LLMasJudgeConfig,
    SearchAgentType,
    SearchConfig,
    SearchOrchestrator,
)
from eigent_search.evaluation import BrowseCompEvaluator

set_log_level(logging.INFO)
logger = get_logger(__name__)

AGENT_TYPES = {
    "eigent_search": SearchAgentType.EIGENT_SEARCH,
    "eigent_search_q+": SearchAgentType.EIGENT_SEARCH_Q_PLUS,
    "search_only": SearchAgentType.SEARCH_ONLY,
}

MODEL_CONFIGS = {
    "gpt-4.1": BackendModelConfig.GPT_4_1,
    "gpt-4.1-mini": BackendModelConfig.GPT_4_1_MINI,
    "gpt-4o": BackendModelConfig.GPT_4O,
    "gpt-4o-mini": BackendModelConfig.GPT_4O_MINI,
    "gpt-oss": BackendModelConfig.GPT_OSS,
}

DATASET_NAME = "browsecomp"

GRADE_EMOJI_MAP = {
    "yes": "✓",
    "no": "✗",
    "ERROR": "🚫",
}


def _derive_key(password: str, length: int) -> bytes:
    """Derive encryption key from password."""
    h = hashlib.sha256()
    h.update(password.encode())
    key = h.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt ciphertext using password."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _json_safe(obj: Any) -> Any:
    """Ensure the object is JSON serializable; if not, convert it to a string."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def set_up_config(
        agent_type: Literal[AGENT_TYPES.keys()], model_name: Literal[MODEL_CONFIGS.keys()]
) -> dict:
    """Set up the search config."""

    class BrowseCompResponse(BaseModel):
        answer: str = Field(..., description="The answer to the research question.")
        search_results: list[str] = Field(
            ..., description="The search results that lead to the answer."
        )

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    working_directory = (
            Path(os.getcwd())
            / "results"
            / f"{DATASET_NAME}_eval_agent={agent_type}_model={model_name}_{time_stamp}"
    )
    working_directory.mkdir(parents=True, exist_ok=True)
    result_file = working_directory / "results.json"

    config = {
        "search_config": SearchConfig(
            working_directory=working_directory,
            **MODEL_CONFIGS[model_name].value,
            agent_type=AGENT_TYPES[agent_type],
            response_format=BrowseCompResponse,
        ),
        "judge_config": LLMasJudgeConfig(
            **MODEL_CONFIGS["gpt-4.1"].value,  # We use gpt-4.1 as the judge model
        ),
        "result_file": result_file,
    }

    set_log_file(config["search_config"].working_directory / f"{DATASET_NAME}_eval.log")

    return config


def run_search_and_evaluate(
        test_sample: dict, search_config: SearchConfig, judge_config: LLMasJudgeConfig
) -> dict:
    """Run the search and evaluation for a single test sample."""

    canary = test_sample.get("canary")
    enc_problem = test_sample.get("problem")
    enc_answer = test_sample.get("answer")

    # If there is no canary for this example, set grade as "ERROR"
    if not canary:
        return {
            "input_sample": test_sample,
            "search_result": {
                "error": "Missing canary",
            },
            "eval_result": None,
        }

    # If decode failed, set grade as "ERROR"
    try:
        problem = _decrypt(enc_problem, canary)
        answer = _decrypt(enc_answer, canary)
    except Exception as e:
        return {
            "input_sample": test_sample,
            "search_result": {
                "error": f"Decrypt failed: {e}",
            },
            "eval_result": None,
        }

    # run the search
    search_orchestrator = SearchOrchestrator(search_config)
    search_result = search_orchestrator.run_agent(
        search_orchestrator.create_search_request(
            input_query=problem, query_id=test_sample["id"]
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
    evaluator = BrowseCompEvaluator(judge_agent)
    eval_request = evaluator.create_request(
        query=problem,
        reference_answer=answer,
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
            "grade": eval_result.metrics.get("grade", "no"),
            "metrics": eval_result.metrics,
        },
    }


def run_search_and_evaluate_multithreaded(
        test_samples: list[dict],
        search_config: SearchConfig,
        judge_config: LLMasJudgeConfig,
        num_workers: int,
        result_file: Path,
) -> list[dict]:
    """Run the search and evaluation for a list of test samples in parallel."""

    process_bar = tqdm(total=len(test_samples), desc=f"{DATASET_NAME} Evaluation")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                run_search_and_evaluate, sample, search_config, judge_config
            )
            for sample in test_samples
        ]
        results = []

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 10 == 0 or i == len(test_samples) - 1:
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {result_file} ...")
            process_bar.update(1)

        return results


@click.command()
@click.option(
    "--agent_type", "-a", type=click.Choice(AGENT_TYPES.keys()), required=True
)
@click.option(
    "--model_name",
    "-m",
    type=click.Choice(MODEL_CONFIGS.keys()),
    default="gpt-4.1-mini",
)
@click.option("--num_workers", "-w", type=int, default=10)
@click.option("--num_questions", "-n", type=int, default=5)
@click.option(
    "--start_idx", "-s", type=int, default=0, help="Start index for the test samples."
)
@click.option(
    "--custom_idx_list",
    "-c",
    type=str,
    default=None,
    help="Customized list of question IDs to evaluate (e.g., '[1,2,3]') If provided, will ignore `num_questions` and `start_idx`.",
)
@click.option(
    "--test-all",
    is_flag=True,
    help="Test all questions in the dataset. If provided, will override the `num_questions`, `start_idx`, and `custom_idx_list`.",
    default=False,
)
def main(
        agent_type: str,
        model_name: str,
        num_workers: int,
        num_questions: int,
        start_idx: int,
        custom_idx_list: str,
        test_all: bool,
):
    # load the dataset
    dataset = load_dataset("smolagents/browse_comp")
    dataset_list = list(dataset["test"])

    if test_all:
        num_questions = len(dataset_list)
        start_idx = 0
        custom_idx_list = None
    elif num_questions > len(dataset_list):
        num_questions = len(dataset_list) - start_idx
        custom_idx_list = None

    test_sample_ids = list(range(start_idx, start_idx + num_questions))
    test_samples = [
        {"id": f"{DATASET_NAME}_{idx}", **dataset_list[idx]} for idx in test_sample_ids
    ]

    if custom_idx_list:
        logger.info(
            f"Overriding `start_idx` and `num_questions` with customized list of question IDs: {custom_idx_list}"
        )
        test_sample_ids = eval(custom_idx_list)
        test_samples = [
            {"id": f"{DATASET_NAME}_{idx}", **dataset_list[idx]} for idx in test_sample_ids
        ]
        num_questions = len(test_sample_ids)

    # Load the search config and judge config
    config = set_up_config(agent_type, model_name)
    search_config = config["search_config"]
    judge_config = config["judge_config"]
    result_file = config["result_file"]

    logger.info(
        f"\n{'=' * 100}\n"
        f"Starting {DATASET_NAME} Evaluation:\n"
        f"[Search Config]\n"
        f"{search_config.model_dump_json(indent=2)}\n"
        f"[Judge Config]\n"
        f"{judge_config.model_dump_json(indent=2)}\n"
        f"\n{'=' * 100}\n"
    )
    saved_config = {
        "search_config": json.loads(search_config.model_dump_json(indent=2)),
        "judge_config": json.loads(judge_config.model_dump_json(indent=2)),
    }
    with open(search_config.working_directory / "config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    # run the search and evaluation
    load_dotenv()  # load openai, google api, and search api keys
    results = run_search_and_evaluate_multithreaded(
        test_samples=test_samples,
        search_config=search_config,
        judge_config=judge_config,
        num_workers=num_workers,
        result_file=result_file,
    )

    # post summary
    error_ids = []
    grade_counts = {"yes": 0, "no": 0, "ERROR": 0}
    total_token_usage = 0
    valid_count = 0

    for result in results:
        if "error" in result["search_result"]:
            error_ids.append(result["input_sample"]["id"])
            grade_counts["ERROR"] += 1
        elif result["eval_result"]:
            grade = result["eval_result"]["grade"]
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            valid_count += 1
            total_token_usage += result["search_result"]["token_usage"]

    accuracy = (grade_counts["yes"] / valid_count * 100) if valid_count else 0.0
    logger.info(
        f"\n{'=' * 50}\n"
        "Summary:\n"
        f"Yes: {grade_counts['yes']}\n"
        f"No: {grade_counts['no']}\n"
        f"Error: {grade_counts['ERROR']}\n"
        f"Accuracy (Excluding Error Cases): {accuracy:.2f}%\n"
        f"Total token usage: {total_token_usage}\n"
        f"Error IDs: {error_ids}\n"
        f"{'=' * 50}\n"
    )


if __name__ == "__main__":
    main()