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

from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Type

from camel.logger import get_logger, set_log_file, set_log_level
import click
from pydantic import BaseModel, Field

from eigent_search.config import LLMasJudgeConfig, SearchConfig
from eigent_search.evaluation import MusiQueEvaluator
from eigent_search.evaluation import utils as run_apated
from eigent_search.evaluation.base import BaseEvaluator
from eigent_search.evaluation.utils import (
    AGENT_TYPES,
    MODEL_CONFIGS,
    run_search_and_evaluate_multithreaded,
)
from eigent_search.orchestrator import SearchOrchestrator

set_log_level(logging.INFO)
logger = get_logger(__name__)

# Define benchmark-specific constants
BENCHMARK_NAME = "musique"
FORMAT_STRING = "In the final response format 'answer', return ONLY the final short answer (entity/number/date/short noun phrase). No sentences, no prefixes, no quotes, no extra text. "
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


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
            input_query=FORMAT_STRING + test_sample["question"],
            query_id=test_sample["id"],
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
    evaluator = MusiQueEvaluator()
    eval_request = evaluator.create_request(
        qid=test_sample["id"],
        reference_answer=test_sample["answer"],
        reference_answer_aliases=test_sample.get("answer_aliases", []),
        answerable=test_sample.get("answerable", True),
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
            "metrics": eval_result.metrics,
        },
    }


class MusiQueResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    evidence: list[str] = Field(
        ...,
        description="The evidence from the search results that lead to the answer.",
    )


@click.command()
@click.option(
    "--agent_type", "-a", type=click.Choice(AGENT_TYPES.keys()), required=True
)
@click.option(
    "--model_name",
    "-m",
    type=click.Choice(MODEL_CONFIGS.keys()),
    default="azure-gpt-4.1-mini",
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
@click.option(
    "--resume-from",
    type=str,
    help="Resume from an existing working directory",
    default=None,
)
def main(
    agent_type: str,
    model_name: str,
    num_workers: int,
    num_questions: int,
    start_idx: int,
    custom_idx_list: list[int],
    test_all: bool,
    resume_from: str | None,
):
    # Set the working directory
    evaluated_question_ids = set()
    existing_results = []
    if resume_from and os.path.exists(resume_from):
        WORKING_DIRECTORY = Path(resume_from)
        logger.info(f"Resuming from existing working directory: {WORKING_DIRECTORY}")
        try:
            with open(WORKING_DIRECTORY / "results.jsonl", "r") as f:
                for line in f:
                    result = json.loads(line)
                    if "error" in result["search_result"]:
                        continue
                    existing_results.append(result)
                    evaluated_question_ids.add(result["input_sample"]["id"])
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            raise e
    else:
        WORKING_DIRECTORY = (
            Path(os.getcwd()) / "results" / f"{BENCHMARK_NAME}_eval_{TIMESTAMP}"
        )
        WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Starting new evaluation in working directory: {WORKING_DIRECTORY}"
        )

    # Set up log file
    set_log_file(WORKING_DIRECTORY / f"{BENCHMARK_NAME}_eval.log")

    # load the dataset
    dataset = list(MusiQueEvaluator.load_dataset())
    if test_all:
        num_questions = len(dataset)
        start_idx = 0
        custom_idx_list = None
    elif num_questions > len(dataset):
        num_questions = len(dataset) - start_idx
        custom_idx_list = None
    test_samples = dataset[start_idx : start_idx + num_questions]
    test_sample_ids = list(range(start_idx, start_idx + num_questions))
    test_samples = [
        {"id": f"{BENCHMARK_NAME}_{idx}", **dataset[idx]} for idx in test_sample_ids
    ]
    test_samples = [
        sample for sample in test_samples if sample["id"] not in evaluated_question_ids
    ]
    if custom_idx_list:
        logger.info(
            f"Overriding `start_idx` and `num_questions` with customized list of question IDs: {custom_idx_list}"
        )
        test_sample_ids = eval(custom_idx_list)
        test_samples = [
            {"id": f"{BENCHMARK_NAME}_{idx}", **dataset[idx]} for idx in test_sample_ids
        ]
        num_questions = len(test_sample_ids)

    # Log the search and judge configs and save a copy to working directory for reference
    # The configs will be re-created for each thread to avoid race conditions
    logger.info(
        f"\n{'=' * 100}\n"
        f"Starting {BENCHMARK_NAME} Evaluation:\n"
        f"[Agent Type]: {agent_type}\n"
        f"[Model]: {model_name}\n"
        f"[Judge Model]: gpt-4.1-2025-04-14"  # constant, not changing
        f"\n{'=' * 100}\n"
    )

    # run the search and evaluation
    run_apated.run_search_and_evaluate = run_search_and_evaluate
    results = run_search_and_evaluate_multithreaded(
        test_samples=test_samples,
        working_directory=WORKING_DIRECTORY,
        benchmark_name=BENCHMARK_NAME,
        agent_type=agent_type,
        model_name=model_name,
        evaluator_class=MusiQueEvaluator,
        response_format=MusiQueResponse,
        num_workers=num_workers,
        existing_results=existing_results,
    )
    results = existing_results + results

    # post summary
    error_ids = []
    per_f1_list, per_em_list, per_acc_list = [], [], []
    total_token_usage = 0
    for result in results:
        if "error" in result["search_result"]:
            error_ids.append(result["input_sample"]["id"])
            continue  # skip error cases
        em, f1, acc = (
            result["eval_result"]["metrics"]["answer_em"],
            result["eval_result"]["metrics"]["answer_f1"],
            result["eval_result"]["metrics"]["answer_acc"],
        )
        per_em_list.append(em)
        per_f1_list.append(f1)
        per_acc_list.append(acc)
        total_token_usage += result["search_result"]["token_usage"]

    em = round(sum(per_em_list) / len(per_em_list), 3) if per_em_list else 0.0
    f1_score = round(sum(per_f1_list) / len(per_f1_list), 3) if per_f1_list else 0.0
    acc = round(sum(per_acc_list) / len(per_acc_list), 3) if per_acc_list else 0.0

    logger.info(
        f"\n{'=' * 50}\n"
        f"Processed {len(results)} questions, {len(error_ids)} of which are error cases with IDs: {error_ids}\n"
        f"Exact_Match: {em * 100:.2f}% (n={len(per_em_list)})\n"
        f"F1 Score: {f1_score * 100:.2f}%\n"
        f"Accuracy: {acc * 100:.2f}%\n"
        f"Total token usage: {total_token_usage}\n"
        f"{'=' * 50}\n"
    )


if __name__ == "__main__":
    main()
