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
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Literal

from camel.logger import get_logger, set_log_file, set_log_level
import click
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
from eigent_search.evaluation.musique.musique_class import MusiQueEvaluator

set_log_level(logging.INFO)
logger = get_logger(__name__)


AGENT_TYPES = {
    "eigent_search": SearchAgentType.EIGENT_SEARCH,
    "eigent_search_q+": SearchAgentType.EIGENT_SEARCH_Q_PLUS,
    "search_only": SearchAgentType.SEARCH_ONLY,
}


MODEL_CONFIGS = {
    "gpt-5-mini": BackendModelConfig.GPT_5_MINI,
    "gpt-4.1": BackendModelConfig.GPT_4_1,
    "gpt-4.1-mini": BackendModelConfig.GPT_4_1_MINI,
    "gpt-4o": BackendModelConfig.GPT_4O,
    "gpt-4o-mini": BackendModelConfig.GPT_4O_MINI,
    "gpt-oss": BackendModelConfig.GPT_OSS,
}

# Define benchmark-specific constants
DATASET_NAME = "musique"
FORMAT_STRING = f"In the final response format \"answer\", return ONLY the final short answer (entity/number/date/short noun phrase). No sentences, no prefixes, no quotes, no extra text.\n"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WORKING_DIRECTORY = Path(os.getcwd()) / "results" / f"{DATASET_NAME}_eval_{TIMESTAMP}"
WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)



def set_up_config(
    agent_type: Literal[AGENT_TYPES.keys()], model_name: Literal[MODEL_CONFIGS.keys()]
) -> dict:
    """Set up the search config."""

    class MusiQueResponse(BaseModel):
        answer: str = Field(..., description="The answer to the research question.")
        evidence: list[str] = Field(
            ...,
            description="The evidence from the search results that lead to the answer.",
        )

    config = {
        "search_config": SearchConfig(
            working_directory=WORKING_DIRECTORY,
            **MODEL_CONFIGS[model_name].value,
            agent_type=AGENT_TYPES[agent_type],
            response_format=MusiQueResponse,
        ),
        "judge_config": "traditional_metric: F1 and EM"
    }

    set_log_file(config["search_config"].working_directory / f"{DATASET_NAME}_eval.log")

    return config


def run_search_and_evaluate(
    test_sample: dict, search_config: SearchConfig
) -> dict:
    """Run the search and evaluation for a single test sample."""

    # run the search
    search_orchestrator = SearchOrchestrator(search_config)
    search_result = search_orchestrator.run_agent(
        search_orchestrator.create_search_request(
            input_query=FORMAT_STRING + test_sample["question"], query_id=test_sample["id"]
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


def run_search_and_evaluate_multithreaded(
    test_samples: list[dict],
    agent_type: Literal[AGENT_TYPES.keys()],
    model_name: Literal[MODEL_CONFIGS.keys()],
    result_file: Path,
    num_workers: int,
) -> list[dict]:
    """Run the search and evaluation for a list of test samples in parallel."""

    process_bar = tqdm(total=len(test_samples), desc=f"{DATASET_NAME} Evaluation")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for sample in test_samples:
            config = set_up_config(agent_type, model_name)
            search_config = config["search_config"]
            judge_config = config["judge_config"]

            futures.append(
                executor.submit(
                    run_search_and_evaluate, sample, search_config
                )
            )
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
    custom_idx_list: list[int],
    test_all: bool,
):
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
        {"id": f"{DATASET_NAME}_{idx}", **dataset[idx]} for idx in test_sample_ids
    ]
    if custom_idx_list:
        logger.info(
            f"Overriding `start_idx` and `num_questions` with customized list of question IDs: {custom_idx_list}"
        )
        test_sample_ids = eval(custom_idx_list)
        test_samples = [
            {"id": f"{DATASET_NAME}_{idx}", **dataset[idx]} for idx in test_sample_ids
        ]
        num_questions = len(test_sample_ids)

    # Log the search and judge configs and save a copy to working directory for reference
    # The configs will be re-created for each thread to avoid race conditions
    config = set_up_config(agent_type, model_name)
    search_config = config["search_config"]
    judge_config = config["judge_config"]
    logger.info(
        f"\n{'=' * 100}\n"
        f"Starting {DATASET_NAME} Evaluation:\n"
        f"[Search Config]\n"
        f"{search_config.model_dump_json(indent=2)}\n"
        f"[Judge Config]\n"
        f"{judge_config}\n"
        f"\n{'=' * 100}\n"
    )
    saved_config = {
        "search_config": json.loads(search_config.model_dump_json(indent=2)),
        "judge_config": judge_config,
    }
    with open(WORKING_DIRECTORY / "config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    # run the search and evaluation
    load_dotenv()  # load openai, google api, and search api keys
    results = run_search_and_evaluate_multithreaded(
        test_samples=test_samples,
        agent_type=agent_type,
        model_name=model_name,
        result_file=WORKING_DIRECTORY / "results.json",
        num_workers=num_workers,
    )

    # post summary
    error_ids = []
    per_f1_list, per_em_list, per_acc_list = [], [], []
    total_token_usage = 0
    for result in results:
        if "error" in result["search_result"]:
            error_ids.append(result["search_result"]["query_id"])
            continue  # skip error cases
        em, f1, acc = result["eval_result"]["metrics"]["answer_em"], result["eval_result"]["metrics"]["answer_f1"], result["eval_result"]["metrics"]["answer_acc"] 
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
