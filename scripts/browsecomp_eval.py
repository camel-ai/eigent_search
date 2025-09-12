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
import time
import base64, hashlib
from typing import Any, Dict

from camel.agents import ChatAgent
from camel.logger import get_logger, set_log_file, set_log_level
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
import click
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from eigent_search.baseline import (
    ChainOfThoughtAgent,
    DirectAnswerAgent,
    KnowledgeThenReasoningAgent,
    SimpleResearchAgent,
)
from eigent_search.evaluation import BrowseCompEvaluator
from eigent_search.research import deep_search_agent_factory
from eigent_search.utils import run_agent_with_retry


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

WORKING_DIRECTORY = Path(f"eigent_search_results_{TIMESTAMP}")
os.makedirs(WORKING_DIRECTORY, exist_ok=True)

AGENTS = {
    "simple_research": SimpleResearchAgent,
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
    "deep_search": lambda model: deep_search_agent_factory(model, WORKING_DIRECTORY),
}

MODEL_NAMES = {
    "gpt-4o-mini": ModelType.GPT_4O_MINI,
    "gpt-4.1-mini": ModelType.GPT_4_1_MINI,
    "gpt-oss": "gpt-oss:120b",  # Ollama model for now
}


GRADE_EMOJI_MAP = {
    "yes": "✅",
    "no": "❌",
    "ERROR": "🚫",
}

set_log_file(WORKING_DIRECTORY / "browsecomp_eval.log")
set_log_level(logging.INFO)
logger = get_logger(__name__)


class BrowseCompResponse(BaseModel):

    answer: str = Field(..., description="The answer to the research question.")
    search_results: list[str] = Field(
        ..., description="The search results that lead to the answer."
    )

# decode problem function
def _derive_key(password: str, length: int) -> bytes:
    h = hashlib.sha256()
    h.update(password.encode())
    key = h.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
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


@click.command()
@click.option("--agent_type", "-a", type=click.Choice(AGENTS.keys()), required=True)
@click.option("--model_name", "-m", type=click.Choice(MODEL_NAMES.keys()), default="gpt-4.1-mini")
@click.option("--num_questions", "-n", type=int, default=5)
@click.option("--start_idx", "-s", type=int, default=0, help="Start index for the test samples.")
def main(agent_type: str, model_name: str, num_questions: int, start_idx: int):
    # Log evaluation configuration
    logger.info(
        f"\n{'=' * 100}\n"
        "Starting BrowseComp Evaluation\n"
        f"Agent Type: {agent_type}\n"
        f"Model: {model_name}\n"
        f"Questions: {num_questions}\n"
        f"Start Index: {start_idx}\n"
        f"Output directory: {WORKING_DIRECTORY}\n"
        f"\n{'=' * 100}\n"
    )

    # setup the agent for evaluation
    load_dotenv()  # load the openai key from .env
    # for ollama models, we need to specify the url that hosts the model
    if model_name == "gpt-oss":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type="gpt-oss:120b",
            url="http://129.212.188.6:7861/v1",
            model_config_dict={"temperature": 0.0},
        )
    else:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=MODEL_NAMES[model_name],
            model_config_dict={"temperature": 0.0},
        )
    agent = AGENTS[agent_type](model=model)

    # setup the evaluator; don't use the same model as the agent
    eval_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={"temperature": 0.0},
    )
    eval_agent = ChatAgent(model=eval_model)
    evaluator = BrowseCompEvaluator(eval_agent)

    # load the dataset
    dataset = load_dataset("smolagents/browse_comp")
    test_samples = list(dataset["test"])[start_idx : start_idx + num_questions]

    results = []
    counter = {"yes": 0, "no": 0, "ERROR": 0}
    valid_count = 0  
    valid_yes = 0     
    output_file = WORKING_DIRECTORY / f"browsecomp_eval_agent={agent_type}_model={model_name}.json"

    try:
        for i, example in enumerate(
            tqdm(test_samples, desc="BrowseComp Evaluation", unit="example", leave=True)
        ):
            problem_id = start_idx + i

            canary = example.get("canary")
            enc_problem = example.get("problem")
            enc_answer = example.get("answer")

            # If there is no canary for this example, set grade as "ERROR"
            if not canary:
                grade = "ERROR"
                item = {
                    "id": problem_id,
                    "problem": enc_problem,
                    "ground_truth_answer": enc_answer,
                    "agent_response": {"error": True, "reason": "Missing canary"},
                    "grade": grade,
                    "tool_trajectory": None,
            
                }
                results.append(item)
                counter[grade] += 1
                logger.info(f"[skip] sample {problem_id} missing canary -> ERROR")
                if (i + 1) % 2 == 0 or i == num_questions - 1:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    logger.info(f"Results saved to {output_file} ...")
                continue

            # if decode failed, set grade as "ERROR"
            try:
                problem = _decrypt(enc_problem, canary)
                answer = _decrypt(enc_answer, canary)
            except Exception as e:
                grade = "ERROR"
                item = {
                    "id": problem_id,
                    "problem": enc_problem,
                    "ground_truth_answer": enc_answer,
                    "agent_response": {"error": True, "reason": f"Decrypt failed: {e}"},
                    "grade": grade,
                    "tool_trajectory": None,
                }
                results.append(item)
                counter[grade] += 1
                logger.info(f"[skip] sample {problem_id} decrypt failed -> ERROR")
                if (i + 1) % 2 == 0 or i == num_questions - 1:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    logger.info(f"Results saved to {output_file} ...")
                continue

            
            run_res = run_agent_with_retry(
                agent=agent,
                input_query=problem,
                response_format=BrowseCompResponse,  
                max_retries=5,
            )
            response = run_res["response"]["answer"]
            tool_trajectory = run_res.get("tool_trajectory", None)
            # Normal evaluation
            predicted = response.get("answer") if isinstance(response, dict) else str(response)
            eval_request = evaluator.create_request(problem=problem, answer=answer, prediction=predicted)
            eval_result = evaluator.evaluate(eval_request)
            grade = eval_result.metrics.get("grade", "no")
            grade_with_emoji = f"{GRADE_EMOJI_MAP.get(grade, '❓')} {grade}"
            

            counter[grade] = counter.get(grade, 0) + 1
            if grade in ("yes", "no"):
                valid_count += 1
                if grade == "yes":
                    valid_yes += 1
            current_accuracy = (valid_yes / valid_count * 100) if valid_count else 0.0

            # save with decode problem and ground truth
            item = {
                "id": problem_id,
                "problem": problem,  
                "ground_truth_answer": answer, 
                "agent_response": _json_safe(response),
                "grade": grade,
                "tool_trajectory": _json_safe(tool_trajectory),
            }
            results.append(item)

            logger.info(
                f"\nQuestion: {i + 1} / {num_questions}\n"
                f"Agent: {agent_type}\n"
                f"Grade: {grade_with_emoji}\n"
                f"Running totals: {counter}\n"
                f"Current accuracy (excl. ERROR): {current_accuracy:.2f}%\n"
                f"Result: {json.dumps(item, ensure_ascii=False)[:1200]}...\n"
            )

            
            if (i + 1) % 2 == 0 or i == num_questions - 1:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                logger.info(f"Results saved to {output_file} ...")

            agent.reset()
            time.sleep(20)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise e

    finally:
        final_accuracy = (valid_yes / valid_count * 100) if valid_count else 0.0
        logger.info(
            f"[{agent_type}] Final Results - Total: {len(results)}, "
            f"yes: {counter['yes']}, no: {counter['no']}, ERROR: {counter['ERROR']}, "
            f"Accuracy (excl. ERROR): {final_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()