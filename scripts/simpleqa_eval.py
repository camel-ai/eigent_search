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
from eigent_search.evaluation import SimpleQAEvaluator
from eigent_search.research import deep_search_agent_factory
from eigent_search.utils import run_agent_with_retry


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

WORKING_DIRECTORY = Path(
    f"eigent_search_results_{TIMESTAMP}",
)
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
    "CORRECT": "✅",
    "INCORRECT": "❌",
    "NOT_ATTEMPTED": "⚠️",
    "ERROR": "🚫",
}

set_log_file(WORKING_DIRECTORY / "simpleqa_eval.log")
set_log_level(logging.INFO)
logger = get_logger(__name__)


class SimpleQAResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    search_results: list[str] = Field(
        ..., description="The search results that lead to the answer."
    )


@click.command()
@click.option("--agent_type", "-a", type=click.Choice(AGENTS.keys()), required=True)
@click.option(
    "--model_name", "-m", type=click.Choice(MODEL_NAMES.keys()), default="gpt-4.1-mini"
)
@click.option("--num_questions", "-n", type=int, default=5)
@click.option(
    "--start_idx", "-s", type=int, default=0, help="Start index for the test samples."
)
def main(agent_type: str, model_name: str, num_questions: int, start_idx: int):
    # Log evaluation configuration
    logger.info(
        f"\n{'=' * 100}\n"
        "Starting SimpleQA Evaluation\n"
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
    evaluator = SimpleQAEvaluator(eval_agent)

    # load the dataset
    dataset = load_dataset("basicv8vc/SimpleQA")

    test_samples = list(dataset["test"])[start_idx : start_idx + num_questions]

    scores = []
    results = []
    counter = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
    output_file = (
        WORKING_DIRECTORY / f"simpleqa_eval_agent={agent_type}_model={model_name}.json"
    )
    total_token_usage = 0
    try:
        for i, example in enumerate(
            tqdm(test_samples, desc="SimpleQA Evaluation", unit="example", leave=True)
        ):
            # Create a unique ID for this problem (dataset index)
            problem_id = start_idx + i

            # Run agent with retry logic
            result = run_agent_with_retry(
                agent=agent,
                input_query=example["problem"],
                response_format=SimpleQAResponse,
                max_retries=5,
                timeout_minutes=5,
            )
            response = result["response"]
            token_usage = result.get("token_usage", 0)
            total_token_usage += token_usage
            logger.info("Total token usage so far: %d", total_token_usage)

            # Handle evaluation - check if response indicates error
            if response.get("error", False):
                # Create a dummy evaluation result for errors
                eval_result = type(
                    "obj",
                    (object,),
                    {"score": -1.0, "metrics": {"grade": "ERROR"}},
                )()
                scores.append(-1.0)
            else:
                # Normal evaluation
                eval_request = evaluator.create_request(
                    problem=example["problem"],
                    answer=example["answer"],
                    prediction=response["answer"],
                )
                eval_result = evaluator.evaluate(eval_request)
                scores.append(eval_result.score)

            # process the evaluation result for logging and saving
            grade = eval_result.metrics["grade"]
            grade_with_emoji = f"{GRADE_EMOJI_MAP.get(grade, '❓')} {grade}"

            result = {
                "id": problem_id,  # Index in the original dataset
                "problem": example["problem"],
                "ground_truth_answer": example["answer"],
                "agent_response": response,
                "grade": grade,
                "metadata": example.get(
                    "metadata", {}
                ),  # Include metadata if available
                "token_usage": token_usage,
                "total_token_usage": total_token_usage,
            }
            results.append(result)
            counter[eval_result.metrics["grade"]] += 1
            current_accuracy = counter["CORRECT"] / (i + 1) * 100

            logger.info(
                f"\nQuestion: {i + 1} / {num_questions}\n"
                f"Agent: {agent_type}\n"
                f"Grade: {grade_with_emoji}\n"
                f"Running totals: {counter}\n"
                f"Current accuracy: {current_accuracy:.2f}%\n"
                f"Result: {json.dumps(result, indent=2)}\n"
            )

            # if agent_type == "research":
            #     logger.info(
            #         f"[{agent_type}] Number of searches: {agent.current_query_toolkit.search_counter}"
            #     )
            #     logger.info(
            #         f"[{agent_type}] Process Graph:\n{agent.current_query_toolkit.trace_graph.render_trace_graph()}"
            #     )

            # Save results periodically
            if (i + 1) % 2 == 0 or i == num_questions - 1:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                logger.info(f"Results saved to {output_file} ...")

            # # Clear browser metrics for next problem
            # if agent_type == "eigent_search" and hasattr(agent, "web_toolkit"):
            #     agent.web_toolkit.clear_metrics()

            agent.reset()
            # time.sleep(20)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        # Todo: Should we also avoid raising error here, to always complete the evaluation?
        raise e

    finally:
        # Always print comprehensive summary, even if evaluation failed
        final_accuracy = sum(scores) / len(scores) * 100 if scores else 0
        logger.info(
            f"[{agent_type}] Final Results - Total: {len(results)}, Correct: {counter['CORRECT']}, "
            f"Incorrect: {counter['INCORRECT']}, Not Attempted: {counter['NOT_ATTEMPTED']}, "
            f"Accuracy: {final_accuracy:.2f}%"
        )
        logger.info("total token usage: %d", total_token_usage)


if __name__ == "__main__":
    main()
