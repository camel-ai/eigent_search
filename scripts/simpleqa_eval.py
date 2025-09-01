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

import os
import time
import click
import json
from datetime import datetime
from dotenv import load_dotenv
from tqdm.auto import tqdm
from datasets import load_dataset
import logging

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

from eigent_search.evaluation import SimpleQAEvaluator
from eigent_search.baseline import (
    DirectAnswerAgent,
    ChainOfThoughtAgent,
    KnowledgeThenReasoningAgent,
    SimpleResearchAgent,
)
from eigent_search.research import deep_search_agent_factory
from eigent_search.utils import run_agent_with_retry

AGENTS = {
    "simple_research": SimpleResearchAgent,
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
    "deep_search": deep_search_agent_factory,
}

MODEL_NAMES = {
    "gpt-4o-mini": ModelType.GPT_4O_MINI,
    "gpt-4.1-mini": ModelType.GPT_4_1_MINI,
    "gpt-oss": "gpt-oss:120b",  # Ollama model for now
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

WORKING_DIRECTORY = os.path.join(
    os.getcwd(),
    "results",
    f"eigent_search_{TIMESTAMP}",
)
os.makedirs(WORKING_DIRECTORY, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(WORKING_DIRECTORY / "simpleqa_eval.log"),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# Set log level of camel.agents.chat_agent to WARNING to reduce noise
# logging.getLogger("camel.agents.chat_agent").setLevel(logging.WARNING)
# logging.getLogger("camel").setLevel(logging.WARNING)
# logging.getLogger("librarian.research.browser_wrapper").setLevel(logging.WARNING)


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
        "Starting SimpleQA Evaluation"
        f"Agent Type: {agent_type}"
        f"Model: {model_name}"
        f"Questions: {num_questions}"
        f"Start Index: {start_idx}"
        f"Output directory: {WORKING_DIRECTORY}"
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
        WORKING_DIRECTORY
        / f"{agent_type}_simpleqa_from={start_idx}_to={start_idx + num_questions}_{TIMESTAMP}.json"
    )

    try:
        for i, example in enumerate(
            tqdm(test_samples, desc="SimpleQA Evaluation", unit="example", leave=True)
        ):
            # Create a unique ID for this problem (dataset index)
            problem_id = start_idx + i

            # Run agent with retry logic
            response = run_agent_with_retry(
                agent=agent,
                input_query=example["problem"],
                max_retries=5,
            )

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

            # Add emoji to grade for visual clarity
            grade_emoji_map = {
                "CORRECT": "✅",
                "INCORRECT": "❌",
                "NOT_ATTEMPTED": "⚠️",
                "ERROR": "🚫",
            }
            grade = eval_result.metrics["grade"]
            grade_with_emoji = f"{grade_emoji_map.get(grade, '❓')} {grade}"

            result = {
                "dataset_index": problem_id,  # Index in the original dataset
                "problem": example["problem"],
                "answer": example["answer"],
                "response": response,
                "grade_emoji": grade_with_emoji,
                "grade": grade,
                "metadata": example.get(
                    "metadata", {}
                ),  # Include metadata if available
            }
            results.append(result)
            counter[eval_result.metrics["grade"]] += 1
            current_accuracy = counter["CORRECT"] / (i + 1) * 100

            logger.info(
                f"[{agent_type}] Index: {problem_id} ({i + 1}/{num_questions}) - Grade: {grade_with_emoji} - Running totals: {counter} - Accuracy: {current_accuracy:.2f}%"
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
                tqdm.write(f"Results saved to {output_file}")

            # # Clear browser metrics for next problem
            # if agent_type == "eigent_search" and hasattr(agent, "web_toolkit"):
            #     agent.web_toolkit.clear_metrics()

            agent.reset()
            time.sleep(20)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise e

    finally:
        # Always print comprehensive summary, even if evaluation failed
        final_accuracy = sum(scores) / len(scores) * 100 if scores else 0
        logger.info(
            f"[{agent_type}] Final Results - Total: {len(results)}, Correct: {counter['CORRECT']}, "
            f"Incorrect: {counter['INCORRECT']}, Not Attempted: {counter['NOT_ATTEMPTED']}, "
            f"Accuracy: {final_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
