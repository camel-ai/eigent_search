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

import click
import json
from dotenv import load_dotenv
from tqdm.auto import tqdm
from datasets import load_dataset
import logging

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

from librarian.evaluation import SimpleQAEvaluator
from librarian.baseline import (
    DirectAnswerAgent,
    ChainOfThoughtAgent,
    KnowledgeThenReasoningAgent,
    SimpleResearchAgent,
)
from librarian.research import ResearchAgent

AGENTS = {
    "research": ResearchAgent,
    "simple_research": SimpleResearchAgent,
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
}

MODEL_NAMES = {
    "gpt-4o-mini": ModelType.GPT_4O_MINI,
    "gpt-4.1-mini": ModelType.GPT_4_1_MINI,
    "gpt-oss": "gpt-oss:120b",  # Ollama model for now
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("results/simpleqa_eval.log"),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# Set log level of camel.agents.chat_agent to WARNING
# logging.getLogger("camel.agents.chat_agent").setLevel(logging.WARNING)


@click.command()
@click.option("--agent_type", "-a", type=click.Choice(AGENTS.keys()), required=True)
@click.option(
    "--model_name", "-m", type=click.Choice(MODEL_NAMES.keys()), default="gpt-4.1-mini"
)
@click.option("--num_questions", "-n", type=int, default=5)
@click.option(
    "--start_idx", "-s", type=int, default=0, help="Start index for the test samples."
)
@click.option(
    "--save_graphs", "-g", is_flag=True, help="Boolean flag for whether to save graphs."
)
def main(
    agent_type: str,
    model_name: str,
    num_questions: int,
    start_idx: int,
    save_graphs: bool,
):
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
    output_file = f"results/{agent_type}_simpleqa_from={start_idx}_to={start_idx + num_questions}.json"
    for i, example in enumerate(
        tqdm(test_samples, desc="SimpleQA Evaluation", unit="example", leave=True)
    ):
        response = agent.step(f"{example['problem']}")
        response = eval(response.msgs[0].content)
        eval_request = evaluator.create_request(
            problem=example["problem"],
            answer=example["answer"],
            prediction=response["answer"],
        )
        eval_result = evaluator.evaluate(eval_request)
        scores.append(eval_result.score)
        results.append(
            {
                "problem": example["problem"],
                "answer": example["answer"],
                "response": response,
                "grade": eval_result.metrics["grade"],
            }
        )
        counter[eval_result.metrics["grade"]] += 1
        tqdm.write(f"[{agent_type}] {counter}")

        if agent_type == "research":
            logger.info(
                f"[{agent_type}] Number of searches: {agent.current_query_toolkit.search_counter}"
            )
            logger.info(
                f"[{agent_type}] Process Graph:\n{agent.current_query_toolkit.trace_graph.render_trace_graph()}"
            )

        if save_graphs and agent_type == "research":
            fig_path = f"results/graphs/{agent_type}_simpleqa_graph={i}.graphml"
            agent.current_query_toolkit.trace_graph.save_graph(fig_path)
            logger.info(f"Process graph figure saved to {fig_path}")

        # save results every 50 examples or at the end
        if (i + 1) % 50 == 0 or i == num_questions - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            tqdm.write(f"Results saved to {output_file}")

        agent.reset()

    tqdm.write(
        f"[{agent_type}] Accuracy (n={num_questions}): {sum(scores) / len(scores)}"
    )


if __name__ == "__main__":
    main()
