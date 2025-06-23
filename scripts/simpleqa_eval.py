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

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

from librarian.evaluation import SimpleQAEvaluator
from librarian.baseline import (
    DirectAnswerAgent,
    ChainOfThoughtAgent,
    KnowledgeThenReasoningAgent,
)


AGENTS = {
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
}


@click.command()
@click.option("--agent_type", "-a", type=click.Choice(AGENTS.keys()), required=True)
@click.option("--num_questions", "-n", type=int, default=5)
def main(agent_type: str, num_questions: int):
    # setup the agent for evaluation
    load_dotenv()  # load the openai key from .env
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={"temperature": 0.5},
    )
    agent = AGENTS[agent_type](model=model)

    # setup the evaluator
    eval_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={"temperature": 0.0},
    )
    eval_agent = ChatAgent(model=eval_model)
    evaluator = SimpleQAEvaluator(eval_agent)

    # load the dataset
    dataset = load_dataset("basicv8vc/SimpleQA")
    test_samples = list(dataset["test"])[:num_questions]

    scores = []
    results = []
    counter = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
    output_file = f"results/{agent_type}_simpleqa_{num_questions}.json"
    for i, example in enumerate(
        tqdm(test_samples, desc="SimpleQA Evaluation", unit="example", leave=True)
    ):
        agent.reset()
        response = agent.step(f"Question: {example['problem']}")
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

        # save results every 50 examples or at the end
        if (i + 1) % 50 == 0 or i == num_questions - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")

    tqdm.write(
        f"[{agent_type}] Accuracy (n={num_questions}): {sum(scores) / len(scores)}"
    )


if __name__ == "__main__":
    main()
