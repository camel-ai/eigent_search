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
    DirectAnswerAgentWithGoogleSearch
)
from librarian.research import ResearchAgent

AGENTS = {
    "research": ResearchAgent,
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
    "direct_w_google": DirectAnswerAgentWithGoogleSearch,
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
@click.option("--num_questions", "-n", type=int, default=5)
@click.option("--start_idx", "-s", type=int, default=0, help="Start index for the test samples.")
@click.option("--browsing", "-b", is_flag=True, default=False, help="Enable browsing capabilities for research agent.")
@click.option("--retry_failed", "-r", type=str, default=None, help="Path to existing results file to retry failed questions.")
def main(agent_type: str, num_questions: int, start_idx: int, browsing: bool, retry_failed: str):
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
    
    if retry_failed:
        # Load existing results and filter for failed questions
        with open(retry_failed, 'r') as f:
            existing_results = json.load(f)
        
        # Find failed questions (INCORRECT or NOT_ATTEMPTED)
        failed_indices = []
        failed_problems = []
        
        for result in existing_results:
            if result['grade'] in ['INCORRECT', 'NOT_ATTEMPTED']:
                # Find the index in the original dataset
                for idx, sample in enumerate(dataset["test"]):
                    if sample['problem'] == result['problem']:
                        failed_indices.append(idx)
                        failed_problems.append(sample)
                        break
        
        test_samples = failed_problems[:num_questions]
        logger.info(f"Retrying {len(test_samples)} failed questions from {retry_failed}")
        logger.info(f"Failed question indices: {failed_indices[:len(test_samples)]}")
        
        # Print summary of failed questions being retried
        failed_grades = [r['grade'] for r in existing_results if r['grade'] in ['INCORRECT', 'NOT_ATTEMPTED']]
        from collections import Counter
        grade_count = Counter(failed_grades)
        logger.info(f"Failed question breakdown: {dict(grade_count)}")
        
        if len(test_samples) > 0:
            logger.info(f"Sample failed questions:")
            for i, sample in enumerate(test_samples[:3]):  # Show first 3
                logger.info(f"  {i+1}. {sample['problem'][:100]}...")
        
        # Update output filename to indicate retry
        import os
        base_name = os.path.splitext(os.path.basename(retry_failed))[0]
        browsing_suffix = "_browsing" if browsing and agent_type == "research" else ""
        output_file = f"results/{base_name}_retry_{agent_type}{browsing_suffix}.json"
    else:
        test_samples = list(dataset["test"])[start_idx: start_idx+num_questions]
        # Include browsing in filename if enabled
        browsing_suffix = "_browsing" if browsing and agent_type == "research" else ""
        output_file = f"results/{agent_type}{browsing_suffix}_simpleqa_from={start_idx}_to={start_idx+num_questions}.json"

    scores = []
    results = []
    counter = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
    for i, example in enumerate(
        tqdm(test_samples, desc="SimpleQA Evaluation", unit="example", leave=True)
    ):
        
        # Prepare step arguments based on agent type
        step_kwargs = {}
        if agent_type == "research":
            step_kwargs["browsing"] = browsing
            
        response = agent.step(f"{example['problem']}", **step_kwargs)
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
