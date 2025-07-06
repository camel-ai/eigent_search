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
from librarian.research import ResearchToolkit
from dev.boerz.researcher_instance import LeadResearcher

import sys
import os
import logging
from datetime import datetime

logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create timestamp-based log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"research_experiment_{timestamp}.log"
log_filepath = os.path.join(logs_dir, log_filename)
# Configure logging to display to stdout and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath, mode='w', encoding='utf-8', delay=False)
    ],
    force=True  # This ensures the configuration is applied
)

# Ensure the root logger level is set correctly
logging.getLogger().setLevel(logging.WARNING)

# Set specific logger levels
logging.getLogger('dev.boerz.researcher_instance').setLevel(logging.INFO)
logging.getLogger('__main__').setLevel(logging.INFO)

# Create logger for this file
logger = logging.getLogger(__name__)

# Test log to verify logging is working
logger.info(f"Logger initialized for module: {__name__}")
logger.info(f"🚀 Starting new research experiment - Log file: {log_filename}")

# Force flush to ensure test messages are written
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.flush()

def main(num_questions: int):
    agent_type = "lead_researcher"
    # setup the agent for evaluation
    load_dotenv()  # load the openai key from .env
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={"temperature": 0.5},
    )
    

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
        toolkit = ResearchToolkit()
        lead_researcher = LeadResearcher(
            system_message="You are a lead research agent. Plan and coordinate research tasks effectively. You have access to Google search and Wikipedia search tools to gather information. Use these tools to find accurate, up-to-date information for your research tasks.",
            model=model,  # Dummy model for MVP
            research_toolkit=toolkit
        )

        response = lead_researcher.research(f"Question: {example['problem']}")
        
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
        if (i + 1) % 1 == 0 or i == num_questions - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")

    tqdm.write(
        f"{agent_type} Accuracy (n={num_questions}): {sum(scores) / len(scores)}"
    )


if __name__ == "__main__":
    main(50)