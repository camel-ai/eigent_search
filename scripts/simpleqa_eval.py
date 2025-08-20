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

import time
import click
import json
import asyncio
import hashlib
import sys
from datetime import datetime
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
from librarian import EigentSearchAgent

AGENTS = {
    "research": ResearchAgent,
    "simple_research": SimpleResearchAgent,
    "eigent_search": EigentSearchAgent,
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
}

MODEL_NAMES = {
    "gpt-4o-mini": ModelType.GPT_4O_MINI,
    "gpt-4.1-mini": ModelType.GPT_4_1_MINI,
    "gpt-oss": "gpt-oss:120b",  # Ollama model for now
}

# Create timestamp for unique log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"results/simpleqa_eval_{timestamp}.log"),  # Unique log file per run
        logging.FileHandler("results/simpleqa_eval.log", mode='a'),  # Cumulative log file
        logging.StreamHandler(),  # Console output
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# Set log level of camel.agents.chat_agent to WARNING to reduce noise
# logging.getLogger("camel.agents.chat_agent").setLevel(logging.WARNING)
# logging.getLogger("camel").setLevel(logging.WARNING)
# logging.getLogger("librarian.research.browser_wrapper").setLevel(logging.WARNING)


def generate_problem_hash(problem: str, answer: str) -> str:
    """Generate a unique hash ID for a problem-answer pair.
    
    Args:
        problem: The problem text
        answer: The answer text
        
    Returns:
        An 8-character hash ID
    """
    content = f"{problem}|{answer}"
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    # Return first 8 characters of the hex digest for a shorter ID
    return hash_obj.hexdigest()[:8]


def run_agent_with_retry(agent, problem: str, agent_type: str, hash_id: str, max_retries: int = 5) -> dict:
    """Run agent.step with exponential retry logic.
    
    Args:
        agent: The agent to run
        problem: The problem text
        agent_type: Type of agent for logging
        hash_id: Hash ID for logging
        max_retries: Maximum number of retry attempts (default 5)
        
    Returns:
        Parsed response dict from agent
        
    Raises:
        Exception: If all retries fail
    """
    retry_delay_minutes = 1  # Start with 1 minute
    
    for attempt in range(max_retries):
        try:
            response = agent.step(problem)
            return eval(response.msgs[0].content)
        except Exception as e:
            if attempt < max_retries - 1:
                retry_delay_seconds = retry_delay_minutes * 60
                logger.warning(f"[{agent_type}] Hash: {hash_id} - Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {retry_delay_minutes} minute(s)...")
                time.sleep(retry_delay_seconds)
                retry_delay_minutes = min(retry_delay_minutes * 2, 10)  # Exponential backoff: 1, 2, 4, 8, capped at 10
            else:
                logger.error(f"[{agent_type}] Hash: {hash_id} - All {max_retries} attempts failed: {str(e)}")
                logger.error(f"Raising exception to exit evaluation.")
                raise e


def print_evaluation_summary(results: list, counter: dict, agent_type: str, num_questions: int):
    """Log comprehensive evaluation summary with browser metrics.
    
    Args:
        results: List of evaluation results
        counter: Grade counter dictionary
        agent_type: Type of agent used
        num_questions: Total number of questions evaluated
    """
    logger.info("\n" + "="*70)
    logger.info("🔍 EVALUATION SUMMARY")
    logger.info("="*70)
    
    # Basic stats
    final_accuracy = counter["CORRECT"] / num_questions * 100 if num_questions > 0 else 0
    logger.info(f"📊 Agent Type: {agent_type}")
    logger.info(f"📈 Total Questions: {num_questions}")
    logger.info(f"✅ Correct: {counter['CORRECT']}")
    logger.info(f"❌ Incorrect: {counter['INCORRECT']}")
    logger.info(f"⚠️  Not Attempted: {counter['NOT_ATTEMPTED']}")
    logger.info(f"🎯 Final Accuracy: {final_accuracy:.2f}%")
    
    # Browser metrics analysis (if available)
    if results and any('browser_metrics' in r for r in results):
        logger.info("\n🌐 BROWSER USAGE ANALYSIS")
        logger.info("-" * 40)
        
        total_browser_calls = 0
        function_counts = {}
        successful_problems = 0
        
        for result in results:
            browser_metrics = result.get('browser_metrics', [])
            if browser_metrics:
                successful_problems += 1
                total_browser_calls += len(browser_metrics)
                
                for metric in browser_metrics:
                    func = metric.get('function', 'unknown')
                    function_counts[func] = function_counts.get(func, 0) + 1
        
        avg_calls = total_browser_calls / successful_problems if successful_problems > 0 else 0
        
        logger.info(f"🔧 Total Browser Calls: {total_browser_calls}")
        logger.info(f"📊 Average Calls per Problem: {avg_calls:.2f}")
        logger.info(f"✅ Problems with Browser Activity: {successful_problems}/{num_questions}")
        
        if function_counts:
            logger.info("\n🔧 Function Usage Breakdown:")
            sorted_functions = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)
            for func, count in sorted_functions:
                avg_per_problem = count / successful_problems if successful_problems > 0 else 0
                logger.info(f"   • {func}: {count} calls ({avg_per_problem:.1f} avg)")
    
    # Performance by grade
    logger.info(f"\n📋 DETAILED BREAKDOWN")
    logger.info("-" * 40)
    for result in results:
        grade_emoji = result.get('grade_emoji', result.get('grade', 'Unknown'))
        hash_id = result.get('hash_id', 'N/A')
        browser_calls = len(result.get('browser_metrics', []))
        logger.info(f"  {grade_emoji} | Hash: {hash_id} | Browser calls: {browser_calls}")
    
    logger.info("="*70)


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
    logger.info("="*60)
    logger.info(f"Starting SimpleQA Evaluation")
    logger.info(f"Agent Type: {agent_type}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Questions: {num_questions}")
    logger.info(f"Start Index: {start_idx}")
    logger.info(f"Log File: results/simpleqa_eval_{timestamp}.log")
    logger.info("="*60)
    
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
    output_file = f"results/{agent_type}_simpleqa_from={start_idx}_to={start_idx + num_questions}_{timestamp}.json"
    
    try:
        for i, example in enumerate(
            tqdm(test_samples, desc="SimpleQA Evaluation", unit="example", leave=True)
        ):
            # Create a unique ID for this problem (dataset index)
            problem_id = start_idx + i
            
            # Generate unique hash ID for this problem
            hash_id = generate_problem_hash(example["problem"], example["answer"])
            
            # Run agent with retry logic 
            response = run_agent_with_retry(
                agent=agent,
                problem=example["problem"],
                agent_type=agent_type,
                hash_id=hash_id
            )
            
            # Handle evaluation - check if response indicates error
            if response.get("error", False):
                # Create a dummy evaluation result for errors
                eval_result = type('obj', (object,), {
                    'score': 0,
                    'metrics': {'grade': 'NOT_ATTEMPTED'}
                })()
                scores.append(0)
            else:
                # Normal evaluation
                eval_request = evaluator.create_request(
                    problem=example["problem"],
                    answer=example["answer"],
                    prediction=response["answer"],
                )
                eval_result = evaluator.evaluate(eval_request)
                scores.append(eval_result.score)
            
            # Collect browser metrics if available (for eigent_search agent)
            browser_metrics = []
            if agent_type == "eigent_search" and hasattr(agent, 'web_toolkit'):
                browser_metrics = agent.web_toolkit.get_usage_metrics()
            
            # Add emoji to grade for visual clarity
            grade_emoji_map = {
                "CORRECT": "✅",
                "INCORRECT": "❌", 
                "NOT_ATTEMPTED": "⚠️"
            }
            grade = eval_result.metrics["grade"]
            grade_with_emoji = f"{grade_emoji_map.get(grade, '❓')} {grade}"
            
            results.append(
                {
                    "hash_id": hash_id,  # Unique hash ID for the problem
                    "dataset_index": problem_id,  # Index in the original dataset
                    "problem": example["problem"],
                    "answer": example["answer"],
                    "response": response,
                    "grade_emoji": grade_with_emoji,
                    "grade": grade,
                    "metadata": example.get("metadata", {}),  # Include metadata if available
                    "browser_metrics": browser_metrics,  # Browser usage metrics
                }
            )
            counter[eval_result.metrics["grade"]] += 1
            current_accuracy = counter["CORRECT"] / (i + 1) * 100
            
            # Log with browser metrics summary if available
            metrics_summary = ""
            if browser_metrics:
                metrics_summary = f" - Browser calls: {len(browser_metrics)}"
                function_counts = {}
                for metric in browser_metrics:
                    func = metric.get('function', 'unknown')
                    function_counts[func] = function_counts.get(func, 0) + 1
                metrics_summary += f" ({', '.join([f'{k}:{v}' for k, v in function_counts.items()])})"
            
            logger.info(f"[{agent_type}] Hash: {hash_id} | Index: {problem_id} ({i+1}/{num_questions}) - Grade: {grade_with_emoji} - Running totals: {counter} - Accuracy: {current_accuracy:.2f}%{metrics_summary}")

            if agent_type == "research":
                logger.info(
                    f"[{agent_type}] Number of searches: {agent.current_query_toolkit.search_counter}"
                )
                logger.info(
                    f"[{agent_type}] Process Graph:\n{agent.current_query_toolkit.trace_graph.render_trace_graph()}"
                )

            # Save results periodically
            if (i + 1) % 2 == 0 or i == num_questions - 1:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                tqdm.write(f"Results saved to {output_file}")

            # Clear browser metrics for next problem
            if agent_type == "eigent_search" and hasattr(agent, 'web_toolkit'):
                agent.web_toolkit.clear_metrics()
            
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
        
        # Print comprehensive summary
        print_evaluation_summary(results, counter, agent_type, len(results))


if __name__ == "__main__":
    main()
