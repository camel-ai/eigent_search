from camel.agents import ChatAgent
from camel.types import ModelType
from datasets import load_dataset
from librarian.model import create_openai_model
from librarian.evaluation import SimpleQAGrader
from librarian.schema import LibrarianResponse, PlainResponse, CoTResponse
from librarian.prompt import PLAIN_PROMPT, COT_PROMPT, LIBRARIAN_PROMPT
import json
import argparse
from typing import Dict
from tqdm import tqdm

AGENT_CONFIGS = {
    'librarian': {
        'prompt': LIBRARIAN_PROMPT,
        'response_format': LibrarianResponse,
        'question_prefix': 'Question: ',
        'show_fields': ['knowledge', 'reasoning', 'answer']
    },
    'plain': {
        'prompt': PLAIN_PROMPT,
        'response_format': PlainResponse,
        'question_prefix': '',
        'show_fields': ['answer']
    },
    'cot': {
        'prompt': COT_PROMPT,
        'response_format': CoTResponse,
        'question_prefix': '',
        'show_fields': ['reasoning', 'answer']
    }
}


def create_agent(agent_type: str) -> ChatAgent:
    """Create a new agent of the specified type."""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    config = AGENT_CONFIGS[agent_type]
    return ChatAgent(
        model=create_openai_model(ModelType.GPT_4O),
        system_message=config['prompt']
    )


def get_agent_response(agent: ChatAgent, problem: str, agent_type: str) -> Dict:
    """Get response from the specified agent."""
    config = AGENT_CONFIGS[agent_type]
    response = agent.step(
        f"Question: {config['question_prefix']}{problem}",
        response_format=config['response_format']
    )
    return eval(response.msgs[0].content)


def print_agent_response(response: Dict, agent_type: str) -> None:
    """Print the response from an agent."""
    tqdm.write(f"\n#### {agent_type.title()} Agent Answer ####\n")
    for field in AGENT_CONFIGS[agent_type]['show_fields']:
        tqdm.write(f"{field}: {response[field]}\n")


def single_result(agent_type: str):
    """Create a new result dictionary for a single agent."""
    return {
        "problem": None,
        "answer": None,
        "prediction": {agent_type: {"response": None, "grade": None}}
    }


def main(agent_type: str, testing: bool = False):
    dataset = list(load_dataset("basicv8vc/SimpleQA")["test"])
    if testing:
        dataset = dataset[6:9]
    
    # Create agents
    agent = create_agent(agent_type)
    evaluation_agent = SimpleQAGrader(
        ChatAgent(model=create_openai_model(ModelType.GPT_4O))
    )
    
    results = []
    correct_count = 0
    total = 0
    
    for dp in tqdm(dataset, desc=f"{agent_type.title()} Agent (SimpleQA)"):
        total += 1
        
        # Reset agent
        agent.reset()
        
        # Create a new result
        result = single_result(agent_type)
        result["problem"] = dp["problem"]
        result["answer"] = dp["answer"]
        
        tqdm.write(f"#### Question ####\n\n{dp['problem']}")
        
        # Get agent response
        response = get_agent_response(agent, dp["problem"], agent_type)
        result["prediction"][agent_type]["response"] = response
        print_agent_response(response, agent_type)
        
        tqdm.write(f"\n#### Gold Answer ####\n\n{dp['answer']}")
        
        tqdm.write("\n#### Grade ####\n")
        grade = eval(
            evaluation_agent.grade(dp["problem"], dp["answer"], response["answer"])
        )["grade"]
        
        result["prediction"][agent_type]["grade"] = grade
        
        if grade == "CORRECT":
            correct_count += 1
        
        tqdm.write(f"{agent_type.title()} Agent Grade: {grade}")
        
        results.append(result)
        
        tqdm.write("\n#### Cumulative Results ###\n")
        tqdm.write(f"{agent_type.title()} Agent Cumulative Correct: {correct_count} / {total}")
        
        tqdm.write("\n")
        tqdm.write("---------")
        tqdm.write("\n")
    
    output_file = f"single_agent_result_{agent_type}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments on SimpleQA...")
    parser.add_argument(
        "--agent_type",
        "-a",
        choices=list(AGENT_CONFIGS.keys()),
        help="Type of agent to use (librarian, plain, or cot)"
    )
    parser.add_argument(
        "--testing",
        "-t",
        action="store_true",
        help="Run in testing mode (only 3 problems)"
    )
    args = parser.parse_args()
    main(args.agent_type, args.testing)
