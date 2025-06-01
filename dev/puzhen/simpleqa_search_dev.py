import json
import os
import re
import requests
from typing import Any, Dict, List
from bs4 import BeautifulSoup

from exa_py import Exa
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

exa_api_key = os.getenv("EXA_API_KEY")
exa = Exa(api_key=exa_api_key)

final_list_history: List[Dict[str, Any]] = []


def save_list_to_json(data_list: List[Dict[str, Any]], file_path: str) -> None:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                print(
                    "Warning: JSON file content is not a list. Overwriting "
                    "with new list.")
                existing_data = []
        except Exception as e:
            print(
                f"Error reading existing JSON file: {e}. Starting with an "
                f"empty list.")
            existing_data = []
    else:
        existing_data = []

    existing_data.extend(data_list)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving JSON to file: {e}")


def search_exa(query: str, max_chars: int = 2000) -> str:
    print(f"Calling Exa API with query: {query}, max_characters={max_chars}")
    result = exa.search_and_contents(
        query,
        text={"max_characters": max_chars}
    )
    return str(result.results)


def browse_webpage(url: str) -> str:
    print(f"🌐 Browse URL: {url}")
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text(separator='\n', strip=True)
        return text if text else "No text content found on the page."
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return f"Error: Could not retrieve content from {url}. Reason: {e}"
    except Exception as e:
        print(f"An unexpected error occurred while browsing {url}: {e}")
        return f"Error: An unexpected error occurred while processing {url}."


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Attempt to extract a JSON object from the given text.
    If the JSON is wrapped in ```json ... ```, extract the inner block.
    Otherwise, try to parse the entire text as JSON.
    """
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        json_text = fence_match.group(1)
    else:
        json_text = text.strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from response: {e}\nResponse Text:\n{text}")


def run_demo(
        question: str,
        exa_max_chars: int = 2000,
        enable_url_browse: bool = True
) -> (Dict[str, Any], List[BaseMessage]):
    global final_list_history
    final_list_history = []

    # ---- Step 1: Create the Knowledge Agent ----

    knowledge_base_prompt = (
        f"You are a diligent Knowledge Agent. Your main goal is to gather "
        f"comprehensive "
        f"information to answer the question: '{question}'.\n\n"
        f"You have access to two tools:\n"
        f"1. Exa Search: For broad information discovery. Use this to find "
        f"relevant "
        f"articles or data sources. When you use Exa Search, provide "
        f"keywords under "
        f"'retrieve_keywords'.\n"
        f"2. Web Browser: To read the full content of a specific URL. Use "
        f"this to dive "
        f"deeper into sources found by Exa or if a URL is directly relevant. "
        f"When you "
        f"use the Web Browser, provide the URL under 'browser_url'.\n\n"
        f"Always respond in JSON format."
    )

    # Instantiate ChatAgent with the system message as a plain string
    knowledge_agent = ChatAgent(system_message=knowledge_base_prompt)

    # Initial user prompt instructing the agent how to start gathering
    # information
    initial_user_prompt = (
        f"Let's start gathering information for the question: '"
        f"{question}'.\n\n"
        f"For your first turn:\n"
        f"- \"sufficiency\" must be False.\n"
        f"- Decide whether to use Exa Search (provide \"retrieve_keywords\") "
        f"or, if you have "
        f"a specific starting URL relevant to the question, use the Web "
        f"Browser (provide "
        f"\"browser_url\"). Typically, you'll start with Exa Search.\n\n"
        f"In subsequent turns, review all gathered information (from Exa "
        f"searches and Web "
        f"Browser) and decide:\n"
        f"1. Is the current information sufficient to answer the main "
        f"question thoroughly? "
        f"Set \"sufficiency\" to true/false.\n"
        f"2. If not sufficient:\n"
        f"   a. Do you need to perform another Exa Search? If so, "
        f"set \"retrieve_keywords\" "
        f"(and \"browser_url\" to null).\n"
        f"   b. Do you need to browse a specific URL (e.g., from previous "
        f"Exa results) for more "
        f"details? If so, set \"browser_url\" (and \"retrieve_keywords\" to "
        f"null).\n\n"
        f"Your JSON response should look like this:\n"
        f"{{\n"
        f"  \"sufficiency\": boolean,\n"
        f"  \"retrieve_keywords\": string or null,\n"
        f"  \"browser_url\": string or null,\n"
        f"  \"reason_for_action\": \"A brief explanation of your decision "
        f"and what you expect to find.\"\n"
        f"}}\n\n"
        f"Ensure that if \"sufficiency\" is False, either "
        f"\"retrieve_keywords\" OR "
        f"\"browser_url\" is a non-null string, but not both. If "
        f"\"sufficiency\" is True, both "
        f"\"retrieve_keywords\" and \"browser_url\" must be null."
    )

    # Send the initial user prompt
    knowledge_agent.step(initial_user_prompt)

    iteration_num = 0
    max_iterations = 10

    # Loop to gather knowledge
    while iteration_num < max_iterations:
        iteration_num += 1
        print(f"\n--- Knowledge Agent Iteration: {iteration_num} ---")

        # Let the agent respond based on its memory; supply an empty string
        # so step() accepts input
        agent_response: ChatAgentResponse = knowledge_agent.step("")
        raw_content = agent_response.msgs[0].content
        try:
            agent_response_dict = parse_json_response(raw_content)
        except ValueError as e:
            print(f"Error: Agent response is not valid JSON: {e}")
            error_msg = (
                "Your previous response was not valid JSON. "
                "Please correct it and follow the specified JSON format."
            )
            knowledge_agent.step(error_msg)
            iteration_num -= 1
            continue

        sufficiency = agent_response_dict.get('sufficiency', False)
        retrieve_keywords = agent_response_dict.get('retrieve_keywords')
        browser_url = agent_response_dict.get('browser_url')
        reason = agent_response_dict.get('reason_for_action',
                                         "No reason provided.")
        print(
            f"Agent's decision: Sufficiency={sufficiency}, "
            f"Search='{retrieve_keywords}', Browse='{browser_url}', "
            f"Reason='{reason}'"
        )

        if sufficiency:
            print("Agent deems information sufficient.")
            break
        else:
            action_taken = False
            user_next = ""

            if retrieve_keywords and isinstance(retrieve_keywords, str):
                if browser_url:
                    print(
                        "⚠️ Agent provided both search keywords and a URL. "
                        "Prioritizing Exa search.")

                exa_results_text = search_exa(retrieve_keywords,
                                              max_chars=exa_max_chars)
                print(
                    f"📄 Exa Search Results (first 300 chars): "
                    f"{exa_results_text[:300]}...")
                user_next = (
                    f"Here are the Exa search results for '"
                    f"{retrieve_keywords}':\n"
                    f"{exa_results_text}\n\n"
                    f"Now, review this and all prior information. Is it "
                    f"sufficient, or do you need "
                    f"to search again or browse a specific URL from these "
                    f"results?"
                )
                action_taken = True

            elif browser_url and isinstance(browser_url, str):
                if enable_url_browse:
                    browsed_content = browse_webpage(browser_url)
                    print(
                        f"📄 Browsed Content from {browser_url} (first 300 "
                        f"chars): {browsed_content[:300]}...")
                    user_next = (
                        f"Here is the content from Browse URL '"
                        f"{browser_url}':\n"
                        f"{browsed_content}\n\n"
                        f"Now, review this and all prior information. Is it "
                        f"sufficient, or do you need "
                        f"to search again or browse another URL?"
                    )
                    action_taken = True
                else:
                    print("🔒 browse_webpage not allowed")
                    user_next = (
                        "Browsing is disabled (enable_browse=False). "
                        "Please decide next steps based on existing "
                        "information."
                    )
                    action_taken = True

            if action_taken:
                knowledge_agent.step(user_next)
            else:
                print(
                    "⚠️ Agent indicated information is not sufficient but "
                    "provided no valid action.")
                no_action_msg = (
                    "You indicated the information is not sufficient, "
                    "but you didn't provide "
                    "'retrieve_keywords' for an Exa search or a "
                    "'browser_url' to browse. "
                    "Please either set 'sufficiency' to True if you have "
                    "enough information, "
                    "or provide one of these actions."
                )
                knowledge_agent.step(no_action_msg)

        if iteration_num >= max_iterations:
            print("Reached max iterations for knowledge gathering.")
            break

    # ---- Step 2: Organize gathered knowledge ----

    organize_knowledge_prompt = (
        "Now, please consolidate and organize all the relevant information "
        "you "
        "have gathered into bullet points. Focus on facts that will help "
        "answer "
        "the main question. Store this organized knowledge in the "
        "\"organized_knowledge\" key.\n\n"
        "Response in JSON:\n"
        "{\n"
        "  \"organized_knowledge\": \"- Point 1\\n- Point 2\\n- ...\"\n"
        "}"
    )
    knowledge_agent.step(organize_knowledge_prompt)

    organized_response_msg: ChatAgentResponse = knowledge_agent.step("")
    raw_organized = organized_response_msg.msgs[0].content
    try:
        organized_response = parse_json_response(raw_organized)
    except ValueError as e:
        raise RuntimeError(f"Failed to parse organized knowledge JSON: {e}")

    # ---- Step 3: Reasoning and final answer ----

    reasoning_user_prompt = (
        "Based on the following organized knowledge:\n"
        f"{organized_response.get('organized_knowledge', 'No organized knowledge provided.')}\n\n"
        f"And the main question: '{question}'\n\n"
        "Perform a step-by-step reasoning process to arrive at the final "
        "answer. "
        "If the provided facts are insufficient, try to answer based on your "
        "general "
        "knowledge but state that the provided facts were not enough.\n\n"
        "Provide your response in JSON format:\n"
        "{\n"
        "  \"step_by_step_reasoning\": \"1. First step...\\n2. Second "
        "step...\\n...\",\n"
        "  \"final_answer\": \"Your answer to the question.\"\n"
        "}"
    )
    knowledge_agent.step(reasoning_user_prompt)

    final_answer_msg: ChatAgentResponse = knowledge_agent.step("")
    raw_final = final_answer_msg.msgs[0].content
    try:
        final_answer_response = parse_json_response(raw_final)
    except ValueError as e:
        raise RuntimeError(f"Failed to parse final answer JSON: {e}")

    print("\n--- Final Answer ---")
    print(
        f"Step-by-step Reasoning:\n"
        f"{final_answer_response.get('step_by_step_reasoning', 'N/A')}")
    print(f"Final Answer: {final_answer_response.get('final_answer', 'N/A')}")

    final_list_history.append({
        "question": question,
        "full_conversation_knowledge_agent": knowledge_agent.chat_history
    })

    return final_answer_response, final_list_history


if __name__ == '__main__':
    question = (
        "How many corners did Barcelona take in the Champions League "
        "semi-final match "
        "between Barcelona and Milan on April 27, 2006?"
    )
    print(f"Starting demo with question: \"{question}\"")
    final_response, final_history= run_demo(
        question,
        exa_max_chars=10000,
        enable_url_browse=True
    )
    save_list_to_json(final_history,
                      'history_single_agent_with_Browse.json')

