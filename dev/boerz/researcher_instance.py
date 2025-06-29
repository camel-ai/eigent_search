from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
import json
import logging

from pydantic.functional_serializers import PlainSerializer

from librarian.research.researcher import LeadResearcher as BaseLeadResearcher, JuniorResearcher as BaseJuniorResearcher    
from librarian.research.research_toolkit import ResearchToolkit
from camel.models import BaseModelBackend

# Set up logging
logger = logging.getLogger(__name__)

# Example of parsing the LLM response
def parse_json_response(response_text):
    """
    Parse the JSON response from LLM into a Python dictionary.
    
    Args:
        response_text (str): The raw response string from LLM
        
    Returns:
        dict: Parsed JSON as Python dictionary
    """
    try:
        # Remove any leading/trailing quotes and whitespace
        cleaned_response = response_text.strip().strip('"\'')
        # Parse JSON string to Python dict
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return None

class LeadResearcher(BaseLeadResearcher):
    
    def create_research_plan(
        self,
        query: str,
        reflection: Optional[Dict[str, Any]] = None,
        max_loops: int = 3,
        max_sub_tasks: int = 3,
        research_results: List[Dict[str, Any]] = [],
    ) -> Dict[str, Any]:
        r"""Create a research plan for a given query."""
        
        logger.info(f"Creating research plan for query: {query}")
        
        
        # TODO: Based on future experiments, perhaps write a dedicated planner agent
        # Rather than asking LeadResearcher to do this.
        plan = {}

        if len(research_results) > 0 and reflection is not None:
            str_results = "\n".join([f"Sub-query: {result['sub_query']}\nResult: {result['sub_result']}" for result in research_results])
            logger.info(f"Re-planning:Sub-queries that have been explored:\n"+ "\n".join([result['sub_query'] for result in research_results]))
            
        planning_prompt = f"""
        We are working on resolving this query: "{query}".

        """

        if len(research_results) > 0 and reflection is not None:
            str_results = "\n".join([f"Sub-query: {result['sub_query']}\nResult: {result['sub_result']}" for result in research_results])
            planning_prompt += f"""
        We have already done some research, and here are the results:
        
        {str_results}

        Our previous answer is:

        {reflection["answer"]}

        We have also done some reflection on previous results:
        
        {reflection["comments"]}

        """

        planning_prompt += f"""
        Please create a step-by-step research plan to better answer the query.

        You will have chance to do more re-planning based on previous reseach results.
        
        You have at most {max_loops} more chances to create research plans to answer the query.
        
        Break this into a few focused sub-tasks that can be researched independently. You can make at most {max_sub_tasks} sub-tasks.
        
        IMPORTANT: Respond with ONLY valid JSON. Do not include any explanatory text before or after the JSON.
        
        Return a JSON object with these exact keys:
        {{
            "sub_tasks": ["list", "of", "subtasks", "to", "answer", "the", "query"],
            
        }}
        
        Example format:
        {{
            "sub_tasks": ["What is X?", "How does Y work?", "What are the applications of Z?"],
        }}
        """


        plans = self.step(planning_prompt)
        
        # Get the response content from the last message
        response_content = plans.msgs[-1].content
        
        # Parse the JSON response into a Python dictionary
        try:
            planning_data = parse_json_response(response_content)
            if planning_data:
                plan["sub_tasks"] = planning_data.get("sub_tasks", [])
                logger.info(f"Parsed planning data: {planning_data}")
            else:
                logger.info("Failed to parse planning response")
                plan["sub_tasks"] = []
        except Exception as e:
            logger.error(f"Error processing planning response: {e}")
            plan["sub_tasks"] = []
        
        
        return plan

    def assign_task(self, sub_query: str) -> Dict[str, Any]:
        r"""Assign a task to a spawned sub-research agent and return the research results."""
        
        # Spawn a new JuniorResearcher agent for this specific sub-task
        # TODO: Based on future experiment results, we might want to consider passing lead researcher's memory into junior researcher. 
        # TODO: Otherwise, junior researchers might lack context of subquery, or do repeated work.
        junior_researcher = JuniorResearcher(
            system_message="You are a junior research assistant. Focus on finding accurate, well-sourced information for your assigned task.",
            # TODO: This is a hack to get the model backend for LeadResearcher. 
            # TODO: We should be able to do it in a more elegant way
            model=self.model_backend.models[0],  
            research_toolkit=self.research_toolkit,  # Share the same toolkit
            parent=self  # Set this lead researcher as the parent
        )
        
        # Let the junior researcher handle the sub-query
        research_result = junior_researcher.research(sub_query)
        
        return research_result

    def reflect(self, research_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        r"""Reflect on the research results.

        Goal of this reflection function:
        - TODO?: Behave like a librarian, summarize current research results for each sub-query to save token, 
        keep the original references in the "evidence" field.
        - TODO?: Also organize previous sub-queries, e.g. manage their order.
        - TODO: Again, maybe use a dedicated agent to do this?

        - Give a summary on current results
        - Give comments on current results and find potential gaps.
        

        Note:
        Compared to Yuan's design,
        I added a (original) query parameter to the reflect function,
        to give the lead researcher more context about the original query.
        Actually lead researcher should already have the context from memory.
        We need to do experiments to see if explicitly passing query to reflect
        function is better than using memory to manage context.
        """
        
        # Simply flatten all research results into a string.
        str_results = "\n".join([f"Sub-query: {result['sub_query']}\nResult: {result['sub_result']}" for result in research_results])

        reflect_prompt = f"""
        We are working on resolving this query: "{query}".

        Our current research results are:
        {str_results}

        Please 
        1. Summarize the current research results to give a short answer to the original query.
        2. Reflect on the current results, and find potential gaps. Give comments on how we should improve our answer.
        3. Decide if we should need further research to answer the query.
        
        Return a JSON object with these exact keys:
        {{
            "answer": "short answer to the original query",
            "comments": "comments on the current results",
            "continue": "True or False, whether to continue the research"
        }}
        """

        reflection_msg = self.step(reflect_prompt)
        response_content = reflection_msg.msgs[-1].content
        logger.debug(f"Raw reflection response: {response_content}")
        
        try:
            reflection = parse_json_response(response_content)
            if reflection:
                logger.debug(f"Successfully parsed reflection data: {reflection}")
            else:
                logger.warning("Failed to parse reflection response - got None")
                reflection = {}
        except Exception as e:
            logger.error(f"Error processing reflection response: {e}")
            reflection = {}

        reflection["evidence"] = "Not implemented yet!"
        logger.info(f"Final reflection result: {reflection}")
        return reflection
        
        
        #return "Research reflection completed"

    def complete_task(self, reflection: Dict[str, Any]) -> bool:
        r"""Todo: Discuss with Yuan about whether/how we should implement this."""
        logger.info(f"Complete_task: Checking if task is complete with reflection.")
        
        continue_value = reflection.get("continue", False)
        logger.info(f"Continue value from reflection: {continue_value} (type: {type(continue_value)})")
        
        # Handle both boolean and string values
        if isinstance(continue_value, bool):
            result = continue_value
            logger.info(f"Continue value is boolean: {result}")
        elif isinstance(continue_value, str):
            result = continue_value.lower() in ['true', 'yes', '1', 'continue']
            logger.info(f"Continue value is string '{continue_value}' -> converted to: {result}")
        else:
            result = False
            logger.warning(f"Continue value is unexpected type {type(continue_value)}: {continue_value} -> defaulting to False")
        
        # Here the logic is a bit tricky, 
        # if continue is True, 
        # the task_complete is False, 
        # and vice versa.
        result = not result
        logger.info(f"Task completion decision: {result}")
        return result

    def research(
        self, 
        query: str, 
        reflection: Optional[Dict[str, Any]] = None, 
        max_loops: int = 3,
        max_sub_tasks: int = 3,
        research_results: List[Dict[str, Any]] = [],
    ) -> Dict[str, Any]:  # noqa: D401
        r"""Temporarily override Yuan's base implementation for testing purposes.

        Args:
            query: The query to be resolved.
            reflection: The reflection on the previous research results.
            max_loops: The maximum number of planning loops to run.
            max_sub_tasks: The maximum number of sub-tasks to create for each planning loop.
            research_results: The previous research results.
        
        """
        logger.info(f"Researching query: {query}")
        plan = self.create_research_plan(
            query = query, 
            reflection = reflection, 
            max_loops = max_loops,
            max_sub_tasks = max_sub_tasks,
            research_results = research_results
            )
        sub_tasks: Sequence[str] = plan.get("sub_tasks", [])  # type: ignore[arg-type]

        # TODO: need to revise async implementation
        
        # research_results: List[Dict[str, Any]] = []
        for task in sub_tasks:
            sub_research_results = self.assign_task(task)
            research_results.append(sub_research_results)

        reflection = self.reflect(research_results, query)
        if not self.complete_task(reflection) and max_loops - 1 > 0:
            # Todo: In Yuan's design, we should refine query during reflection
            # I am using original query for simplicity. In the future, we should refine query during reflection.
            # Btw, I think we should always keep the original query even if we refine it.
            return self.research(query, 
                reflection=reflection, 
                max_loops=max_loops-1,
                max_sub_tasks=max_sub_tasks, 
                research_results=research_results,
                )

        return {
            "query": query,
            "answer": reflection["answer"],
            "evidence": reflection["evidence"],
        }

    

class JuniorResearcher(BaseJuniorResearcher):

    
    def research(self, query: str) -> Dict[str, Any]:
        r"""Research the query and return results with evidence."""
        logger.info(f"Junior Researcher: Researching query: {query}")
        result = {}
        result["sub_result"] = self.step(query).msgs[-1].content
        result["sub_query"] = query
        # logger.info(f"Junior Researcher: Research result: {result['sub_result']}")
        return result

    def reflect(self, research_results: List[Dict[str, Any]]) -> str:
        r"""Reflect on the research results."""
        # Currently a place holder, we do not always need this.
        return "Research reflection completed"

    def complete_task(self, reflection: str) -> bool:
        r"""Determine if the research task is complete."""
        # Currently a place holder, we do not always need this.
        return True
        

# Usage example:
# response = llm_response  # Your LLM response string
# planning_data = parse_planning_response(response)
# if planning_data:
#     sub_tasks = planning_data.get('sub_tasks', [])
#     strategy = planning_data.get('strategy', '')
#     iterations = planning_data.get('estimated_iterations', 1)