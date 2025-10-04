# query_processing_v4.py
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

from typing import List, Set
from functools import wraps
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.logger import get_logger

logger = get_logger(__name__)



def validate_input_query_in_frontier(func):
    """Decorator to validate that the input query is in the current frontier.

    This decorator should be applied to methods that take a query parameter
    and need to ensure it exists in the current frontier before processing.
    """

    @wraps(func)
    def wrapper(self, **kwargs):
        # Get the query parameter - could be 'query' or 'current_query'
        query = kwargs.get("query") or kwargs.get("current_query") or kwargs.get("selected_query")
        if query and query not in self.frontier:
            error_message = (
                f"❌ Invalid operation: Candidate query '{query}' must be selected "
                f"from the current frontier listed below.\n{self.get_frontier_str()}"
            )
            logger.error(f"[{func.__name__}] {error_message}")
            return error_message
        return func(self, **kwargs)

    return wrapper



def validate_output_query_not_explored(func):
    """Decorator to validate that the output query is not already explored.

    This decorator should be applied to methods that take an output query parameter
    and need to ensure it hasn't been explored before processing.
    """

    @wraps(func)
    def wrapper(self, **kwargs):
        # Find output query parameters
        output_params = {
            "select_query": "selected_query",
            "local_expand_query": "expanded_query",
            "local_refine_query": "refined_query",
            "global_refine_query": "refined_query",
            "global_expand_query": "expanded_query",
        }
        output_param = output_params.get(func.__name__)
        output_query = kwargs.get(output_param)

        if output_query:
            # Validation logic
            was_single = not isinstance(output_query, list)
            if was_single:
                output_query = [output_query]

            output_query_not_explored = [
                oq for oq in output_query if oq not in self.explored
            ]

            if not output_query_not_explored:
                error_message = (
                    f"❌ Invalid operation: All {output_query} are already explored."
                )
                logger.error(error_message)
                return error_message

            # Update kwargs with filtered queries
            if was_single:
                kwargs[output_param] = output_query_not_explored[0]
            else:
                kwargs[output_param] = output_query_not_explored

        return func(self, **kwargs)

    return wrapper


class QueryProcessingToolkit(BaseToolkit):
    r"""A comprehensive toolkit for query processing and relevance feedback in web search.

    This toolkit combines query selection logic with relevance feedback mechanisms,
    allowing agents to iteratively refine their search strategy.
    """

    def __init__(self, initial_query: str = None):
        super().__init__()
        self.frontier: Set[str] = set()  # set of queries to be explored
        self.explored: Set[str] = set()  # set of queries that have been explored

        # If initial query provided, add to frontier
        if initial_query:
            self.frontier.add(initial_query)

    def get_frontier_str(self) -> str:
        """Display the current frontier as a string."""
        return "📋 current frontier:\n  - " + "\n  - ".join(list(self.frontier))


    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def select_query(
            self,
            frontier: str,
            selected_query: str,
            selection_reason: str
    ) -> dict[str, list[str]]:
        r"""Select the best query from the current frontier.

        The agent should choose based on specificity, clarity, and search potential, in order to minimize the number of searches and the cost of the search.

        If the search results are not sufficient to answer the user's initial query,
        the agent should process and select another query from the current frontier.

        After selection, call search_google with the selected query.



        Args:
            frontier (str): The input query from the current frontier being considered.
            selected_query (str): The query you've chosen (should be same as query).
            selection_reason (str): Why you selected this query.

        Returns:
            dict[str, list[str]]: The current frontier after the selection process.
        """
        # Move from frontier to explored
        if selected_query in self.frontier:
            self.frontier.remove(selected_query)
        self.explored.add(selected_query)

        logger.info(f"[select_query] Selected: '{selected_query}'")
        logger.info(f"[select_query] Reason: {selection_reason}")

        return {"frontier": list(self.frontier)}




    def extract_relevant_details(
            self,
            snapshot: str,
            query: str,
            question: str,
            relevant_information: str,
            page_url: str = ""
    ) -> str:
        r"""Use this tool to extract relevant information from the page that answers the question.

        When extracting, carefully read the ENTIRE snapshot including:
        - Structured sections like tables, info boxes, and labeled fields (these often contain direct answers)
        - Main article text and paragraphs
        - All sections that might contain relevant facts

        CRITICAL - Precision Requirements:
        - Extract information that EXACTLY matches what the question asks for, not just related information
        - If the question asks for specific terms (e.g., "Gold"), don't substitute with related terms (e.g., "Platinum")
        - If the question asks for complete details (e.g., "day, month, and year"), ensure you capture all components
        - If you find information that's close but not exact, note what's missing and mark it as incomplete

        Extraction guidelines:
        - Look for explicit, direct answers first (especially in tables/info boxes with structured data)
        - If you find conflicting or multiple pieces of information, include ALL of them
        - Be thorough - don't stop at the first relevant snippet; scan the entire page
        - If the exact information requested is NOT on this page, explicitly state what's missing

        Args:
            snapshot (str): The complete page content from browser_visit_page
            query (str): The search query you used to find this page
            question (str): The original question - what EXACTLY does it ask for?
            relevant_information (str): Information you extract (must precisely match question requirements)
            page_url (str): The URL of the page

        Returns:
            str: Confirmation that your extracted information has been recorded
        """
        logger.info(f"[extract_relevant_details] From: {page_url}")
        return f"Relevant details recorded:\n\n{relevant_information}"

    def analyze_search_progress(
            self,
            question: str,
            current_query: str,
            findings_so_far: str,
            your_analysis: str
    ) -> str:
        r"""Call this to analyse whether you have enough information to answer the question.

        In your_analysis parameter, write your evaluation by:
        1. Comparing your findings_so_far against what the question asks
        2. Identifying any gaps or missing details
        3. Determining if you need to refine your search

        Your analysis should explain what you have vs what you still need.

        Args:
            question (str): The original research question
            current_query (str): The search query you last used
            findings_so_far (str): All relevant details you've extracted from visited pages
            your_analysis (str): Your written evaluation - do findings answer the question completely?

        Returns:
            str: Your analysis, recorded
        """
        logger.info(f"[analyze_search_progress] Query: '{current_query}'")
        return f"Analysis recorded:\n\n{your_analysis}"

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def local_expand_query(
            self,
            question: str,
            current_query: str,
            what_you_know: str,
            what_exactly_missing: str,
            refined_query: str
    ) -> str:
        r"""Call this to create a refined search query based on identified gaps (LOCAL EXPAND).

        Use the parameters to build your refined_query:
        - Look at what_you_know to avoid searching for information you already have
        - Focus refined_query specifically on what_exactly_missing
        - Make the query more targeted than current_query to fill the precise gap

        Your refined_query should target the specific missing piece, not repeat previous searches.

        Args:
            question (str): The original research question (for context)
            current_query (str): The search query you last used
            what_you_know (str): Summary of information you've already found
            what_exactly_missing (str): The specific information gap you identified
            refined_query (str): Your NEW search query (3-7 words) targeting what_exactly_missing

        Returns:
            str: Confirmation of your refined query to use in search_google
        """
        # Add the refined query to the frontier
        self.frontier.add(refined_query)
        logger.info(f"[local_expand_query] '{current_query}' → '{refined_query}'")
        return f"""
        REFINED QUERY (LOCAL EXPAND)

        Previous Query: "{current_query}"
        Already Found: {what_you_know}
        Missing: {what_exactly_missing}
        New Query: "{refined_query}"
        Frontier Queries: {list(self.frontier)}
        Use this refined query in your next search_google call.
        """

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def local_refine_query(
            self,
            question: str,
            current_query: str,
            search_results_summary: str,
            refined_query: str
    ) -> str:
        r"""Refine the current query based on search results without changing the core meaning (LOCAL REFINE).

        This tool helps you rephrase the query for better results while maintaining the same search intent.
        Use this when your current query didn't return good results, but you want to search for the same information
        using different phrasing, synonyms, or alternative formulations.

        Key differences from local_expand_query:
        - local_refine: Same meaning, different phrasing (e.g., "CEO of Apple" → "Apple chief executive")
        - local_expand_query: Different scope/focus based on gaps (e.g., "CEO of Apple" → "Tim Cook leadership style")

        Args:
            question (str): The original research question
            current_query (str): The search query that didn't work well
            search_results_summary (str): Brief summary of why current results were insufficient
            refined_query (str): Your rephrased query with same meaning but different wording

        Returns:
            str: Confirmation of your refined query
        """
        # Add the refined query to the frontier
        self.frontier.add(refined_query)
        logger.info(f"[local_refine_query] '{current_query}' → '{refined_query}'")
        return f"""
        QUERY REFINEMENT (LOCAL REFINE)

        Original Query: "{current_query}"
        Search Results Issue: {search_results_summary}
        Refined Query: "{refined_query}"
        Frontier Queries: {list(self.frontier)}
        Use this refined query to search for the same information with better phrasing.
        """

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def global_refine_query(
            self,
            question: str,
            current_query: str,
            refinement_reason: str,
            refined_query: str
    ) -> str:
        r"""Refine the query based on your understanding without using search results (GLOBAL REFINE).

        This tool helps you improve query clarity, remove ambiguity, or fix issues you identify
        through your own analysis, not based on search results. Use this when you realize
        the current query has problems that you can fix through better formulation.

        Common use cases:
        - Remove ambiguous terms
        - Make query more specific
        - Fix grammatical issues
        - Use more standard terminology
        - Clarify temporal aspects (add year, "current", etc.)

        Args:
            question (str): The original research question
            current_query (str): The current search query
            refinement_reason (str): Why you think the query needs refinement
            refined_query (str): Your improved query with same core meaning

        Returns:
            str: Confirmation of your refined query
        """
        # Add the refined query to the frontier
        self.frontier.add(refined_query)
        logger.info(f"[global_refine_query] '{current_query}' → '{refined_query}'")
        return f"""
        GLOBAL QUERY REFINEMENT

        Original Query: "{current_query}"
        Refinement Reason: {refinement_reason}
        Refined Query: "{refined_query}"
        Frontier Queries: {list(self.frontier)}
        Use this globally refined query in your next search.
        """

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def global_expand_query(
            self,
            question: str,
            current_query: str,
            expansion_strategy: str,
            expanded_query: str
    ) -> str:
        r"""Expand the query with additional terms without using search results (GLOBAL EXPAND).

        This tool helps you add synonyms, related terms, morphological variants, or context
        to improve search coverage. Use your knowledge to broaden the query scope or
        add relevant context that might help find better results.

        Expansion strategies:
        - Add synonyms or alternative terms
        - Include related concepts or domains
        - Add temporal context (years, periods)
        - Include morphological variants
        - Add context from your knowledge
        - Include related entities or organizations

        Args:
            question (str): The original research question
            current_query (str): The current search query
            expansion_strategy (str): What type of expansion you're applying and why
            expanded_query (str): Your expanded query with additional relevant terms

        Returns:
            str: Confirmation of your expanded query
        """
        # Add the expanded query to the frontier
        self.frontier.add(expanded_query)
        logger.info(f"[global_expand_query] '{current_query}' → '{expanded_query}'")
        return f"""
        GLOBAL QUERY EXPANSION

        Original Query: "{current_query}"
        Expansion Strategy: {expansion_strategy}
        Expanded Query: "{expanded_query}"
        Frontier Queries: {list(self.frontier)}
        Use this globally expanded query to cast a wider search net.
        """

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns all available tools in the toolkit."""
        return [
            FunctionTool(self.select_query),
            FunctionTool(self.extract_relevant_details),
            FunctionTool(self.analyze_search_progress),
            FunctionTool(self.local_expand_query),  # LOCAL EXPAND
            FunctionTool(self.local_refine_query),  # LOCAL REFINE
            FunctionTool(self.global_refine_query),  # GLOBAL REFINE
            FunctionTool(self.global_expand_query),  # GLOBAL EXPAND
        ]