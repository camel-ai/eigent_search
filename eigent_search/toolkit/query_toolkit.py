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

from functools import wraps
from textwrap import dedent
import difflib

from camel.logger import get_logger
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.toolkits.search_toolkit import SearchToolkit


logger = get_logger(__name__)


def fuzzy_contains(query: str, candidates: set[str], cutoff: float = 0.7) -> bool:
    """Check if the query has a fuzzy match in the candidates set.

    Args:
        query (str): The query to check for similarity.
        candidates (set[str]): The set of candidate queries to match against.
        cutoff (float): The minimum similarity threshold (0.0 to 1.0).

    Returns:
        bool: True if a similar query is found, False otherwise.
    """
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=cutoff)
    return len(matches) > 0


def validate_input_query_in_frontier(func):
    """Decorator to validate that the input query is in the current frontier.

    This decorator should be applied to methods that take a query parameter
    and need to ensure it exists in the current frontier before processing.
    """

    @wraps(func)
    def wrapper(self, **kwargs):
        # Get the query parameter
        query = kwargs.get("query")
        if query and not fuzzy_contains(query, self.frontier):
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
            "local_expand_query": "expanded_queries",
            "local_refine_query": "refined_queries",
            "global_refine_query": "refined_queries",
            "global_expand_query": "expanded_queries",
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


def validate_input_query_in_frontier_or_explored(func):
    """Decorator to validate that the input query is in the current frontier or already explored.

    This decorator should be applied to methods that take a query parameter
    and need to ensure it exists in the current frontier OR has been previously explored.
    """

    @wraps(func)
    def wrapper(self, **kwargs):
        # Get the query parameter - could be 'query' or 'current_query'
        query = kwargs.get("query") or kwargs.get("current_query")
        if (
            query
            and not fuzzy_contains(query, self.frontier)
            and query not in self.explored
        ):
            error_message = (
                f"❌ Invalid operation: Candidate query '{query}' must be selected "
                f"from the current frontier listed below.\n{self.get_frontier_str()}"
                f"OR from the explored queries listed below.\n{self.get_explored_str()}"
            )
            logger.error(f"[{func.__name__}] {error_message}")
            return error_message
        return func(self, **kwargs)

    return wrapper


class QueryProcessingToolkit(BaseToolkit):
    r"""A comprehensive toolkit for query processing and relevance feedback in web search.

    This toolkit combines query selection logic with relevance feedback mechanisms,
    allowing agents to iteratively refine their search strategy.
    """

    def __init__(self, exclude_domains: list[str] | None = None):
        super().__init__()
        self.frontier = set()
        self.explored = set()
        self.initial_query = None
        self._search_toolkit = SearchToolkit(exclude_domains=exclude_domains)
        self.search = lambda query: self._search_toolkit.search_google(
            query=query,
            search_type="web",
            number_of_result_pages=10,
            start_page=1,
        )
        self.search_counter = 0

    def load_initial_query(self, initial_query: str):
        """Reset with the new initial query."""
        self.initial_query = initial_query
        self.frontier = {initial_query}
        self.explored = set()
        return f"Initial query: {initial_query}\nCurrent Frontier:\n  - " + "\n  - ".join(list(self.frontier))

    def get_frontier_str(self) -> str:
        """Display the current frontier as a string."""
        return "Current Frontier:\n  - " + "\n  - ".join(list(self.frontier))

    def get_explored_str(self) -> str:
        """Display the explored queries as a string."""
        return "Explored Queries:\n  - " + "\n  - ".join(list(self.explored))

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def select_query_and_search(self, query: str) -> dict[str, dict[str, str]]:
        r"""Select the best query from the current frontier and perform web search.
        The agent should choose based on specificity, clarity, and search potential,
        in order to minimize the number of searches and the cost of the search.
        If the search results are not sufficient to answer the user's initial query,
        the agent should process and select another query from the current frontier
        and perform web search again, or generate new queries based on the search results.

        Args:
            query (str): The input query from the current frontier that is selected
                for web search.
        Returns:
            dict[str, dict[str, str]]: The search results from the web search. The key is "search_results" and the value is a dict where each key is a URL and each value is the string of the title, description, and long description of the result. If the search fails, the key is "None" and the value is the error message.
        """
        # Update frontier and explored sets; the search will be conducted anyway
        if query in self.frontier:
            self.frontier.remove(query)
        self.explored.add(query) # for fuzzy contain

        # Helper function to perform search and handle results
        def search_and_record(query_str: str):
            results = self.search(query_str)
            self.search_counter += 1

            # Check if search has returned anything valid
            if "error" in results[0]:
                logger.error(f"[search] Error for '{query_str}': {results[0]['error']}")
                return {"None": results[0]["error"]}

            # Linearize valid search results to dictionary of strings
            results: dict[str, str] = {
                result["url"]: (
                    f"Title: {result['title']}\n"
                    f"Description: {result['description']}\n"
                    f"Long Description: {result['long_description']}"
                )
                for result in results
            }

            logger.info(f"[search] Found {len(results)} results for '{query_str}'")
            return results

        # Perform search
        return {"search_results": search_and_record(query)}

    def extract_relevant_details(
        self,
        snapshot: str,
        query: str,
        question: str,
        relevant_information: str,
        page_url: str = "",
    ) -> str:
        r"""Use this tool to extract relevant information from the page that answers
        the question.

        When extracting, carefully read the ENTIRE snapshot including:
        - Structured sections like tables, info boxes, and labeled fields (these often
          contain direct answers)
        - Main article text and paragraphs
        - All sections that might contain relevant facts
        CRITICAL - Precision Requirements:
        - Extract information that EXACTLY matches what the question asks for, not just
          related information
        - If the question asks for specific terms (e.g., "Gold"), don't substitute with
          related terms (e.g., "Platinum")
        - If the question asks for complete details (e.g., "day, month, and year"),
          ensure you capture all components
        - If you find information that's close but not exact, note what's missing and
          mark it as incomplete

        Extraction guidelines:
        - Look for explicit, direct answers first (especially in tables/info boxes with
          structured data)
        - If you find conflicting or multiple pieces of information, include ALL of them
        - Be thorough - don't stop at the first relevant snippet; scan the entire page
        - If the exact information requested is NOT on this page, explicitly state
          what's missing

        Args:
            snapshot (str): The complete page content from browser_visit_page
            query (str): The search query you used to find this page
            question (str): The original question - what EXACTLY does it ask for?
            relevant_information (str): Information you extract (must precisely match
                question requirements)
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
        your_analysis: str,
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

    @validate_input_query_in_frontier_or_explored
    @validate_output_query_not_explored
    def local_expand_query(
        self,
        question: str,
        current_query: str,
        what_you_know: str,
        what_exactly_missing: str,
        expanded_queries: list[str],
    ) -> str:
        r"""Call this to create refined search queries based on identified gaps (LOCAL EXPAND).

        Use the parameters to build your expanded_queries:
        - Look at what_you_know to avoid searching for information you already have
        - Focus each query in expanded_queries specifically on what_exactly_missing
        - Make queries more targeted than current_query to fill precise gaps
        - Generate multiple query variations to explore different angles
        Your expanded_queries should target the specific missing pieces from different
        perspectives.

        'current_query' must be an existing query (either in frontier or explored).

        Args:
            question (str): The original research question (for context)
            current_query (str): The search query you last used
            what_you_know (str): Summary of information you've already found
            what_exactly_missing (str): The specific information gap you identified
            expanded_queries (List[str]): Multiple NEW search queries targeting
                what_exactly_missing
        Returns:
            str: Confirmation of your expanded queries to use in select_query_and_search
        """
        # Add all expanded queries to the frontier
        self.frontier.update(expanded_queries)

        logger.info(f"[local_expand_query] '{current_query}' → {expanded_queries}")

        return dedent(
            f"""
            REFINED QUERIES (LOCAL EXPAND)
            Previous Query: "{current_query}"
            Already Found: {what_you_know}
            Missing: {what_exactly_missing}
            New Queries: {expanded_queries}
            Current Frontier: {list(self.frontier)}
            Use these refined queries in your next searches.
            """
        ).strip()

    @validate_input_query_in_frontier_or_explored
    @validate_output_query_not_explored
    def local_refine_query(
        self,
        question: str,
        current_query: str,
        search_results_summary: str,
        refined_queries: list[str],
    ) -> str:
        r"""Refine the current query based on search results without changing core
        meaning (LOCAL REFINE).

        This tool helps you rephrase the query for better results while maintaining
        the same search intent. Generate multiple reformulations to try different
        phrasings, synonyms, or alternative formulations.

        Key differences from local_expand_query:
        - local_refine: Same meaning, different phrasings
          (e.g., "CEO of Apple" → ["Apple chief executive", "Apple CEO name"])
        - local_expand_query: Different scope/focus based on gaps
          (e.g., "CEO of Apple" → ["Tim Cook leadership", "Apple executive team"])
        'current_query' must be an existing query (either in frontier or explored).

        Args:
            question (str): The original research question
            current_query (str): The search query that didn't work well
            search_results_summary (str): Brief summary of why current results were
                insufficient
            refined_queries (List[str]): Multiple rephrased queries with same meaning
                but different wording
        Returns:
            str: Confirmation of your refined queries
        """
        # Add all refined queries to the frontier
        self.frontier.update(refined_queries)

        logger.info(f"[local_refine_query] '{current_query}' → {refined_queries}")

        return dedent(
            f"""
            QUERY REFINEMENTS (LOCAL REFINE)
            Original Query: "{current_query}"
            Search Results Issue: {search_results_summary}
            Refined Queries: {refined_queries}
            Current Frontier: {list(self.frontier)}
            Use these refined queries to search for the same information with better phrasing.
            """
        ).strip()

    @validate_input_query_in_frontier_or_explored
    @validate_output_query_not_explored
    def global_refine_query(
        self,
        question: str,
        current_query: str,
        refinement_reason: str,
        refined_queries: list[str],
    ) -> str:
        r"""Refine the query based on your understanding without using search results (GLOBAL REFINE).

        This tool helps you improve query clarity, remove ambiguity, or fix issues
        through your own analysis. Generate multiple refined versions to try different
        improvements.

        Common use cases:
        - Remove ambiguous terms
        - Make query more specific
        - Fix grammatical issues
        - Use more standard terminology
        - Clarify temporal aspects (add year, "current", etc.)
        'current_query' must be an existing query (either in frontier or explored).

        Args:
            question (str): The original research question
            current_query (str): The current search query
            refinement_reason (str): Why you think the query needs refinement
            refined_queries (List[str]): Multiple improved queries with same core meaning
        Returns:
            str: Confirmation of your refined queries
        """
        # Add all refined queries to the frontier
        self.frontier.update(refined_queries)

        logger.info(f"[global_refine_query] '{current_query}' → {refined_queries}")

        return dedent(
            f"""
            GLOBAL QUERY REFINEMENTS
            Original Query: "{current_query}"
            Refinement Reason: {refinement_reason}
            Refined Queries: {refined_queries}
            Current Frontier: {list(self.frontier)}
            Use these globally refined queries in your next searches.
            """
        ).strip()

    @validate_input_query_in_frontier_or_explored
    @validate_output_query_not_explored
    def global_expand_query(
        self,
        question: str,
        current_query: str,
        expansion_strategy: str,
        expanded_queries: list[str],
    ) -> str:
        r"""Expand the query with additional terms without using search results (GLOBAL EXPAND).

        This tool helps you add synonyms, related terms, morphological variants, or
        context to improve search coverage. Generate multiple expanded versions using
        different strategies.

        Expansion strategies:
        - Add synonyms or alternative terms
        - Include related concepts or domains
        - Add temporal context (years, periods)
        - Include morphological variants
        - Add context from your knowledge
        - Include related entities or organizations
        'current_query' must be an existing query (either in frontier or explored).

        Args:
            question (str): The original research question
            current_query (str): The current search query
            expansion_strategy (str): What type of expansion you're applying and why
            expanded_queries (List[str]): Multiple expanded queries with additional
                relevant terms
        Returns:
            str: Confirmation of your expanded queries
        """
        # Add all expanded queries to the frontier
        self.frontier.update(expanded_queries)

        logger.info(f"[global_expand_query] '{current_query}' → {expanded_queries}")

        return dedent(
            f"""
            GLOBAL QUERY EXPANSIONS
            Original Query: "{current_query}"
            Expansion Strategy: {expansion_strategy}
            Expanded Queries: {expanded_queries}
            Current Frontier: {list(self.frontier)}
            Use these globally expanded queries to cast a wider search net."""
        ).strip()

    def get_tools(self) -> list[FunctionTool]:
        r"""Returns all available tools in the toolkit."""
        return [
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.extract_relevant_details),
            FunctionTool(self.analyze_search_progress),
            FunctionTool(self.local_expand_query),
            FunctionTool(self.local_refine_query),
            FunctionTool(self.global_refine_query),
            FunctionTool(self.global_expand_query),
        ]
