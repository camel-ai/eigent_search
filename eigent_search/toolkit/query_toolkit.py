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

from camel.logger import get_logger
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits.base import BaseToolkit
from camel.toolkits.search_toolkit import SearchToolkit


logger = get_logger(__name__)


def validate_output_query_not_explored(func):
    """Decorator to validate that output queries haven't been explored.

    Filters out any queries that have already been searched.
    """
    @wraps(func)
    def wrapper(self, **kwargs):
        output_param = "search_queries"
        output_queries = kwargs.get(output_param)

        if output_queries:
            # Filter out already-explored queries
            filtered_queries = [
                q for q in output_queries if q not in self.explored
            ]

            if not filtered_queries:
                error_message = (
                    f"❌ Invalid operation: All proposed queries have already been explored.\n"
                    f"Explored queries: {list(self.explored)[:10]}..."
                )
                logger.error(error_message)
                return error_message

            # Update kwargs with filtered queries
            kwargs[output_param] = filtered_queries

        return func(self, **kwargs)

    return wrapper


class QueryProcessingToolkit(BaseToolkit):
    r"""A unified query processing toolkit for web search with configurable pages.

    This toolkit provides:
    - Single query planning tool (plan_next_searches)
    - Strong cognitive scaffolding (forces progress tracking and gap identification)
    - Configurable number of result pages (agent can control via select_query_and_search)
    - Extract and analyze tools for information tracking
    - Natural knowledge integration (no current_query constraint)
    """

    def __init__(self, exclude_domains: list[str] | None = None):
        super().__init__()
        self.frontier = set()
        self.explored = set()
        self.initial_query = None
        self._search_toolkit = SearchToolkit(exclude_domains=exclude_domains)
        self.search_counter = 0

    def load_initial_query(self, initial_query: str):
        """Reset with the new initial query."""
        self.initial_query = initial_query
        self.frontier = {initial_query}
        self.explored = set()
        return (
            f"Initial query: {initial_query}\nCurrent Frontier:\n  - "
            + "\n  - ".join(list(self.frontier))
        )

    def get_frontier_str(self) -> str:
        """Display the current frontier as a string."""
        return "Current Frontier:\n  - " + "\n  - ".join(list(self.frontier))

    def get_explored_str(self) -> str:
        """Display the explored queries as a string."""
        return "Explored Queries:\n  - " + "\n  - ".join(list(self.explored))

    def select_query_and_search(self, query: str, number_of_result_pages: int = 10) -> dict[str, dict[str, str]]:
        """Select the best query from the current frontier and perform web search.
        The agent should select or generate query based on specificity, clarity, and search potential,
        in order to minimize the number of searches and the cost of the search.
        If the search results are not sufficient to answer the user's initial query,
        the agent should process and select another query from the current frontier
        and perform web search again, or generate new queries based on the search results.

        Args:
            query (str): The input query from the current frontier that is selected
                for web search.
            number_of_result_pages (int): The number of result pages to
                retrieve. Must be a positive integer between 1 and 10.
                Google Custom Search API limits results to 10 per request.
                If a value greater than 10 is provided, it will be capped
                at 10 with a warning. Adjust this based on your task - use
                fewer results for focused searches and more for comprehensive
                searches. (default: :obj:`10`)
        Returns:
            dict[str, dict[str, str]]: The search results from the web search. The key is "search_results" and the value is a dict where each key is a URL and each value is the string of the title, description, and long description of the result. If the search fails, the key is "None" and the value is the error message.
        """
        # Validate and cap number_of_result_pages
        if number_of_result_pages < 1:
            logger.warning(f"number_of_result_pages must be at least 1, got {number_of_result_pages}. Using 1.")
            number_of_result_pages = 1
        elif number_of_result_pages > 10:
            logger.warning(f"number_of_result_pages capped at 10 (requested {number_of_result_pages})")
            number_of_result_pages = 10

        # Update frontier and explored sets
        if query in self.frontier:
            self.frontier.remove(query)
        self.explored.add(query)

        # Helper function to perform search and handle results
        def search_and_record(query_str: str, num_pages: int):
            results = self._search_toolkit.search_google(
                query=query_str,
                search_type="web",
                number_of_result_pages=num_pages,
                start_page=1,
            )
            self.search_counter += 1

            # Check if search returned empty results
            if not results:
                logger.warning(f"[search] No results found for '{query_str}'")
                return {
                    "None": f"No search results found for query: '{query_str}'. "
                    f"This may indicate the query is too specific, uses uncommon terms, "
                    f"or targets information that doesn't exist on the web. "
                    f"Consider refining the query with more common terms or trying alternative phrasings."
                }

            # Check if search has returned an error
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
        return {"search_results": search_and_record(query, number_of_result_pages)}

    @validate_output_query_not_explored
    def plan_next_searches(
        self,
        question: str,
        current_understanding: str,
        missing_information: str,
        search_queries: list[str],
    ) -> str:
        r"""Plan your next search queries based on research gaps, and add them to the frontier.

        This tool provides structured space for deliberate search planning.
        Use it whenever you need to search for information.

        Forces you to:
        1. Review what you've learned (current_understanding)
        2. Identify what's still missing (missing_information)
        3. Generate strategic queries targeting the gaps (search_queries)

        Args:
            question (str): The original research question you're trying to answer

            current_understanding (str): What you know so far about the question.
                Be specific - what facts, entities, relationships, or details have you found?
                Include ALL relevant information discovered from searches and page visits.
                Example: "Found that Tim Cook is Apple CEO since 2011. He studied
                         industrial engineering at Auburn University (BS 1982).
                         Previously worked at IBM for 12 years."

            missing_information (str): What EXACTLY is still missing to answer the question?
                Be precise - what specific facts, numbers, dates, names, or details do you need?
                Example: "Still need: (1) His specific roles at IBM (job titles, departments),
                         (2) Which other companies he worked at before Apple,
                         (3) Exact date he joined Apple"

            search_queries (List[str]): 3-6 search queries that target the missing information.
                Each query should approach the gaps from different angles or phrasings.
                Queries can incorporate newly discovered information (not constrained to
                deriving from previous queries).
                Example: ["Tim Cook IBM career job title",
                         "Tim Cook career history before Apple CEO",
                         "Tim Cook employment Compaq 1997-1998"]

        Returns:
            str: Confirmation that queries were added to frontier with summary
        """
        # Add all queries to the frontier
        self.frontier.update(search_queries)

        logger.info(f"[plan_next_searches] Added {len(search_queries)} queries to frontier")

        frontier_list = list(self.frontier)
        # Format all frontier queries with clear delimiters
        frontier_items = "\n".join(f'- {q}' for q in frontier_list)

        # Build the message without dedent to avoid indentation issues
        return (
            f"Queries added to frontier. Current Frontier ({len(frontier_list)} total queries):\n"
            f"{'='*60}\n"
            f"{frontier_items}\n"
            f"{'='*60}"
        )

    def extract_relevant_details(
        self,
        query: str,
        question: str,
        relevant_information: str,
        snapshot: str = "",
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
            query (str): The search query you used to find this page
            question (str): The original question - what EXACTLY does it ask for?
            relevant_information (str): Information you extract (must precisely match
                question requirements)
            snapshot (str, optional): The complete page content from browser_visit_page
            page_url (str, optional): The URL of the page
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

    def get_tools(self) -> list[FunctionTool]:
        r"""Returns all available tools in the toolkit."""
        return [
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.plan_next_searches),
            FunctionTool(self.extract_relevant_details),
            FunctionTool(self.analyze_search_progress),
        ]
