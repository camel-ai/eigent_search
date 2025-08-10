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

from __future__ import annotations

import networkx as nx
import logging
from functools import wraps

from camel.toolkits import FunctionTool, BaseToolkit, SearchToolkit, ThinkingToolkit

logger = logging.getLogger(__name__)


def validate_input_query_in_frontier(func):
    """Decorator to validate that the input query is in the current frontier.

    This decorator should be applied to methods that take a query parameter
    and need to ensure it exists in the current frontier before processing.
    """

    @wraps(func)
    def wrapper(self, **kwargs):
        # Get the query parameter from kwargs
        query = kwargs["query"]
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
            "rewrite_query": "rewritten_query",
            "expand_query": "expanded_queries",
            "select_query_and_search": "final_query",
            "generate_new_queries": "new_queries",
        }
        output_param = output_params[func.__name__]
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
    """Toolkit for processing queries for deep research.

    This toolkit provides methods for query rewriting, expansion, selection, and generation. Each method follows a pattern where it takes the original input and the intended output as parameters, then returns the intended output. This allows LLMs to understand the expected behavior through the docstring and call the function appropriately.
    """

    def __init__(self, initial_query: str) -> None:
        super().__init__()
        self.initial_query = initial_query
        self.frontier = {initial_query}  # set of queries to be explored
        self.explored = set()  # set of queries that have been explored
        self.trace_graph = QueryProcessingGraph(initial_query)
        self.search_tool = SearchToolkit().search_google
        self.search_counter = 0  # For counting the number of searches

    def get_frontier_str(self) -> str:
        """Display the current frontier as a string."""
        return "📋 current frontier:\n  - " + "\n  - ".join(list(self.frontier))

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def rewrite_query(self, query: str, rewritten_query: str) -> dict[str, list[str]]:
        """Rewrite a selected query from the current frontier to improve search effectiveness.

        This step is OPTIONAL and should only be used when the input query is vague, fuzzy, or doesn't clearly convey the user's intent. For clear, specific queries, this step can be skipped. The agent should provide a more specific, detailed, or differently phrased version of the original query when rewriting is needed.

        Args:
            query (str): The input query from the current frontier that requires rewriting.
            rewritten_query (str): The intended rewritten version of the query.

        Returns:
            dict[str, list[str]]: The current frontier after the rewriting process.
        """
        # Add the rewritten query to the frontier if it has passed the frontier and explored validation
        self.frontier.add(rewritten_query)
        self.trace_graph.record_process(query, rewritten_query, "rewrite_query")
        return {"frontier": list(self.frontier)}

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def expand_query(
        self, query: str, expanded_queries: list[str]
    ) -> dict[str, list[str]]:
        """Expand the input query selected from the current frontier into multiple related search queries.

        The agent should expand queries using two main strategies:

        (1) associations - related concepts, synonyms, broader/narrower terms, or
        (2) decomposition - breaking down complex queries into simpler, more focused components.

        This helps capture different aspects and variations of the input query for comprehensive research.

        Args:
            query (str): The input query from the current frontier that requires expanding.
            expanded_queries (list[str]): A list of expanded queries.

        Returns:
            dict[str, list[str]]: The current frontier after the expanding process.
        """
        # Add the new queries to the frontier if they have passed the frontier and explored validation
        self.frontier.update(expanded_queries)
        for new_query in expanded_queries:
            self.trace_graph.record_process(query, new_query, "expand_query")
        return {"frontier": list(self.frontier)}

    @validate_input_query_in_frontier
    @validate_output_query_not_explored
    def select_query_and_search(
        self, query: str, final_query: str
    ) -> dict[str, list[str]]:
        """Select the best query from the current frontier and perform web search.

        The agent should choose based on specificity, clarity, and search potential, in order to minimize the number of searches and the cost of the search. Then, OPTIONALLY add advanced search operators (AND, OR, NOT, quotes, site:, filetype:, etc.) to improve search precision and relevance. Finally, perform an actual web search using the final query and return the results. The agent should return the search results as a list of strings.

        If the search results are not sufficient to answer the user's initial query, the agent should process and select another query from the current frontier and perform web search again, or generate new queries based on the search results.

        Args:
            query (str): The input query from the current frontier that is selected for web search.
            final_query (str): The final query with optional advanced search operators added to the selected query that will be used for searching the web.

        Returns:
            dict[str, list[str]]: The search results from the web search.
        """

        # NOTE: prevent searching huggingface website to avoid answer leakage
        search_results = self.search_tool(final_query + " -site:huggingface.co")
        self.search_counter += 1
        self.trace_graph.record_process(query, final_query, "enhance_query_for_search")

        # If search on the final query returns an error, handle it gracefully
        if "error" in search_results[0]:
            error_message = search_results[0]["error"]
            self.explored.add(final_query)

            # Fall back to the original query, if enhanced query is different from the original query
            if final_query != query:
                self.explored.add(final_query)
                logger.info(
                    f"[select_query_and_search] No valid search results found for the enhanced query: {final_query}. Falling back with the original query: {query}."
                )
                search_results_orig = self.search_tool(query + " -site:huggingface.co")
                self.search_counter += 1
                self.trace_graph.record_process(
                    query, query, "enhance_query_for_search"
                )

                if "error" in search_results_orig[0]:
                    error_message_orig = search_results_orig[0]["error"]
                    logger.error(
                        f"[select_query_and_search] Error searching original query: {query}. Error message from search tool:{error_message_orig}"
                    )
                    self.explored.add(query)
                    self.frontier.remove(query)
                    return {"None": error_message_orig}

                for result_orig in search_results_orig:
                    self.trace_graph.record_process(
                        query, str(result_orig["url"]), "search_google"
                    )

                self.explored.add(query)
                self.frontier.remove(query)
                return {
                    "search_results": [str(result) for result in search_results_orig]
                }

            # If the final query is the same as the original query, return an error message
            else:
                logger.error(
                    f"[select_query_and_search] Error searching both enhanced and original query: {query}. Error message from search tool: {error_message}"
                )
                return {"None": error_message}

        for result in search_results:
            if "url" in result:
                self.trace_graph.record_process(
                    final_query, str(result["url"]), "search_google"
                )
            else:
                logger.warning(f"Unexpected search result:{str(result)}")

        # Modify the frontier and explored set after the search
        self.explored.add(query)
        self.frontier.remove(query)
        if final_query != query:
            self.explored.add(final_query)

        return {"search_results": [str(result) for result in search_results]}

    def _extract_urls_from_search_results(
        self, search_results: list[str]
    ) -> list[str] | str:
        """Extract URLs from the search results formatted as JSON strings.

        Args:
            search_results (list[str]): The search results from the web search, each formatted as a JSON string.

        Returns:
            list[str] | str: A list of URLs extracted from the search results, or an error message string.
        """
        import ast

        # Extract URLs from search results
        urls = []
        for i, result_str in enumerate(search_results, 1):
            # 1) Check if result_str is JSON-like format
            json_validation_error = self._validate_json_format(result_str, i)
            if json_validation_error:
                logger.error(json_validation_error)
                return json_validation_error

            try:
                # Parse the string representation of the dictionary
                result_dict = ast.literal_eval(result_str)

                # Validate it's actually a dictionary
                if not isinstance(result_dict, dict):
                    error_message = f"ERROR: Search result {i} parsed successfully but is not a dictionary. Got {type(result_dict).__name__}: {str(result_dict)[:100]}..."
                    logger.error(error_message)
                    return error_message

                # 2) Check if 'url' field is present in the search result
                if "url" not in result_dict:
                    available_keys = list(result_dict.keys())
                    error_message = f"ERROR: Search result {i} is missing required 'url' field. Available fields: {available_keys}. Content: {str(result_dict)[:200]}..."
                    logger.error(error_message)
                    return error_message

                # Validate URL value
                url_value = result_dict["url"]
                if not isinstance(url_value, str) or not url_value.strip():
                    error_message = f"ERROR: Search result {i} has invalid 'url' field. Expected non-empty string, got {type(url_value).__name__}: '{url_value}'"
                    logger.error(error_message)
                    return error_message

                # URL looks good, add it to the list
                urls.append(url_value.strip())

            except (ValueError, SyntaxError) as e:
                # Parsing failed after JSON format validation
                error_message = f"ERROR: Search result {i} failed to parse despite passing JSON validation. Parse error: {str(e)[:100]}... Original: {result_str[:200]}..."
                logger.error(error_message)
                return error_message

        # Success - return the extracted URLs
        return urls

    def _validate_json_format(self, result_str: str, result_num: int) -> str | None:
        """Validate that a result string looks like JSON format.

        Args:
            result_str: The string to validate
            result_num: The result number for error messages

        Returns:
            str | None: Error message if invalid, None if valid
        """
        # Check basic type and content
        if not isinstance(result_str, str):
            return f"ERROR: Search result {result_num} must be a string, got {type(result_str).__name__}: {result_str}"

        if not result_str or not result_str.strip():
            return f"ERROR: Search result {result_num} is empty or whitespace-only"

        result_str = result_str.strip()

        # Must look like a dictionary (starts with { and ends with })
        if not (result_str.startswith("{") and result_str.endswith("}")):
            return f"ERROR: Search result {result_num} is not in dictionary format. Expected format: {{'key': 'value', ...}}. Got: {result_str[:100]}..."

        # Check for basic JSON structure patterns
        import re

        # Should contain key-value patterns
        if not re.search(r'["\'][\w\s_-]+["\']:\s*["\']', result_str):
            return f"ERROR: Search result {result_num} does not contain valid key-value pairs. Expected format like 'key': 'value'. Got: {result_str[:100]}..."

        # Check for balanced braces (basic check)
        open_braces = result_str.count("{")
        close_braces = result_str.count("}")
        if open_braces != close_braces:
            return f"ERROR: Search result {result_num} has unbalanced braces. Found {open_braces} opening '{{' and {close_braces} closing '}}'. Content: {result_str[:200]}..."

        # Check for the problematic repeated bracket pattern
        if re.search(r"(\}|\]){3,}", result_str):
            return f"ERROR: Search result {result_num} contains repeated closing brackets/braces (like }}}}}} or ]]]]]). This indicates malformed JSON. Content: {result_str[:200]}..."

        # Check for required field patterns (should contain 'url' somewhere)
        if "'url'" not in result_str and '"url"' not in result_str:
            return f"ERROR: Search result {result_num} does not appear to contain a 'url' field. Expected search results should have 'url' field. Content: {result_str[:200]}..."

        # Looks valid
        return None

    @validate_output_query_not_explored
    def generate_new_queries(
        self, search_results: list[str], new_queries: list[str]
    ) -> dict[str, list[str]] | str:
        """Generate new queries when the search results are not sufficient to answer the user's initial query.

        The agent should reflect on the search results and generate new queries to continue the deep research.

        The search_results should be formatted as strings representing JSON objects following the Google Search API schema. Do not make up search results, only use previously seen search results.
        Each search result must contain these fields:
        - result_id: Unique identifier (integer)
        - title: Title of the web page or document (string)
        - description: Brief description or snippet from the content (string)
        - long_description: Extended description, may be 'N/A' if not available (string)
        - url: Web address of the source (string)

        Example search result format:
        "{'result_id': 5, 'title': 'CIS Awards - IEEE Computational Intelligence Society', 'description': 'Prize items include a bronze medal, certificate and honorarium. View past IEEE Frank Rosenblatt Award Recipients (PDF) ... 2010 to 2021. He was a \"Finland\\xa0...', 'long_description': 'N/A', 'url': 'https://cis.ieee.org/awards/13-cis-awards'}"

        Args:
            search_results (list[str]): The search results from the web search.
            new_queries (list[str]): The new queries to be added to the frontier.

        Returns:
            dict[str, list[str]]: The current frontier after the generating process.
        """

        urls = self._extract_urls_from_search_results(search_results)
        # Check if urls is a string (error message)
        if isinstance(urls, str):
            return urls  # Return the error message directly

        self.frontier.update(new_queries)
        for url in urls:
            for new_query in new_queries:
                self.trace_graph.record_process(str(url), new_query, "generate_queries")
        return {"frontier": list(self.frontier)}

    def complete_task(
        self, search_results: list[str], final_answer: str
    ) -> dict[str, str | list[str]] | str:
        """Complete the deep research when search results are sufficient to answer the user's initial query.

        The agent should return the final answer and the search results to terminate the deep research.

        The search_results should be formatted as strings representing JSON objects following the Google Search API schema. Do not make up search results, only use previously seen search results.
        Each search result must contain these fields:
        - result_id: Unique identifier (integer)
        - title: Title of the web page or document (string)
        - description: Brief description or snippet from the content (string)
        - long_description: Extended description, may be 'N/A' if not available (string)
        - url: Web address of the source (string)

        Example search result format
        "{'result_id': 5, 'title': 'CIS Awards - IEEE Computational Intelligence Society', 'description': 'Prize items include a bronze medal, certificate and honorarium. View past IEEE Frank Rosenblatt Award Recipients (PDF) ... 2010 to 2021. He was a \"Finland\\xa0...', 'long_description': 'N/A', 'url': 'https://cis.ieee.org/awards/13-cis-awards'}"

        Args:
            search_results (list[str]): The search results from web search, each formatted as a JSON string with the above schema.
            final_answer (str): The final answer to the user's initial query.

        Returns:
            dict[str, str | list[str]]: The final answer and the search results.
        """

        urls = self._extract_urls_from_search_results(search_results)
        # Check if urls is a string (error message)
        if isinstance(urls, str):
            return urls  # Return the error message directly

        for url in urls:
            self.trace_graph.record_process(str(url), final_answer, "complete_task")

        return {"answer": final_answer, "search_results": search_results}

    def get_tools(self) -> list[FunctionTool]:
        """Get the tools for the query processing toolkit."""
        return [
            FunctionTool(self.rewrite_query),
            FunctionTool(self.expand_query),
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.generate_new_queries),
            FunctionTool(self.complete_task),
            FunctionTool(ThinkingToolkit().think),
            FunctionTool(ThinkingToolkit().reflect),
        ]


class QueryProcessingGraph:
    """A graph representing the query processing process.

    The `initial_query` is the root node of the graph. The agent will call the `QueryProcessingToolkit` to process the query and return the processed query or queries. This process will be recorded as a node in the graph.
    """

    def __init__(self, initial_query: str):
        self.initial_query = initial_query
        self._graph = nx.DiGraph()  # node_id to query mapping
        self._data_to_node = {}  # map query strings to their latest node IDs
        self.new_node_id = 0
        # Add the initial query as the root node
        # NOTE: the initial query is the root node of the graph
        self._graph.add_node(self.new_node_id, data=initial_query)
        self._data_to_node[initial_query] = self.new_node_id
        self.new_node_id += 1

    @property
    def trace_graph(self) -> nx.DiGraph:
        """Get the trace graph."""
        return self._graph

    def record_process(self, from_data: str, to_data: str, action: str) -> str:
        """Record a process in the trace graph.

        Args:
            from_data (str): The source data.
            to_data (str): The target data after processing.
            action (str): The type of processing action performed.

        Returns:
            str: The node ID of the newly created target node.
        """
        # O(1) lookup for source node ID using the mapping
        source_node_id = self._data_to_node.get(from_data)
        target_node_id = self._data_to_node.get(to_data)

        # If source not found, create it (shouldn't happen in normal usage)
        if source_node_id is None:
            source_node_id = self.new_node_id
            self._graph.add_node(source_node_id, data=from_data)
            self._data_to_node[from_data] = source_node_id
            self.new_node_id += 1

        # Add the target node
        if target_node_id is None:
            target_node_id = self.new_node_id
            self._graph.add_node(target_node_id, data=to_data)
            self._data_to_node[to_data] = target_node_id
            self.new_node_id += 1

        # Add the edge
        self._graph.add_edge(source_node_id, target_node_id, action=action)

        # Log the process
        logger.info(f"[{action}]: '{from_data}' -> '{to_data}'")

        return str(target_node_id)

    def render_trace_graph(self) -> str:
        """Render the process graph as a string representation."""
        if not self._graph.nodes():
            return "Empty graph"

        def format_data(data):
            """Format data for display, truncating long strings and handling complex objects."""
            if isinstance(data, str):
                # Truncate long strings
                if len(data) > 50:
                    return f'"{data[:47]}..."'
                return f'"{data}"'
            elif isinstance(data, list):
                if len(data) > 3:
                    return f"[{len(data)} items: {format_data(data[0])}, {format_data(data[1])}, ...]"
                return f"[{', '.join(format_data(item) for item in data)}]"
            elif isinstance(data, dict):
                return f"{{dict with {len(data)} keys}}"
            else:
                return str(data)

        def find_roots(graph):
            """Find all root nodes (nodes with no predecessors)."""
            return [n for n in graph.nodes() if not list(graph.predecessors(n))]

        def render_component(graph, root, visited, indent="  "):
            """Render a single connected component starting from root."""
            if root in visited:
                return []

            visited.add(root)
            node_data = graph.nodes[root].get("data", "")
            formatted_data = format_data(node_data)

            # Use human-readable node ID for display
            readable_id = str(root)
            lines = [f"{indent}{readable_id}: {formatted_data}\n"]

            # Get all successors and sort them for consistent output
            successors = sorted(graph.successors(root))
            for i, successor in enumerate(successors):
                edge_data = graph.edges[root, successor].get("action", "")
                is_last = i == len(successors) - 1

                # Add edge indicator
                prefix = "└─" if is_last else "├─"
                lines.append(f"{indent}{prefix}[{edge_data}]─>\n")

                # Handle cycles and already visited nodes differently
                if successor not in visited:
                    child_indent = indent + ("    " if is_last else "│   ")
                    lines.extend(
                        render_component(graph, successor, visited, child_indent)
                    )
                else:
                    # Show what we're cycling back to
                    successor_data = graph.nodes[successor].get("data", "")
                    formatted_successor = format_data(successor_data)
                    lines.append(
                        f"{indent}{'    ' if is_last else '│   '}↺ {successor}: {formatted_successor} (back reference)\n"
                    )

            return lines

        # Find all root nodes
        roots = find_roots(self._graph)

        if not roots:
            # If no roots found, start from any node (for cyclic graphs)
            roots = [list(self._graph.nodes())[0]]

        # Render all components
        visited = set()
        result_lines = []

        for i, root in enumerate(roots):
            if i > 0:
                result_lines.append("\n")  # Separate components
            result_lines.extend(render_component(self._graph, root, visited))

        return "".join(result_lines)
