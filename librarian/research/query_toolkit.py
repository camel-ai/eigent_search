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
from .web_content_toolkit import WebContentToolkit

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
        self.web_content_toolkit = WebContentToolkit()  # Available for agent to use
        self.content_extraction_counter = 0  # For counting content extractions
        self.extracted_contents = {}  # Store extracted content by URL

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
        self, query: str, enhanced_query: str
    ) -> dict[str, dict[str, str]]:
        """Select the best query from the current frontier and perform web search.

        The agent should choose based on specificity, clarity, and search potential, in order to minimize the number of searches and the cost of the search. Then, OPTIONALLY add advanced search operators (AND, OR, NOT, quotes, site:, filetype:, etc.) to improve search precision and relevance. Finally, perform an actual web search using the enhanced query first, and fall back to the original query if the enhanced query fails.

        If the search results are not sufficient to answer the user's initial query, the agent should process and select another query from the current frontier and perform web search again, or generate new queries based on the search results.

        Args:
            query (str): The input query from the current frontier that is selected for web search.
            enhanced_query (str): The enhanced query with optional advanced search operators added to the selected query that will be used for searching the web. If the enhanced query leads to an error in search, search results of the original query will be returned.

        Returns:
            dict[str, dict[str, str]]: The search results from the web search. The key is the URL and the value is the string of the title, description, and long description of the result. If the search fails, the key is "None" and the value is the error message.
        """
        # Record enhancement and add to frontier if needed
        if enhanced_query != query:
            self.trace_graph.record_process(
                query, enhanced_query, "enhance_query_for_search"
            )
            self.frontier.add(enhanced_query)
        else:
            # If queries are identical, no need to search twice
            logger.info(
                "[select_query_and_search] Query and enhanced query are identical, performing single search"
            )

        # Update frontier and explored sets; the search will be conducted anyway
        for q in [enhanced_query, query]:
            if q in self.frontier:
                self.explored.add(q)
                self.frontier.remove(q)

        # Helper function to perform search and handle results
        def search_and_record(query_str: str, action: str = "search_google"):
            results = self.search_tool(query_str + " -site:huggingface.co")
            self.search_counter += 1
            # Check if search has returned anything valid
            if "error" in results[0]:
                self.trace_graph.record_process(query_str, results[0]["error"], action)
                return {"None": results[0]["error"]}
            
            # linearize valid search results to dictionary of strings
            results: dict[str, str] = {
                result["url"]: (
                    f"Title: {result['title']}\n"
                    f"Description: {result['description']}\n"
                    f"Long Description: {result['long_description']}"
                )
                for result in results
            }
            # Record results in trace graph
            for url in results.keys():
                self.trace_graph.record_process(query_str, url, action)
            return results

        # Try enhanced query first
        enhanced_results = search_and_record(enhanced_query)
        if "None" not in enhanced_results:
            return {"search_results": enhanced_results}
        else:
            if query != enhanced_query:
                # Fall back to original query
                self.trace_graph.record_process(enhanced_query, query, "query_fallback")
                return {"search_results": search_and_record(query)}
            else:
                return {"search_results": enhanced_results}

    @validate_output_query_not_explored
    def generate_new_queries(
        self, search_results: dict[str, str] | None, new_queries: list[str]
    ) -> dict[str, list[str]] | str:
        """Generate new queries when the search results are not sufficient to answer the user's initial query.

        The agent should reflect on the search results and generate new queries to continue the deep research.

        The search_results are formatted as a dictionary of strings, where the key is the URL and the value is the string of the title, description, and long description of the result.

        Args:
            search_results (dict[str, str]): The search results from the web search.
            new_queries (list[str]): The new queries to be added to the frontier.

        Returns:
            dict[str, list[str]]: The current frontier after the generating process.
        """

        self.frontier.update(new_queries)
        for new_query in new_queries:
            for url in search_results.keys():
                self.trace_graph.record_process(url, new_query, "generate_new_queries")
        return {"frontier": list(self.frontier)}

    def extract_web_content(self, url: str) -> dict[str, str]:
        """Extract the main content from a web page URL.
        
        Use this tool when search result snippets don't contain enough detail to answer the query.
        This will fetch and extract the full content from the webpage.

        Args:
            url (str): The URL of the web page to extract content from.

        Returns:
            dict[str, str]: A dictionary containing:
                - "title": The page title
                - "content": The extracted main content (limited to 3000 characters)
                - "status": Success or error status
                - "word_count": Number of words in extracted content
        """
        extracted = self.web_content_toolkit.extract_web_content(url)
        self.content_extraction_counter += 1
        
        if extracted["status"].startswith("✅"):
            original_length = len(extracted["content"])
            # Store full content for complete_task to use
            self.extracted_contents[url] = extracted["content"]
            
            logger.info(f"[extract_web_content] Successfully extracted {extracted['word_count']} words from {url}")
            self.trace_graph.record_process(url, f"Extracted {extracted['word_count']} words", "extract_web_content")
        else:
            logger.warning(f"[extract_web_content] Failed to extract from {url}: {extracted['status']}")
            
        return extracted

    def complete_task(
        self, search_results: dict[str, str] | None, final_answer: str
    ) -> dict[str, str | list[str]] | str:
        """Complete the deep research when search results are sufficient to answer the user's initial query.

        IMPORTANT: This method should only be called after extract_web_content has been used to get 
        detailed information from the most promising URLs. 

        Args:
            search_results (dict[str, str]): The search results from web search and extract_web_content. 
            final_answer (str): The final answer to the user's initial query.
        
        Example:
            https://en.wikipedia.org/wiki/Isaac_Newton: https://en.wikipedia.org/wiki/Isaac_Newton


        Returns:
            dict[str, str | list[str]]: The final answer and the search results. key is url, value is the quote supporting the answer. MUST be original quotes from the web page, not paraphrased.
        """
        
        for url in search_results.keys():
            self.trace_graph.record_process(url, final_answer, "complete_task")

        return {"answer": final_answer, "search_results": search_results}

    def get_tools(self) -> list[FunctionTool]:
        """Get the tools for the query processing toolkit."""
        return [
            FunctionTool(self.rewrite_query),
            FunctionTool(self.expand_query),
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.generate_new_queries),
            FunctionTool(self.extract_web_content),
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
