"""Wrapper for HybridBrowserToolkit that tracks visited pages and provides reminders."""

import logging
import functools
import inspect
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, urlunparse
from collections import defaultdict
from datetime import datetime
from camel.toolkits import HybridBrowserToolkit
from camel.toolkits.function_tool import FunctionTool

if TYPE_CHECKING:
    from camel.toolkits import FunctionTool

# Set up logging for the wrapper
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def track_metrics(func):
    """Decorator to automatically track metrics for browser functions.
    
    This decorator:
    1. Captures function name, arguments, and response
    2. Records metrics automatically
    3. Handles both sync and async functions
    """
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        # Get function name and prepare args dict
        func_name = func.__name__
        
        # Get argument names from function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        
        # Remove 'self' from arguments
        args_dict = dict(bound_args.arguments)
        args_dict.pop('self', None)
        
        # Truncate long values in args
        truncated_args = {}
        for k, v in args_dict.items():
            if isinstance(v, str) and len(v) > 100:
                truncated_args[k] = v[:100] + "..."
            else:
                truncated_args[k] = v
        
        # Call the original function
        result = await func(self, *args, **kwargs)
        
        # Prepare response summary
        response_summary = "N/A"
        if isinstance(result, dict):
            if 'result' in result:
                response_summary = str(result['result'])[:100]
            elif 'blocked' in result and result['blocked']:
                response_summary = f"BLOCKED: {result.get('reason', 'unknown')}"
            elif 'snapshot' in result:
                response_summary = f"Snapshot: {result['snapshot'][:50]}..."
        elif isinstance(result, str):
            response_summary = result[:100]
        
        # Record the metric
        self._record_metric(func_name, truncated_args, response_summary)
        
        return result
    
    return async_wrapper


class BrowserToolkitWrapper:
    """Wrapper for HybridBrowserToolkit that tracks page visits and provides reminders."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the wrapper with the same arguments as HybridBrowserToolkit.
        
        Args:
            *args: Positional arguments for HybridBrowserToolkit
            **kwargs: Keyword arguments for HybridBrowserToolkit
                domain_blacklist: Optional list of domains to block (e.g., ['huggingface.co', 'hf.co'])
        """
        logger.info("[BrowserWrapper] Initializing BrowserToolkitWrapper")
        
        # Extract our custom kwargs before passing to HybridBrowserToolkit
        self.domain_blacklist = kwargs.pop('domain_blacklist', ['huggingface.co', 'hf.co'])
        
        self.toolkit = HybridBrowserToolkit(*args, **kwargs)
        self.visit_history: Dict[str, List[datetime]] = defaultdict(list)
        self.page_content_cache: Dict[str, str] = {}
        self.max_cache_size = 10  # Maximum number of pages to cache
        
        # Usage metrics tracking
        self.usage_metrics: List[Dict[str, Any]] = []
        
        logger.info(f"[BrowserWrapper] Wrapper initialized with {len(kwargs.get('enabled_tools', []))} tools")
        logger.info(f"[BrowserWrapper] Domain blacklist: {self.domain_blacklist}")
        
    def _record_metric(self, function_name: str, args: Dict[str, Any], response_summary: str):
        """Record usage metrics for a function call.
        
        Args:
            function_name: Name of the function called
            args: Arguments passed to the function
            response_summary: Short summary of the response
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "function": function_name,
            "args": args,
            "response_summary": response_summary[:100] if isinstance(response_summary, str) else str(response_summary)[:100]
        }
        self.usage_metrics.append(metric)
    
    def get_usage_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded usage metrics.
        
        Returns:
            List of metrics dictionaries
        """
        return self.usage_metrics.copy()
    
    def clear_metrics(self):
        """Clear all usage metrics."""
        self.usage_metrics.clear()
    
    def _is_url_blacklisted(self, url: str) -> bool:
        """Check if a URL contains any blacklisted domain.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is blacklisted, False otherwise
        """
        url_lower = url.lower()
        for domain in self.domain_blacklist:
            if domain.lower() in url_lower:
                return True
        return False
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison purposes.
        
        Removes fragments, converts to lowercase domain, and removes trailing slashes.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL string
        """
        try:
            parsed = urlparse(url.lower())
            # Remove fragment and normalize path
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path.rstrip('/') if parsed.path else '',
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            return normalized
        except Exception:
            return url.lower()
    
    def _get_visit_count(self, url: str) -> int:
        """Get the number of times a URL has been visited.
        
        Args:
            url: The URL to check
            
        Returns:
            Number of visits
        """
        normalized_url = self._normalize_url(url)
        return len(self.visit_history[normalized_url])
    
    def _add_reminder_to_result(self, result: Any, url: str, visit_count: int) -> Any:
        """Add a reminder message to the result if the page has been visited before.
        
        Args:
            result: The original result from browser_visit_page
            url: The URL that was visited
            visit_count: Number of times this URL has been visited
            
        Returns:
            Modified result with reminder message
        """
        if visit_count > 1:
            reminder = f"\n⚠️ REMINDER: This page ({url}) has been visited {visit_count} times already."
            
            if visit_count == 2:
                reminder += " Consider if you need different information or should try a different approach."
            elif visit_count >= 3:
                reminder += " Multiple visits detected - you may be stuck in a loop. Try a different search query or website."
            
            # Add last visit times
            visits = self.visit_history[self._normalize_url(url)]
            if len(visits) > 1:
                last_visit = visits[-2]
                time_diff = (datetime.now() - last_visit).total_seconds()
                if time_diff < 60:
                    reminder += f" Last visited {time_diff:.0f} seconds ago."
                elif time_diff < 3600:
                    reminder += f" Last visited {time_diff/60:.0f} minutes ago."
            
            # Modify result based on type
            if isinstance(result, dict):
                if 'result' in result:
                    result['result'] = str(result['result']) + reminder
                else:
                    result['reminder'] = reminder
            elif isinstance(result, str):
                result = result + reminder
                
        return result
    
    @track_metrics
    async def browser_visit_page(self, url: str, *args, **kwargs) -> Dict[str, Any]:
        r"""Opens a URL in a new browser tab and switches to it.
        
        This wrapper adds visit tracking and provides reminders when pages are revisited.
        After 2 visits, the page will not be visited again and only a reminder is returned.

        Args:
            url (str): The web address to load. This should be a valid and
                existing URL.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action (may include reminder if revisiting).
                - "snapshot" (str): A textual snapshot of the new page.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the new active tab.
                - "total_tabs" (int): Total number of open tabs.
                - "reminder" (str, optional): Warning message if page was visited before.
        """
        # Check if URL is blacklisted
        if self._is_url_blacklisted(url):
            blacklisted_domains = [d for d in self.domain_blacklist if d.lower() in url.lower()]
            logger.warning(f"[BrowserWrapper] Blocked blacklisted URL: {url} (domains: {blacklisted_domains})")
            
            # Get current tab info if available
            try:
                session = await self.toolkit._get_session()
                tabs = await session.get_all_tabs()
                total_tabs = len(tabs)
            except:
                tabs = []
                total_tabs = 1
            
            return {
                "result": f"⛔ BLOCKED: Access to blacklisted domain(s) {blacklisted_domains} is not allowed. URL: {url}",
                "snapshot": f"[Content from {blacklisted_domains} blocked - please search for alternative sources]",
                "tabs": tabs,
                "total_tabs": total_tabs,
                "blocked": True,
                "reason": "blacklisted_domain"
            }
        
        normalized_url = self._normalize_url(url)
        
        # Check visit count before recording new visit
        current_visit_count = len(self.visit_history[normalized_url])
        
        # If already visited twice or more, don't visit again
        if current_visit_count >= 1:
            logger.warning(f"[BrowserWrapper] Page {url} already visited {current_visit_count} times - BLOCKING actual visit")
            
            # Get current tab info if available
            try:
                session = await self.toolkit._get_session()
                tabs = await session.get_all_tabs()
                total_tabs = len(tabs)
            except:
                tabs = []
                total_tabs = 1
            
            # Create a synthetic result without actually visiting
            blocked_result = {
                "result": f"⛔ BLOCKED: Page {url} has already been visited {current_visit_count} times. "
                         f"You are stuck in a loop visiting the same page repeatedly. "
                         f"You MUST NOT visit this page agin."
                         f"Try a different search query or website to find new information.",
                # "snapshot": self.page_content_cache.get(normalized_url, 
                        #    f"[No cached content available. Page was visited {current_visit_count} times previously.]"),
                "tabs": tabs,
                # "current_tab": current_tab_id,
                "total_tabs": total_tabs,
                "blocked": True,
                "visit_count": current_visit_count
            }
            
            # Still record this attempt for tracking
            self.visit_history[normalized_url].append(datetime.now())
            
            return blocked_result
        
        # Record the visit
        self.visit_history[normalized_url].append(datetime.now())
        visit_count = len(self.visit_history[normalized_url])
        
        logger.info(f"[BrowserWrapper] browser_visit_page called for: {url}")
        logger.info(f"[BrowserWrapper] Visit count for {normalized_url}: {visit_count}")
        
        # Call the original method
        result = await self.toolkit.browser_visit_page(url, *args, **kwargs)
        
        # Add reminder if needed
        result = self._add_reminder_to_result(result, url, visit_count)
        
        if visit_count > 1:
            logger.warning(f"[BrowserWrapper] Page {url} visited {visit_count} times - reminder added")
        
        # Cache management - cache first and second visits
        if visit_count <= 2 and isinstance(result, dict) and 'snapshot' in result:
            # Cache content for potential future blocking
            self.page_content_cache[normalized_url] = result['snapshot']
            # Limit cache size
            if len(self.page_content_cache) > self.max_cache_size:
                oldest_url = next(iter(self.page_content_cache))
                del self.page_content_cache[oldest_url]
        
        return result
    
    def get_visit_summary(self) -> str:
        """Get a summary of all visited pages.
        
        Returns:
            String summary of visit history
        """
        if not self.visit_history:
            return "No pages visited yet."
        
        summary = "Page Visit Summary:\n"
        summary += "-" * 40 + "\n"
        
        for url, visits in self.visit_history.items():
            summary += f"URL: {url}\n"
            summary += f"  Visits: {len(visits)}\n"
            if len(visits) > 1:
                summary += "  Visit times:\n"
                for i, visit_time in enumerate(visits, 1):
                    summary += f"    {i}. {visit_time.strftime('%H:%M:%S')}\n"
            summary += "\n"
        
        return summary
    
    def get_repeated_visits(self) -> List[str]:
        """Get list of URLs that have been visited more than once.
        
        Returns:
            List of URLs visited multiple times
        """
        return [url for url, visits in self.visit_history.items() if len(visits) > 1]
    
    def clear_history(self):
        """Clear the visit history and cache."""
        self.visit_history.clear()
        self.page_content_cache.clear()
    
    async def reset(self):
        """Reset the browser wrapper by closing browser and clearing history.
        
        This method:
        1. Closes the browser if it's open
        2. Clears all visit history
        3. Clears the page content cache
        4. Clears usage metrics
        
        Returns:
            str: Confirmation message
        """
        try:
            # Try to close the browser if it's open
            agent = self.toolkit._playwright_agent
            session = self.toolkit._session
            if agent is not None:
                await agent.close()
            if session is not None:
                await session.close()
        except Exception as e:
            logger.error(f"[BrowserWrapper] Error resetting browser: {e}")
        
        # Clear all tracking data
        self.clear_history()
        self.clear_metrics()
        
        return "Browser wrapper reset: browser closed, history and metrics cleared"
    
    # Delegate all other methods to the wrapped toolkit
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped toolkit.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute from the wrapped toolkit
        """
        return getattr(self.toolkit, name)
    
    @track_metrics
    async def browser_open(self, *args, **kwargs) -> Dict[str, Any]:
        r"""Starts a new browser session. This must be the first browser
        action.

        This method initializes the browser and navigates to a default start
        page. To visit a specific URL, use `visit_page` after this.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of interactive
                elements.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_open(*args, **kwargs)
    
    @track_metrics
    async def browser_close(self, *args, **kwargs) -> str:
        r"""Closes the browser session, releasing all resources.

        This should be called at the end of a task for cleanup.

        Returns:
            str: A confirmation message.
        """
        result = await self.toolkit.browser_close(*args, **kwargs)
        # Optionally clear history when browser closes
        # self.clear_history()
        return result
    
    @track_metrics
    async def browser_get_page_snapshot(self, *args, **kwargs) -> str:
        r"""Gets a textual snapshot of the page's interactive elements.

        The snapshot lists elements like buttons, links, and inputs,
        each with
        a unique `ref` ID. This ID is used by other tools (e.g., `click`,
        `type`) to interact with a specific element. This tool provides no
        visual information.

        Returns:
            str: A formatted string representing the interactive elements and
                their `ref` IDs. For example:
                '- link "Sign In" [ref=1]'
                '- textbox "Username" [ref=2]'
        """
        return await self.toolkit.browser_get_page_snapshot(*args, **kwargs)
    
    @track_metrics
    async def browser_click(self, *, ref: str) -> Dict[str, Any]:
        r"""Performs a click on an element on the page.

        Args:
            ref (str): The `ref` ID of the element to click. This ID is
                obtained from a page snapshot (`get_page_snapshot` or
                `get_som_screenshot`).

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the page after the
                  click.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_click(ref=ref)
    
    @track_metrics
    async def browser_scroll(self, *, direction: str, amount: int) -> Dict[str, Any]:
        r"""Scrolls the current page window.

        Args:
            direction (str): The direction to scroll: 'up' or 'down'.
            amount (int): The number of pixels to scroll.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A snapshot of the page after scrolling.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_scroll(direction=direction, amount=amount)
    
    @track_metrics
    async def browser_back(self, *args, **kwargs) -> Dict[str, Any]:
        r"""Goes back to the previous page in the browser history.

        This action simulates using the browser's "back" button in the
        currently active tab.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the previous page.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_back(*args, **kwargs)
    
    @track_metrics
    async def browser_forward(self, *args, **kwargs) -> Dict[str, Any]:
        r"""Goes forward to the next page in the browser history.

        This action simulates using the browser's "forward" button in the
        currently active tab.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the next page.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_forward(*args, **kwargs)
    
    @track_metrics
    async def browser_get_som_screenshot(
        self,
        read_image: bool = True,
        instruction: Optional[str] = None,
    ):
        r"""Captures a screenshot with interactive elements highlighted.

        "SoM" stands for "Set of Marks". This tool takes a screenshot and
        draws
        boxes around clickable elements, overlaying a `ref` ID on each. Use
        this for a visual understanding of the page, especially when the
        textual snapshot is not enough.

        Args:
            read_image (bool, optional): If `True`, the agent will analyze
                the screenshot. Requires agent to be registered.
            instruction (Optional[str]): Specific instruction for
                screenshot analysis.

        Returns:
            The screenshot result from the toolkit.
        """
        return await self.toolkit.browser_get_som_screenshot(
            read_image=read_image,
            instruction=instruction
        )
        
    def get_tools(self) -> List['FunctionTool']:
        """Get tools from the wrapped toolkit with our wrapper methods.
        
        This creates FunctionTool objects that point to our wrapper methods
        instead of the original toolkit methods, ensuring visit tracking works.
        """
        from camel.toolkits import FunctionTool
        
        # Get the original tools to see which ones are enabled
        original_tools = self.toolkit.get_tools()
        original_tool_names = [tool.func.__name__ for tool in original_tools]
        
        logger.info(f"[BrowserWrapper] get_tools called, original tools: {original_tool_names}")
        
        # Create wrapped tools that use our methods
        wrapped_tools = []
        
        # Map of method names to our wrapper methods
        wrapper_methods = {
            'browser_open': self.browser_open,
            'browser_close': self.browser_close,
            'browser_visit_page': self.browser_visit_page,  # This is the key one with tracking
            'browser_back': self.browser_back,
            'browser_forward': self.browser_forward,
            'browser_click': self.browser_click,
            'browser_get_page_snapshot': self.browser_get_page_snapshot,
            'browser_scroll': self.browser_scroll,
            'browser_get_som_screenshot': self.browser_get_som_screenshot,
        }
        
        # Only create tools for methods that were in the original enabled tools
        for tool_name in original_tool_names:
            if tool_name in wrapper_methods:
                wrapped_tools.append(FunctionTool(wrapper_methods[tool_name]))
                logger.info(f"[BrowserWrapper] Wrapped tool: {tool_name}")
            else:
                # For any tools we haven't wrapped, use the original
                # This ensures we don't break if new tools are added
                for original_tool in original_tools:
                    if original_tool.func.__name__ == tool_name:
                        wrapped_tools.append(original_tool)
                        logger.info(f"[BrowserWrapper] Using original tool: {tool_name}")
                        break
        
        logger.info(f"[BrowserWrapper] Returning {len(wrapped_tools)} wrapped tools")
        return wrapped_tools
