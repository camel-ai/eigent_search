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

from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Any, Callable, Optional

from camel.toolkits import FunctionTool, BaseToolkit, SearchToolkit


class ResearchToolkit(BaseToolkit):
    """Toolkit for deep research.

    This class provides methods for researching information on the web. A search tool that uses search engines like Google and DuckDuckGo is a must. After searching, a scrape tool can be used to read the content of the searched urls, or a browse tool can be used to browse the content of the searched urls. The main difference between scraping and browsing is that scraping can only read static html pages, while browsing can read dynamic pages with dom, js, etc.

    Args:
        search_tool (FunctionTool): A function that searches the web using a search engine.
        scrape_tool (FunctionTool, optional): A function that scrapes the web using a scraper.
        browse_tool (FunctionTool, optional): A function that browses the web using a browser.
        max_workers (int): The maximum number of workers to use for asynchronous tasks.
    """

    def __init__(
        self,
        search_tool: FunctionTool = SearchToolkit().search_google,
        scrape_tool: Optional[FunctionTool] = None,
        browse_tool: Optional[FunctionTool] = None,
        max_workers: int = 8,
    ) -> None:
        super().__init__()
        self.search_tool: FunctionTool = search_tool or SearchToolkit().search_google
        self.scrape_tool: FunctionTool = scrape_tool
        self.browse_tool: FunctionTool = browse_tool
        # TODO: maybe we do not need this executor because CAMEL has implemented such functionality? Please check.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ---------------- helpers ---------------- #

    def dispatch_async(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Future:
        return self._executor.submit(fn, *args, **kw)

    # -------------- BaseToolkit -------------- #

    def get_tools(self) -> List[FunctionTool]:
        tools = [self.search_tool]
        if self.scrape_tool:
            tools.append(self.scrape_tool)
        if self.browse_tool:
            tools.append(self.browse_tool)
        return tools
