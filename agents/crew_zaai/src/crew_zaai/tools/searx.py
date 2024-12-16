from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
from langchain_community.utilities import SearxSearchWrapper


class SearxSearchToolInput(BaseModel):
    """Input schema for SearxSearchTool."""

    query: str = Field(..., description="The search query.")
    num_results: int = Field(10, description="The number of results to retrieve.")


class SearxSearchTool(BaseTool):
    name: str = "searx_search_tool"
    description: str = (
        "A tool to perform searches using the Searx metasearch engine. "
        "Specify a query and optionally limit by engines, categories, or number of results."
    )
    args_schema: Type[BaseModel] = SearxSearchToolInput
    _searx_wrapper: SearxSearchWrapper = PrivateAttr()

    def __init__(self, searx_host: str, unsecure: bool = False):
        """Initialize the SearxSearchTool with SearxSearchWrapper."""
        super().__init__()
        self._searx_wrapper = SearxSearchWrapper(
            searx_host=searx_host, unsecure=unsecure
        )

    def _run(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict]:
        """Perform a search using the Searx API."""
        try:
            results = self._searx_wrapper.results(
                query=query + f" :youtube",
                num_results=num_results,
            )
            return results
        except Exception as e:
            return [{"Error": str(e)}]
