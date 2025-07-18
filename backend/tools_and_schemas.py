from langchain_tavily import TavilySearch
from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


class SearchQueryList(BaseModel):
    query: list[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_map: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: list[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


# search_tool = DuckDuckGoSearchResults()
# search_tools = load_tools(["serpapi"])
search_tool = TavilySearch(max_results=3)
