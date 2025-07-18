from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import AsyncGenerator
from langchain_core.runnables.config import RunnableConfig
from graph import graph, builder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应更严格
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


async def stream_agent_events_old(query: str) -> AsyncGenerator[str, None]:
    """
    这是一个异步生成器，用于运行代理并以 SSE 格式流式传输事件。
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    config = RunnableConfig(
        configurable={
            "number_of_initial_queries": 2,
            "max_research_loops": 2,
        },
        recursion_limit=50,
    )

    # astream_events 返回一个异步迭代器
    async for event in graph.astream_events(inputs, config, version="v2"):
        kind = event["event"]
        name = event.get("name")

        # 将我们关心的事件格式化为 JSON 并发送
        # 我们将发送三种类型的消息: status, chunk, sources

        # 1. 状态更新：哪个节点正在运行
        if kind == "on_chain_start" and name in builder.nodes:
            data = {"type": "status", "message": f"Thinking: Entering {name}..."}
            yield f"data: {json.dumps(data)}\n\n"

        # 2. 最终答案的文本流
        # 我们只关心 finalize_answer 节点的 LLM 流
        if kind == "on_llm_stream" and name == "finalize_answer":
            content = event["data"]["chunk"].content
            if content:
                data = {"type": "chunk", "content": content}
                yield f"data: {json.dumps(data)}\n\n"

        # 3. 最终的来源信息
        # 当 finalize_answer 节点完成时，我们可以从其输出中提取来源
        if kind == "on_chain_end" and name == "finalize_answer":
            sources = event["data"]["output"].get("sources_gathered", [])
            data = {"type": "sources", "sources": sources}
            yield f"data: {json.dumps(data)}\n\n"

# 找到你的 stream_agent_events 函数并用下面的版本替换它


async def stream_agent_events(query: str) -> AsyncGenerator[str, None]:
    """
    这是一个异步生成器，用于运行代理并以 SSE 格式流式传输事件。
    现在它监听 on_chain_stream 事件来处理流式节点。
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    config = RunnableConfig(
        configurable={
            "number_of_initial_queries": 2,
            "max_research_loops": 2,
        },
        recursion_limit=50,
    )

    # 用于跟踪上一次发送的文本，只发送增量部分
    last_sent_content = ""

    async for event in graph.astream_events(inputs, config, version="v2"):
        kind = event["event"]
        name = event.get("name")

        # 状态更新：哪个节点正在运行
        if kind == "on_chain_start" and name in builder.nodes:
            data = {"type": "status", "message": f"Thinking: Entering {name}..."}
            yield f"data: {json.dumps(data)}\n\n"

        # 核心改动：监听 on_chain_stream 事件
        if kind == "on_chain_stream" and name == "finalize_answer":
            # 获取节点 yield 的数据块
            chunk = event["data"]["chunk"]

            # 检查是否有 messages 更新
            if "messages" in chunk and chunk["messages"]:
                full_content = chunk["messages"].content
                # 计算并发送增量内容
                delta = full_content[len(last_sent_content):]
                last_sent_content = full_content
                if delta:
                    data = {"type": "chunk", "content": delta}
                    yield f"data: {json.dumps(data)}\n\n"

            # 检查是否有 sources 更新（在流的最后）
            if "sources_gathered" in chunk:
                sources = chunk.get("sources_gathered", [])
                data = {"type": "sources", "sources": sources}
                yield f"data: {json.dumps(data)}\n\n"


@app.post("/research")
async def research(request: ResearchRequest):
    """
    API 端点，接收 POST 请求并返回一个流式响应。
    """
    return StreamingResponse(
        stream_agent_events(request.query),
        media_type="text/event-stream"
    )
