# AI-ChatBot
Langgraph AI ChatBot
# LangGraph + Groq Chatbot

A simple conversational chatbot built with **LangGraph**, powered by an LLM hosted on **Groq**, with tracing via **LangSmith**.

## Overview

This project demonstrates the basics of LangGraph: defining a shared state, building a graph of nodes and edges, and running a simple chat loop that streams responses from an LLM.

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph-based orchestration framework for LLM apps
- [LangChain](https://github.com/langchain-ai/langchain) — core LLM app framework
- [Groq](https://groq.com/) — fast LLM inference (via `langchain_groq`)
- [LangSmith](https://smith.langchain.com/) — tracing and observability

## Setup

### 1. Install dependencies

```bash
pip install langgraph langsmith
pip install langchain langchain_groq langchain_community
```

### 2. Set up API keys

You'll need:
- A **Groq API key** — get one at [console.groq.com](https://console.groq.com)
- A **LangSmith API key** — get one at [smith.langchain.com](https://smith.langchain.com) (Settings → API Keys)

In Google Colab, store these as secrets (key icon in the sidebar) under:
- `groq_api_key`
- `LANGSMITH_API_KEY`

```python
from google.colab import userdata

groq_api_key = userdata.get('groq_api_key')
langsmith = userdata.get('LANGSMITH_API_KEY')
```

### 3. Enable tracing

```python
import os

os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"
```

## Building the Graph

### Define the LLM

```python
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
```

> ⚠️ Groq's available models change over time. Always verify the current list for your account before hardcoding a model name:
> ```python
> import requests
>
> url = "https://api.groq.com/openai/v1/models"
> headers = {"Authorization": f"Bearer {groq_api_key}"}
> response = requests.get(url, headers=headers)
> for m in response.json()["data"]:
>     print(m["id"])
> ```

### Define the state

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### Define the node

```python
def chatbot(state: State):
    return {"messages": llm.invoke(state['messages'])}
```

### Build the graph

```python
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
```

## Running the Chatbot

```python
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye")
        break
    for event in graph.stream({'messages': ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"].content)
```

Type `quit` or `q` to exit the chat loop.

## Notes

- Use a **chat/instruction model** (e.g. `openai/gpt-oss-20b`, `openai/gpt-oss-120b`) — not a classifier model like `llama-prompt-guard-2-*`, which only outputs a score, not a reply.
- All conversation history is kept in memory only for the current session (not persisted). For persistence across restarts, use LangGraph's checkpointer feature with a real database (e.g. SQLite/Postgres).
- Every run is automatically traced in LangSmith under the `CourseLanggraph` project.

## License

MIT
