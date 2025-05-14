#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import openai

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rag_tool.langgraph_tool import RAGTool

# ————————————————————————————————————————————————————————————————
# 1) Define our state
class MyState(TypedDict):
    messages: list[dict]    # just the user message
    collection: str
    answer: bool
    tool_output: dict       # RAG API response
    final_answer: str       # LLM’s final answer

# ————————————————————————————————————————————————————————————————
# 2) Bootstrap
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")
rag_tool = RAGTool(api_url=os.getenv("RAG_API_URL"))

# ————————————————————————————————————————————————————————————————
# 3) Node A: call the RAG API (tool)
def call_rag(state: MyState) -> MyState:
    try:
        out = rag_tool.rag_query(
            state["collection"],
            state["messages"][0]["content"],
            state["answer"],
        )
    except Exception as e:
        out = {"snippets": [], "error": str(e)}
    state["tool_output"] = out
    return state

# ————————————————————————————————————————————————————————————————
# 4) Node B: call the LLM to synthesize a user‑friendly answer
def synthesize_answer(state: MyState) -> MyState:
    # build a mini‑conversation for the model
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": state["messages"][0]["content"]},
        {
            "role": "function",
            "name": "rag_query",
            "content": str(state["tool_output"])
        },
    ]
    resp = openai.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=msgs
    )
    state["final_answer"] = resp.choices[0].message.content or ""
    return state

# ————————————————————————————————————————————————————————————————
# 5) Wire up the graph
graph = StateGraph(MyState)
graph.add_node("rag", call_rag)
graph.add_node("synth", synthesize_answer)
graph.add_edge(START,  "rag")
graph.add_edge("rag",  "synth")
graph.add_edge("synth", END)
agent = graph.compile()

# ————————————————————————————————————————————————————————————————
# 6) Print the workflow — Mermaid and ASCII
g = agent.get_graph()
print("\n=== Agent Workflow (Mermaid) ===")
print(g.draw_mermaid())
print("\n=== Agent Workflow (ASCII) ===")
print(g.draw_ascii())

agent = graph.compile()

# ————————————————————————————————————————————————————————————————
# 7) Run it
def main():
    collection = os.getenv("RAG_COLLECTION", "test")
    questions = [
        "What are the main services offered?",
        "How can I apply for financial aid?",
        "What documents do I need to provide?"
    ]
    for q in questions:
        print(f"\n>>> Question: {q}")
        init_state: MyState = {
            "messages":    [{"role": "user", "content": q}],
            "collection":  collection,
            "answer":      True,
            "tool_output": {},
            "final_answer": ""
        }
        result = agent.invoke(init_state)
        print("→ RAG raw:",   result["tool_output"])
        print("→ Answer:",    result["final_answer"])

if __name__ == "__main__":
    main()
