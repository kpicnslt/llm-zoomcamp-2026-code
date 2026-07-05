"""
Решение ДЗ модуля 01 — Agentic RAG.
LLM Zoomcamp 2026, DataTalksClub.

Запуск:
    uv add gitsource minsearch openai toyaikit python-dotenv
    # положите OPENAI_API_KEY в .env
    uv run python solution.py

Скрипт последовательно отвечает на вопросы Q1–Q6 и печатает результаты.
Локальный rag_helper.py (рядом) адаптирован под схему `filename`/`content`
и отдаёт usage (входные токены).
"""

from dotenv import load_dotenv
load_dotenv()

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI

from rag_helper import RAGBase

QUERY = "How does the agentic loop keep calling the model until it stops?"


# ---------------------------------------------------------------------------
# Preparation: тянем страницы уроков с фиксированного коммита
# ---------------------------------------------------------------------------
def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    files = reader.read()

    documents = []
    for file in files:
        documents.append(file.parse())  # {'filename': ..., 'content': ...}
    return documents


def build_index(docs):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(docs)
    return index


# ---------------------------------------------------------------------------
# Q6: агент с инструментом search через toyaikit
# ---------------------------------------------------------------------------
def run_agent(chunk_index, openai_client):
    from toyaikit.llm import OpenAIClient
    from toyaikit.tools import Tools
    from toyaikit.chat import IPythonChatInterface
    from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

    def search(query: str) -> list[dict]:
        """Search the course lessons for chunks matching the given query."""
        return chunk_index.search(query, num_results=5)

    agent_tools = Tools()
    agent_tools.add_tool(search)  # схема выводится из type hint + docstring

    instructions = (
        "You're a course teaching assistant. Answer the student's question "
        "using the search tool. Make multiple searches with different keywords "
        "before answering."
    )

    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)
    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=instructions,
        chat_interface=chat_interface,
        llm_client=OpenAIClient(model="gpt-5.4-mini"),
    )

    result = runner.loop(
        prompt="How does the agentic loop work, and how is it different from plain RAG?",
        callback=callback,
    )

    # считаем, сколько раз вызван инструмент search
    search_calls = sum(
        1 for m in result.all_messages
        if getattr(m, "type", None) == "function_call"
        and getattr(m, "name", None) == "search"
    )
    return search_calls


def main():
    openai_client = OpenAI()

    # ----- Preparation / Q1 -----
    documents = load_documents()
    print(f"Q1. Lesson pages: {len(documents)}")  # -> 72

    # ----- Q2 -----
    index = build_index(documents)
    results = index.search(QUERY, num_results=5)
    print(f"Q2. First result filename: {results[0]['filename']}")
    # -> 01-agentic-rag/lessons/14-agentic-loop.md

    # ----- Q3 -----
    rag = RAGBase(index=index, llm_client=openai_client)
    res = rag.rag(QUERY)
    print(f"Q3. Input tokens (full pages): {res.input_tokens}")  # ~7000

    # ----- Q4 -----
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Q4. Chunks: {len(chunks)}")  # ~295

    # ----- Q5 -----
    chunk_index = build_index(chunks)
    rag_chunked = RAGBase(index=chunk_index, llm_client=openai_client)
    res_chunked = rag_chunked.rag(QUERY)
    print(f"Q5. Input tokens (chunked): {res_chunked.input_tokens}")
    ratio = res.input_tokens / res_chunked.input_tokens
    print(f"Q5. Reduction factor: ~{ratio:.1f}x fewer")  # ~3x

    # ----- Q6 -----
    n = run_agent(chunk_index, openai_client)
    print(f"Q6. Agent called search: {n} times")  # ~4


if __name__ == "__main__":
    main()
