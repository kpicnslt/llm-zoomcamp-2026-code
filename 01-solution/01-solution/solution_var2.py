"""
Решение ДЗ модуля 01 — Agentic RAG. ВАРИАНТ 2.

Отличие от solution.py: Q6 реализован ручным агентным циклом из урока 14
(`while True` + `responses.create(..., tools=[search_tool])`), без toyaikit
и без IPython-интерфейса. Формат `function_call` гарантированно тот, что мы
считаем, поэтому подсчёт вызовов `search` надёжен и не зависит от Jupyter.

Q1–Q5 идентичны solution.py и переиспользуют тот же rag_helper.py.

Запуск:
    uv add gitsource minsearch openai python-dotenv   # toyaikit для var2 не нужен
    # положите OPENAI_API_KEY в .env
    uv run python solution_var2.py
"""

import json

from dotenv import load_dotenv
load_dotenv()

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI

from rag_helper import RAGBase

QUERY = "How does the agentic loop keep calling the model until it stops?"
MODEL = "gpt-5.4-mini"


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    files = reader.read()
    return [file.parse() for file in files]  # {'filename', 'content'}


def build_index(docs):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(docs)
    return index


# ---------------------------------------------------------------------------
# Q6 (var2): ручной агентный цикл из урока 14
# ---------------------------------------------------------------------------
def run_agent_manual(chunk_index, openai_client):
    def search(query):
        """Search the course lessons for chunks matching the given query."""
        return chunk_index.search(query, num_results=5)

    search_tool = {
        "type": "function",
        "name": "search",
        "description": "Search the course lessons for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course lessons.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }

    def make_call(call):
        args = json.loads(call.arguments)
        if call.name == "search":
            result = search(**args)
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, indent=2),
        }

    instructions = (
        "You're a course teaching assistant. Answer the student's question "
        "using the search tool. Make multiple searches with different keywords "
        "before answering."
    )
    question = "How does the agentic loop work, and how is it different from plain RAG?"

    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question},
    ]

    search_calls = 0
    it = 1
    while True:
        has_function_calls = False
        response = openai_client.responses.create(
            model=MODEL,
            input=messages,
            tools=[search_tool],
        )
        messages.extend(response.output)

        for item in response.output:
            if item.type == "function_call":
                if item.name == "search":
                    search_calls += 1
                print(f"iteration #{it} -> function_call:", item.name, item.arguments)
                messages.append(make_call(item))
                has_function_calls = True
            elif item.type == "message":
                print("ASSISTANT:", item.content[0].text)

        it += 1
        if not has_function_calls:
            break

    return search_calls


def main():
    openai_client = OpenAI()

    documents = load_documents()
    print(f"Q1. Lesson pages: {len(documents)}")

    index = build_index(documents)
    results = index.search(QUERY, num_results=5)
    print(f"Q2. First result filename: {results[0]['filename']}")

    rag = RAGBase(index=index, llm_client=openai_client)
    res = rag.rag(QUERY)
    print(f"Q3. Input tokens (full pages): {res.input_tokens}")

    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Q4. Chunks: {len(chunks)}")

    chunk_index = build_index(chunks)
    rag_chunked = RAGBase(index=chunk_index, llm_client=openai_client)
    res_chunked = rag_chunked.rag(QUERY)
    print(f"Q5. Input tokens (chunked): {res_chunked.input_tokens}")
    print(f"Q5. Reduction factor: ~{res.input_tokens / res_chunked.input_tokens:.1f}x fewer")

    n = run_agent_manual(chunk_index, openai_client)
    print(f"Q6. Agent called search: {n} times")


if __name__ == "__main__":
    main()
