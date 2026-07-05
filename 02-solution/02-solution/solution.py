"""
Решение ДЗ модуля 02 — Vector Search.
LLM Zoomcamp 2026, DataTalksClub.

Запуск:
    # установка зависимостей (через uv в курсе, тут — pip)
    uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
    uv add --dev huggingface-hub jupyter
    python download.py          # один раз — скачать ONNX-модель в models/
    python solution.py          # печатает ответы Q1..Q6

Используются локальные helper-скрипты embedder.py и download.py из embed/.
Эмбеддинги считаются лёгким ONNX-рантаймом (без PyTorch/CUDA).
"""

import numpy as np
from tqdm.auto import tqdm

from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import VectorSearch, Index


# ---------------------------------------------------------------------------
# Q1. Embedding a query
# ---------------------------------------------------------------------------
embed = Embedder()

QUERY_Q1 = "How does approximate nearest neighbor search work?"
v = embed.encode(QUERY_Q1)

print(f"Q1. len(v) = {len(v)}; v[0] = {v[0]:.4f}")  # 384 чисел, v[0] ~ -0.02


# ---------------------------------------------------------------------------
# Loading the data — те же 72 страницы уроков с коммита 8c1834d
# ---------------------------------------------------------------------------
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"    Loaded {len(documents)} lesson pages")  # 72


# ---------------------------------------------------------------------------
# Q2. Cosine similarity
# ---------------------------------------------------------------------------
target_name = "02-vector-search/lessons/07-sqlitesearch-vector.md"
target_doc = next(d for d in documents if d["filename"] == target_name)

doc_vec = embed.encode(target_doc["content"])
cos = float(doc_vec.dot(v))  # нормированные векторы => dot == cosine
print(f"Q2. cosine similarity = {cos:.4f}")  # ~0.37


# ---------------------------------------------------------------------------
# Q3. Chunking and search by hand
# ---------------------------------------------------------------------------
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"    Chunks: {len(chunks)}")

chunk_texts = [c["content"] for c in chunks]

batch_size = 50
X = []
for i in tqdm(range(0, len(chunk_texts), batch_size), desc="embedding chunks"):
    X.extend(embed.encode_batch(chunk_texts[i:i + batch_size]))
X = np.array(X)

scores = X.dot(v)
best = int(np.argmax(scores))
print(f"Q3. top chunk filename = {chunks[best]['filename']} (score={scores[best]:.4f})")


# ---------------------------------------------------------------------------
# Q4. Vector search with minsearch
# ---------------------------------------------------------------------------
vindex = VectorSearch()
vindex.fit(X, chunks)

QUERY_Q4 = "What metric do we use to evaluate a search engine?"
v_q4 = embed.encode(QUERY_Q4)
vres = vindex.search(v_q4, num_results=5)
print(f"Q4. first result filename = {vres[0]['filename']}")


# ---------------------------------------------------------------------------
# Q5. Text search vs vector search
# ---------------------------------------------------------------------------
tindex = Index(text_fields=["content"], keyword_fields=["filename"])
tindex.fit(chunks)

QUERY_Q5 = "How do I store vectors in PostgreSQL?"
v_q5 = embed.encode(QUERY_Q5)

vector_results_q5 = vindex.search(v_q5, num_results=5)
text_results_q5 = tindex.search(QUERY_Q5, num_results=5)

vector_files = {r["filename"] for r in vector_results_q5}
text_files = {r["filename"] for r in text_results_q5}
only_vector = vector_files - text_files
print(f"Q5. in vector but not text = {sorted(only_vector)}")


# ---------------------------------------------------------------------------
# Q6. Hybrid search (RRF)
# ---------------------------------------------------------------------------
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


QUERY_Q6 = "How do I give the model access to tools?"
v_q6 = embed.encode(QUERY_Q6)

vector_results = vindex.search(v_q6, num_results=5)
text_results = tindex.search(QUERY_Q6, num_results=5)
fused = rrf([vector_results, text_results])
print(f"Q6. RRF first result filename = {fused[0]['filename']}")
