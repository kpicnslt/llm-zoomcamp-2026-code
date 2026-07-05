# Решение ДЗ — Модуль 02: Vector Search

LLM Zoomcamp 2026, DataTalksClub. Разбор модуля — в [../02-description.md](../02-description.md).

## Состав

- `solution.py` — скрипт, печатает ответы Q1..Q6.
- `solution.ipynb` — то же решение в виде Jupyter-ноутбука (с пояснениями по каждому вопросу).
- `download.py` — helper из `embed/`: скачивает ONNX-модель с HuggingFace.
- `embedder.py` — helper из `embed/`: класс `Embedder` (encode/encode_batch на ONNX Runtime).

## Запуск

```bash
# зависимости (в курсе через uv; можно и pip)
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub jupyter

python download.py    # один раз — модель ляжет в models/Xenova/all-MiniLM-L6-v2/
python solution.py    # ответы в stdout
# либо открыть solution.ipynb
```

Эмбеддинги считаются лёгким ONNX-рантаймом (без PyTorch/CUDA). База знаний —
72 страницы уроков курса, снятые с коммита `8c1834d`.

## Ответы

| Вопрос | Ответ |
|--------|-------|
| Q1. `v[0]` запроса | **-0.02** (`-0.0206`) |
| Q2. косинус с `07-sqlitesearch-vector.md` | **0.37** (`0.3611`) |
| Q3. файл топ-чанка | **`02-vector-search/lessons/07-sqlitesearch-vector.md`** |
| Q4. первый результат VectorSearch | **`04-evaluation/lessons/05-search-metrics.md`** |
| Q5. в векторных, но не в текстовых | **`02-vector-search/lessons/08-pgvector.md`** |
| Q6. первый после RRF | **`01-agentic-rag/lessons/13-function-calling.md`** |
