"""
RAG helper для ДЗ модуля 01 (Agentic RAG).

Отличия от учебного rag_helper.py:
- база знаний — страницы уроков (схема `filename` + `content`),
  а не FAQ (`section`/`question`/`answer`);
- `llm()` возвращает весь объект ответа (а не только текст),
  чтобы можно было прочитать usage;
- `rag()` возвращает датакласс RAGResult с ответом и usage (входные токены).
"""

from dataclasses import dataclass


INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


@dataclass
class RAGResult:
    answer: str
    input_tokens: int
    output_tokens: int


class RAGBase:
    """RAG поверх индекса со схемой `filename` / `content`."""

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini',
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        # наши документы: text-поле `content`, keyword-поле `filename`.
        # фильтра по курсу больше нет — ищем по всей базе уроков.
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        """Возвращает весь объект ответа (а не только текст)."""
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt},
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )
        return response

    def rag(self, query) -> RAGResult:
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)

        usage = response.usage
        # Responses API: input_tokens / output_tokens.
        # Chat Completions API: prompt_tokens / completion_tokens.
        input_tokens = getattr(usage, 'input_tokens', None) \
            or getattr(usage, 'prompt_tokens', None)
        output_tokens = getattr(usage, 'output_tokens', None) \
            or getattr(usage, 'completion_tokens', None)

        return RAGResult(
            answer=response.output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
