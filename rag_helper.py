INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''  # системный промпт: определяет роль и ограничения ассистента (отвечать строго по контексту)

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()  # шаблон пользовательского сообщения: {question} и {context} заменяются через .format()


class RAGBase:  # инкапсулирует полный RAG-пайплайн: поиск → сборка промпта → вызов LLM

    def __init__(
        self,
        index,                           # поисковый индекс (minsearch.Index или sqlitesearch.TextSearchIndex)
        llm_client,                      # инициализированный клиент OpenAI
        instructions=INSTRUCTIONS,       # системный промпт (можно заменить при создании объекта)
        prompt_template=PROMPT_TEMPLATE, # шаблон промпта (можно заменить при создании объекта)
        course='llm-zoomcamp',           # фильтр: поиск ведётся только по документам этого курса
        model='gpt-5.4-mini'             # модель OpenAI для генерации ответов
    ):
        self.index = index                        # сохраняем ссылку на поисковый индекс
        self.llm_client = llm_client              # сохраняем ссылку на клиент OpenAI
        self.instructions = instructions          # сохраняем системный промпт
        self.course = course                      # сохраняем название курса для фильтрации
        self.prompt_template = prompt_template    # сохраняем шаблон промпта
        self.model = model                        # сохраняем название модели

    def search(self, query, num_results=5):
        boost_dict = {'question': 3.0, 'section': 0.5}  # question в 3× важнее answer; section менее значим
        filter_dict = {'course': self.course}            # ограничиваем поиск только нужным курсом

        return self.index.search(
            query,
            num_results=num_results,   # возвращаем топ-N наиболее релевантных документов
            boost_dict=boost_dict,     # применяем веса полей при ранжировании
            filter_dict=filter_dict    # применяем фильтр по курсу
        )

    def build_context(self, search_results):
        lines = []  # список строк, из которых собирается текстовый блок контекста

        for doc in search_results:                # для каждого найденного документа
            lines.append(doc['section'])           # раздел FAQ (например, "Module 1: Intro")
            lines.append('Q: ' + doc['question'])  # вопрос из FAQ
            lines.append('A: ' + doc['answer'])    # ответ из FAQ
            lines.append('')                       # пустая строка — визуальный разделитель между документами

        return '\n'.join(lines).strip()  # объединяем строки в текст, убираем пробелы по краям

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)  # преобразуем список документов в текстовый контекст
        return self.prompt_template.format(
            question=query, context=context  # подставляем вопрос пользователя и контекст в шаблон
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},  # системное сообщение: правила поведения
            {'role': 'user', 'content': prompt}                   # пользовательский запрос с контекстом
        ]

        response = self.llm_client.responses.create(
            model=self.model,      # используем заданную модель (gpt-5.4-mini по умолчанию)
            input=input_messages   # передаём список сообщений (developer + user)
        )

        return response.output_text  # возвращаем только текст ответа, без метаданных usage и т.д.

    def rag(self, query):
        search_results = self.search(query)               # шаг 1: находим релевантные документы по запросу
        prompt = self.build_prompt(query, search_results)  # шаг 2: строим промпт (вопрос + найденный контекст)
        answer = self.llm(prompt)                         # шаг 3: отправляем промпт в LLM, получаем ответ
        return answer                                     # возвращаем итоговый ответ пользователю
