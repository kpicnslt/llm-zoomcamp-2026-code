import requests          # HTTP-библиотека для скачивания данных из внешних API
from minsearch import Index  # лёгкий поисковый движок, хранит индекс в оперативной памяти


def load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'  # URL со списком всех курсов DataTalks.Club
    response = requests.get(docs_url)   # GET-запрос: получаем JSON со списком курсов
    courses_raw = response.json()       # разбираем JSON → список словарей {name, path, ...}

    documents = []                      # сюда накапливаем FAQ-документы со всех курсов
    url_prefix = 'https://datatalks.club/faq'  # базовый URL — к нему добавляем path каждого курса

    for course in courses_raw:          # перебираем каждый курс (mlops, llm, de-zoomcamp, ...)
        course_url = f'{url_prefix}{course["path"]}'   # строим полный URL FAQ этого курса
        course_response = requests.get(course_url)     # скачиваем список вопрос-ответов курса
        course_response.raise_for_status()             # бросает HTTPError, если код ответа 4xx/5xx
        course_data = course_response.json()           # парсим JSON в список документов курса

        documents.extend(course_data)   # добавляем документы курса в общий список (итого ~1154)

    return documents  # возвращаем полный список FAQ-документов всех курсов


def build_index(documents):
    index = Index(
        text_fields=['question', 'section', 'answer'],  # по этим полям ведётся полнотекстовый поиск (TF-IDF)
        keyword_fields=['course']                       # это поле используется только для точной фильтрации
    )
    index.fit(documents)  # строим поисковый индекс по всем переданным документам
    return index          # возвращаем готовый индекс для последующих поисковых запросов