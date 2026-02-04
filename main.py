"""
AI Agent для поиска документов в научных базах данных
Использует LangGraph для оркестрации процесса поиска
"""

import os
# FIX: Решение проблемы OpenMP conflict с FAISS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Загрузка переменных окружения
load_dotenv()


# ============================================================================
# УТИЛИТЫ: Надёжный парсинг JSON
# ============================================================================

def safe_json_parse(response_text: str, fallback: Any = None) -> Any:
    """
    Надёжный парсинг JSON из ответа LLM с автоматическим исправлением типичных ошибок
    """
    try:
        # Пробуем найти JSON в тексте
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            # Пробуем найти массив
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            return fallback
        
        json_str = response_text[start_idx:end_idx + 1]
        
        # Пробуем стандартный парсинг
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Пытаемся исправить простые ошибки вручную
            try:
                # Убираем trailing commas
                json_str = json_str.replace(',}', '}').replace(',]', ']')
                # Убираем комментарии (если есть)
                lines = json_str.split('\n')
                cleaned = [line.split('//')[0] for line in lines]
                json_str = '\n'.join(cleaned)
                return json.loads(json_str)
            except Exception:
                return fallback
    
    except Exception as e:
        return fallback


# ============================================================================
# PYDANTIC МОДЕЛИ ДЛЯ ВАЛИДАЦИИ
# ============================================================================

class TopicCardModel(BaseModel):
    """Модель для TopicCard"""
    must: List[str] = Field(default_factory=list)
    should: List[str] = Field(default_factory=list)
    must_not: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    expanded_queries: List[str] = Field(default_factory=list)
    fields_of_study: List[str] = Field(default_factory=list)


class StructuredSummaryModel(BaseModel):
    """Модель для структурированного конспекта статьи"""
    problem: str = ""
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    key_findings: str = ""
    limitations: str = ""
    future_work: str = ""
    contributions: str = ""  # Вклад работы
    related_work_summary: str = ""  # Краткий обзор связанных работ
    experimental_setup: str = ""  # Детали экспериментов
    reproducibility_info: str = ""  # Наличие кода/данных
    discussion: str = ""  # Обсуждение результатов
    conclusion: str = ""  # Выводы


class ResearchGapModel(BaseModel):
    """Модель для research gap"""
    gap: str
    type: str  # methodological|data|metric|reproducibility|contradiction|temporal|scalability|cross_domain
    severity: str  # high|medium|low
    evidence: List[str] = Field(default_factory=list)
    reasoning: str = ""  # Обоснование почему это gap
    potential_impact: str = ""  # Потенциальное влияние исследования
    related_methods: List[str] = Field(default_factory=list)  # Связанные методы которые можно использовать
    feasibility: str = ""  # Оценка выполнимости исследования


class ResearchIdeaModel(BaseModel):
    """Модель для исследовательской идеи"""
    hypothesis: str
    experiment_plan: Dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str = ""
    risks: List[str] = Field(default_factory=list)
    related_gap: str = ""


# Определяем State агента (расширенная версия)
class AgentState(TypedDict):
    """
    Состояние агента для поиска документов - полная архитектура
    """
    # === Входные данные ===
    query: str  # Исходный запрос пользователя
    time_window: int  # Окно времени (лет назад)
    max_papers: int  # Максимум статей для анализа
    
    # === Анализ запроса ===
    selected_databases: List[str]  # Выбранные базы данных для поиска
    refined_query: str  # Улучшенный запрос
    
    # === TopicCard (QueryBuilder) ===
    topic_card: Dict[str, Any]  # must[], should[], must_not[], synonyms[]
    query_strings: List[str]  # Расширенные запросы
    
    # === Результаты поиска ===
    search_results: Dict[str, List[Dict[str, Any]]]  # Результаты по источникам
    
    # === SeedResults (Retriever) ===
    seed_results: List[Dict[str, Any]]  # Сырые результаты из API
    
    # === CorpusIndex (Deduper/Normalizer) ===
    corpus_index: List[Dict[str, Any]]  # Унифицированные метаданные
    
    # === Ranked papers (Ranker) ===
    ranked_papers: List[Dict[str, Any]]  # Ранжированные статьи
    
    # === PDF Reader ===
    pdf_texts: Dict[str, str]  # Полные тексты PDF: {paper_id: full_text}
    
    # === Citation graph (Snowballer) ===
    citation_graph: Dict[str, Any]  # nodes[], edges[], centrality{}
    
    # === LitMatrix (Summarizer) ===
    lit_matrix: List[Dict[str, Any]]  # Литературная матрица
    
    # === Research gaps (GapMiner) ===
    gap_list: List[Dict[str, Any]]  # Список лакун
    
    # === Ideas (Ideator) ===
    idea_bank: List[Dict[str, Any]]  # Гипотезы и идеи
    
    # === Служебные поля ===
    final_response: str  # Финальный отчёт
    messages: List[Any]  # История сообщений LLM
    next_step: str  # Следующий шаг в pipeline
    budget: Dict[str, int]  # Бюджет API вызовов
    retry_count: int  # Количество попыток поиска
    search_quality_score: float  # Оценка качества результатов
    replanning_history: List[str]  # История перепланирований


# Инициализация LLM
def get_llm():
    """Создаёт и возвращает экземпляр LLM"""
    from config import LLM_CONFIG
    return ChatOpenAI(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        max_retries=LLM_CONFIG["max_retries"],
        api_key=os.getenv("OPENAI_API_KEY")
    )


# Инициализация Embeddings
def get_embeddings():
    """Создаёт и возвращает модель для embeddings"""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # Дешёвая и быстрая модель
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ============================================================================
# ФУНКЦИЯ 1: Анализ запроса пользователя
# ============================================================================

def analyze_query(state: AgentState) -> AgentState:
    """
    Анализирует запрос пользователя и определяет:
    1. Какие базы данных нужно использовать
    2. Как переформулировать запрос для лучшего поиска
    """
    print("\n🔍 Функция 1: Анализ запроса...")
    
    query = state["query"]
    llm = get_llm()
    
    # Системный промпт для анализа запроса
    system_prompt = """Ты - эксперт по поиску научных статей. 
    Проанализируй запрос пользователя и определи:
    1. Какие базы данных лучше использовать (arxiv, pubmed, или оба)
    2. Переформулируй запрос для улучшения результатов поиска (на английском)
    
    Выбирай базы данных по следующим критериям:
    - arxiv: физика, математика, computer science, биология (препринты)
    - pubmed: медицина, биомедицина, клинические исследования
    
    Ответь СТРОГО в формате JSON:
    {
        "databases": ["arxiv", "pubmed"],
        "refined_query": "переформулированный запрос"
    }
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Запрос пользователя: {query}")
    ]
    
    response = llm.invoke(messages)
    
    # Парсим ответ с валидацией
    parsed = safe_json_parse(response.content, {})
    
    state["selected_databases"] = parsed.get("databases", ["arxiv"]) if isinstance(parsed, dict) else ["arxiv"]
    state["refined_query"] = parsed.get("refined_query", query) if isinstance(parsed, dict) else query
    
    if not isinstance(parsed, dict):
        print(f"⚠️  Ошибка парсинга, используем значения по умолчанию")
    
    state["messages"] = messages + [response]
    state["next_step"] = "search"
    
    print(f"   📊 Выбранные БД: {state['selected_databases']}")
    print(f"   📝 Улучшенный запрос: {state['refined_query']}")
    
    return state


# ============================================================================
# ФУНКЦИЯ 2: Поиск в ArXiv
# ============================================================================

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Ищет статьи в базе данных ArXiv
    """
    print(f"\n📚 Функция 2a: Поиск в ArXiv по запросу '{query}'...")
    
    import arxiv
    
    try:
        # Создаём клиент для поиска
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in search.results():
            # Извлекаем ArXiv ID из entry_id (формат: http://arxiv.org/abs/1234.5678v1)
            arxiv_id = None
            if paper.entry_id:
                # Извлекаем ID из URL
                import re
                match = re.search(r'/(\d{4}\.\d{4,5})(?:v\d+)?', paper.entry_id)
                if match:
                    arxiv_id = match.group(1)
            
            results.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                "url": paper.entry_id,
                "arxiv_id": arxiv_id,
                "published": paper.published.strftime("%Y-%m-%d"),
                "categories": paper.categories,
                "source": "arxiv"
            })
        
        print(f"   ✓ Найдено {len(results)} статей в ArXiv")
        return results
    
    except Exception as e:
        print(f"   ⚠️  Ошибка поиска в ArXiv: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 3: Поиск в PubMed
# ============================================================================

def search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Ищет статьи в базе данных PubMed через Entrez API
    """
    print(f"\n🏥 Функция 2b: Поиск в PubMed по запросу '{query}'...")
    
    from Bio import Entrez
    import xml.etree.ElementTree as ET
    
    # Устанавливаем email для Entrez (требование NCBI)
    Entrez.email = "dru4inin.dmitry@gmail.com"
    
    try:
        # Поиск ID статей
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        
        if not id_list:
            print("   ✓ Статьи в PubMed не найдены")
            return []
        
        # Получаем детали статей
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
        xml_data = handle.read()
        handle.close()
        
        # Парсим XML
        root = ET.fromstring(xml_data)
        
        results = []
        for article in root.findall(".//PubmedArticle"):
            try:
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "No title"
                
                abstract_elem = article.find(".//AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                if abstract and len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                
                # Авторы
                authors = []
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None and forename is not None:
                        authors.append(f"{forename.text} {lastname.text}")
                
                # PMID
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # Дата публикации
                pub_date = article.find(".//PubDate/Year")
                year = pub_date.text if pub_date is not None else "Unknown"
                
                results.append({
                    "title": title,
                    "authors": authors,
                    "summary": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "published": year,
                    "source": "pubmed",
                    "pmid": pmid
                })
            except Exception as e:
                print(f"   ⚠️  Ошибка парсинга статьи: {e}")
                continue
        
        print(f"   ✓ Найдено {len(results)} статей в PubMed")
        return results
    
    except Exception as e:
        print(f"   ⚠️  Ошибка поиска в PubMed: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 4: Узел графа для выполнения поиска
# ============================================================================

def perform_search(state: AgentState) -> AgentState:
    """
    Выполняет поиск в выбранных базах данных
    """
    print("\n🔎 Функция 4: Выполнение поиска в базах данных...")
    
    query = state["refined_query"]
    databases = state["selected_databases"]
    
    search_results = {}
    
    # Поиск в каждой выбранной базе данных
    if "arxiv" in databases:
        search_results["arxiv"] = search_arxiv(query, max_results=5)
    
    if "pubmed" in databases:
        search_results["pubmed"] = search_pubmed(query, max_results=5)
    
    state["search_results"] = search_results
    state["next_step"] = "synthesize"
    
    # Подсчитываем общее количество найденных статей
    total_results = sum(len(results) for results in search_results.values())
    print(f"\n   📊 Всего найдено статей: {total_results}")
    
    return state


# ============================================================================
# ФУНКЦИЯ 5: Поиск в OpenAlex
# ============================================================================

def search_openalex(query: str, max_results: int = 50, from_year: int = 2019) -> List[Dict[str, Any]]:
    """
    Ищет статьи в OpenAlex - мощная база данных научных публикаций
    """
    print(f"\n🌐 Функция 5: Поиск в OpenAlex по запросу '{query}'...")
    
    import requests
    import time
    
    try:
        base_url = "https://api.openalex.org/works"
        
        params = {
            "search": query,
            "filter": f"from_publication_date:{from_year}-01-01",
            "per_page": min(max_results, 50),
            "mailto": "researcher@example.com"  # Вежливость для API
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for work in data.get("results", []):
            # Извлекаем DOI
            doi = work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None
            
            # Авторы
            authors = []
            for authorship in work.get("authorships", [])[:10]:
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])
            
            # Год публикации
            year = work.get("publication_year", "Unknown")
            
            # Цитирования
            citations_total = work.get("cited_by_count", 0)
            
            # Abstract (может быть инвертированный индекс)
            abstract = ""
            abstract_inv = work.get("abstract_inverted_index", {})
            if abstract_inv:
                # Восстанавливаем текст из инвертированного индекса
                word_positions = []
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join([word for _, word in word_positions])
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
            
            results.append({
                "title": work.get("title", "No title"),
                "authors": authors,
                "summary": abstract or "No abstract available",
                "url": work.get("id", ""),
                "doi": doi,
                "published": str(year),
                "citations_total": citations_total,
                "source": "openalex",
                "venue": work.get("primary_location", {}).get("source", {}).get("display_name", "Unknown"),
                "type": work.get("type", "article")
            })
        
        print(f"   ✓ Найдено {len(results)} статей в OpenAlex")
        time.sleep(0.1)  # Вежливая задержка
        return results
    
    except Exception as e:
        print(f"   ⚠️  Ошибка поиска в OpenAlex: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 6: Поиск в Semantic Scholar
# ============================================================================

def search_semantic_scholar(query: str, max_results: int = 50, from_year: int = 2019) -> List[Dict[str, Any]]:
    """
    Ищет статьи в Semantic Scholar - отличный источник с метриками влияния
    """
    print(f"\n🎓 Функция 6: Поиск в Semantic Scholar по запросу '{query}'...")
    
    import requests
    import time
    
    try:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,authors,abstract,year,citationCount,url,externalIds,venue,publicationTypes,influentialCitationCount",
            "year": f"{from_year}-"
        }
        
        headers = {
            "Accept": "application/json"
        }
        
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for paper in data.get("data", []):
            # Авторы
            authors = [author.get("name", "") for author in paper.get("authors", [])]
            
            # DOI и другие ID
            ext_ids = paper.get("externalIds", {})
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            pmid = ext_ids.get("PubMed")
            
            # Abstract
            abstract = paper.get("abstract", "No abstract available")
            if abstract and len(abstract) > 500:
                abstract = abstract[:500] + "..."
            
            results.append({
                "title": paper.get("title", "No title"),
                "authors": authors,
                "summary": abstract,
                "url": paper.get("url", ""),
                "doi": doi,
                "arxiv_id": arxiv_id,
                "pmid": pmid,
                "published": str(paper.get("year", "Unknown")),
                "citations_total": paper.get("citationCount", 0),
                "influential_citations": paper.get("influentialCitationCount", 0),
                "source": "semantic_scholar",
                "venue": paper.get("venue", "Unknown"),
                "type": paper.get("publicationTypes", ["article"])[0] if paper.get("publicationTypes") else "article"
            })
        
        print(f"   ✓ Найдено {len(results)} статей в Semantic Scholar")
        time.sleep(0.1)  # Вежливая задержка
        return results
    
    except Exception as e:
        print(f"   ⚠️  Ошибка поиска в Semantic Scholar: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 7: Поиск в Crossref
# ============================================================================

def search_crossref(query: str, max_results: int = 50, from_year: int = 2019) -> List[Dict[str, Any]]:
    """
    Ищет статьи в Crossref - источник метаданных по DOI
    """
    print(f"\n🔗 Функция 7: Поиск в Crossref по запросу '{query}'...")
    
    import requests
    import time
    
    try:
        base_url = "https://api.crossref.org/works"
        
        params = {
            "query": query,
            "filter": f"from-pub-date:{from_year}",
            "rows": min(max_results, 100),
            "mailto": "researcher@example.com"
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("message", {}).get("items", []):
            # Авторы
            authors = []
            for author in item.get("author", [])[:10]:
                given = author.get("given", "")
                family = author.get("family", "")
                if given or family:
                    authors.append(f"{given} {family}".strip())
            
            # Год
            pub_date = item.get("published-print") or item.get("published-online") or item.get("created")
            year = "Unknown"
            if pub_date and "date-parts" in pub_date:
                year = str(pub_date["date-parts"][0][0])
            
            # Abstract
            abstract = item.get("abstract", "No abstract available")
            if abstract and len(abstract) > 500:
                abstract = abstract[:500] + "..."
            
            # Название журнала/конференции
            venue = "Unknown"
            if item.get("container-title"):
                venue = item["container-title"][0] if isinstance(item["container-title"], list) else item["container-title"]
            
            results.append({
                "title": item.get("title", ["No title"])[0] if isinstance(item.get("title"), list) else item.get("title", "No title"),
                "authors": authors,
                "summary": abstract,
                "url": item.get("URL", ""),
                "doi": item.get("DOI"),
                "published": year,
                "citations_total": item.get("is-referenced-by-count", 0),
                "source": "crossref",
                "venue": venue,
                "type": item.get("type", "article")
            })
        
        print(f"   ✓ Найдено {len(results)} статей в Crossref")
        time.sleep(0.1)
        return results
    
    except Exception as e:
        print(f"   ⚠️  Ошибка поиска в Crossref: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 8: QueryBuilder - расширение и нормализация запроса
# ============================================================================

def build_topic_card(state: AgentState) -> AgentState:
    """
    Создаёт TopicCard: извлекает ключевые термины, синонимы,
    негативные фильтры для улучшения поиска
    """
    print("\n🔨 Функция 8: QueryBuilder - построение TopicCard...")
    
    query = state["query"]
    llm = get_llm()
    
    system_prompt = """Ты - эксперт по формированию научных поисковых запросов.
    
Твоя задача: проанализировать тему исследования и создать структурированный TopicCard.

Верни СТРОГО JSON формат:
{
  "must": ["обязательные термины"],
  "should": ["желательные термины", "синонимы", "смежные понятия"],
  "must_not": ["исключения", "например: review, survey, tutorial"],
  "synonyms": ["список всех синонимов и аббревиатур"],
  "expanded_queries": ["расширенный запрос 1", "запрос с синонимами 2", "запрос по подтеме 3"],
  "fields_of_study": ["области науки, например: machine learning, biology"]
}

Правила:
1. Извлекай ключевые термины (5-10)
2. Добавляй синонимы и аббревиатуры (ML = Machine Learning)
3. Создай 3-5 вариантов запросов с разными формулировками
4. Опционально исключай reviews/surveys если нужны только исследования
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Тема исследования: {query}\n\nВремя: последние {state.get('time_window', 5)} лет")
    ]
    
    response = llm.invoke(messages)
    
    # Применяем фильтры из конфига
    try:
        from config import SEARCH_FILTERS
        exclude_reviews = SEARCH_FILTERS.get("exclude_reviews", False)
        novelty_level = SEARCH_FILTERS.get("novelty_level", None)
        domain = SEARCH_FILTERS.get("domain", None)
        
        filter_hints = []
        if exclude_reviews:
            filter_hints.append("ИСКЛЮЧИТЬ: review, survey, tutorial, overview статьи")
        if novelty_level == "high":
            filter_hints.append("ПРИОРИТЕТ: очень свежие работы (последние 1-2 года)")
        elif novelty_level == "low":
            filter_hints.append("ВКЛЮЧИТЬ: классические и старые работы")
        if domain == "medicine":
            filter_hints.append("ФОКУС: медицинские и биомедицинские исследования")
        elif domain == "cs":
            filter_hints.append("ФОКУС: computer science, AI, ML исследования")
        
        if filter_hints:
            filter_text = "\n".join(filter_hints)
            messages[1].content += f"\n\nДополнительные требования:\n{filter_text}"
    except:
        pass
    
    response = llm.invoke(messages)
    
    # Парсим TopicCard с валидацией
    parsed = safe_json_parse(response.content, {})
    
    # Валидируем через Pydantic
    try:
        topic_card_model = TopicCardModel(**parsed)
        topic_card = topic_card_model.model_dump()
    except ValidationError:
        # Fallback на сырой dict
        topic_card = parsed if isinstance(parsed, dict) else {
            "must": [query],
            "should": [],
            "must_not": [],
            "synonyms": [],
            "expanded_queries": [query],
            "fields_of_study": []
        }
    
    # Применяем фильтры к must_not
    if exclude_reviews:
        if "review" not in topic_card.get("must_not", []):
            topic_card.setdefault("must_not", []).extend(["review", "survey", "tutorial", "overview"])
    
    state["topic_card"] = topic_card
    state["query_strings"] = topic_card.get("expanded_queries", [query])
    
    print(f"   ✓ TopicCard создан:")
    print(f"     - Обязательные термины: {len(topic_card.get('must', []))}")
    print(f"     - Синонимы: {len(topic_card.get('synonyms', []))}")
    print(f"     - Исключения: {len(topic_card.get('must_not', []))}")
    print(f"     - Расширенных запросов: {len(state['query_strings'])}")
    
    state["messages"] = messages + [response]
    state["next_step"] = "retrieve"
    
    return state


# ============================================================================
# ФАЗА 4: Умный выбор источников на основе темы (Задача 4.1)
# ============================================================================

def analyze_query_and_select_sources(state: AgentState) -> AgentState:
    """
    Анализирует запрос через LLM и выбирает оптимальные источники данных
    на основе научной области
    """
    print("\n🧠 Анализ темы и выбор источников...")
    
    query = state["query"]
    llm = get_llm()
    
    prompt = f"""Проанализируй исследовательский запрос и определи научную область.

Запрос: "{query}"

Выбери 1-2 наиболее подходящих области из списка:
- computer_science (AI, ML, NLP, computer vision, algorithms, software engineering)
- biomedicine (medicine, biology, genetics, pharmaceuticals, healthcare)
- physics (physics, astronomy, quantum mechanics, astrophysics)
- mathematics (pure math, applied math, statistics, computational math)
- chemistry (chemistry, materials science, nanotechnology, biochemistry)
- social_sciences (psychology, sociology, economics, political science, education)
- engineering (mechanical, electrical, civil engineering, robotics)
- general (междисциплинарные или неопределенные темы)

Верни СТРОГО JSON:
{{
    "primary_field": "название_области",
    "secondary_field": "название_области или null",
    "confidence": 0.0-1.0,
    "reasoning": "краткое объяснение (1-2 предложения)"
}}"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    analysis = safe_json_parse(response.content, {
        "primary_field": "general",
        "secondary_field": None,
        "confidence": 0.5,
        "reasoning": "Could not determine field"
    })
    
    # Выбор источников на основе области
    primary_field = analysis.get("primary_field", "general")
    secondary_field = analysis.get("secondary_field")
    
    source_weights = select_sources_by_field(primary_field, secondary_field)
    
    state["selected_databases"] = source_weights
    state["field_analysis"] = analysis
    
    print(f"   📚 Detected field: {primary_field} (confidence: {analysis.get('confidence', 0.5):.2f})")
    if secondary_field:
        print(f"   📚 Secondary field: {secondary_field}")
    print(f"   📊 Selected sources: {source_weights}")
    print(f"   💡 Reasoning: {analysis.get('reasoning', '')[:100]}")
    
    return state


def select_sources_by_field(primary_field: str, secondary_field: str = None) -> Dict[str, float]:
    """
    Возвращает словарь {источник: вес} на основе научной области
    
    Веса:
    - 1.0 = основной источник для этой области
    - 0.5-0.9 = дополнительные источники
    - 0.0-0.3 = не используем (фильтруются)
    """
    source_configs = {
        "computer_science": {
            "arxiv": 1.0,
            "semantic_scholar": 1.0,
            "openalex": 0.8,
            "crossref": 0.5,
            "pubmed": 0.0,  # Не используем для CS
        },
        "biomedicine": {
            "pubmed": 1.0,
            "openalex": 0.9,
            "crossref": 0.7,
            "semantic_scholar": 0.5,
            "arxiv": 0.3,
        },
        "physics": {
            "arxiv": 1.0,
            "openalex": 0.9,
            "crossref": 0.7,
            "semantic_scholar": 0.6,
            "pubmed": 0.0,
        },
        "mathematics": {
            "arxiv": 1.0,
            "openalex": 0.9,
            "crossref": 0.7,
            "semantic_scholar": 0.6,
            "pubmed": 0.0,
        },
        "chemistry": {
            "openalex": 1.0,
            "crossref": 0.9,
            "pubmed": 0.6,
            "semantic_scholar": 0.5,
            "arxiv": 0.4,
        },
        "social_sciences": {
            "openalex": 1.0,
            "crossref": 0.9,
            "semantic_scholar": 0.7,
            "pubmed": 0.3,
            "arxiv": 0.0,
        },
        "engineering": {
            "openalex": 1.0,
            "semantic_scholar": 0.9,
            "crossref": 0.8,
            "arxiv": 0.6,
            "pubmed": 0.0,
        },
        "general": {
            "openalex": 0.8,
            "semantic_scholar": 0.8,
            "crossref": 0.8,
            "arxiv": 0.8,
            "pubmed": 0.8,
        }
    }
    
    weights = source_configs.get(primary_field, source_configs["general"])
    
    # Если есть вторичная область - усредняем веса
    if secondary_field and secondary_field in source_configs:
        secondary_weights = source_configs[secondary_field]
        weights = {
            source: (weights.get(source, 0) * 0.7 + secondary_weights.get(source, 0) * 0.3)
            for source in set(list(weights.keys()) + list(secondary_weights.keys()))
        }
    
    # Фильтруем источники с весом > 0.3 (убираем нерелевантные)
    filtered_weights = {
        source: weight
        for source, weight in weights.items()
        if weight > 0.3
    }
    
    return filtered_weights


# ============================================================================
# ФУНКЦИЯ 9: Retriever - мульти-поиск по всем источникам
# ============================================================================

def multi_source_retriever(state: AgentState) -> AgentState:
    """
    Выполняет параллельный поиск по источникам с использованием
    расширенных запросов из TopicCard
    
    УЛУЧШЕНИЕ (Фаза 4): Использует умный выбор источников на основе темы
    """
    print("\n🔍 Функция 9: Multi-source Retriever - поиск по источникам...")
    
    query_strings = state.get("query_strings", [state["query"]])
    time_window = state.get("time_window", 5)
    from_year = 2025 - time_window  # Текущий год минус окно
    
    # Используем первый (основной) запрос для поиска
    main_query = query_strings[0] if query_strings else state["query"]
    
    # ============================================================================
    # УМНЫЙ ВЫБОР ИСТОЧНИКОВ (Фаза 4, Задача 4.1)
    # ============================================================================
    source_weights = state.get("selected_databases", {})
    
    # Если нет анализа темы - используем все источники
    if not source_weights:
        print("   ⚠️  Source selection not performed, using all sources")
        source_weights = {
            "openalex": 1.0,
            "semantic_scholar": 1.0,
            "crossref": 1.0,
            "arxiv": 1.0,
            "pubmed": 1.0
        }
    
    seed_results = []
    per_source = 30  # Количество результатов с каждого источника
    
    print(f"   📌 Основной запрос: '{main_query}'")
    print(f"   📅 Период: с {from_year} года")
    print(f"   🎯 Выбранные источники: {list(source_weights.keys())}")
    
    # Формируем список источников на основе весов (только с весом > 0)
    sources = []
    
    if source_weights.get("openalex", 0) > 0:
        sources.append(("OpenAlex", lambda: search_openalex(main_query, max_results=per_source, from_year=from_year), source_weights["openalex"]))
    
    if source_weights.get("semantic_scholar", 0) > 0:
        sources.append(("Semantic Scholar", lambda: search_semantic_scholar(main_query, max_results=per_source, from_year=from_year), source_weights["semantic_scholar"]))
    
    if source_weights.get("crossref", 0) > 0:
        sources.append(("Crossref", lambda: search_crossref(main_query, max_results=per_source, from_year=from_year), source_weights["crossref"]))
    
    if source_weights.get("arxiv", 0) > 0:
        sources.append(("ArXiv", lambda: search_arxiv(main_query, max_results=per_source), source_weights["arxiv"]))
    
    if source_weights.get("pubmed", 0) > 0:
        sources.append(("PubMed", lambda: search_pubmed(main_query, max_results=per_source), source_weights["pubmed"]))
    
    for source_name, search_func, weight in sources:
        try:
            results = search_func()
            # Добавляем вес источника к каждой статье (для ранжирования)
            for paper in results:
                paper["_source_weight"] = weight
            seed_results.extend(results)
        except Exception as e:
            print(f"   ⚠️  Ошибка в {source_name}: {e}")
    
    # Применяем фильтры к результатам
    filtered_results = apply_search_filters(seed_results, state.get("topic_card", {}))
    
    state["seed_results"] = filtered_results
    state["next_step"] = "deduplicate"
    
    print(f"\n   📊 Собрано сырых результатов: {len(seed_results)}")
    print(f"   🔍 После фильтрации: {len(filtered_results)}")
    
    # Обновляем бюджет
    if "budget" not in state:
        state["budget"] = {}
    state["budget"]["api_calls"] = state["budget"].get("api_calls", 0) + len(sources)
    
    return state


# ============================================================================
# ФУНКЦИЯ: Применение фильтров к результатам поиска
# ============================================================================

def apply_search_filters(results: List[Dict[str, Any]], topic_card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Применяет фильтры к результатам поиска:
    - Исключение обзоров/surveys
    - Фильтрация по новизне
    - Фильтрация по домену
    """
    try:
        from config import SEARCH_FILTERS
        exclude_reviews = SEARCH_FILTERS.get("exclude_reviews", False)
        novelty_level = SEARCH_FILTERS.get("novelty_level", None)
        domain = SEARCH_FILTERS.get("domain", None)
    except:
        exclude_reviews = False
        novelty_level = None
        domain = None
    
    filtered = []
    must_not_terms = topic_card.get("must_not", [])
    
    for paper in results:
        title = paper.get("title", "").lower()
        summary = (paper.get("summary") or "").lower() if paper.get("summary") else ""
        text = f"{title} {summary}"
        
        # Фильтр: исключение обзоров
        if exclude_reviews or must_not_terms:
            exclude_terms = ["review", "survey", "tutorial", "overview", "overview of"]
            exclude_terms.extend([term.lower() for term in must_not_terms])
            
            is_review = any(term in text for term in exclude_terms)
            if is_review:
                continue
        
        # Фильтр: новизна
        if novelty_level:
            try:
                year = int(paper.get("published", "2000"))
                current_year = 2025
                age = current_year - year
                
                if novelty_level == "high" and age > 3:
                    continue
                elif novelty_level == "low" and age < 5:
                    continue
            except:
                pass
        
        # Фильтр: домен (упрощённая версия)
        if domain:
            venue = paper.get("venue", "").lower()
            categories = paper.get("categories", [])
            categories_str = " ".join(categories).lower() if isinstance(categories, list) else ""
            
            if domain == "medicine":
                medical_keywords = ["medical", "clinical", "biomedical", "pubmed", "nejm", "lancet", "jama"]
                if not any(kw in venue or kw in categories_str for kw in medical_keywords):
                    # Проверяем по source
                    if paper.get("source") != "pubmed":
                        continue
            
            elif domain == "cs":
                cs_keywords = ["computer", "ai", "machine learning", "neural", "algorithm", "arxiv", "neurips", "icml"]
                if not any(kw in venue or kw in categories_str for kw in cs_keywords):
                    if paper.get("source") == "pubmed":
                        continue
        
        filtered.append(paper)
    
    return filtered


# ============================================================================
# ФУНКЦИЯ 10: Deduper/Normalizer - очистка дубликатов
# ============================================================================

def deduplicate_and_normalize(state: AgentState) -> AgentState:
    """
    Удаляет дубликаты по DOI/ArXiv ID/названию (fuzzy match)
    и унифицирует метаданные
    """
    print("\n🧹 Функция 10: Deduper/Normalizer - очистка дубликатов...")
    
    seed_results = state["seed_results"]
    
    # Словарь для отслеживания уникальных статей
    seen_identifiers = {}
    corpus_index = []
    
    from difflib import SequenceMatcher
    
    def normalize_title(title):
        """Нормализует название для сравнения"""
        import re
        title = title.lower()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def is_similar_title(title1, title2, threshold=0.85):
        """Проверяет схожесть названий"""
        norm1 = normalize_title(title1)
        norm2 = normalize_title(title2)
        ratio = SequenceMatcher(None, norm1, norm2).ratio()
        return ratio >= threshold
    
    duplicates_count = 0
    
    for paper in seed_results:
        # Проверяем по DOI
        doi = paper.get("doi")
        if doi and doi in seen_identifiers:
            duplicates_count += 1
            # Обогащаем существующую запись
            existing = seen_identifiers[doi]
            if paper.get("citations_total", 0) > existing.get("citations_total", 0):
                existing["citations_total"] = paper["citations_total"]
            continue
        
        # Проверяем по ArXiv ID
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id and arxiv_id in seen_identifiers:
            duplicates_count += 1
            continue
        
        # Проверяем по PMID
        pmid = paper.get("pmid")
        if pmid and pmid in seen_identifiers:
            duplicates_count += 1
            continue
        
        # Проверяем по названию (fuzzy match)
        title = paper.get("title", "")
        is_duplicate = False
        
        for existing_paper in corpus_index[-20:]:  # Проверяем только последние 20 для скорости
            if is_similar_title(title, existing_paper.get("title", "")):
                duplicates_count += 1
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
        
        # Добавляем уникальную статью
        normalized_paper = {
            **paper,
            "normalized_title": normalize_title(title),
            "citations_per_year": 0,
            "recency_score": 0,
            "relevance_score": 0
        }
        
        # Вычисляем citations_per_year
        try:
            year = int(paper.get("published", "2020"))
            years_since = max(2025 - year, 1)
            normalized_paper["citations_per_year"] = paper.get("citations_total", 0) / years_since
        except:
            pass
        
        # Вычисляем recency_score (более свежие = выше)
        try:
            year = int(paper.get("published", "2000"))
            normalized_paper["recency_score"] = max(0, (year - 2000) / 25.0)  # 0 to 1
        except:
            normalized_paper["recency_score"] = 0
        
        corpus_index.append(normalized_paper)
        
        # Регистрируем идентификаторы
        if doi:
            seen_identifiers[doi] = normalized_paper
        if arxiv_id:
            seen_identifiers[arxiv_id] = normalized_paper
        if pmid:
            seen_identifiers[pmid] = normalized_paper
    
    state["corpus_index"] = corpus_index
    state["next_step"] = "rank"
    
    print(f"   ✓ Удалено дубликатов: {duplicates_count}")
    print(f"   ✓ Уникальных статей в корпусе: {len(corpus_index)}")
    
    return state


# ============================================================================
# ФУНКЦИЯ 11: Ranker - гибридное ранжирование статей
# ============================================================================

def hybrid_ranker(state: AgentState) -> AgentState:
    """
    Упрощённый ранкер на основе результатов part_1:
    - Semantic search (dense embeddings) - ОСНОВНОЙ компонент (вес 0.75)
    - Остальные компоненты опциональны и можно включать/выключать через config
    
    Формула (с автоматической нормализацией весов):
    score = w_semantic * semantic_score + [опциональные компоненты]
    """
    print("\n📊 Функция 11: Simplified Ranker - ранжирование статей...")
    print("   💡 На основе результатов part_1: semantic search - основной метод")
    
    import math
    from collections import Counter
    import re
    import numpy as np
    
    corpus = state["corpus_index"]
    query = state["query"]
    
    # Загружаем конфигурацию компонентов ранжирования
    try:
        from config import (
            EXPERIMENTAL, VECTOR_SEARCH_CONFIG, 
            RANKING_COMPONENTS, RANKING_WEIGHTS
        )
        use_vector_search = (
            EXPERIMENTAL.get("enable_semantic_search", True) and 
            RANKING_COMPONENTS.get("semantic_search", True)
        )
        vec_weight = VECTOR_SEARCH_CONFIG.get("weight", 0.75)
        batch_size = VECTOR_SEARCH_CONFIG.get("batch_size", 100)
        embedding_model = VECTOR_SEARCH_CONFIG.get("embedding_model", "text-embedding-3-small")
        
        # Проверяем какие компоненты включены
        use_recency = RANKING_COMPONENTS.get("recency", True)
        use_cpy = RANKING_COMPONENTS.get("citations_per_year", True)
        use_ctotal = RANKING_COMPONENTS.get("citations_total", True)
        use_venue = RANKING_COMPONENTS.get("venue", True)
        use_keywords = RANKING_COMPONENTS.get("keywords_bm25", False)
        
        # Базовые веса из конфига
        w_vec = vec_weight if use_vector_search else 0.0
        w_recency = RANKING_WEIGHTS.get("recency", 0.10) if use_recency else 0.0
        w_cpy = RANKING_WEIGHTS.get("cpy", 0.08) if use_cpy else 0.0
        w_ctotal = RANKING_WEIGHTS.get("ctotal", 0.05) if use_ctotal else 0.0
        w_venue = RANKING_WEIGHTS.get("venue", 0.02) if use_venue else 0.0
        w_kw = RANKING_WEIGHTS.get("keywords", 0.00) if use_keywords else 0.0
        
    except Exception as e:
        print(f"   ⚠️  Ошибка загрузки конфигурации: {e}, используем значения по умолчанию")
        use_vector_search = True
        vec_weight = 0.75
        batch_size = 100
        embedding_model = "text-embedding-3-small"
        use_recency = use_cpy = use_ctotal = use_venue = True
        use_keywords = False
        w_vec = 0.75
        w_recency = 0.10
        w_cpy = 0.08
        w_ctotal = 0.05
        w_venue = 0.02
        w_kw = 0.00
    
    # Нормализуем веса (сумма должна быть 1.0)
    total_weight = w_vec + w_recency + w_cpy + w_ctotal + w_venue + w_kw
    if total_weight > 0:
        w_vec /= total_weight
        w_recency /= total_weight
        w_cpy /= total_weight
        w_ctotal /= total_weight
        w_venue /= total_weight
        w_kw /= total_weight
    
    # Выводим информацию о включённых компонентах
    enabled_components = []
    if use_vector_search:
        enabled_components.append(f"semantic ({w_vec:.1%})")
    if use_recency:
        enabled_components.append(f"recency ({w_recency:.1%})")
    if use_cpy:
        enabled_components.append(f"citations/year ({w_cpy:.1%})")
    if use_ctotal:
        enabled_components.append(f"citations/total ({w_ctotal:.1%})")
    if use_venue:
        enabled_components.append(f"venue ({w_venue:.1%})")
    if use_keywords:
        enabled_components.append(f"keywords ({w_kw:.1%})")
    
    print(f"   📋 Компоненты ранжирования: {', '.join(enabled_components)}")
    
    # Venue scores (упрощённая версия, можно расширить)
    venue_scores = {
        "nature": 1.0, "science": 1.0, "cell": 1.0,
        "nejm": 0.95, "lancet": 0.95, "jama": 0.95,
        "neurips": 0.90, "icml": 0.90, "iclr": 0.90, "cvpr": 0.90,
        "acl": 0.85, "emnlp": 0.85, "naacl": 0.85
    }
    
    def get_venue_score(venue):
        """Получает оценку venue"""
        if not venue or venue == "Unknown":
            return 0.3
        venue_lower = venue.lower()
        for key, score in venue_scores.items():
            if key in venue_lower:
                return score
        return 0.5  # Default для неизвестных venues
    
    def simple_bm25(query_text, document_text):
        """Упрощённый BM25 для keyword matching"""
        query_tokens = set(re.findall(r'\w+', query_text.lower()))
        doc_tokens = re.findall(r'\w+', document_text.lower())
        doc_freq = Counter(doc_tokens)
        
        score = 0
        for token in query_tokens:
            if token in doc_freq:
                tf = doc_freq[token]
                score += (tf / (tf + 1.5)) * 2.0
        
        return score / max(len(query_tokens), 1)
    
    # Векторный поиск (если включён)
    query_embedding = None
    if use_vector_search:
        try:
            print("   🔍 Создание embeddings для векторного поиска...")
            embeddings = get_embeddings()
            query_embedding = np.array(embeddings.embed_query(query))
            print(f"   ✓ Embeddings созданы (модель: {embedding_model})")
        except Exception as e:
            print(f"   ⚠️  Ошибка создания embeddings: {e}")
            use_vector_search = False
            w_vec = 0.0
            w_kw = 0.20
    
    # Нормализация значений для рассчёта
    max_cpy = max([p.get("citations_per_year", 0) for p in corpus] + [1])
    max_ctotal = max([p.get("citations_total", 0) for p in corpus] + [1])
    
    # Создаём FAISS индекс для документов (если включён векторный поиск)
    faiss_index = None
    doc_embeddings_list = []
    if use_vector_search and query_embedding is not None:
        try:
            print("   📊 Создание embeddings и FAISS индекса для документов...")
            embeddings = get_embeddings()
            
            # Создаём тексты для embedding (title + summary)
            doc_texts = []
            for paper in corpus:
                doc_text = f"{paper.get('title', '')} {paper.get('summary', '')[:500]}"
                doc_texts.append(doc_text)
            
            # Создаём embeddings батчами (чтобы не превышать лимиты)
            for i in range(0, len(doc_texts), batch_size):
                batch = doc_texts[i:i+batch_size]
                batch_embeddings = embeddings.embed_documents(batch)
                doc_embeddings_list.extend(batch_embeddings)
            
            # Создаём FAISS индекс
            import faiss
            
            # Определяем размерность векторов
            dimension = len(doc_embeddings_list[0])
            
            # Создаём индекс FAISS (L2 расстояние, можно использовать InnerProduct для косинусного)
            # Используем IndexFlatIP (Inner Product) для косинусного сходства после нормализации
            faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусного сходства
            
            # Нормализуем векторы для косинусного сходства (L2 нормализация)
            doc_embeddings_array = np.array(doc_embeddings_list).astype('float32')
            faiss.normalize_L2(doc_embeddings_array)
            
            # Добавляем векторы в индекс
            faiss_index.add(doc_embeddings_array)
            
            print(f"   ✓ Создан FAISS индекс с {faiss_index.ntotal} векторами (размерность: {dimension})")
        except Exception as e:
            print(f"   ⚠️  Ошибка создания FAISS индекса: {e}")
            use_vector_search = False
            w_vec = 0.0
            w_kw = 0.20
            faiss_index = None
    
    # Вычисляем векторные сходства через FAISS (если включено) ПЕРЕД расчетом финальных оценок
    vec_scores_dict = {}
    if use_vector_search and faiss_index is not None and query_embedding is not None:
        try:
            # Нормализуем query embedding для косинусного сходства
            query_vec = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vec)
            
            # Ищем похожие документы (k = количество документов)
            k = min(len(corpus), faiss_index.ntotal)
            distances, indices = faiss_index.search(query_vec, k)
            
            # Создаём словарь для быстрого доступа к сходствам
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(corpus):
                    # Inner Product после нормализации = косинусное сходство
                    # Нормализуем от [-1, 1] к [0, 1]
                    vec_scores_dict[idx] = (distance + 1) / 2
            
            print(f"   ✓ Векторные сходства вычислены через FAISS")
        except Exception as e:
            print(f"   ⚠️  Ошибка поиска через FAISS: {e}")
            vec_scores_dict = {}
    
    # Рассчитываем финальные оценки
    for i, paper in enumerate(corpus):
        # Recency score уже вычислен
        recency = paper.get("recency_score", 0)
        
        # Citations per year (нормализовано)
        cpy = paper.get("citations_per_year", 0) / max_cpy if max_cpy > 0 else 0
        
        # Total citations (log scale, нормализовано)
        ctotal = paper.get("citations_total", 0)
        log_ctotal = math.log1p(ctotal) / math.log1p(max_ctotal) if max_ctotal > 0 else 0
        
        # Venue score
        venue = paper.get("venue", "")
        venue_score = get_venue_score(venue)
        
        # Keyword matching (BM25-like) - только если включено
        kw_score = 0.0
        if use_keywords:
            doc_text = f"{paper.get('title', '')} {paper.get('summary', '')}"
            kw_score = simple_bm25(query.lower(), doc_text)
        
        # Vector similarity из FAISS - ОСНОВНОЙ компонент
        vec_score = vec_scores_dict.get(i, 0.0) if use_vector_search else 0.0
        
        # Итоговая оценка (semantic search - основной компонент)
        final_score = w_vec * vec_score
        
        # Добавляем опциональные компоненты
        if use_recency:
            final_score += w_recency * recency
        if use_cpy:
            final_score += w_cpy * cpy
        if use_ctotal:
            final_score += w_ctotal * log_ctotal
        if use_venue:
            final_score += w_venue * venue_score
        if use_keywords:
            final_score += w_kw * kw_score
        
        # ============================================================================
        # ФАЗА 4: Применяем вес источника (Задача 4.1)
        # ============================================================================
        source_weight = paper.get("_source_weight", 1.0)
        final_score = final_score * source_weight
        
        paper["relevance_score"] = final_score
        
        # Сохраняем компоненты оценки (только включённые)
        score_components = {}
        if use_vector_search:
            score_components["semantic"] = vec_score * w_vec
        if use_recency:
            score_components["recency"] = recency * w_recency
        if use_cpy:
            score_components["citations_per_year"] = cpy * w_cpy
        if use_ctotal:
            score_components["citations_total"] = log_ctotal * w_ctotal
        if use_venue:
            score_components["venue"] = venue_score * w_venue
        if use_keywords:
            score_components["keywords"] = kw_score * w_kw
        
        paper["score_components"] = score_components
    
    # Сортируем по оценке
    ranked_papers = sorted(corpus, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Берём топ-N для reranking
    top_n = min(state.get("max_papers", 40), len(ranked_papers))
    top_papers_for_rerank = ranked_papers[:top_n]
    
    # LLM-rerank поверх топ-N (если включено)
    try:
        from config import EXPERIMENTAL, RERANK_CONFIG
        enable_rerank = EXPERIMENTAL.get("enable_llm_rerank", False)
        rerank_top_k = RERANK_CONFIG.get("top_k", 20)
    except:
        enable_rerank = False
        rerank_top_k = 20
    
    if enable_rerank and len(top_papers_for_rerank) > 5:
        print(f"   🔄 LLM-rerank для топ-{min(rerank_top_k, len(top_papers_for_rerank))} статей...")
        reranked_papers = llm_rerank(
            query=query,
            papers=top_papers_for_rerank[:rerank_top_k],
            llm=get_llm()
        )
        # Проверяем, что reranked_papers не None
        if reranked_papers:
            # Объединяем reranked с остальными
            reranked_indices = {p.get("_original_index") for p in reranked_papers if "_original_index" in p}
            final_ranked = reranked_papers + [p for i, p in enumerate(top_papers_for_rerank) if i not in reranked_indices]
            # Удаляем временные поля
            for p in final_ranked:
                p.pop("_original_index", None)
            state["ranked_papers"] = final_ranked[:top_n]
        else:
            # Если rerank не удался, используем исходное ранжирование
            state["ranked_papers"] = top_papers_for_rerank
    else:
        state["ranked_papers"] = top_papers_for_rerank
    
    # Citation snowballing (если включено)
    try:
        from config import EXPERIMENTAL, SNOWBALL_CONFIG
        if EXPERIMENTAL.get("enable_snowballer", False) and SNOWBALL_CONFIG.get("enabled", False):
            print(f"   🔗 Citation snowballing...")
            expanded_papers = citation_snowballing(
                seed_papers=state["ranked_papers"][:10],  # Топ-10 для расширения
                max_expansion=SNOWBALL_CONFIG.get("max_expansion", 20),
                min_citations=SNOWBALL_CONFIG.get("min_citations", 5),
                forward=SNOWBALL_CONFIG.get("forward_citations", True),
                backward=SNOWBALL_CONFIG.get("backward_citations", True)
            )
            if expanded_papers:
                # Добавляем новые статьи в корпус
                existing_ids = {p.get("doi") or p.get("arxiv_id") or p.get("url") for p in state["ranked_papers"]}
                new_papers = [p for p in expanded_papers if (p.get("doi") or p.get("arxiv_id") or p.get("url")) not in existing_ids]
                if new_papers:
                    state["ranked_papers"].extend(new_papers[:SNOWBALL_CONFIG.get("max_expansion", 20)])
                    print(f"      ✓ Добавлено {len(new_papers)} статей через citation snowballing")
    except Exception as e:
        print(f"      ⚠️  Ошибка citation snowballing: {e}")
    
    state["next_step"] = "read_pdfs"
    
    print(f"   ✓ Ранжировано статей: {len(ranked_papers)}")
    print(f"   ✓ Выбрано топ-{top_n} для дальнейшего анализа")
    
    # Показываем топ-3 с их оценками
    print("\n   🏆 Топ-3 статьи:")
    for i, paper in enumerate(ranked_papers[:3], 1):
        print(f"      {i}. {paper.get('title', 'No title')[:60]}...")
        print(f"         Score: {paper['relevance_score']:.3f} | Год: {paper.get('published')} | "
              f"Цит.: {paper.get('citations_total', 0)}")
    
    return state


# ============================================================================
# ФУНКЦИИ: LLM-rerank и Citation Snowballing
# ============================================================================

def llm_rerank(query: str, papers: List[Dict[str, Any]], llm) -> Optional[List[Dict[str, Any]]]:
    """
    LLM-rerank: использует LLM для переранжирования топ-N статей
    Работает как cross-encoder, сравнивая запрос с каждой статьёй
    Возвращает None если ранжирование не удалось
    """
    if len(papers) == 0:
        return papers
    
    system_prompt = """Ты - эксперт по оценке релевантности научных статей к запросу.

Оцени релевантность каждой статьи к запросу по шкале от 0.0 до 1.0, где:
- 1.0 = идеально соответствует запросу
- 0.8-0.9 = очень релевантна
- 0.6-0.7 = релевантна
- 0.4-0.5 = частично релевантна
- 0.0-0.3 = не релевантна

Верни JSON массив с оценками:
[
  {"index": 0, "relevance_score": 0.95, "reason": "краткое объяснение"},
  {"index": 1, "relevance_score": 0.87, "reason": "краткое объяснение"},
  ...
]
"""
    
    # Формируем промпт с информацией о статьях
    papers_info = []
    for i, paper in enumerate(papers):
        papers_info.append(f"""
Статья {i}:
Название: {paper.get('title', 'Unknown')}
Аннотация: {paper.get('summary', 'No abstract')[:300]}
Год: {paper.get('published')}
Цитирований: {paper.get('citations_total', 0)}
""")
    
    user_prompt = f"""Запрос: {query}

Статьи для оценки:
{chr(10).join(papers_info)}

Оцени релевантность каждой статьи к запросу."""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Парсим ответ с валидацией
        parsed = safe_json_parse(response.content, [])
        
        if not isinstance(parsed, list) or len(parsed) == 0:
            print(f"      ⚠️  Не удалось распарсить результаты LLM-rerank")
            return None
        
        # Создаём словарь оценок
        rerank_scores = {}
        for item in parsed:
            if isinstance(item, dict) and "index" in item and "relevance_score" in item:
                idx = item["index"]
                score = float(item.get("relevance_score", 0.5))
                rerank_scores[idx] = score
        
        # Переранжируем статьи по новым оценкам
        reranked = []
        for i, paper in enumerate(papers):
            new_score = rerank_scores.get(i, paper.get("relevance_score", 0.5))
            paper_copy = paper.copy()
            paper_copy["rerank_score"] = new_score
            paper_copy["relevance_score"] = new_score  # Обновляем общий score
            paper_copy["_original_index"] = i
            reranked.append(paper_copy)
        
        # Сортируем по новым оценкам
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        print(f"      ✓ LLM-rerank завершён для {len(reranked)} статей")
        return reranked
    
    except Exception as e:
        print(f"      ⚠️  Ошибка при LLM-rerank: {e}")
        return None
    
    except Exception as e:
        print(f"      ⚠️  Ошибка LLM-rerank: {e}")
        return papers


def citation_snowballing(
    seed_papers: List[Dict[str, Any]],
    max_expansion: int = 20,
    min_citations: int = 5,
    forward: bool = True,
    backward: bool = True
) -> List[Dict[str, Any]]:
    """
    Расширяет корпус статей через citation snowballing:
    - Forward citations: кто цитирует seed papers
    - Backward citations: на кого ссылаются seed papers
    """
    expanded_papers = []
    
    try:
        import requests
        import time
        
        for paper in seed_papers[:5]:  # Ограничиваем для скорости
            paper_id = paper.get("doi") or paper.get("arxiv_id")
            if not paper_id:
                continue
            
            # Forward citations через Semantic Scholar
            if forward:
                try:
                    # Используем Semantic Scholar API для получения citations
                    if paper.get("doi"):
                        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper['doi']}/citations"
                    elif paper.get("arxiv_id"):
                        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper['arxiv_id']}/citations"
                    else:
                        continue
                    
                    response = requests.get(
                        s2_url,
                        params={"limit": 10, "fields": "title,authors,abstract,year,citationCount,url,externalIds"},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for cited_paper in data.get("data", [])[:5]:
                            if cited_paper.get("citationCount", 0) >= min_citations:
                                expanded_papers.append({
                                    "title": cited_paper.get("title", ""),
                                    "authors": [a.get("name", "") for a in cited_paper.get("authors", [])],
                                    "summary": cited_paper.get("abstract", "")[:500],
                                    "url": cited_paper.get("url", ""),
                                    "doi": cited_paper.get("externalIds", {}).get("DOI"),
                                    "published": str(cited_paper.get("year", "Unknown")),
                                    "citations_total": cited_paper.get("citationCount", 0),
                                    "source": "semantic_scholar",
                                    "_snowball_type": "forward"
                                })
                    
                    time.sleep(0.1)
                except Exception as e:
                    pass
            
            # Backward citations (references)
            if backward and len(expanded_papers) < max_expansion:
                try:
                    if paper.get("doi"):
                        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{paper['doi']}/references"
                    elif paper.get("arxiv_id"):
                        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper['arxiv_id']}/references"
                    else:
                        continue
                    
                    response = requests.get(
                        s2_url,
                        params={"limit": 10, "fields": "title,authors,abstract,year,citationCount,url,externalIds"},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for ref_paper in data.get("data", [])[:5]:
                            if ref_paper.get("citationCount", 0) >= min_citations:
                                expanded_papers.append({
                                    "title": ref_paper.get("title", ""),
                                    "authors": [a.get("name", "") for a in ref_paper.get("authors", [])],
                                    "summary": ref_paper.get("abstract", "")[:500],
                                    "url": ref_paper.get("url", ""),
                                    "doi": ref_paper.get("externalIds", {}).get("DOI"),
                                    "published": str(ref_paper.get("year", "Unknown")),
                                    "citations_total": ref_paper.get("citationCount", 0),
                                    "source": "semantic_scholar",
                                    "_snowball_type": "backward"
                                })
                    
                    time.sleep(0.1)
                except Exception as e:
                    pass
            
            if len(expanded_papers) >= max_expansion:
                break
        
        # Дедупликация
        seen_ids = set()
        unique_papers = []
        for p in expanded_papers:
            paper_id = p.get("doi") or p.get("arxiv_id") or p.get("url")
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(p)
        
        return unique_papers[:max_expansion]
    
    except Exception as e:
        print(f"      ⚠️  Ошибка citation snowballing: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 12: PDF Reader - чтение полных текстов по ссылкам
# ============================================================================

def get_pdf_url_from_paper(paper: Dict[str, Any]) -> str:
    """
    Получает URL для скачивания PDF из метаданных статьи
    Использует прямые ссылки из метаданных
    """
    # 1. Для ArXiv - строим прямую ссылку на PDF
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        # Убираем префикс если есть
        arxiv_clean = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()
        return f"https://arxiv.org/pdf/{arxiv_clean}.pdf"
    
    # 2. Проверяем прямую ссылку (может быть PDF)
    url = paper.get("url", "")
    if url and url.endswith(".pdf"):
        return url
    
    # 3. Для DOI - некоторые журналы предоставляют PDF через DOI
    # Но это сложнее, пока пропускаем
    
    return None


def extract_pdf_sections(pdf_text: str) -> Dict[str, str]:
    """
    Извлекает структурированные секции из PDF текста
    Использует regex для поиска заголовков секций
    """
    import re
    
    sections = {
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "related_work": "",
        "limitations": "",
        "future_work": "",
        "contributions": "",
        "experimental_setup": ""
    }
    
    # Паттерны для поиска секций
    section_patterns = {
        "introduction": [
            r'##?\s*\d*\.?\s*Introduction',
            r'##?\s*\d*\.?\s*Background',
            r'##?\s*\d*\.?\s*Motivation',
        ],
        "methods": [
            r'##?\s*\d*\.?\s*Methods?',
            r'##?\s*\d*\.?\s*Methodology',
            r'##?\s*\d*\.?\s*Approach',
            r'##?\s*\d*\.?\s*Algorithm',
        ],
        "results": [
            r'##?\s*\d*\.?\s*Results?',
            r'##?\s*\d*\.?\s*Experiments?',
            r'##?\s*\d*\.?\s*Evaluation',
        ],
        "discussion": [
            r'##?\s*\d*\.?\s*Discussion',
            r'##?\s*\d*\.?\s*Analysis',
        ],
        "conclusion": [
            r'##?\s*\d*\.?\s*Conclusion',
            r'##?\s*\d*\.?\s*Conclusions?',
            r'##?\s*\d*\.?\s*Summary',
        ],
        "related_work": [
            r'##?\s*\d*\.?\s*Related\s+Work',
            r'##?\s*\d*\.?\s*Related\s+Literature',
            r'##?\s*\d*\.?\s*Background\s+and\s+Related\s+Work',
        ],
        "limitations": [
            r'##?\s*\d*\.?\s*Limitations?',
            r'##?\s*\d*\.?\s*Discussion\s+of\s+Limitations',
            r'##?\s*\d*\.?\s*Threats\s+to\s+Validity',
        ],
        "future_work": [
            r'##?\s*\d*\.?\s*Future\s+(Work|Directions|Research)',
            r'##?\s*\d*\.?\s*Open\s+Problems?',
        ],
        "contributions": [
            r'##?\s*\d*\.?\s*Contributions?',
            r'##?\s*\d*\.?\s*Main\s+Contributions?',
        ],
        "experimental_setup": [
            r'##?\s*\d*\.?\s*Experimental\s+Setup',
            r'##?\s*\d*\.?\s*Implementation',
            r'##?\s*\d*\.?\s*Experimental\s+Settings?',
        ]
    }
    
    # Находим все секции
    found_sections = {}
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, pdf_text, re.IGNORECASE)
            if match:
                section_start = match.start()
                # Ищем следующую секцию или конец
                next_section_match = re.search(
                    r'\n##?\s*\d+\.',
                    pdf_text[section_start + 10:]
                )
                section_end = (
                    section_start + 10 + next_section_match.start()
                    if next_section_match
                    else len(pdf_text)
                )
                found_sections[section_name] = pdf_text[section_start:section_end]
                break
    
    return found_sections


def download_and_parse_pdf(url: str) -> str:
    """
    Скачивает и парсит PDF по ссылке
    Возвращает текст или None при ошибке
    """
    try:
        import requests
        import pdfplumber
        import io
        
        # Скачиваем PDF
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # Проверяем content-type
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not url.endswith(".pdf"):
            return None
        
        pdf_content = response.content
        
        # Парсим PDF
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            # Ограничиваем количество страниц
            max_pages = min(50, len(pdf.pages))
            for page in pdf.pages[:max_pages]:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except:
                    continue
        
        full_text = "\n\n".join(text_parts)
        
        # Ограничиваем размер (чтобы не превышать лимиты токенов)
        max_chars = 50000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[... текст обрезан ...]"
        
        return full_text
    
    except Exception as e:
        # Fallback на PyMuPDF если pdfplumber не работает
        try:
            import fitz  # PyMuPDF
            import requests
            import io
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            doc = fitz.open(stream=response.content, filetype="pdf")
            text_parts = []
            for i in range(min(50, len(doc))):
                text_parts.append(doc[i].get_text())
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            if len(full_text) > 50000:
                full_text = full_text[:50000] + "\n\n[... текст обрезан ...]"
            
            return full_text
        
        except Exception as e2:
            return None


def read_pdfs(state: AgentState) -> AgentState:
    """
    Читает полные тексты PDF для топ-статей по прямым ссылкам
    """
    print("\n📄 Функция 12: PDF Reader - чтение полных текстов...")
    
    ranked_papers = state.get("ranked_papers", [])
    
    # Берём топ-5 статей для чтения PDF (чтобы не тратить много времени)
    top_for_pdf = min(5, len(ranked_papers))
    papers_to_read = ranked_papers[:top_for_pdf]
    
    print(f"   📥 Пытаемся прочитать PDF для топ-{top_for_pdf} статей...")
    
    pdf_texts = {}
    successful = 0
    
    for i, paper in enumerate(papers_to_read, 1):
        title = paper.get("title", "Unknown")[:60]
        print(f"      [{i}/{top_for_pdf}] {title}...")
        
        # Получаем URL для PDF
        pdf_url = get_pdf_url_from_paper(paper)
        
        if not pdf_url:
            print(f"         ⚠️  Нет доступной ссылки на PDF")
            continue
        
        # Скачиваем и парсим PDF
        pdf_text = download_and_parse_pdf(pdf_url)
        
        if not pdf_text or len(pdf_text) < 100:
            print(f"         ⚠️  Не удалось прочитать PDF")
            continue
        
        # Сохраняем текст
        paper_id = paper.get("doi") or paper.get("arxiv_id") or paper.get("url", f"paper_{i}")
        pdf_texts[paper_id] = pdf_text
        successful += 1
        print(f"         ✓ PDF прочитан ({len(pdf_text)} символов)")
    
    state["pdf_texts"] = pdf_texts
    state["next_step"] = "summarize"
    
    print(f"\n   ✓ Успешно прочитано PDF: {successful}/{top_for_pdf}")
    
    return state


# ============================================================================
# ФУНКЦИИ: Поиск GitHub репозиториев
# ============================================================================

def search_github_repositories(paper: Dict[str, Any], max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Ищет GitHub репозитории связанные со статьёй через GitHub API
    """
    try:
        from config import GITHUB_CONFIG
        if not GITHUB_CONFIG.get("enabled", True) or not GITHUB_CONFIG.get("use_api", True):
            return []
    except:
        pass
    
    import requests
    import os
    
    github_token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    # Формируем поисковый запрос
    title = paper.get("title", "")
    authors = paper.get("authors", [])
    first_author = authors[0].split()[-1] if authors else ""
    
    # Пробуем разные варианты поиска
    search_queries = []
    if title:
        # Извлекаем ключевые слова из названия
        keywords = title.split()[:5]  # Первые 5 слов
        search_queries.append(" ".join(keywords))
    if first_author:
        search_queries.append(first_author)
    
    repos = []
    for query in search_queries[:2]:  # Максимум 2 запроса
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"{query} in:name,description,readme",
                "sort": "stars",
                "order": "desc",
                "per_page": max_results
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for repo in data.get("items", []):
                    repos.append({
                        "name": repo.get("full_name", ""),
                        "url": repo.get("html_url", ""),
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "language": repo.get("language", ""),
                        "description": repo.get("description", ""),
                        "updated_at": repo.get("updated_at", "")
                    })
                break  # Если нашли результаты, прекращаем поиск
        except Exception as e:
            continue
    
    return repos[:max_results]


def web_search_github(paper: Dict[str, Any], max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Ищет GitHub репозитории через веб-поиск (DuckDuckGo) как fallback
    """
    try:
        from config import GITHUB_CONFIG
        if not GITHUB_CONFIG.get("enabled", True) or not GITHUB_CONFIG.get("use_web_search", True):
            return []
    except:
        pass
    
    try:
        from duckduckgo_search import DDGS
        
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        first_author = authors[0] if authors else ""
        
        # Формируем поисковый запрос
        search_query = f"{title} github" if title else f"{first_author} github"
        
        repos = []
        with DDGS() as ddgs:
            results = ddgs.text(
                search_query,
                max_results=max_results * 2  # Берём больше, потом фильтруем
            )
            
            for result in results:
                url = result.get("href", "")
                if "github.com" in url:
                    # Извлекаем информацию из URL
                    parts = url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
                    if len(parts) >= 2:
                        repos.append({
                            "name": f"{parts[0]}/{parts[1]}",
                            "url": url,
                            "stars": 0,  # Не доступно через веб-поиск
                            "forks": 0,
                            "language": "",
                            "description": result.get("body", "")[:200],
                            "updated_at": ""
                        })
                        if len(repos) >= max_results:
                            break
        
        return repos
    except ImportError:
        # DuckDuckGo не установлен
        return []
    except Exception as e:
        return []


def find_github_for_paper(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Комбинированный поиск GitHub: сначала API, потом веб-поиск
    """
    repos = []
    
    # Пробуем GitHub API
    repos = search_github_repositories(paper)
    
    # Если не нашли через API, пробуем веб-поиск
    if not repos:
        repos = web_search_github(paper)
    
    return repos


# ============================================================================
# ФУНКЦИЯ 13: Summarizer - создание литературной матрицы
# ============================================================================

def create_literature_matrix(state: AgentState) -> AgentState:
    """
    Создаёт структурированные конспекты топ-статей с использованием LLM
    Использует полный текст PDF если доступен
    """
    print("\n📝 Функция 13: Summarizer - создание литературной матрицы...")
    
    ranked_papers = state["ranked_papers"]
    pdf_texts = state.get("pdf_texts", {})
    llm = get_llm()
    lit_matrix = []
    
    # Берём топ-10 для детального анализа (чтобы не тратить много токенов)
    top_papers = ranked_papers[:min(10, len(ranked_papers))]
    
    system_prompt = """Ты - эксперт по анализу научных статей.
    
На основе названия, аннотации и метаданных статьи создай структурированный конспект.
Если доступен полный текст PDF, используй его для более детального анализа.

ВАЖНО: Извлекай информацию из следующих секций статьи:
- Introduction: проблема, мотивация, цели
- Methods/Methodology: методы, подходы, алгоритмы
- Results: результаты, метрики, эксперименты
- Discussion: обсуждение результатов, интерпретация
- Conclusion: выводы, итоги
- Related Work: обзор связанных работ
- Contributions: вклад работы (может быть отдельной секцией или в Introduction)
- Experimental Setup: детали экспериментов, гиперпараметры, конфигурация
- Limitations: ограничения (может быть в Discussion или отдельной секцией)
- Future Work: будущие направления (может быть в Conclusion или отдельной секцией)

Обрати особое внимание на:
- Упоминания GitHub репозиториев, кода, данных (для reproducibility_info)
- Ссылки на датасеты, код, дополнительные материалы
- Детали экспериментальной настройки

Верни СТРОГО JSON:
{
  "problem": "краткое описание проблемы из Introduction",
  "methods": ["метод 1", "метод 2"],
  "datasets": ["датасет 1"],
  "metrics": ["метрика 1"],
  "key_findings": "ключевые выводы из Results/Discussion",
  "limitations": "ограничения исследования",
  "future_work": "направления будущих исследований",
  "contributions": "основной вклад работы",
  "related_work_summary": "краткий обзор связанных работ из Related Work",
  "experimental_setup": "детали экспериментов: гиперпараметры, конфигурация, оборудование",
  "reproducibility_info": "наличие кода/данных: GitHub ссылки, dataset links, упоминания кода",
  "discussion": "обсуждение результатов, интерпретация, сравнение с литературой",
  "conclusion": "выводы, итоги работы"
}
"""
    
    print(f"   Анализируем топ-{len(top_papers)} статей...")
    
    for i, paper in enumerate(top_papers, 1):
        try:
            paper_id = paper.get("doi") or paper.get("arxiv_id") or paper.get("url", f"paper_{i}")
            pdf_text = pdf_texts.get(paper_id)
            has_full_text = pdf_text is not None
            
            paper_info = f"""
Название: {paper.get('title', 'Unknown')}
Авторы: {', '.join(paper.get('authors', [])[:5])}
Год: {paper.get('published')}
Venue: {paper.get('venue')}
Аннотация: {paper.get('summary', 'No abstract')[:800]}
"""
            
            # Если есть полный текст PDF, добавляем его структурированно
            if has_full_text:
                # Извлекаем секции из PDF
                pdf_sections = extract_pdf_sections(pdf_text)
                
                # Формируем структурированное представление
                sections_text = ""
                for section_name, section_text in pdf_sections.items():
                    if section_text:
                        # Ограничиваем размер каждой секции
                        section_excerpt = section_text[:3000] if len(section_text) > 3000 else section_text
                        sections_text += f"\n\n=== {section_name.upper()} ===\n{section_excerpt}"
                
                # Если секции не найдены, используем полный текст
                if not sections_text:
                    pdf_excerpt = pdf_text[:20000] if len(pdf_text) > 20000 else pdf_text
                    sections_text = f"\n\n=== ПОЛНЫЙ ТЕКСТ СТАТЬИ (PDF) ===\n{pdf_excerpt}"
                paper_info += sections_text
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=paper_info)
            ]
            
            # ============================================================================
            # УЛУЧШЕНИЕ: Retry логика с exponential backoff (Задача 1.2)
            # ============================================================================
            response = None
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    response = llm.invoke(messages)
                    break  # Успешно получили ответ
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 секунд
                        print(f"         ⚠️  Retry {retry_count}/{max_retries} after {wait_time}s: {str(e)[:50]}")
                        import time
                        time.sleep(wait_time)
                    else:
                        # Исчерпали все попытки
                        raise last_error
            
            # Парсим ответ с валидацией
            parsed = safe_json_parse(response.content, {})
            
            # Валидируем через Pydantic
            try:
                summary_model = StructuredSummaryModel(**parsed)
                summary = summary_model.model_dump()
            except ValidationError:
                # Fallback на сырой dict с дефолтными значениями
                summary = {
                    "problem": parsed.get("problem", ""),
                    "methods": parsed.get("methods", []),
                    "datasets": parsed.get("datasets", []),
                    "metrics": parsed.get("metrics", []),
                    "key_findings": parsed.get("key_findings", ""),
                    "limitations": parsed.get("limitations", ""),
                    "future_work": parsed.get("future_work", ""),
                    "contributions": parsed.get("contributions", ""),
                    "related_work_summary": parsed.get("related_work_summary", ""),
                    "experimental_setup": parsed.get("experimental_setup", ""),
                    "reproducibility_info": parsed.get("reproducibility_info", ""),
                    "discussion": parsed.get("discussion", ""),
                    "conclusion": parsed.get("conclusion", "")
                }
            except Exception as e:
                # ============================================================================
                # УЛУЧШЕНИЕ: Fallback на аннотацию (Задача 1.2)
                # ============================================================================
                # Если вообще не удалось распарсить, используем аннотацию статьи
                abstract = paper.get('summary', paper.get('abstract', ''))
                
                summary = {
                    "problem": abstract[:300] if abstract else "Не удалось извлечь",
                    "methods": [],
                    "datasets": [],
                    "metrics": [],
                    "key_findings": abstract if abstract else "Не удалось извлечь детали",
                    "limitations": "LLM не смог извлечь детали. Используется аннотация.",
                    "future_work": "",
                    "contributions": "",
                    "related_work_summary": "",
                    "experimental_setup": "",
                    "reproducibility_info": "",
                    "discussion": "",
                    "conclusion": ""
                }
                print(f"         ⚠️  Fallback на аннотацию для статьи {i}: {str(e)[:50]}")
            
            # Ищем GitHub репозитории для статьи
            github_repos = []
            if has_full_text or summary.get("reproducibility_info"):
                # Если есть полный текст или упоминания кода, ищем GitHub
                try:
                    github_repos = find_github_for_paper(paper)
                    if github_repos:
                        print(f"         🔗 Найдено GitHub репозиториев: {len(github_repos)}")
                except Exception as e:
                    pass
            
            lit_matrix.append({
                **paper,
                "structured_summary": summary,
                "has_full_text": has_full_text,
                "github_repos": github_repos
            })
            
            status_icon = "📄" if has_full_text else "📝"
            print(f"      {status_icon} {i}/{len(top_papers)}: {paper.get('title', 'Unknown')[:50]}...")
            
        except Exception as e:
            # ============================================================================
            # УЛУЧШЕНИЕ: Fallback на аннотацию при критической ошибке (Задача 1.2)
            # ============================================================================
            print(f"      ⚠️  Ошибка анализа статьи {i}: {e}")
            
            # Используем аннотацию как fallback
            abstract = paper.get('summary', paper.get('abstract', ''))
            
            lit_matrix.append({
                **paper,
                "structured_summary": {
                    "problem": abstract[:300] if abstract else "Ошибка анализа",
                    "methods": [],
                    "datasets": [],
                    "metrics": [],
                    "key_findings": abstract if abstract else "Не удалось извлечь",
                    "limitations": f"Критическая ошибка LLM: {str(e)[:100]}",
                    "future_work": "",
                    "contributions": "",
                    "related_work_summary": "",
                    "experimental_setup": "",
                    "reproducibility_info": "",
                    "discussion": "",
                    "conclusion": ""
                },
                "has_full_text": False,
                "github_repos": []
            })
    
    state["lit_matrix"] = lit_matrix
    state["next_step"] = "find_gaps"
    
    papers_with_pdf = sum(1 for p in lit_matrix if p.get("has_full_text", False))
    print(f"   ✓ Литературная матрица создана: {len(lit_matrix)} статей проанализировано")
    if papers_with_pdf > 0:
        print(f"   📄 Из них с полным текстом PDF: {papers_with_pdf}")
    
    return state


# ============================================================================
# ФУНКЦИИ: Стратегии анализа для GapMiner
# ============================================================================

def analyze_temporal_evolution(lit_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Анализ эволюции методов во времени
    Выявляет новые тренды, устаревшие подходы, методы без развития
    """
    gaps = []
    
    # Группируем методы по годам
    methods_by_year = {}
    for paper in lit_matrix:
        year = paper.get("published", "Unknown")
        try:
            year_int = int(year) if year != "Unknown" else None
        except:
            year_int = None
        
        if year_int is None:
            continue
            
        summary = paper.get("structured_summary", {})
        methods = summary.get("methods", [])
        
        if year_int not in methods_by_year:
            methods_by_year[year_int] = []
        methods_by_year[year_int].extend(methods)
    
    if len(methods_by_year) < 2:
        return gaps
    
    # Находим методы которые появились недавно (новые тренды)
    recent_years = sorted(methods_by_year.keys())[-2:]
    old_years = sorted(methods_by_year.keys())[:-2] if len(methods_by_year) > 2 else []
    
    recent_methods = set()
    for year in recent_years:
        recent_methods.update(methods_by_year[year])
    
    old_methods = set()
    for year in old_years:
        old_methods.update(methods_by_year[year])
    
    # Методы которые были популярны раньше, но исчезли
    disappeared_methods = old_methods - recent_methods
    if disappeared_methods:
        gaps.append({
            "gap": f"Методы {', '.join(list(disappeared_methods)[:3])} были популярны ранее, но не используются в последних работах. Возможно устарели или заменены.",
            "type": "temporal",
            "severity": "medium",
            "evidence": [p.get("title", "") for p in lit_matrix if any(m in old_methods for m in p.get("structured_summary", {}).get("methods", []))][:3],
            "reasoning": "Временной анализ показывает сдвиг в использовании методов",
            "potential_impact": "Понимание эволюции методов может помочь предсказать будущие тренды",
            "related_methods": list(recent_methods)[:5],
            "feasibility": "medium"
        })
    
    return gaps


def detect_contradictions(lit_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Выявление противоречивых результатов на одинаковых датасетах/метриках
    """
    gaps = []
    
    # Группируем статьи по датасетам
    papers_by_dataset = {}
    for paper in lit_matrix:
        summary = paper.get("structured_summary", {})
        datasets = summary.get("datasets", [])
        for dataset in datasets:
            if dataset not in papers_by_dataset:
                papers_by_dataset[dataset] = []
            papers_by_dataset[dataset].append(paper)
    
    # Ищем противоречия
    for dataset, papers in papers_by_dataset.items():
        if len(papers) < 2:
            continue
        
        # Сравниваем результаты
        findings = []
        for paper in papers:
            summary = paper.get("structured_summary", {})
            findings.append({
                "title": paper.get("title", ""),
                "findings": summary.get("key_findings", ""),
                "methods": summary.get("methods", [])
            })
        
        # Если есть противоречивые выводы (простая эвристика)
        if len(set(f["findings"][:100] for f in findings)) > 1:
            gaps.append({
                "gap": f"Противоречивые результаты на датасете {dataset}. Разные работы дают разные выводы.",
                "type": "contradiction",
                "severity": "high",
                "evidence": [f["title"] for f in findings],
                "reasoning": "Несколько работ на одном датасете дают разные результаты",
                "potential_impact": "Высокое - требует разрешения противоречий",
                "related_methods": [m for f in findings for m in f["methods"]],
                "feasibility": "high"
            })
    
    return gaps


def find_methodological_gaps(lit_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Анализ неисследованных комбинаций метод×датасет×метрика
    """
    gaps = []
    
    all_methods = set()
    all_datasets = set()
    all_metrics = set()
    
    for paper in lit_matrix:
        summary = paper.get("structured_summary", {})
        all_methods.update(summary.get("methods", []))
        all_datasets.update(summary.get("datasets", []))
        all_metrics.update(summary.get("metrics", []))
        
    # Находим комбинации которые логичны но не исследованы
    # Простая эвристика: если метод A успешен на датасете X, но не тестировался на похожем датасете Y
    if len(all_methods) > 0 and len(all_datasets) > 1:
        gaps.append({
            "gap": f"Методы {', '.join(list(all_methods)[:3])} не тестировались на всех доступных датасетах. Есть потенциал для кросс-валидации.",
            "type": "methodological",
            "severity": "medium",
            "evidence": [p.get("title", "") for p in lit_matrix[:5]],
            "reasoning": "Анализ комбинаций методов и датасетов показывает пробелы",
            "potential_impact": "Среднее - может улучшить понимание применимости методов",
            "related_methods": list(all_methods)[:5],
            "feasibility": "high"
        })
    
    return gaps


def analyze_reproducibility(lit_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Анализ воспроизводимости: проверка наличия кода/данных
    """
    gaps = []
    
    papers_without_code = []
    for paper in lit_matrix:
        summary = paper.get("structured_summary", {})
        github_repos = paper.get("github_repos", [])
        reproducibility_info = summary.get("reproducibility_info", "").lower()
        
        has_code = len(github_repos) > 0 or "github" in reproducibility_info or "code" in reproducibility_info
        
        if not has_code:
            papers_without_code.append(paper.get("title", ""))
    
    if len(papers_without_code) > len(lit_matrix) * 0.5:  # Больше 50% без кода
        gaps.append({
            "gap": f"Большинство работ ({len(papers_without_code)}/{len(lit_matrix)}) не предоставляют код или данные для воспроизведения результатов.",
            "type": "reproducibility",
            "severity": "high",
            "evidence": papers_without_code[:5],
            "reasoning": "Отсутствие кода/данных затрудняет воспроизведение и валидацию результатов",
            "potential_impact": "Высокое - критично для научной воспроизводимости",
            "related_methods": [],
            "feasibility": "medium"
        })
    
    return gaps


def analyze_scalability(lit_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Анализ масштабируемости: проверка тестирования на больших данных/моделях
    """
    gaps = []
    
    scalability_mentions = []
    for paper in lit_matrix:
        summary = paper.get("structured_summary", {})
        experimental_setup = summary.get("experimental_setup", "").lower()
        discussion = summary.get("discussion", "").lower()
        
        # Ищем упоминания масштаба
        text = experimental_setup + " " + discussion
        if any(keyword in text for keyword in ["large scale", "scalability", "computational cost", "efficiency"]):
            scalability_mentions.append(paper.get("title", ""))
    
    if len(scalability_mentions) < len(lit_matrix) * 0.3:  # Меньше 30% обсуждают масштабируемость
        gaps.append({
            "gap": "Большинство работ не обсуждают масштабируемость или computational cost методов. Неясно как методы работают на больших данных/моделях.",
            "type": "scalability",
            "severity": "medium",
            "evidence": [p.get("title", "") for p in lit_matrix if p.get("title") not in scalability_mentions][:5],
            "reasoning": "Отсутствие анализа масштабируемости ограничивает практическое применение",
            "potential_impact": "Среднее - важно для production deployment",
            "related_methods": [],
            "feasibility": "medium"
        })
    
    return gaps


def find_cross_domain_opportunities(lit_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Выявление возможностей переноса методов между доменами
    """
    gaps = []
    
    # Простая эвристика: если методы из одного домена не применялись в другом
    # Здесь можно добавить более сложную логику определения доменов
    
    all_methods = []
    for paper in lit_matrix:
        summary = paper.get("structured_summary", {})
        methods = summary.get("methods", [])
        all_methods.extend(methods)
    
    if len(set(all_methods)) > 5:
        gaps.append({
            "gap": "Обнаружено разнообразие методов, но неясно какие из них можно перенести в другие домены. Требуется междисциплинарный анализ.",
            "type": "cross_domain",
            "severity": "low",
            "evidence": [p.get("title", "") for p in lit_matrix[:5]],
            "reasoning": "Методы могут быть применимы в других доменах, но это не исследовано",
            "potential_impact": "Среднее - может открыть новые применения методов",
            "related_methods": list(set(all_methods))[:5],
            "feasibility": "medium"
        })
    
    return gaps


def llm_deep_gap_analysis(lit_matrix: List[Dict[str, Any]], query: str, llm) -> List[Dict[str, Any]]:
    """
    Глубокий семантический анализ gaps через LLM
    """
    try:
        # Подготавливаем информацию для анализа
        papers_summary = []
        for paper in lit_matrix[:10]:  # Топ-10 для анализа
            summary = paper.get("structured_summary", {})
            papers_summary.append({
                "title": paper.get("title", ""),
                "methods": summary.get("methods", []),
                "datasets": summary.get("datasets", []),
                "limitations": summary.get("limitations", ""),
                "future_work": summary.get("future_work", "")
            })
        
        prompt = f"""Проанализируй корпус из {len(lit_matrix)} научных статей по теме "{query}" и найди глубокие research gaps.

Статьи:
{chr(10).join([f"- {p['title']}: методы={p['methods']}, limitations={p['limitations'][:100]}" for p in papers_summary])}

Найди неочевидные пробелы через reasoning:
1. Логичные но не реализованные комбинации методов
2. Методы которые должны работать но не тестировались
3. Проблемы которые упоминаются но не решаются
4. Противоречия в выводах которые требуют разрешения

Верни JSON:
{{
  "gaps": [
    {{
      "gap": "описание лакуны",
      "type": "methodological|data|metric|reproducibility|contradiction|temporal|scalability|cross_domain",
      "severity": "high|medium|low",
      "evidence": ["названия статей"],
      "reasoning": "обоснование почему это gap",
      "potential_impact": "потенциальное влияние",
      "related_methods": ["методы"],
      "feasibility": "high|medium|low"
    }}
  ]
}}
"""
    
        messages = [
            SystemMessage(content="Ты - эксперт по глубокому анализу научной литературы и поиску неочевидных research gaps."),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        parsed = safe_json_parse(response.content, {})
        
        gaps = []
        if isinstance(parsed, dict):
            for gap_dict in parsed.get("gaps", []):
                try:
                    gap_model = ResearchGapModel(**gap_dict)
                    gaps.append(gap_model.model_dump())
                except:
                    gaps.append({
                        "gap": gap_dict.get("gap", ""),
                        "type": gap_dict.get("type", "general"),
                        "severity": gap_dict.get("severity", "medium"),
                        "evidence": gap_dict.get("evidence", []),
                        "reasoning": gap_dict.get("reasoning", ""),
                        "potential_impact": gap_dict.get("potential_impact", ""),
                        "related_methods": gap_dict.get("related_methods", []),
                        "feasibility": gap_dict.get("feasibility", "medium")
                    })
        
        return gaps
    except Exception as e:
        return []


# ============================================================================
# ФУНКЦИЯ 13: GapMiner - поиск research gaps
# ============================================================================

def find_research_gaps(state: AgentState) -> AgentState:
    """
    Анализирует литературную матрицу и выявляет research gaps используя множественные стратегии
    """
    print("\n🔬 Функция 13: GapMiner - поиск research gaps...")
    
    lit_matrix = state["lit_matrix"]
    llm = get_llm()
    query = state.get("query", "")
    
    gap_list = []
    
    # Применяем все стратегии анализа
    print("   📊 Применяем множественные стратегии анализа...")
    
    # 1. Temporal Analysis
    temporal_gaps = analyze_temporal_evolution(lit_matrix)
    gap_list.extend(temporal_gaps)
    if temporal_gaps:
        print(f"      ✓ Temporal analysis: найдено {len(temporal_gaps)} gaps")
    
    # 2. Contradiction Detection
    contradiction_gaps = detect_contradictions(lit_matrix)
    gap_list.extend(contradiction_gaps)
    if contradiction_gaps:
        print(f"      ✓ Contradiction detection: найдено {len(contradiction_gaps)} gaps")
    
    # 3. Methodological Gaps
    methodological_gaps = find_methodological_gaps(lit_matrix)
    gap_list.extend(methodological_gaps)
    if methodological_gaps:
        print(f"      ✓ Methodological analysis: найдено {len(methodological_gaps)} gaps")
    
    # 4. Reproducibility Analysis
    repro_gaps = analyze_reproducibility(lit_matrix)
    gap_list.extend(repro_gaps)
    if repro_gaps:
        print(f"      ✓ Reproducibility analysis: найдено {len(repro_gaps)} gaps")
    
    # 5. Scalability Analysis
    scale_gaps = analyze_scalability(lit_matrix)
    gap_list.extend(scale_gaps)
    if scale_gaps:
        print(f"      ✓ Scalability analysis: найдено {len(scale_gaps)} gaps")
    
    # 6. Cross-domain Opportunities
    cross_domain_gaps = find_cross_domain_opportunities(lit_matrix)
    gap_list.extend(cross_domain_gaps)
    if cross_domain_gaps:
        print(f"      ✓ Cross-domain analysis: найдено {len(cross_domain_gaps)} gaps")
    
    # 7. LLM-based Deep Analysis
    try:
        llm_gaps = llm_deep_gap_analysis(lit_matrix, query, llm)
        gap_list.extend(llm_gaps)
        if llm_gaps:
            print(f"      ✓ LLM deep analysis: найдено {len(llm_gaps)} gaps")
    except Exception as e:
        print(f"      ⚠️  Ошибка LLM deep analysis: {e}")
    
    # Валидируем все gaps через Pydantic
    validated_gaps = []
    for gap_dict in gap_list:
        try:
            gap_model = ResearchGapModel(**gap_dict)
            validated_gaps.append(gap_model.model_dump())
        except ValidationError:
            # Fallback с дефолтными значениями для новых полей
            validated_gaps.append({
                "gap": gap_dict.get("gap", ""),
                "type": gap_dict.get("type", "general"),
                "severity": gap_dict.get("severity", "medium"),
                "evidence": gap_dict.get("evidence", []),
                "reasoning": gap_dict.get("reasoning", ""),
                "potential_impact": gap_dict.get("potential_impact", ""),
                "related_methods": gap_dict.get("related_methods", []),
                "feasibility": gap_dict.get("feasibility", "medium")
            })
    
    gap_list = validated_gaps
    
    state["gap_list"] = gap_list
    
    print(f"   ✓ Найдено research gaps: {len(gap_list)}")
    for i, gap in enumerate(gap_list[:3], 1):
        print(f"      {i}. [{gap.get('severity', 'N/A')}] {gap.get('gap', 'N/A')[:80]}")
    
    # Агентное перепланирование (если включено)
    try:
        from config import EXPERIMENTAL, REPLAN_CONFIG
        if EXPERIMENTAL.get("enable_replanning", False) and REPLAN_CONFIG.get("enabled", False):
            # Проверяем покрытие тем
            coverage_issues = analyze_coverage(lit_matrix, gap_list)
            if coverage_issues:
                print(f"   🔄 Перепланирование запросов из-за провалов покрытия...")
                new_queries = replan_queries(state["query"], coverage_issues, llm)
                if new_queries:
                    state["query_strings"].extend(new_queries)
                    print(f"      ✓ Добавлено {len(new_queries)} новых запросов")
    except Exception as e:
        pass
    
    state["next_step"] = "generate_ideas"
    
    return state


# ============================================================================
# ФУНКЦИИ: Агентное поведение - новые узлы
# ============================================================================

def retry_search_with_expansion(state: AgentState) -> AgentState:
    """
    Расширяет запрос и повторяет поиск если найдено мало результатов
    """
    print("\n🔄 Retry Search - расширение запроса и повторный поиск...")
    
    llm = get_llm()
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    
    try:
        from config import AGENT_CONFIG
        max_retries = AGENT_CONFIG.get("max_retries", 2)
    except:
        max_retries = 2
    
    if retry_count >= max_retries:
        print("   ⚠️  Достигнут лимит попыток, продолжаем с текущими результатами")
        state["next_step"] = "rank"
        return state
    
    # Расширяем запрос с помощью LLM
    expansion_prompt = f"""Текущий запрос дал мало результатов. Расширь запрос для более широкого поиска.

Исходный запрос: {query}
Текущее количество результатов: {len(state.get('corpus_index', []))}

Создай 3 альтернативных расширенных запроса которые:
1. Используют синонимы и смежные термины
2. Расширяют область поиска
3. Включают связанные концепции

Верни JSON:
{{
  "expanded_queries": ["запрос 1", "запрос 2", "запрос 3"]
}}
"""
    
    try:
        messages = [
            SystemMessage(content="Ты - эксперт по формулированию научных поисковых запросов."),
            HumanMessage(content=expansion_prompt)
        ]
        response = llm.invoke(messages)
        parsed = safe_json_parse(response.content, {})
        expanded_queries = parsed.get("expanded_queries", [query])
        
        # Обновляем query_strings
        state["query_strings"] = expanded_queries
        state["retry_count"] = retry_count + 1
        
        print(f"   ✓ Создано {len(expanded_queries)} расширенных запросов")
        print(f"   🔍 Повторный поиск...")
        
        # Увеличиваем количество результатов с источников
        state["next_step"] = "retrieve"
    except Exception as e:
        print(f"   ⚠️  Ошибка расширения запроса: {e}")
        state["next_step"] = "rank"
    
    return state


def replan_search_queries(state: AgentState) -> AgentState:
    """
    Перепланирует поисковые запросы на основе найденных gaps
    """
    print("\n🔄 Replan Search - перепланирование запросов на основе gaps...")
    
    llm = get_llm()
    gap_list = state.get("gap_list", [])
    query = state["query"]
    
    # Анализируем gaps для создания целевых запросов
    high_gaps = [g for g in gap_list if g.get("severity") == "high"]
    
    if not high_gaps:
        print("   ℹ️  Нет критических gaps для перепланирования")
        state["next_step"] = "generate_ideas"
        return state
    
    replan_prompt = f"""На основе найденных research gaps создай новые целевые поисковые запросы.

Исходная тема: {query}

Критические gaps:
{chr(10).join([f"- {g.get('gap', '')[:200]}" for g in high_gaps[:5]])}

Создай 3-5 новых запросов которые помогут найти статьи для заполнения этих пробелов.

Верни JSON:
{{
  "new_queries": ["запрос 1", "запрос 2", "запрос 3"]
}}
"""
    
    try:
        messages = [
            SystemMessage(content="Ты - эксперт по формулированию научных поисковых запросов для заполнения research gaps."),
            HumanMessage(content=replan_prompt)
        ]
        response = llm.invoke(messages)
        parsed = safe_json_parse(response.content, {})
        new_queries = parsed.get("new_queries", [])
        
        # Добавляем новые запросы
        existing_queries = state.get("query_strings", [query])
        state["query_strings"] = existing_queries + new_queries
        
        # Обновляем историю перепланирований
        replanning_history = state.get("replanning_history", [])
        replanning_history.append(f"Gaps-based replan: {len(new_queries)} queries")
        state["replanning_history"] = replanning_history
        
        print(f"   ✓ Создано {len(new_queries)} новых целевых запросов")
        state["next_step"] = "retrieve"
    except Exception as e:
        print(f"   ⚠️  Ошибка перепланирования: {e}")
    state["next_step"] = "generate_ideas"
    
    return state


# ============================================================================
# ФУНКЦИИ: Агентное перепланирование
# ============================================================================

def analyze_coverage(lit_matrix: List[Dict[str, Any]], gap_list: List[Dict[str, Any]]) -> List[str]:
    """
    Анализирует покрытие тем и выявляет провалы
    """
    issues = []
    
    # Проверяем наличие high-severity gaps
    high_gaps = [g for g in gap_list if g.get("severity") == "high"]
    if len(high_gaps) > 3:
        issues.append(f"Обнаружено {len(high_gaps)} критических research gaps")
    
    # Проверяем разнообразие методов
    all_methods = set()
    for paper in lit_matrix:
        summary = paper.get("structured_summary", {})
        all_methods.update(summary.get("methods", []))
    
    if len(all_methods) < 5:
        issues.append("Недостаточное разнообразие методов в найденных статьях")
    
    return issues


def replan_queries(original_query: str, coverage_issues: List[str], llm) -> List[str]:
    """
    Перепланирует запросы на основе провалов покрытия
    """
    if not coverage_issues:
        return []
    
    system_prompt = """Ты - эксперт по формулированию научных поисковых запросов.

На основе исходного запроса и выявленных проблем покрытия тем, создай новые запросы для расширения поиска.

Верни JSON:
{
  "new_queries": ["новый запрос 1", "новый запрос 2", ...]
}
"""
    
    user_prompt = f"""Исходный запрос: {original_query}

Выявленные проблемы:
{chr(10).join([f"- {issue}" for issue in coverage_issues])}

Создай 2-3 новых запроса для расширения поиска и улучшения покрытия тем."""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        parsed = safe_json_parse(response.content, {})
        
        if isinstance(parsed, dict):
            return parsed.get("new_queries", [])
        return []
    
    except Exception as e:
        return []


# ============================================================================
# ФУНКЦИЯ 14: Ideator - генерация исследовательских идей
# ============================================================================

def generate_research_ideas(state: AgentState) -> AgentState:
    """
    Генерирует проверяемые гипотезы на основе найденных gaps
    """
    print("\n💡 Функция 14: Ideator - генерация исследовательских идей...")
    
    gap_list = state["gap_list"]
    lit_matrix = state["lit_matrix"]
    llm = get_llm()
    
    ideation_prompt = f"""На основе найденных research gaps предложи 5 конкретных, проверяемых идей для исследований.

**Research Gaps:**
{chr(10).join([f"- {g.get('gap', 'N/A')}" for g in gap_list[:5]])}

**Контекст (топ работы в области):**
{chr(10).join([f"- {p.get('title', 'Unknown')[:60]}" for p in lit_matrix[:3]])}

Для каждой идеи укажи:
1. Гипотезу (что проверяем)
2. План эксперимента (методы, датасеты, бейзлайны)
3. Ожидаемый результат
4. Риски и альтернативы

Верни JSON:
{{
  "ideas": [
    {{
      "hypothesis": "описание гипотезы",
      "experiment_plan": {{
        "methods": ["метод 1"],
        "datasets": ["датасет 1"],
        "baselines": ["бейзлайн 1"],
        "metrics": ["метрика 1"]
      }},
      "expected_outcome": "что ожидаем",
      "risks": ["риск 1"],
      "related_gap": "на какой gap отвечает"
    }}
  ]
}}
"""
    
    try:
        messages = [
            SystemMessage(content="Ты - эксперт по генерации исследовательских идей на основе анализа литературы."),
            HumanMessage(content=ideation_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Парсим с валидацией
        parsed = safe_json_parse(response.content, {})
        ideas_data = parsed if isinstance(parsed, dict) else {}
        
        ideas_raw = ideas_data.get("ideas", [])
        
        # Валидируем через Pydantic
        idea_bank = []
        for idea_dict in ideas_raw:
            try:
                idea_model = ResearchIdeaModel(**idea_dict)
                idea_bank.append(idea_model.model_dump())
            except ValidationError:
                # Fallback
                idea_bank.append({
                    "hypothesis": idea_dict.get("hypothesis", ""),
                    "experiment_plan": idea_dict.get("experiment_plan", {}),
                    "expected_outcome": idea_dict.get("expected_outcome", ""),
                    "risks": idea_dict.get("risks", []),
                    "related_gap": idea_dict.get("related_gap", "")
                })
        
    except Exception as e:
        print(f"   ⚠️  Ошибка генерации идей: {e}")
        idea_bank = [
            {
                "hypothesis": "Требуется дополнительный анализ для генерации конкретных идей",
                "experiment_plan": {},
                "expected_outcome": "N/A",
                "risks": [],
                "related_gap": "general"
            }
        ]
    
    state["idea_bank"] = idea_bank
    state["next_step"] = "report"
    
    print(f"   ✓ Сгенерировано идей: {len(idea_bank)}")
    for i, idea in enumerate(idea_bank[:3], 1):
        print(f"      {i}. {idea.get('hypothesis', 'N/A')[:70]}...")
    
    return state


# ============================================================================
# ФУНКЦИЯ 15: Reporter - создание финального отчёта
# ============================================================================

def generate_final_report(state: AgentState) -> AgentState:
    """
    Генерирует итоговый отчёт в формате Markdown
    """
    print("\n📄 Функция 16: Reporter - создание итогового отчёта...")
    
    query = state["query"]
    lit_matrix = state.get("lit_matrix", [])
    gap_list = state.get("gap_list", [])
    idea_bank = state.get("idea_bank", [])
    ranked_papers = state.get("ranked_papers", [])
    
    # Формируем отчёт
    report = f"""# Отчёт по научным исследованиям

## Тема: {query}

**Дата:** 2025-10-20  
**Проанализировано статей:** {len(ranked_papers)}  
**Детально изучено:** {len(lit_matrix)}

---

## 1. Executive Summary

Проведён систематический поиск и анализ научной литературы по теме "{query}".
Использованы 5 источников (OpenAlex, Semantic Scholar, Crossref, ArXiv, PubMed).
Выявлено {len(gap_list)} research gaps и предложено {len(idea_bank)} исследовательских идей.

---

## 2. Топ-статьи по релевантности

"""
    
    # Добавляем топ-10 статей
    for i, paper in enumerate(ranked_papers[:10], 1):
        authors_str = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors_str += " et al."
        
        report += f"""### {i}. {paper.get('title', 'No title')}

- **Авторы:** {authors_str}
- **Год:** {paper.get('published')}
- **Venue:** {paper.get('venue', 'Unknown')}
- **Цитирований:** {paper.get('citations_total', 0)}
- **Score:** {paper.get('relevance_score', 0):.3f}
- **URL:** {paper.get('url', 'N/A')}

"""
        if paper.get("doi"):
            report += f"- **DOI:** [{paper['doi']}](https://doi.org/{paper['doi']})\n"
        
        report += f"\n**Аннотация:** {paper.get('summary', 'No abstract')[:300]}...\n\n---\n\n"
    
    # Литературная матрица (детальный анализ)
    if lit_matrix:
        report += f"""## 3. Литературная матрица (детальный анализ топ-{len(lit_matrix)})

| Работа | Проблема | Методы | Датасеты | Метрики | Ограничения | PDF |
|--------|----------|--------|----------|---------|-------------|-----|
"""
        
        for paper in lit_matrix:
            summary = paper.get("structured_summary", {})
            # ============================================================================
            # УЛУЧШЕНИЕ: Увеличены лимиты обрезания (Задача 1.3)
            # ============================================================================
            title_short = paper.get('title', 'N/A')[:60]  # было 40
            problem = summary.get('problem', 'N/A')[:100]  # было 50
            methods = ", ".join(summary.get('methods', [])[:3])[:80]  # было [:2][:30]
            datasets = ", ".join(summary.get('datasets', [])[:3])[:60]  # было [:2][:30]
            metrics = ", ".join(summary.get('metrics', [])[:3])[:60]  # было [:2][:30]
            limitations = summary.get('limitations', 'N/A')[:80]  # было 40
            has_pdf = "📄" if paper.get('has_full_text', False) else "—"
            
            report += f"| {title_short}... | {problem} | {methods} | {datasets} | {metrics} | {limitations} | {has_pdf} |\n"
        
        report += "\n---\n\n"
        
        # ============================================================================
        # НОВЫЙ РАЗДЕЛ: Integrated Synthesis (Задача 1.1)
        # ============================================================================
        report += """## 3.5. 📊 Интегрированный синтез литературы

### Основные выводы из проанализированных работ:

"""
        
        for idx, paper in enumerate(lit_matrix, 1):
            title = paper.get('title', 'Unknown')
            summary = paper.get("structured_summary", {})
            
            # Извлекаем полные (не обрезанные) данные
            findings = summary.get('key_findings', '')
            methods = summary.get('methods', [])
            contributions = summary.get('contributions', '')
            limitations = summary.get('limitations', '')
            
            report += f"**📄 [{idx}] {title}**\n\n"
            
            # Ключевые выводы (ПОЛНЫЙ текст, не обрезанный)
            if findings and findings != "N/A" and findings != "Не удалось извлечь":
                report += f"*Ключевые выводы:* {findings}\n\n"
            
            # Методы (до 5 методов)
            if methods and len(methods) > 0:
                methods_str = ", ".join(methods[:5])
                report += f"*Методы:* {methods_str}\n\n"
            
            # Вклад работы
            if contributions and contributions != "N/A" and contributions != "":
                report += f"*Вклад:* {contributions}\n\n"
            
            # Ограничения
            if limitations and limitations != "N/A" and limitations != "Не удалось извлечь детали":
                report += f"*Ограничения:* {limitations}\n\n"
            
            report += "---\n\n"
        
        # Общий синтез трендов
        report += "### 🔍 Общие тренды и паттерны:\n\n"
        
        # Собираем все методы
        all_methods = []
        all_datasets = []
        for paper in lit_matrix:
            summary = paper.get("structured_summary", {})
            all_methods.extend(summary.get('methods', []))
            all_datasets.extend(summary.get('datasets', []))
        
        if all_methods:
            # Подсчитываем частоту методов
            from collections import Counter
            method_counts = Counter(all_methods)
            top_methods = method_counts.most_common(5)
            
            report += "**Наиболее популярные методы:**\n"
            for method, count in top_methods:
                report += f"- {method} (используется в {count} работах)\n"
            report += "\n"
        
        if all_datasets:
            dataset_counts = Counter(all_datasets)
            top_datasets = dataset_counts.most_common(5)
            
            report += "**Наиболее используемые датасеты:**\n"
            for dataset, count in top_datasets:
                report += f"- {dataset} (используется в {count} работах)\n"
            report += "\n"
        
        report += "\n---\n\n"
    
    # Research Gaps
    if gap_list:
        report += f"""## 4. Research Gaps (Исследовательские лакуны)

Выявлено **{len(gap_list)}** потенциальных направлений для исследований:

"""
        for i, gap in enumerate(gap_list, 1):
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(gap.get('severity', 'medium'), "⚪")
            report += f"""### Gap {i}: {severity_emoji} {gap.get('severity', 'N/A').upper()}

**Тип:** {gap.get('type', 'N/A')}

**Описание:** {gap.get('gap', 'N/A')}

**Подтверждающие источники:**
"""
            for evidence in gap.get('evidence', [])[:3]:
                report += f"- {evidence}\n"
            
            report += "\n---\n\n"
    
    # Research Ideas
    if idea_bank:
        report += f"""## 5. Предлагаемые исследовательские идеи

Сгенерировано **{len(idea_bank)}** проверяемых гипотез:

"""
        for i, idea in enumerate(idea_bank, 1):
            report += f"""### Идея {i}

**Гипотеза:** {idea.get('hypothesis', 'N/A')}

**План эксперимента:**
"""
            plan = idea.get('experiment_plan', {})
            if plan.get('methods'):
                report += f"- **Методы:** {', '.join(plan['methods'])}\n"
            if plan.get('datasets'):
                report += f"- **Датасеты:** {', '.join(plan['datasets'])}\n"
            if plan.get('baselines'):
                report += f"- **Бейзлайны:** {', '.join(plan['baselines'])}\n"
            if plan.get('metrics'):
                report += f"- **Метрики:** {', '.join(plan['metrics'])}\n"
            
            report += f"\n**Ожидаемый результат:** {idea.get('expected_outcome', 'N/A')}\n\n"
            
            if idea.get('risks'):
                report += f"**Риски:**\n"
                for risk in idea['risks']:
                    report += f"- {risk}\n"
            
            report += f"\n**Связанный Gap:** {idea.get('related_gap', 'N/A')}\n\n---\n\n"
    
    # Метрики и статистика
    pdf_texts = state.get('pdf_texts', {})
    papers_with_pdf = len(pdf_texts)
    
    report += f"""## 6. Метрики и статистика

- **Всего найдено статей (с дубликатами):** {len(state.get('seed_results', []))}
- **Уникальных статей:** {len(state.get('corpus_index', []))}
- **Топ-статей для анализа:** {len(ranked_papers)}
- **Детально проанализировано:** {len(lit_matrix)}
- **PDF прочитано:** {papers_with_pdf}
- **API вызовов:** {state.get('budget', {}).get('api_calls', 0)}

---

## 7. Заключение

Данный отчёт предоставляет систематический обзор научной литературы по теме "{query}".
Выявленные research gaps и предложенные идеи могут служить основой для дальнейших исследований.

**Примечание:** Все утверждения подкреплены ссылками на научные статьи. 
Для более глубокого анализа рекомендуется ознакомиться с полными текстами статей.

---

*Отчёт сгенерирован AI агентом на базе LangGraph*
"""
    
    state["final_response"] = report
    state["next_step"] = "end"
    
    print(f"   ✓ Отчёт создан: {len(report)} символов")
    print(f"   ✓ Включено разделов: 7")
    
    return state


# ============================================================================
# СОЗДАНИЕ LANGGRAPH АГЕНТА
# ============================================================================

def create_research_agent():
    """
    Создаёт и возвращает LangGraph агента для поиска научных статей
    С условными переходами для агентного поведения
    """
    print("\n🔧 Создание LangGraph агента с агентным поведением...")
    
    # Создаём граф
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы (nodes)
    workflow.add_node("build_topic_card", build_topic_card)
    workflow.add_node("select_sources", analyze_query_and_select_sources)  # Новый узел (Фаза 4)
    workflow.add_node("retrieve", multi_source_retriever)
    workflow.add_node("deduplicate", deduplicate_and_normalize)
    workflow.add_node("retry_search", retry_search_with_expansion)  # Новый узел
    workflow.add_node("rank", hybrid_ranker)
    workflow.add_node("read_pdfs", read_pdfs)
    workflow.add_node("summarize", create_literature_matrix)
    workflow.add_node("find_gaps", find_research_gaps)
    workflow.add_node("replan_search", replan_search_queries)  # Новый узел
    workflow.add_node("generate_ideas", generate_research_ideas)
    workflow.add_node("report", generate_final_report)
    
    # Определяем рёбра (edges) - последовательность выполнения
    workflow.set_entry_point("build_topic_card")
    workflow.add_edge("build_topic_card", "select_sources")  # Сначала выбираем источники
    workflow.add_edge("select_sources", "retrieve")  # Затем поиск
    workflow.add_edge("retrieve", "deduplicate")
    
    # Условный переход после deduplicate
    def route_after_dedup(state: AgentState) -> str:
        try:
            from config import AGENT_CONFIG
            min_papers = AGENT_CONFIG.get("min_papers_threshold", 10)
            enable_retry = AGENT_CONFIG.get("enable_retry", True)
        except:
            min_papers = 10
            enable_retry = True
        
        corpus_size = len(state.get("corpus_index", []))
        retry_count = state.get("retry_count", 0)
        
        if enable_retry and corpus_size < min_papers and retry_count < 2:
            return "retry_search"
        return "rank"
    
    workflow.add_conditional_edges(
        "deduplicate",
        route_after_dedup,
        {
            "retry_search": "retry_search",
            "rank": "rank"
        }
    )
    workflow.add_edge("retry_search", "retrieve")  # Цикл обратно к поиску
    
    workflow.add_edge("rank", "read_pdfs")
    workflow.add_edge("read_pdfs", "summarize")
    workflow.add_edge("summarize", "find_gaps")
    
    # Условный переход после find_gaps
    def route_after_gaps(state: AgentState) -> str:
        try:
            from config import AGENT_CONFIG
            enable_replan = AGENT_CONFIG.get("enable_replanning", True)
            replan_threshold = AGENT_CONFIG.get("replan_gap_threshold", 5)
        except:
            enable_replan = True
            replan_threshold = 5
        
        if not enable_replan:
            return "generate_ideas"
        
        gap_list = state.get("gap_list", [])
        high_gaps = [g for g in gap_list if g.get("severity") == "high"]
        
        if len(high_gaps) > replan_threshold:
            return "replan_search"
        return "generate_ideas"
    
    workflow.add_conditional_edges(
        "find_gaps",
        route_after_gaps,
        {
            "replan_search": "replan_search",
            "generate_ideas": "generate_ideas"
        }
    )
    workflow.add_edge("replan_search", "retrieve")  # Вернуться к поиску
    
    workflow.add_edge("generate_ideas", "report")
    workflow.add_edge("report", END)
    
    # Компилируем граф
    app = workflow.compile()
    
    print("   ✓ LangGraph агент создан с 11 узлами (включая агентные)")
    print("   ✓ Pipeline с условными переходами:")
    print("      - Retry search если мало результатов")
    print("      - Replan search если много критических gaps")
    
    return app


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def run_research_pipeline(
    query: str,
    time_window: int = 5,
    max_papers: int = 40,
    save_report: bool = True
) -> str:
    """
    Запускает полный pipeline поиска и анализа научных статей
    
    Args:
        query: Тема исследования
        time_window: Окно времени (лет назад)
        max_papers: Максимум статей для анализа
        save_report: Сохранять ли отчёт в файл
    
    Returns:
        Финальный отчёт (markdown)
    """
    print("="*70)
    print("🚀 ЗАПУСК AI АГЕНТА ДЛЯ ПОИСКА НАУЧНЫХ СТАТЕЙ")
    print("="*70)
    print(f"Тема: {query}")
    print(f"Период: последние {time_window} лет")
    print(f"Максимум статей: {max_papers}")
    print("="*70)
    
    # Создаём агента
    agent = create_research_agent()
    
    # Инициализируем начальное состояние
    initial_state = {
        "query": query,
        "time_window": time_window,
        "max_papers": max_papers,
        "selected_databases": [],
        "refined_query": query,
        "topic_card": {},
        "query_strings": [],
        "seed_results": [],
        "search_results": {},
        "corpus_index": [],
        "ranked_papers": [],
        "pdf_texts": {},
        "citation_graph": {},
        "lit_matrix": [],
        "gap_list": [],
        "idea_bank": [],
        "final_response": "",
        "messages": [],
        "next_step": "",
        "budget": {"api_calls": 0, "llm_calls": 0},
        "retry_count": 0,
        "search_quality_score": 0.0,
        "replanning_history": []
    }
    
    # Запускаем pipeline
    print("\n🔄 Запуск pipeline...\n")
    
    try:
        # Выполняем граф
        final_state = agent.invoke(initial_state)
        
        print("\n" + "="*70)
        print("✅ PIPELINE ЗАВЕРШЁН УСПЕШНО!")
        print("="*70)
        
        # Получаем финальный отчёт
        report = final_state.get("final_response", "Отчёт не создан")
        
        # Сохраняем отчёт
        if save_report:
            filename = f"research_report_{query.replace(' ', '_')[:30]}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n📁 Отчёт сохранён: {filename}")
        
        # Выводим краткую статистику
        print(f"\n📊 Итоговая статистика:")
        print(f"   - Найдено уникальных статей: {len(final_state.get('corpus_index', []))}")
        print(f"   - Проанализировано детально: {len(final_state.get('lit_matrix', []))}")
        print(f"   - Выявлено research gaps: {len(final_state.get('gap_list', []))}")
        print(f"   - Сгенерировано идей: {len(final_state.get('idea_bank', []))}")
        print(f"   - API вызовов: {final_state.get('budget', {}).get('api_calls', 0)}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Демонстрация работы агента с примерами
    """
    print("\n" + "="*70)
    print("🤖 AI АГЕНТ ДЛЯ ПОИСКА НАУЧНЫХ СТАТЕЙ")
    print("Версия: 1.0 | Архитектура: LangGraph | Источники: 5")
    print("="*70)
    
    # Проверяем наличие API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  ВНИМАНИЕ: Не найден OPENAI_API_KEY")
        print("Создайте файл .env и добавьте: OPENAI_API_KEY=your_key_here")
        print("\nДля демонстрации используется пример с ограниченным функционалом.\n")
    
    # Примеры запросов
    examples = [
        "transformer models for natural language processing",
        "CRISPR gene editing in cancer treatment",
        "quantum computing algorithms",
        "climate change machine learning predictions"
    ]
    
    print("\n📋 Примеры тем для исследования:")
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")
    
    print("\n" + "-"*70)
    print("ℹ️  Для запуска используйте:")
    print("   report = run_research_pipeline('ваша тема', time_window=5, max_papers=40)")
    print("-"*70)
    
    # Демо-запуск (закомментировано, раскомментируйте для реального запуска)
    # if os.getenv("OPENAI_API_KEY"):
    #     print("\n🎬 Запускаем демо с темой: transformer models...")
    #     report = run_research_pipeline(
    #         query="transformer models for natural language processing",
    #         time_window=3,
    #         max_papers=20
    #     )
    #     if report:
    #         print("\n✅ Демо завершено! Проверьте созданный файл отчёта.")


if __name__ == "__main__":
    main()


print("\n✅ Все функции созданы! Агент готов к использованию.")
print("   Запустите: python main.py")
print("   Или используйте: run_research_pipeline('ваша тема')")

