"""
Конфигурация AI агента для поиска научных статей
"""

# ============================================================================
# LLM НАСТРОЙКИ
# ============================================================================

LLM_CONFIG = {
    "model": "gpt-4o-mini",        # Модель OpenAI
    "temperature": 0.7,             # Температура генерации (0-1)
    "max_retries": 3,               # Количество попыток при ошибке
}

# ============================================================================
# ПОИСКОВЫЕ НАСТРОЙКИ
# ============================================================================

SEARCH_CONFIG = {
    "per_source_limit": 30,         # Результатов с каждого источника
    "default_time_window": 5,       # Лет назад (по умолчанию)
    "default_max_papers": 40,       # Максимум статей для анализа
    "top_papers_for_analysis": 10,  # Для детального анализа (LLM)
}

# ============================================================================
# ИСТОЧНИКИ (включить/выключить)
# ============================================================================

SOURCES_ENABLED = {
    "openalex": True,
    "semantic_scholar": True,
    "crossref": True,
    "arxiv": True,
    "pubmed": True,
}

# ============================================================================
# ДЕДУПЛИКАЦИЯ
# ============================================================================

DEDUP_CONFIG = {
    "title_similarity_threshold": 0.85,  # Порог для fuzzy match (0-1)
    "check_last_n_papers": 20,           # Сколько последних проверять
}

# ============================================================================
# РАНЖИРОВАНИЕ
# ============================================================================

# Компоненты ранжирования (можно включать/выключать)
# На основе результатов part_1: semantic search показал лучший результат
# Рекомендуется: semantic_search=True, остальное опционально
RANKING_COMPONENTS = {
    "semantic_search": True,    # Semantic search (dense embeddings) - ОСНОВНОЙ
    "recency": True,            # Свежесть публикации (опционально)
    "citations_per_year": True, # Цитаты в год (опционально)
    "citations_total": True,    # Общее количество цитат (опционально)
    "venue": True,              # Престижность venue (опционально)
    "keywords_bm25": False,     # Keyword matching (BM25) - по умолчанию ВЫКЛЮЧЕНО
                                 # (по результатам part_1: BM25 может мешать)
}

# Веса компонентов (используются только для включённых компонентов)
# Автоматически нормализуются в зависимости от включённых компонентов
RANKING_WEIGHTS = {
    "semantic_search": 0.75,    # Основной вес для semantic search
    "recency": 0.10,            # Вес свежести публикации
    "cpy": 0.08,                # Citations per year
    "ctotal": 0.05,             # Total citations (log scale)
    "venue": 0.02,              # Престижность venue
    "keywords": 0.00,           # Keyword matching (BM25) - по умолчанию 0
}

VENUE_SCORES = {
    # Топовые журналы
    "nature": 1.0,
    "science": 1.0,
    "cell": 1.0,
    "nature neuroscience": 1.0,
    "nature biotechnology": 1.0,
    
    # Медицинские
    "nejm": 0.95,  # New England Journal of Medicine
    "lancet": 0.95,
    "jama": 0.95,
    "bmj": 0.90,
    
    # ML/AI конференции
    "neurips": 0.90,
    "icml": 0.90,
    "iclr": 0.90,
    "cvpr": 0.90,
    "eccv": 0.85,
    "iccv": 0.85,
    "aaai": 0.85,
    
    # NLP конференции
    "acl": 0.85,
    "emnlp": 0.85,
    "naacl": 0.85,
    "coling": 0.80,
    
    # Системные конференции
    "sosp": 0.90,
    "osdi": 0.90,
    "sigcomm": 0.85,
    
    # Databases
    "sigmod": 0.85,
    "vldb": 0.85,
    
    # Default для неизвестных
    "default": 0.5,
}

# ============================================================================
# GAP MINING
# ============================================================================

GAP_MINING_CONFIG = {
    "min_gaps": 3,                  # Минимум gaps для поиска
    "max_gaps": 10,                 # Максимум gaps
    "freshness_threshold_months": 18,  # Считается "старым" если нет работ за N мес.
}

# ============================================================================
# IDEA GENERATION
# ============================================================================

IDEATION_CONFIG = {
    "num_ideas": 5,                 # Количество генерируемых идей
    "require_experiment_plan": True, # Обязательно ли план эксперимента
}

# ============================================================================
# ОТЧЁТ
# ============================================================================

REPORT_CONFIG = {
    "save_by_default": True,        # Сохранять отчёт автоматически
    "format": "markdown",           # markdown | latex (в будущем)
    "include_abstracts": True,      # Включать аннотации в отчёт
    "abstract_max_length": 300,     # Максимум символов аннотации
    "top_papers_in_report": 10,     # Топ-N статей в разделе 2
}

# ============================================================================
# API RATE LIMITS
# ============================================================================

RATE_LIMITS = {
    "openalex_delay": 0.1,          # Секунд между запросами
    "semantic_scholar_delay": 0.1,
    "crossref_delay": 0.1,
    "arxiv_delay": 3.0,             # ArXiv требует 3 сек
    "pubmed_delay": 0.33,           # PubMed: 3 req/sec = 0.33s
}

# ============================================================================
# БЮДЖЕТ
# ============================================================================

BUDGET_CONFIG = {
    "max_api_calls": 100,           # Максимум API вызовов
    "max_llm_calls": 50,            # Максимум LLM вызовов
    "warn_threshold": 0.8,          # Предупреждение при 80% бюджета
}

# ============================================================================
# КЭШИРОВАНИЕ (для будущей реализации)
# ============================================================================

CACHE_CONFIG = {
    "enabled": True,                # Включено! (Задача 2.2)
    "backend": "sqlite",            # sqlite | redis
    "ttl_days": 7,                  # Time to live (дней)
    "db_path": "data/cache.db",     # Путь к файлу базы данных
}

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",                # DEBUG | INFO | WARNING | ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,                   # Путь к файлу логов (None = stdout)
}

# ============================================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ФИЧИ
# ============================================================================

EXPERIMENTAL = {
    "enable_snowballer": False,     # Citation snowballing (РЕАЛИЗОВАНО ✅)
    "enable_pdf_reader": False,     # Парсинг PDF (не реализовано)
    "enable_semantic_search": True,  # Векторный поиск (РЕАЛИЗОВАНО ✅) - ОСНОВНОЙ
    "enable_llm_rerank": False,     # LLM-rerank поверх топ-N (РЕАЛИЗОВАНО ✅) - опционально
    "enable_replanning": False,     # Агентное перепланирование (РЕАЛИЗОВАНО ✅)
    "async_api_calls": False,       # Асинхронные запросы (не реализовано)
}

# ============================================================================
# ВЕКТОРНЫЙ ПОИСК (SEMANTIC SEARCH)
# ============================================================================

VECTOR_SEARCH_CONFIG = {
    "embedding_model": "text-embedding-3-small",  # OpenAI модель для embeddings
    "batch_size": 100,                           # Размер батча для создания embeddings
    "weight": 0.75,                              # Вес векторного поиска (основной компонент)
                                                 # На основе part_1: semantic search показал лучший результат
}

# ============================================================================
# LLM-RERANK НАСТРОЙКИ
# ============================================================================

RERANK_CONFIG = {
    "top_k": 20,                                 # Топ-K статей для reranking
    "enabled": True,                             # Включить/выключить rerank
}

# ============================================================================
# ФИЛЬТРЫ ПОИСКА
# ============================================================================

SEARCH_FILTERS = {
    "exclude_reviews": False,                    # Исключать обзоры/surveys
    "novelty_level": "medium",                  # "high" | "medium" | "low" | None
    "domain": None,                              # "medicine" | "cs" | None (для фильтрации)
}

# ============================================================================
# CITATION SNOWBALLING
# ============================================================================

SNOWBALL_CONFIG = {
    "enabled": False,                            # Включить citation snowballing
    "max_expansion": 20,                          # Максимум новых статей для расширения
    "min_citations": 5,                          # Минимум цитирований для включения
    "forward_citations": True,                   # Искать forward citations (кто цитирует)
    "backward_citations": True,                  # Искать backward citations (на кого ссылается)
}

# ============================================================================
# АГЕНТНОЕ ПЕРЕПЛАНИРОВАНИЕ
# ============================================================================

REPLAN_CONFIG = {
    "enabled": False,                            # Включить перепланирование запросов
    "min_gaps_for_replan": 3,                    # Минимум gaps для запуска перепланирования
}

# ============================================================================
# GITHUB ПОИСК
# ============================================================================

GITHUB_CONFIG = {
    "enabled": True,                    # Включить поиск GitHub репозиториев
    "use_api": True,                    # Использовать GitHub API
    "api_token": None,                  # Опционально, из env (GITHUB_TOKEN)
    "use_web_search": True,            # Fallback на веб-поиск (DuckDuckGo)
    "max_repos_per_paper": 3,          # Максимум репозиториев на статью
}

# ============================================================================
# АГЕНТНЫЕ НАСТРОЙКИ
# ============================================================================

AGENT_CONFIG = {
    "enable_retry": True,              # Включить повторный поиск при малом количестве результатов
    "min_papers_threshold": 10,         # Минимум статей для продолжения без retry
    "enable_replanning": True,          # Включить перепланирование поиска
    "replan_gap_threshold": 5,          # Количество high-severity gaps для replan
    "max_retries": 2,                   # Максимум попыток retry
}

# ============================================================================
# КОНТАКТЫ ДЛЯ API (вежливость)
# ============================================================================

API_CONTACT = {
    "email": "dru4inin.dmitry@gmail.com",  # Замените на ваш email
    "user_agent": "AI-Research-Agent/1.0",
}


def get_config():
    """Возвращает полную конфигурацию"""
    return {
        "llm": LLM_CONFIG,
        "search": SEARCH_CONFIG,
        "sources": SOURCES_ENABLED,
        "dedup": DEDUP_CONFIG,
        "ranking": {
            "components": RANKING_COMPONENTS,
            "weights": RANKING_WEIGHTS,
            "venue_scores": VENUE_SCORES,
        },
        "gap_mining": GAP_MINING_CONFIG,
        "ideation": IDEATION_CONFIG,
        "report": REPORT_CONFIG,
        "rate_limits": RATE_LIMITS,
        "budget": BUDGET_CONFIG,
        "cache": CACHE_CONFIG,
        "logging": LOGGING_CONFIG,
        "experimental": EXPERIMENTAL,
        "vector_search": VECTOR_SEARCH_CONFIG,
        "rerank": RERANK_CONFIG,
        "filters": SEARCH_FILTERS,
        "snowball": SNOWBALL_CONFIG,
        "replan": REPLAN_CONFIG,
        "github": GITHUB_CONFIG,
        "agent": AGENT_CONFIG,
        "api_contact": API_CONTACT,
    }


def print_config():
    """Выводит текущую конфигурацию"""
    import json
    config = get_config()
    print("=" * 70)
    print("ТЕКУЩАЯ КОНФИГУРАЦИЯ АГЕНТА")
    print("=" * 70)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("=" * 70)


if __name__ == "__main__":
    print_config()

