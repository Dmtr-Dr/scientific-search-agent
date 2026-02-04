"""
Асинхронные функции поиска для Research Agent
Фаза 2, Задача 2.1: Async API calls
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# ASYNC SEARCH: OpenAlex
# ============================================================================

async def search_openalex_async(query: str, max_results: int = 30, from_year: int = 2020) -> List[Dict[str, Any]]:
    """
    Асинхронный поиск в OpenAlex API
    """
    logger.info(f"🔍 OpenAlex async: searching for '{query}'")
    
    try:
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"default.search:{query},from_publication_date:{from_year}-01-01",
            "per_page": max_results,
            "mailto": "dru4inin.dmitry@gmail.com"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"OpenAlex returned status {response.status}")
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                
                papers = []
                for item in results:
                    # Извлекаем авторов
                    authors = []
                    for authorship in item.get("authorships", [])[:5]:
                        author_info = authorship.get("author", {})
                        if author_info.get("display_name"):
                            authors.append(author_info["display_name"])
                    
                    # Извлекаем DOI
                    doi = None
                    if item.get("doi"):
                        doi = item["doi"].replace("https://doi.org/", "")
                    
                    # Извлекаем venue
                    venue = "Unknown"
                    primary_location = item.get("primary_location", {})
                    if primary_location:
                        source = primary_location.get("source", {})
                        venue = source.get("display_name", "Unknown")
                    
                    # Извлекаем аннотацию
                    abstract = item.get("abstract_inverted_index")
                    summary = ""
                    if abstract:
                        # Восстанавливаем текст из inverted index
                        words_positions = []
                        for word, positions in abstract.items():
                            for pos in positions:
                                words_positions.append((pos, word))
                        words_positions.sort()
                        summary = " ".join([word for _, word in words_positions])
                        if len(summary) > 500:
                            summary = summary[:500] + "..."
                    
                    # Дата публикации
                    pub_date = item.get("publication_date", "")
                    if not pub_date:
                        pub_year = item.get("publication_year")
                        pub_date = f"{pub_year}-01-01" if pub_year else "Unknown"
                    
                    papers.append({
                        "title": item.get("title", "No title"),
                        "authors": authors,
                        "summary": summary or "No abstract available",
                        "url": item.get("id", ""),
                        "doi": doi,
                        "published": pub_date,
                        "venue": venue,
                        "citations_total": item.get("cited_by_count", 0),
                        "source": "openalex"
                    })
                
                logger.info(f"✓ OpenAlex: found {len(papers)} papers")
                return papers
    
    except asyncio.TimeoutError:
        logger.error("OpenAlex: timeout")
        return []
    except Exception as e:
        logger.error(f"OpenAlex error: {e}")
        return []


# ============================================================================
# ASYNC SEARCH: Semantic Scholar
# ============================================================================

async def search_semantic_scholar_async(query: str, max_results: int = 30, from_year: int = 2020) -> List[Dict[str, Any]]:
    """
    Асинхронный поиск в Semantic Scholar API
    """
    logger.info(f"🔍 Semantic Scholar async: searching for '{query}'")
    
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "year": f"{from_year}-",
            "fields": "title,authors,abstract,year,citationCount,venue,externalIds,url,publicationDate"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"Semantic Scholar returned status {response.status}")
                    return []
                
                data = await response.json()
                papers_data = data.get("data", [])
                
                papers = []
                for item in papers_data:
                    # Извлекаем авторов
                    authors = []
                    for author in item.get("authors", [])[:5]:
                        if author.get("name"):
                            authors.append(author["name"])
                    
                    # Извлекаем DOI
                    external_ids = item.get("externalIds", {})
                    doi = external_ids.get("DOI")
                    arxiv_id = external_ids.get("ArXiv")
                    
                    # Аннотация
                    abstract = item.get("abstract", "")
                    if abstract and len(abstract) > 500:
                        abstract = abstract[:500] + "..."
                    
                    papers.append({
                        "title": item.get("title", "No title"),
                        "authors": authors,
                        "summary": abstract or "No abstract available",
                        "url": item.get("url", ""),
                        "doi": doi,
                        "arxiv_id": arxiv_id,
                        "published": item.get("publicationDate", item.get("year", "Unknown")),
                        "venue": item.get("venue", "Unknown"),
                        "citations_total": item.get("citationCount", 0),
                        "source": "semantic_scholar"
                    })
                
                logger.info(f"✓ Semantic Scholar: found {len(papers)} papers")
                return papers
    
    except asyncio.TimeoutError:
        logger.error("Semantic Scholar: timeout")
        return []
    except Exception as e:
        logger.error(f"Semantic Scholar error: {e}")
        return []


# ============================================================================
# ASYNC SEARCH: Crossref
# ============================================================================

async def search_crossref_async(query: str, max_results: int = 30, from_year: int = 2020) -> List[Dict[str, Any]]:
    """
    Асинхронный поиск в Crossref API
    """
    logger.info(f"🔍 Crossref async: searching for '{query}'")
    
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "filter": f"from-pub-date:{from_year}",
            "mailto": "dru4inin.dmitry@gmail.com"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"Crossref returned status {response.status}")
                    return []
                
                data = await response.json()
                items = data.get("message", {}).get("items", [])
                
                papers = []
                for item in items:
                    # Извлекаем авторов
                    authors = []
                    for author in item.get("author", [])[:5]:
                        if author.get("given") and author.get("family"):
                            authors.append(f"{author['given']} {author['family']}")
                        elif author.get("family"):
                            authors.append(author["family"])
                    
                    # DOI
                    doi = item.get("DOI")
                    
                    # Venue (container-title)
                    venue = "Unknown"
                    container = item.get("container-title", [])
                    if container and len(container) > 0:
                        venue = container[0]
                    
                    # Аннотация
                    abstract = item.get("abstract", "")
                    if abstract and len(abstract) > 500:
                        abstract = abstract[:500] + "..."
                    
                    # Дата публикации
                    pub_date = "Unknown"
                    date_parts = item.get("published", {}).get("date-parts", [[]])
                    if date_parts and len(date_parts[0]) > 0:
                        year = date_parts[0][0]
                        month = date_parts[0][1] if len(date_parts[0]) > 1 else 1
                        day = date_parts[0][2] if len(date_parts[0]) > 2 else 1
                        pub_date = f"{year}-{month:02d}-{day:02d}"
                    
                    # Title
                    title_list = item.get("title", [])
                    title = title_list[0] if title_list else "No title"
                    
                    papers.append({
                        "title": title,
                        "authors": authors,
                        "summary": abstract or "No abstract available",
                        "url": item.get("URL", ""),
                        "doi": doi,
                        "published": pub_date,
                        "venue": venue,
                        "citations_total": item.get("is-referenced-by-count", 0),
                        "source": "crossref"
                    })
                
                logger.info(f"✓ Crossref: found {len(papers)} papers")
                return papers
    
    except asyncio.TimeoutError:
        logger.error("Crossref: timeout")
        return []
    except Exception as e:
        logger.error(f"Crossref error: {e}")
        return []


# ============================================================================
# ASYNC SEARCH: ArXiv
# ============================================================================

async def search_arxiv_async(query: str, max_results: int = 30) -> List[Dict[str, Any]]:
    """
    Асинхронный поиск в ArXiv API
    """
    logger.info(f"🔍 ArXiv async: searching for '{query}'")
    
    try:
        # ArXiv API endpoint
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"ArXiv returned status {response.status}")
                    return []
                
                xml_text = await response.text()
                
                # Парсим XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_text)
                
                # Namespace для ArXiv
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                papers = []
                for entry in root.findall('atom:entry', ns):
                    # Title
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None else "No title"
                    
                    # Authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    # Abstract
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""
                    if abstract and len(abstract) > 500:
                        abstract = abstract[:500] + "..."
                    
                    # ArXiv ID and URL
                    id_elem = entry.find('atom:id', ns)
                    arxiv_url = id_elem.text if id_elem is not None else ""
                    
                    # Extract ArXiv ID from URL
                    import re
                    arxiv_id = None
                    if arxiv_url:
                        match = re.search(r'/(\d{4}\.\d{4,5})(?:v\d+)?', arxiv_url)
                        if match:
                            arxiv_id = match.group(1)
                    
                    # Published date
                    published_elem = entry.find('atom:published', ns)
                    pub_date = "Unknown"
                    if published_elem is not None:
                        pub_date = published_elem.text.split('T')[0]
                    
                    # Categories
                    categories = []
                    for category in entry.findall('atom:category', ns):
                        term = category.get('term')
                        if term:
                            categories.append(term)
                    
                    papers.append({
                        "title": title,
                        "authors": authors,
                        "summary": abstract or "No abstract available",
                        "url": arxiv_url,
                        "arxiv_id": arxiv_id,
                        "published": pub_date,
                        "categories": categories,
                        "venue": "arXiv",
                        "source": "arxiv"
                    })
                
                logger.info(f"✓ ArXiv: found {len(papers)} papers")
                return papers
    
    except asyncio.TimeoutError:
        logger.error("ArXiv: timeout")
        return []
    except Exception as e:
        logger.error(f"ArXiv error: {e}")
        return []


# ============================================================================
# ASYNC SEARCH: PubMed
# ============================================================================

async def search_pubmed_async(query: str, max_results: int = 30) -> List[Dict[str, Any]]:
    """
    Асинхронный поиск в PubMed API
    """
    logger.info(f"🔍 PubMed async: searching for '{query}'")
    
    try:
        # PubMed E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Step 1: Search for IDs
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": "dru4inin.dmitry@gmail.com"
        }
        
        async with aiohttp.ClientSession() as session:
            # Get article IDs
            async with session.get(search_url, params=search_params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"PubMed search returned status {response.status}")
                    return []
                
                search_data = await response.json()
                id_list = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not id_list:
                    logger.info("✓ PubMed: no papers found")
                    return []
            
            # Step 2: Fetch article details
            fetch_url = f"{base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "xml",
                "retmode": "xml"
            }
            
            async with session.get(fetch_url, params=fetch_params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"PubMed fetch returned status {response.status}")
                    return []
                
                xml_text = await response.text()
                
                # Parse XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_text)
                
                papers = []
                for article in root.findall(".//PubmedArticle"):
                    try:
                        # Title
                        title_elem = article.find(".//ArticleTitle")
                        title = title_elem.text if title_elem is not None else "No title"
                        
                        # Abstract
                        abstract_elem = article.find(".//AbstractText")
                        abstract = abstract_elem.text if abstract_elem is not None else ""
                        if abstract and len(abstract) > 500:
                            abstract = abstract[:500] + "..."
                        
                        # Authors
                        authors = []
                        for author in article.findall(".//Author")[:5]:
                            lastname = author.find("LastName")
                            forename = author.find("ForeName")
                            if lastname is not None and forename is not None:
                                authors.append(f"{forename.text} {lastname.text}")
                        
                        # PMID
                        pmid_elem = article.find(".//PMID")
                        pmid = pmid_elem.text if pmid_elem is not None else ""
                        
                        # Publication date
                        pub_date_elem = article.find(".//PubDate")
                        pub_date = "Unknown"
                        if pub_date_elem is not None:
                            year = pub_date_elem.find("Year")
                            month = pub_date_elem.find("Month")
                            day = pub_date_elem.find("Day")
                            
                            if year is not None:
                                y = year.text
                                m = month.text if month is not None else "01"
                                d = day.text if day is not None else "01"
                                # Handle month names
                                month_map = {
                                    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                                    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                                    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
                                }
                                m = month_map.get(m, m)
                                try:
                                    pub_date = f"{y}-{m}-{d}"
                                except:
                                    pub_date = f"{y}-01-01"
                        
                        # Journal
                        journal_elem = article.find(".//Journal/Title")
                        venue = journal_elem.text if journal_elem is not None else "Unknown"
                        
                        papers.append({
                            "title": title,
                            "authors": authors,
                            "summary": abstract or "No abstract available",
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                            "pmid": pmid,
                            "published": pub_date,
                            "venue": venue,
                            "source": "pubmed"
                        })
                    
                    except Exception as e:
                        logger.warning(f"Error parsing PubMed article: {e}")
                        continue
                
                logger.info(f"✓ PubMed: found {len(papers)} papers")
                return papers
    
    except asyncio.TimeoutError:
        logger.error("PubMed: timeout")
        return []
    except Exception as e:
        logger.error(f"PubMed error: {e}")
        return []


# ============================================================================
# MAIN ASYNC RETRIEVER
# ============================================================================

async def multi_source_retriever_async(
    query: str,
    from_year: int = 2020,
    max_results_per_source: int = 30,
    source_weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Асинхронный поиск по всем источникам параллельно
    
    Args:
        query: Поисковый запрос
        from_year: Начальный год для фильтрации
        max_results_per_source: Максимум результатов с каждого источника
        source_weights: Веса источников {source_name: weight}
    
    Returns:
        Список всех найденных статей со всех источников
    """
    logger.info(f"🚀 Starting async multi-source retrieval for: '{query}'")
    
    # Если веса не указаны, используем все источники с равными весами
    if source_weights is None:
        source_weights = {
            "openalex": 1.0,
            "semantic_scholar": 1.0,
            "crossref": 1.0,
            "arxiv": 1.0,
            "pubmed": 1.0
        }
    
    # Создаем задачи только для выбранных источников (вес > 0)
    tasks = []
    sources_used = []
    
    if source_weights.get("openalex", 0) > 0:
        tasks.append(search_openalex_async(query, max_results_per_source, from_year))
        sources_used.append(("OpenAlex", source_weights["openalex"]))
    
    if source_weights.get("semantic_scholar", 0) > 0:
        tasks.append(search_semantic_scholar_async(query, max_results_per_source, from_year))
        sources_used.append(("Semantic Scholar", source_weights["semantic_scholar"]))
    
    if source_weights.get("crossref", 0) > 0:
        tasks.append(search_crossref_async(query, max_results_per_source, from_year))
        sources_used.append(("Crossref", source_weights["crossref"]))
    
    if source_weights.get("arxiv", 0) > 0:
        tasks.append(search_arxiv_async(query, max_results_per_source))
        sources_used.append(("ArXiv", source_weights["arxiv"]))
    
    if source_weights.get("pubmed", 0) > 0:
        tasks.append(search_pubmed_async(query, max_results_per_source))
        sources_used.append(("PubMed", source_weights["pubmed"]))
    
    # Параллельный запуск всех источников
    logger.info(f"⏳ Querying {len(tasks)} sources in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Обработка результатов и ошибок
    all_papers = []
    for (source_name, weight), result in zip(sources_used, results):
        if isinstance(result, Exception):
            logger.error(f"❌ {source_name} failed: {result}")
        else:
            logger.info(f"✅ {source_name}: {len(result)} papers (weight: {weight})")
            # Добавляем вес источника к каждой статье
            for paper in result:
                paper["_source_weight"] = weight
            all_papers.extend(result)
    
    logger.info(f"🎉 Total papers found: {len(all_papers)} from {len(sources_used)} sources")
    return all_papers


# ============================================================================
# HELPER: Run async function from sync context
# ============================================================================

def run_async_retriever(query: str, from_year: int, max_results: int, source_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Wrapper для запуска async retriever из синхронного кода
    """
    return asyncio.run(multi_source_retriever_async(query, from_year, max_results, source_weights))
