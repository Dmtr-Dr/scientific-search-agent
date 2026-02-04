"""
Система кэширования для Research Agent
Фаза 2, Задача 2.2: SQLite кэширование результатов
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
import logging
import os

logger = logging.getLogger(__name__)


class ResultCache:
    """
    SQLite кэш для результатов API запросов
    
    Особенности:
    - TTL (Time To Live) для автоматической инвалидации
    - Хэширование ключей для уникальности
    - Thread-safe операции
    """
    
    def __init__(self, db_path: str = "data/cache.db", ttl_days: int = 7):
        """
        Инициализация кэша
        
        Args:
            db_path: Путь к файлу SQLite базы данных
            ttl_days: Время жизни кэша в днях
        """
        self.db_path = db_path
        self.ttl_days = ttl_days
        
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        self._init_db()
        logger.info(f"✅ Cache initialized: {db_path} (TTL: {ttl_days} days)")
    
    def _init_db(self):
        """Создает таблицу кэша если не существует"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Создаем индексы для быстрого поиска
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON cache(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)")
        
        conn.commit()
        conn.close()
        logger.info("📊 Cache database initialized")
    
    def _make_key(self, source: str, query: str, params: Dict) -> str:
        """
        Генерирует уникальный ключ кэша на основе источника, запроса и параметров
        
        Args:
            source: Название источника (openalex, arxiv, etc.)
            query: Поисковый запрос
            params: Параметры поиска (max_results, from_year, etc.)
        
        Returns:
            MD5 хэш строки ключа
        """
        # Сортируем параметры для консистентности
        params_str = json.dumps(params, sort_keys=True)
        key_str = f"{source}:{query}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, source: str, query: str, params: Dict) -> Optional[Any]:
        """
        Получает значение из кэша
        
        Args:
            source: Название источника
            query: Поисковый запрос
            params: Параметры поиска
        
        Returns:
            Кэшированное значение или None если не найдено/устарело
        """
        key = self._make_key(source, query, params)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value, created_at FROM cache WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            value_json, created_at_str = row
            created_at = datetime.fromisoformat(created_at_str)
            
            # Проверяем TTL
            if datetime.now() - created_at < timedelta(days=self.ttl_days):
                # Обновляем статистику доступа
                conn.execute(
                    "UPDATE cache SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                    (datetime.now().isoformat(), key)
                )
                conn.commit()
                conn.close()
                
                logger.info(f"✅ Cache HIT: {source} - {query[:30]}...")
                return json.loads(value_json)
            else:
                # Устаревшая запись - удаляем
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                conn.close()
                logger.info(f"🗑️ Cache EXPIRED: {source} - {query[:30]}...")
                return None
        
        conn.close()
        logger.info(f"❌ Cache MISS: {source} - {query[:30]}...")
        return None
    
    def set(self, source: str, query: str, params: Dict, value: Any):
        """
        Сохраняет значение в кэш
        
        Args:
            source: Название источника
            query: Поисковый запрос
            params: Параметры поиска
            value: Значение для кэширования
        """
        key = self._make_key(source, query, params)
        value_json = json.dumps(value, ensure_ascii=False)
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, source, query, created_at, accessed_at, access_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, value_json, source, query[:200], now, now, 0)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"💾 Cache SET: {source} - {query[:30]}... ({len(value)} items)")
    
    def clear(self, source: Optional[str] = None):
        """
        Очищает кэш
        
        Args:
            source: Если указан, очищает только для этого источника
        """
        conn = sqlite3.connect(self.db_path)
        
        if source:
            conn.execute("DELETE FROM cache WHERE source = ?", (source,))
            logger.info(f"🗑️ Cache cleared for source: {source}")
        else:
            conn.execute("DELETE FROM cache")
            logger.info("🗑️ Cache cleared completely")
        
        conn.commit()
        conn.close()
    
    def clear_expired(self):
        """Удаляет все устаревшие записи из кэша"""
        cutoff_date = (datetime.now() - timedelta(days=self.ttl_days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM cache WHERE created_at < ?",
            (cutoff_date,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"🗑️ Removed {deleted} expired cache entries")
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        conn = sqlite3.connect(self.db_path)
        
        # Общее количество записей
        cursor = conn.execute("SELECT COUNT(*) FROM cache")
        total = cursor.fetchone()[0]
        
        # По источникам
        cursor = conn.execute("SELECT source, COUNT(*) FROM cache GROUP BY source")
        by_source = dict(cursor.fetchall())
        
        # Наиболее часто используемые
        cursor = conn.execute(
            "SELECT source, query, access_count FROM cache ORDER BY access_count DESC LIMIT 10"
        )
        top_accessed = cursor.fetchall()
        
        # Размер базы данных
        import os
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        db_size_mb = db_size / (1024 * 1024)
        
        conn.close()
        
        return {
            "total_entries": total,
            "by_source": by_source,
            "top_accessed": top_accessed,
            "db_size_mb": round(db_size_mb, 2)
        }
    
    def print_stats(self):
        """Выводит статистику кэша в консоль"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📊 CACHE STATISTICS")
        print("="*70)
        print(f"Total entries: {stats['total_entries']}")
        print(f"Database size: {stats['db_size_mb']} MB")
        print(f"\nBy source:")
        for source, count in stats['by_source'].items():
            print(f"  - {source}: {count} entries")
        
        if stats['top_accessed']:
            print(f"\nTop accessed:")
            for source, query, count in stats['top_accessed'][:5]:
                print(f"  - [{source}] {query[:50]}... ({count} accesses)")
        
        print("="*70 + "\n")


# ============================================================================
# ИНТЕГРАЦИЯ С ASYNC SEARCH
# ============================================================================

# Глобальный экземпляр кэша
_global_cache = None

def get_cache() -> ResultCache:
    """Возвращает глобальный экземпляр кэша"""
    global _global_cache
    if _global_cache is None:
        from config import CACHE_CONFIG
        enabled = CACHE_CONFIG.get("enabled", False)
        
        if enabled:
            db_path = CACHE_CONFIG.get("db_path", "data/cache.db")
            ttl_days = CACHE_CONFIG.get("ttl_days", 7)
            _global_cache = ResultCache(db_path=db_path, ttl_days=ttl_days)
        else:
            logger.info("⚠️ Cache is DISABLED in config")
            _global_cache = None
    
    return _global_cache


# ============================================================================
# ДЕКОРАТОР ДЛЯ КЭШИРОВАНИЯ
# ============================================================================

def cached_search(source_name: str):
    """
    Декоратор для автоматического кэширования результатов поиска
    
    Usage:
        @cached_search("openalex")
        async def search_openalex_async(query, max_results, from_year):
            ...
    """
    def decorator(func):
        async def wrapper(query: str, *args, **kwargs):
            cache = get_cache()
            
            if cache is None:
                # Кэш отключен - вызываем функцию напрямую
                return await func(query, *args, **kwargs)
            
            # Формируем параметры для ключа
            params = {
                "max_results": kwargs.get("max_results") or (args[0] if len(args) > 0 else 30),
                "from_year": kwargs.get("from_year") or (args[1] if len(args) > 1 else None)
            }
            
            # Проверяем кэш
            cached_result = cache.get(source_name, query, params)
            if cached_result is not None:
                return cached_result
            
            # Вызываем функцию
            result = await func(query, *args, **kwargs)
            
            # Сохраняем в кэш
            if result:  # Только успешные результаты
                cache.set(source_name, query, params, result)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Тестирование кэша
    logging.basicConfig(level=logging.INFO)
    
    cache = ResultCache(db_path="test_cache.db", ttl_days=7)
    
    # Тестовые данные
    test_query = "machine learning"
    test_params = {"max_results": 30, "from_year": 2020}
    test_value = [{"title": "Test Paper", "authors": ["John Doe"]}]
    
    # Сохраняем
    cache.set("openalex", test_query, test_params, test_value)
    
    # Загружаем
    result = cache.get("openalex", test_query, test_params)
    print(f"Retrieved: {result}")
    
    # Статистика
    cache.print_stats()
    
    # Очистка
    import os
    os.remove("test_cache.db")
