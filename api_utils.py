"""
API утилиты для Research Agent
Фаза 3: Retry логика и Circuit Breaker для надежности API
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Callable, Any
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# ЗАДАЧА 3.1: RETRY ЛОГИКА С EXPONENTIAL BACKOFF
# ============================================================================

class RetryConfig:
    """Конфигурация retry логики"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_wait: float = 2.0,
        max_wait: float = 10.0,
        exponential_base: float = 2.0
    ):
        """
        Args:
            max_attempts: Максимальное количество попыток
            initial_wait: Начальная задержка в секундах
            max_wait: Максимальная задержка в секундах
            exponential_base: База для экспоненциального роста
        """
        self.max_attempts = max_attempts
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.exponential_base = exponential_base
    
    def get_wait_time(self, attempt: int) -> float:
        """
        Вычисляет время ожидания для попытки
        
        Args:
            attempt: Номер попытки (0-indexed)
        
        Returns:
            Время ожидания в секундах
        """
        wait = self.initial_wait * (self.exponential_base ** attempt)
        return min(wait, self.max_wait)


async def async_retry(
    func: Callable,
    *args,
    config: RetryConfig = None,
    retry_exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Асинхронный wrapper с retry логикой и exponential backoff
    
    Args:
        func: Асинхронная функция для выполнения
        *args: Позиционные аргументы для функции
        config: Конфигурация retry (опционально)
        retry_exceptions: Кортеж исключений для retry
        **kwargs: Именованные аргументы для функции
    
    Returns:
        Результат выполнения функции
    
    Raises:
        Последнее пойманное исключение если все попытки исчерпаны
    
    Example:
        result = await async_retry(
            search_api,
            query="test",
            config=RetryConfig(max_attempts=3),
            retry_exceptions=(TimeoutError, ConnectionError)
        )
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            result = await func(*args, **kwargs)
            
            # Успех!
            if attempt > 0:
                logger.info(f"✅ Retry successful on attempt {attempt + 1}")
            
            return result
        
        except retry_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                wait_time = config.get_wait_time(attempt)
                logger.warning(
                    f"⚠️ Attempt {attempt + 1}/{config.max_attempts} failed: {str(e)[:100]}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"❌ All {config.max_attempts} attempts failed. Last error: {str(e)[:100]}"
                )
                raise last_exception
        
        except Exception as e:
            # Исключение не подлежит retry
            logger.error(f"❌ Non-retryable error: {str(e)[:100]}")
            raise e
    
    # Shouldn't reach here, but just in case
    if last_exception:
        raise last_exception


def sync_retry(
    func: Callable,
    *args,
    config: RetryConfig = None,
    retry_exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Синхронный wrapper с retry логикой и exponential backoff
    
    Аналогичен async_retry, но для синхронных функций
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            result = func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"✅ Retry successful on attempt {attempt + 1}")
            
            return result
        
        except retry_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                wait_time = config.get_wait_time(attempt)
                logger.warning(
                    f"⚠️ Attempt {attempt + 1}/{config.max_attempts} failed: {str(e)[:100]}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                import time
                time.sleep(wait_time)
            else:
                logger.error(
                    f"❌ All {config.max_attempts} attempts failed. Last error: {str(e)[:100]}"
                )
                raise last_exception
        
        except Exception as e:
            logger.error(f"❌ Non-retryable error: {str(e)[:100]}")
            raise e
    
    if last_exception:
        raise last_exception


# ============================================================================
# ЗАДАЧА 3.2: CIRCUIT BREAKER
# ============================================================================

class CircuitState:
    """Состояния circuit breaker"""
    CLOSED = "CLOSED"  # Нормальная работа
    OPEN = "OPEN"  # API отключен (много ошибок)
    HALF_OPEN = "HALF_OPEN"  # Пробуем восстановить


class CircuitBreaker:
    """
    Circuit Breaker для защиты от постоянно падающих API
    
    Паттерн работы:
    1. CLOSED: Нормальная работа, запросы проходят
    2. Если ошибок > failure_threshold → переход в OPEN
    3. OPEN: Все запросы блокируются без попыток
    4. После recovery_timeout → переход в HALF_OPEN
    5. HALF_OPEN: Пробуем один запрос
       - Успех → возврат в CLOSED
       - Ошибка → возврат в OPEN
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Args:
            failure_threshold: Количество ошибок для перехода в OPEN
            recovery_timeout: Секунд до перехода в HALF_OPEN
            success_threshold: Успешных запросов для перехода в CLOSED
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # Состояние по API
        self.failures: Dict[str, int] = {}
        self.successes: Dict[str, int] = {}
        self.states: Dict[str, str] = {}
        self.opened_at: Dict[str, datetime] = {}
        
        logger.info(
            f"⚡ Circuit Breaker initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )
    
    def record_success(self, api_name: str):
        """Регистрирует успешный вызов API"""
        current_state = self.states.get(api_name, CircuitState.CLOSED)
        
        if current_state == CircuitState.HALF_OPEN:
            # В HALF_OPEN считаем успехи для восстановления
            self.successes[api_name] = self.successes.get(api_name, 0) + 1
            
            if self.successes[api_name] >= self.success_threshold:
                # Достаточно успехов - восстанавливаем
                self._transition_to_closed(api_name)
        
        elif current_state == CircuitState.CLOSED:
            # Сбрасываем счетчик ошибок при успехе
            self.failures[api_name] = 0
    
    def record_failure(self, api_name: str):
        """Регистрирует неудачный вызов API"""
        current_state = self.states.get(api_name, CircuitState.CLOSED)
        
        if current_state == CircuitState.CLOSED:
            self.failures[api_name] = self.failures.get(api_name, 0) + 1
            
            if self.failures[api_name] >= self.failure_threshold:
                self._transition_to_open(api_name)
        
        elif current_state == CircuitState.HALF_OPEN:
            # Неудача в HALF_OPEN - обратно в OPEN
            self._transition_to_open(api_name)
    
    def is_call_allowed(self, api_name: str) -> bool:
        """
        Проверяет, разрешен ли вызов API
        
        Returns:
            True если вызов разрешен, False если заблокирован
        """
        current_state = self.states.get(api_name, CircuitState.CLOSED)
        
        if current_state == CircuitState.CLOSED:
            return True
        
        elif current_state == CircuitState.OPEN:
            # Проверяем, не пора ли перейти в HALF_OPEN
            if api_name in self.opened_at:
                elapsed = (datetime.now() - self.opened_at[api_name]).seconds
                
                if elapsed >= self.recovery_timeout:
                    self._transition_to_half_open(api_name)
                    return True  # Разрешаем пробный запрос
            
            return False  # Блокируем вызов
        
        elif current_state == CircuitState.HALF_OPEN:
            return True  # Разрешаем пробный запрос
        
        return False
    
    def _transition_to_open(self, api_name: str):
        """Переход в состояние OPEN"""
        self.states[api_name] = CircuitState.OPEN
        self.opened_at[api_name] = datetime.now()
        logger.warning(f"🔴 Circuit breaker OPEN for '{api_name}' (failures: {self.failures[api_name]})")
    
    def _transition_to_half_open(self, api_name: str):
        """Переход в состояние HALF_OPEN"""
        self.states[api_name] = CircuitState.HALF_OPEN
        self.successes[api_name] = 0
        logger.info(f"🟡 Circuit breaker HALF_OPEN for '{api_name}' (attempting recovery)")
    
    def _transition_to_closed(self, api_name: str):
        """Переход в состояние CLOSED"""
        self.states[api_name] = CircuitState.CLOSED
        self.failures[api_name] = 0
        self.successes[api_name] = 0
        if api_name in self.opened_at:
            del self.opened_at[api_name]
        logger.info(f"🟢 Circuit breaker CLOSED for '{api_name}' (recovered)")
    
    def get_state(self, api_name: str) -> str:
        """Возвращает текущее состояние API"""
        return self.states.get(api_name, CircuitState.CLOSED)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику circuit breaker"""
        return {
            "states": dict(self.states),
            "failures": dict(self.failures),
            "successes": dict(self.successes),
            "opened_at": {k: v.isoformat() for k, v in self.opened_at.items()}
        }
    
    def reset(self, api_name: str = None):
        """
        Сбрасывает circuit breaker
        
        Args:
            api_name: Если указан, сбрасывает только для этого API
        """
        if api_name:
            self.states.pop(api_name, None)
            self.failures.pop(api_name, None)
            self.successes.pop(api_name, None)
            self.opened_at.pop(api_name, None)
            logger.info(f"🔄 Circuit breaker reset for '{api_name}'")
        else:
            self.states.clear()
            self.failures.clear()
            self.successes.clear()
            self.opened_at.clear()
            logger.info("🔄 Circuit breaker reset for all APIs")


# ============================================================================
# ГЛОБАЛЬНЫЙ CIRCUIT BREAKER
# ============================================================================

_global_circuit_breaker = None

def get_circuit_breaker() -> CircuitBreaker:
    """Возвращает глобальный экземпляр circuit breaker"""
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2
        )
    return _global_circuit_breaker


# ============================================================================
# ИНТЕГРАЦИЯ: Async Search с Circuit Breaker
# ============================================================================

async def call_with_circuit_breaker(
    api_name: str,
    func: Callable,
    *args,
    fallback_value: Any = None,
    **kwargs
) -> Any:
    """
    Вызывает функцию с защитой circuit breaker
    
    Args:
        api_name: Название API для tracking
        func: Асинхронная функция для вызова
        *args: Аргументы для функции
        fallback_value: Значение при блокировке (по умолчанию None)
        **kwargs: Именованные аргументы для функции
    
    Returns:
        Результат функции или fallback_value если заблокирован
    """
    breaker = get_circuit_breaker()
    
    if not breaker.is_call_allowed(api_name):
        logger.warning(f"⛔ Call to '{api_name}' blocked by circuit breaker (state: {breaker.get_state(api_name)})")
        return fallback_value
    
    try:
        result = await func(*args, **kwargs)
        breaker.record_success(api_name)
        return result
    
    except Exception as e:
        breaker.record_failure(api_name)
        logger.error(f"❌ '{api_name}' failed: {str(e)[:100]}")
        raise e


# ============================================================================
# КОМБИНИРОВАННЫЙ WRAPPER: Retry + Circuit Breaker
# ============================================================================

async def resilient_api_call(
    api_name: str,
    func: Callable,
    *args,
    retry_config: RetryConfig = None,
    fallback_value: Any = None,
    retry_exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Комбинированный wrapper: Circuit Breaker + Retry логика
    
    Порядок выполнения:
    1. Проверка circuit breaker
    2. Если разрешено - retry с exponential backoff
    3. Отслеживание успехов/ошибок для circuit breaker
    
    Args:
        api_name: Название API
        func: Асинхронная функция
        *args: Аргументы функции
        retry_config: Конфигурация retry (опционально)
        fallback_value: Значение при блокировке
        retry_exceptions: Исключения для retry
        **kwargs: Именованные аргументы
    
    Returns:
        Результат функции или fallback_value
    
    Example:
        result = await resilient_api_call(
            "openalex",
            search_openalex_async,
            query="test",
            max_results=30,
            retry_config=RetryConfig(max_attempts=3),
            fallback_value=[]
        )
    """
    breaker = get_circuit_breaker()
    
    # Проверяем circuit breaker
    if not breaker.is_call_allowed(api_name):
        logger.warning(f"⛔ '{api_name}' blocked by circuit breaker")
        return fallback_value
    
    # Выполняем с retry
    try:
        result = await async_retry(
            func,
            *args,
            config=retry_config,
            retry_exceptions=retry_exceptions,
            **kwargs
        )
        breaker.record_success(api_name)
        return result
    
    except Exception as e:
        breaker.record_failure(api_name)
        logger.error(f"❌ '{api_name}' failed after all retries: {str(e)[:100]}")
        return fallback_value


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Тестовая функция
    call_count = 0
    
    async def test_api(should_fail: bool = False):
        global call_count
        call_count += 1
        print(f"📞 API call #{call_count}")
        
        if should_fail:
            raise Exception("API error")
        
        return {"status": "success", "data": "test"}
    
    async def main():
        # Тест 1: Успешный вызов с retry
        print("\n=== Test 1: Successful call ===")
        result = await async_retry(test_api, should_fail=False)
        print(f"Result: {result}")
        
        # Тест 2: Retry после ошибок
        print("\n=== Test 2: Retry after failures ===")
        call_count = 0
        try:
            await async_retry(test_api, should_fail=True, config=RetryConfig(max_attempts=3))
        except Exception as e:
            print(f"Failed after {call_count} attempts: {e}")
        
        # Тест 3: Circuit breaker
        print("\n=== Test 3: Circuit breaker ===")
        breaker = get_circuit_breaker()
        
        # Симулируем 5 ошибок
        for i in range(5):
            try:
                await call_with_circuit_breaker("test_api", test_api, should_fail=True, fallback_value={})
            except:
                pass
        
        # Следующий вызов должен быть заблокирован
        print("\nNext call should be blocked:")
        result = await call_with_circuit_breaker("test_api", test_api, should_fail=False, fallback_value={"blocked": True})
        print(f"Result: {result}")
        
        print(f"\nCircuit breaker stats: {breaker.get_stats()}")
    
    asyncio.run(main())
