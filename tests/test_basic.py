"""
Простой тест основных компонентов агента
"""

from main import create_research_agent, get_llm
from config import LLM_CONFIG
import os

def test_llm_connection():
    """Тестируем подключение к OpenAI"""
    print("🔧 Тестирую подключение к LLM...")

    try:
        llm = get_llm()
        response = llm.invoke("Say 'Hello, AI Agent!' in one word")
        print(f"✅ LLM работает: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")
        return False

def test_agent_creation():
    """Тестируем создание LangGraph агента"""
    print("\n🔧 Тестирую создание агента...")

    try:
        agent = create_research_agent()
        print("✅ Агент создан успешно")
        print(f"   - Тип: {type(agent)}")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания агента: {e}")
        return False

def test_config():
    """Тестируем конфигурацию"""
    print("\n🔧 Тестирую конфигурацию...")

    try:
        from config import get_config
        config = get_config()

        print("✅ Конфигурация загружена")
        print(f"   - LLM модель: {config['llm']['model']}")
        print(f"   - Температура: {config['llm']['temperature']}")
        print(f"   - Источники: {list(config['sources'].keys())}")

        return True
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return False

def main():
    print("="*60)
    print("🧪 БАЗОВОЕ ТЕСТИРОВАНИЕ AI АГЕНТА")
    print("="*60)

    # Проверяем API ключ
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден!")
        return False

    print(f"✅ OPENAI_API_KEY найден (длина: {len(os.getenv('OPENAI_API_KEY'))})")

    # Запускаем тесты
    results = []
    results.append(test_config())
    results.append(test_llm_connection())
    results.append(test_agent_creation())

    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")

    passed = sum(results)
    total = len(results)

    print(f"✅ Пройдено: {passed}/{total}")

    if passed == total:
        print("🎉 Все тесты пройдены! Агент готов к работе.")
    else:
        print("⚠️  Некоторые тесты не пройдены.")

    return passed == total

if __name__ == "__main__":
    main()
