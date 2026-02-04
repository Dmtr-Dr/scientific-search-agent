"""
Тест реального поиска с ограниченными параметрами
"""

from main import run_research_pipeline
import time

def test_mini_research():
    """Тестируем полный цикл с минимальными параметрами"""
    print("="*70)
    print("🧪 ТЕСТ МИНИ-ИССЛЕДОВАНИЯ")
    print("="*70)

    # Очень маленькие параметры для быстрого теста
    query = "machine learning basics"
    time_window = 1  # Только последний год
    max_papers = 5   # Всего 5 статей

    print(f"Тема: {query}")
    print(f"Период: последние {time_window} год")
    print(f"Максимум статей: {max_papers}")
    print("="*70)

    start_time = time.time()

    try:
        report = run_research_pipeline(
            query=query,
            time_window=time_window,
            max_papers=max_papers,
            save_report=True  # Сохраняем файл
        )

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "="*70)
        print("✅ ТЕСТ ЗАВЕРШЁН УСПЕШНО!")
        print("="*70)
        print(f"Время выполнения: {duration:.1f} сек")
        print(f"Длина отчёта: {len(report)} символов")

        # Проверяем основные секции отчёта
        sections = [
            "Executive Summary",
            "Top Papers",
            "Literature Matrix",
            "Research Gaps",
            "Research Ideas"
        ]

        print("\n📋 Проверка секций отчёта:")
        for section in sections:
            if section.lower() in report.lower():
                print(f"   ✅ {section}")
            else:
                print(f"   ❌ {section}")

        return True, duration, len(report)

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n❌ ОШИБКА: {e}")
        print(f"Время выполнения: {duration:.1f} сек")
        return False, duration, 0

def test_error_handling():
    """Тестируем обработку ошибок"""
    print("\n" + "="*70)
    print("🧪 ТЕСТ ОБРАБОТКИ ОШИБОК")
    print("="*70)

    # Тест с пустым запросом
    try:
        report = run_research_pipeline(
            query="",
            time_window=1,
            max_papers=5,
            save_report=False
        )
        print("❌ Пустой запрос не вызвал ошибку")
        return False
    except Exception as e:
        print(f"✅ Пустой запрос правильно обработан: {type(e).__name__}")

    # Тест с очень большим time_window
    try:
        report = run_research_pipeline(
            query="test",
            time_window=100,  # Слишком большой период
            max_papers=5,
            save_report=False
        )
        print("✅ Большой time_window обработан")
        return True
    except Exception as e:
        print(f"❌ Большой time_window вызвал ошибку: {e}")
        return False

def main():
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ПОЛНОГО ЦИКЛА\n")

    # Основной тест
    success, duration, report_length = test_mini_research()

    # Тест обработки ошибок
    error_handling_ok = test_error_handling()

    # Итоги
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*70)

    if success:
        print("✅ Мини-исследование: УСПЕШНО")
        print(f"   Время выполнения: {duration:.1f} сек")
        print(f"   Длина отчёта: {report_length} символов")
    else:
        print("❌ Мини-исследование: НЕУДАЧА")

    if error_handling_ok:
        print("✅ Обработка ошибок: УСПЕШНО")
    else:
        print("❌ Обработка ошибок: ПРОБЛЕМЫ")

    overall_success = success and error_handling_ok

    if overall_success:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("Агент полностью готов к работе.")
    else:
        print("\n⚠️  ИМЕЮТСЯ ПРОБЛЕМЫ")
        print("Рекомендуется проверить конфигурацию и API ключи.")

    return overall_success

if __name__ == "__main__":
    main()
