"""
Скрипт для проверки настройки всех API
"""

import os
import sys
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def check_openai():
    """Проверяет настройку OpenAI API"""
    print("🔑 Проверка OpenAI API...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ❌ OPENAI_API_KEY не найден в .env")
        print("      → Добавьте в .env: OPENAI_API_KEY=sk-...")
        return False
    
    if not api_key.startswith("sk-"):
        print("   ⚠️  OPENAI_API_KEY выглядит неверно (должен начинаться с 'sk-')")
        return False
    
    # Пробуем подключиться
    try:
        from main import get_llm
        llm = get_llm()
        response = llm.invoke("Say 'OK'")
        print(f"   ✅ OpenAI API работает: {response.content[:50]}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка подключения к OpenAI: {e}")
        return False

def check_email_config():
    """Проверяет настройку email в config.py"""
    print("\n📧 Проверка email в config.py...")
    
    try:
        from config import API_CONTACT
        email = API_CONTACT.get("email", "")
        
        if not email or email == "researcher@example.com":
            print("   ⚠️  Email не настроен (используется пример)")
            print("      → Откройте config.py и измените API_CONTACT['email']")
            return False
        
        if "@" not in email or "." not in email.split("@")[1]:
            print(f"   ⚠️  Email выглядит неверно: {email}")
            return False
        
        print(f"   ✅ Email настроен: {email}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка чтения config.py: {e}")
        return False

def check_pubmed_email():
    """Проверяет настройку email для PubMed в main.py"""
    print("\n📧 Проверка email для PubMed в main.py...")
    
    try:
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
            
            # Ищем строку с Entrez.email
            if 'Entrez.email = "your.email@example.com"' in content:
                print("   ⚠️  Email для PubMed не настроен (используется пример)")
                print("      → Откройте main.py (строка ~189) и измените Entrez.email")
                return False
            
            if 'Entrez.email =' in content:
                # Извлекаем email
                import re
                match = re.search(r'Entrez\.email\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    email = match.group(1)
                    if email and "@" in email and email != "your.email@example.com":
                        print(f"   ✅ Email для PubMed настроен: {email}")
                        return True
                    else:
                        print(f"   ⚠️  Email для PubMed выглядит неверно: {email}")
                        return False
            
            print("   ⚠️  Не найдена строка Entrez.email в main.py")
            return False
    except Exception as e:
        print(f"   ❌ Ошибка чтения main.py: {e}")
        return False

def test_public_apis():
    """Тестирует публичные API (без ключей)"""
    print("\n🌐 Тестирование публичных API...")
    
    results = {}
    
    # ArXiv
    print("   📚 Тестирую ArXiv...")
    try:
        import arxiv
        search = arxiv.Search(query="machine learning", max_results=1)
        next(search.results(), None)  # Пробуем получить один результат
        print("      ✅ ArXiv работает")
        results["arxiv"] = True
    except Exception as e:
        print(f"      ❌ ArXiv: {e}")
        results["arxiv"] = False
    
    # Semantic Scholar
    print("   🎓 Тестирую Semantic Scholar...")
    try:
        import requests
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": "machine learning", "limit": 1},
            timeout=10
        )
        if response.status_code == 200:
            print("      ✅ Semantic Scholar работает")
            results["semantic_scholar"] = True
        else:
            print(f"      ⚠️  Semantic Scholar вернул код {response.status_code}")
            results["semantic_scholar"] = False
    except Exception as e:
        print(f"      ❌ Semantic Scholar: {e}")
        results["semantic_scholar"] = False
    
    # OpenAlex (без ключа, но с email)
    print("   🌐 Тестирую OpenAlex...")
    try:
        import requests
        response = requests.get(
            "https://api.openalex.org/works",
            params={"search": "machine learning", "per_page": 1, "mailto": "test@example.com"},
            timeout=10
        )
        if response.status_code == 200:
            print("      ✅ OpenAlex работает")
            results["openalex"] = True
        else:
            print(f"      ⚠️  OpenAlex вернул код {response.status_code}")
            results["openalex"] = False
    except Exception as e:
        print(f"      ❌ OpenAlex: {e}")
        results["openalex"] = False
    
    return results

def main():
    print("="*70)
    print("🔍 ПРОВЕРКА НАСТРОЙКИ API")
    print("="*70)
    
    results = {}
    
    # Обязательные проверки
    results["openai"] = check_openai()
    results["email_config"] = check_email_config()
    results["email_pubmed"] = check_pubmed_email()
    
    # Публичные API
    public_apis = test_public_apis()
    results.update(public_apis)
    
    # Итоги
    print("\n" + "="*70)
    print("📊 ИТОГИ ПРОВЕРКИ")
    print("="*70)
    
    required = ["openai", "email_config", "email_pubmed"]
    optional = ["arxiv", "semantic_scholar", "openalex"]
    
    print("\n✅ Обязательные настройки:")
    for key in required:
        status = "✅" if results.get(key) else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print("\n🌐 Публичные API:")
    for key in optional:
        status = "✅" if results.get(key) else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    # Финальный вердикт
    all_required = all(results.get(k, False) for k in required)
    all_public = all(results.get(k, False) for k in optional)
    
    print("\n" + "="*70)
    if all_required:
        print("🎉 ВСЕ ОБЯЗАТЕЛЬНЫЕ НАСТРОЙКИ В ПОРЯДКЕ!")
        if all_public:
            print("✅ Все публичные API работают")
        else:
            print("⚠️  Некоторые публичные API недоступны (может быть временная проблема)")
        print("\n✅ Агент готов к работе!")
        return True
    else:
        print("⚠️  ТРЕБУЮТСЯ ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ")
        print("\n📋 Следующие шаги:")
        if not results.get("openai"):
            print("   1. Настройте OPENAI_API_KEY в .env")
        if not results.get("email_config"):
            print("   2. Настройте email в config.py (API_CONTACT)")
        if not results.get("email_pubmed"):
            print("   3. Настройте email в main.py (Entrez.email)")
        print("\n📖 Подробности: см. API_SETUP.md")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

