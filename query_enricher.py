"""
Обогащение запроса пользователя: уточнение и конкретизация для анализа корпорации Синергия.
"""
import os
import sys

from openai import OpenAI

import config

# Убеждаемся, что стандартная кодировка установлена на UTF-8
if sys.platform != 'win32':
    import locale
    if locale.getpreferredencoding() != 'UTF-8':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

SYSTEM = """Ты — аналитический ассистент. Руководитель задает краткий вопрос.
Твоя задача — кратко уточнить и конкретизировать запрос для анализа, добавив контекст корпорации Синергия.
НЕ делай запрос слишком длинным или развернутым. Достаточно 1-2 предложений, которые уточняют исходный вопрос.
Верни только улучшенный запрос, без дополнительных комментариев.
Если указан год ниже 2026, значит это относится к прошлому и больше указывает на аналитику.
"""


def _client() -> OpenAI:
    artemox_key = os.getenv("ARTEMOX_API_KEY", "").strip()
    if not artemox_key:
        raise ValueError("ARTEMOX_API_KEY не задан.")
    return OpenAI(
        base_url=config.ARTEMOX_BASE,
        api_key=artemox_key,
    )


def enrich_query(original_query: str) -> str:
    """
    Обогащает пользовательский запрос, расширяя и углубляя его.
    Обрабатывает проблемы с кодировкой.
    
    Args:
        original_query: Исходный запрос пользователя
    
    Returns:
        Обогащенный запрос
    """
    if not original_query or not original_query.strip():
        return original_query
    
    try:
        # Убеждаемся, что запрос правильно обработан как UTF-8
        query_str = original_query.strip()
        if isinstance(query_str, bytes):
            query_str = query_str.decode('utf-8')
        
        user_prompt = f"""Исходный запрос:
{query_str}

Кратко уточни и конкретизируй запрос для анализа корпорации Синергия."""

        client = _client()
        model_name = config.ARTEMOX_MODEL
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        
        enriched = (resp.choices[0].message.content or "").strip()
        
        # Убеждаемся, что результат правильно обработан
        if isinstance(enriched, bytes):
            enriched = enriched.decode('utf-8')
        
        # Если модель вернула что-то странное, возвращаем исходный запрос
        if not enriched or len(enriched) < 3:
            return original_query.strip()
        
        return str(enriched).strip()
    
    except UnicodeEncodeError as e:
        # Если возникла ошибка кодировки, пробуем исправить
        import traceback
        print(f"UnicodeEncodeError в enrich_query: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Возвращаем оригинальный запрос как fallback
        return original_query.strip()
    except Exception as e:
        # Для других ошибок также возвращаем оригинальный запрос
        import traceback
        print(f"Ошибка в enrich_query: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return original_query.strip()
