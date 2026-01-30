"""
Генерация ответа по контексту RAG через OpenRouter (gpt-4o).
Модель извлекает самое важное из найденных топ-k результатов BM25 и семантического поиска.
"""
import os

from openai import OpenAI

import config

SYSTEM = """Ты анализируешь найденные фрагменты из корпоративных документов (брифинги, стенограммы, презентации) и извлекаешь самое важное, чтобы ответить на вопрос пользователя.

Инструкция:
1. Внимательно прочитай ВСЕ найденные фрагменты (они отсортированы по релевантности)
2. Выдели ключевую информацию, относящуюся к вопросу
3. Объедини информацию из разных фрагментов, если она дополняет друг друга
4. Извлеки самое важное и существенное из всех найденных результатов
5. Сформулируй полный, структурированный ответ на основе найденных данных
6. Если в контексте нет ответа — честно напиши об этом. Не придумывай факты.

Важно: используй информацию ТОЛЬКО из предоставленных фрагментов. Не добавляй информацию, которой нет в контексте."""


def _client() -> OpenAI:
    artemox_key = os.getenv("ARTEMOX_API_KEY", "").strip()
    if not artemox_key:
        raise ValueError("ARTEMOX_API_KEY не задан.")
    return OpenAI(
        base_url=config.ARTEMOX_BASE,
        api_key=artemox_key,
    )


def _format_context(docs: list[dict]) -> str:
    """Форматирует контекст для промпта."""
    parts = []
    for i, d in enumerate(docs, 1):
        t = d.get("text", "")
        src = d.get("source", "")
        f = d.get("file", "")
        dt = d.get("date", "")
        page = d.get("page", None)
        score = d.get("_score", None)
        source_type = d.get("_source", "")
        
        # Формируем метаинформацию
        meta_parts = [f"source: {src}", f"file: {f}"]
        if dt:
            meta_parts.append(f"date: {dt}")
        if page is not None:
            meta_parts.append(f"page: {page}")
        if score is not None:
            meta_parts.append(f"relevance: {score:.4f}")
        if source_type:
            meta_parts.append(f"found_by: {source_type}")
        
        meta = ", ".join(meta_parts)
        parts.append(f"[Фрагмент {i}] ({meta})\n{t}")
    return "\n\n---\n\n".join(parts)


def generate(query: str, docs: list[dict], stream: bool = False):
    """
    Генерирует ответ по запросу и списку документов из retriever.
    Модель извлекает самое важное из найденных топ-k результатов.
    
    Args:
        query: Поисковый запрос пользователя
        docs: Список найденных документов (отсортированных по релевантности)
        stream: Если True, возвращает итератор по кускам текста
    
    Returns:
        str (если stream=False) или iterator (если stream=True)
    """
    if not docs:
        return "Не найдено релевантных документов для ответа на вопрос."
    
    ctx = _format_context(docs)
    
    user = f"""Ниже приведены найденные фрагменты документов (топ-{len(docs)} по релевантности из BM25 и семантического поиска):

{ctx}

Задача: проанализируй ВСЕ эти фрагменты, извлеки самое важное и существенное из них, объедини информацию из разных источников (если она дополняет друг друга), и сформулируй полный, структурированный ответ на вопрос пользователя.

Вопрос пользователя: {query}

Важно: 
- Используй информацию ТОЛЬКО из предоставленных фрагментов
- Извлеки самое важное из всех найденных результатов
- Объедини информацию из разных фрагментов, если она дополняет друг друга
- Если в контексте нет ответа — честно напиши об этом"""

    client = _client()
    model_name = config.ARTEMOX_MODEL if os.getenv("ARTEMOX_API_KEY") else config.OPENROUTER_MODEL
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        stream=stream,
    )

    if not stream:
        return (resp.choices[0].message.content or "").strip()

    def _gen():
        for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and getattr(delta, "content", None):
                yield delta.content

    return _gen()