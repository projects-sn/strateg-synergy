"""
Прогнозный агент: формирует предложения на будущее (1-3 года)
по обогащенному запросу.
"""
import os
import re
from typing import Dict, Any, List
from dataclasses import dataclass, field

from openai import OpenAI

import config

def _model_name() -> str:
    return config.ARTEMOX_MODEL

SYSTEM = """Ты — стратегический аналитик корпорации Синергия (это частный российский вуз), специализирующийся на анализе будущих перспектив и трендов.
Твоя задача — предложить 3 высокоуровневых варианта решения/подхода, ориентированных на будущее развитие.
Отвечай только на основе общеизвестных и безопасных формулировок, избегай чувствительных данных и закрытой информации.
Если в исходном запросе есть спорные или чувствительные моменты — переформулируй их нейтрально и дай общие рекомендации.

Каждый вариант должен быть:
- Конкретным и реализуемым
- Ориентированным на долгосрочную перспективу
- Учитывающим современные тренды и технологии
- С обоснованием перспективности

Формат ответа — структурированный текст с 3 вариантами, каждый из которых содержит:
- Название варианта
- Краткое описание подхода/решения (2–4 предложения)
- Обоснование перспективности (1–2 предложения)
- Ключевые действия для реализации (3–5 пунктов)

Варианты должны быть ранжированы по приоритету (первый — наиболее перспективный).
Сроки для вариантов НЕ указывать.
Пиши цельным, аккуратным текстом без "служебных" фрагментов и без некорректных разрывов формата."""


def _client() -> OpenAI:
    artemox_key = os.getenv("ARTEMOX_API_KEY", "").strip()
    if not artemox_key:
        raise ValueError("ARTEMOX_API_KEY не задан.")
    return OpenAI(
        base_url=config.ARTEMOX_BASE,
        api_key=artemox_key,
    )


@dataclass
class FutureResult:
    session_id: str
    answer_text: str
    options: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""


def future_chat(session_id: str, user_query: str) -> FutureResult:
    """
    Основной вход: на входе session_id и текст пользователя.
    Возвращает анализ будущих перспектив с вариантами решения.
    """
    client = _client()
    resp = client.chat.completions.create(
        model=_model_name(),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_query},
        ],
        temperature=0.7,
    )
    result_text = (resp.choices[0].message.content or "").strip()
    return FutureResult(
        session_id=session_id,
        answer_text=result_text,
        options=[],
        raw=result_text
    )
