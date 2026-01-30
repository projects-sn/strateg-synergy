"""
Агент-классификатор: извлекает из запроса параметры
локация, время, действующее лицо, действие, контрагент.
Через OpenRouter (gpt-4o).
"""
import json
import os

from openai import OpenAI

import config

# Быстрый классификатор + таймаут
CLASSIFIER_MODEL = "gpt-4o"
CLASSIFIER_TIMEOUT = 30  # секунд

# Поля для извлечения
FIELDS = ["location", "time", "actor", "action", "counterparty"]
FIELDS_RU = {
    "location": "локация",
    "time": "время",
    "actor": "действующее лицо",
    "action": "действие",
    "counterparty": "контрагент",
}

SYSTEM = """Ты извлекаешь структурированные параметры из запроса пользователя к корпоративной базе (брифинги, стенограммы, презентации).

Параметры (на русском, если пользователь по-русски):
- location (локация): место, город, регион, страна
- time (время): дата, период, год, «прошлый брифинг» и т.п.
- actor (действующее лицо): ФИО, должность, кто выступает/действует
- action (действие): о чём речь — стратегия, HR, IT, экспорт, проблемы и т.п.
- counterparty (контрагент): партнёр, организация, кому/с кем

Ответ — только валидный JSON без markdown и комментариев, с ключами: location, time, actor, action, counterparty.
Значение — строка или null, если не удалось определить. Пустые — null."""

USER_TEMPLATE = "Запрос: {query}"


def _client() -> tuple[OpenAI, str]:
    api_key = os.getenv("ARTEMOX_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ARTEMOX_API_KEY не задан. Укажи в .env или в .streamlit/secrets.toml."
        )
    return (
        OpenAI(base_url=config.ARTEMOX_BASE, api_key=api_key, timeout=CLASSIFIER_TIMEOUT),
        CLASSIFIER_MODEL,
    )


def classify(query: str) -> dict[str, str | None]:
    """Извлекает 5 параметров из запроса. Возвращает dict с ключами FIELDS."""
    if not query or not query.strip():
        return {f: None for f in FIELDS}

    client, model = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(query=query.strip())},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    # Убрать обёртки ```json ... ```
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {f: None for f in FIELDS}

    out: dict[str, str | None] = {}
    for f in FIELDS:
        v = data.get(f)
        if isinstance(v, str) and v.strip():
            out[f] = v.strip()
        else:
            out[f] = None
    return out


def params_to_keywords(params: dict[str, str | None]) -> str:
    """Собирает из параметров строку ключевых слов для добавления к запросу."""
    parts = [v for v in params.values() if v]
    return " ".join(parts)
