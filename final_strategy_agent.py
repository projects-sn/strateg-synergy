"""
Итоговый стратегический агент: формирует 3 стратегии с ранжированием и SWOT.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any

from openai import OpenAI

import config

def _model_name() -> str:
    return config.ARTEMOX_MODEL

SYSTEM = """Ты — стратегический агент корпорации Синергия.
Твоя задача — на основе трех источников (внутренние данные, внешние кейсы вузов, прогнозные идеи)
сформировать 3 итоговые стратегии и (отдельно) SWOT-анализ по каждой.

Требования:
1) Используй только предоставленные данные, без выдумок.
2) Стратегии должны опираться на то, что уже есть в Синергии (из внутренних данных).
3) Учитывай внешние кейсы вузов и прогнозные идеи.
4) Для каждой стратегии оцени 5 критериев по шкале 0–10:
   - Затратность (10 = очень дорого)
   - Рисковость (10 = очень рискованно)
   - Время (10 = долго реализовать)
   - Эффект (10 = максимальный эффект)
   - Оптимальность (суммарная оценка)
5) Ранжируй стратегии по оптимальности (1 — наиболее предпочтительная).
6) SWOT НЕ выводи в основном блоке. SWOT оформи отдельным блоком между маркерами.

Формат ответа — аккуратный Markdown и строго следующая структура:

## Итоговые стратегии

Для каждой стратегии:
### Стратегия 1: <название>
Краткое описание (3–6 предложений).
Оценки (0-10): Затратность=X; Рисковость=Y; Время=Z; Эффект=W; Оптимальность=O

### Стратегия 2: ...
...

### Стратегия 3: ...
...

<!--SWOT_START-->
## SWOT (скрываемый блок)
### Стратегия 1: <название>
S: 2–3 пункта (каждый пункт с новой строки, начни с "- ")  
W: 2–3 пункта (каждый пункт с новой строки, начни с "- ")  
O: 2–3 пункта (каждый пункт с новой строки, начни с "- ")  
T: 2–3 пункта (каждый пункт с новой строки, начни с "- ")  

### Стратегия 2: ...
...

### Стратегия 3: ...
...
<!--SWOT_END-->

Не выводи JSON. Не упоминай отсутствующие источники."""


def _client() -> OpenAI:
    artemox_key = os.getenv("ARTEMOX_API_KEY", "").strip()
    if not artemox_key:
        raise ValueError("ARTEMOX_API_KEY не задан.")
    return OpenAI(
        base_url=config.ARTEMOX_BASE,
        api_key=artemox_key,
    )


@dataclass
class FinalStrategyResult:
    main_text: str
    swot_text: str
    raw: Dict[str, Any] = field(default_factory=dict)


def build_final_strategy(
    rag_summary: str,
    web_summary: str,
    web_bullets: list[str],
    future_text: str,
) -> FinalStrategyResult:
    """
    Основной вход: агрегирует результаты трех агентов и формирует стратегии.
    """
    client = _client()

    user_prompt = f"""Данные для анализа:

1) Внутренние данные (RAG):
{rag_summary}

Делай на внутренние данные больше всего акцент. Важно составить стратегии на базе уже того, что есть в Синергии.

2) Внешние кейсы вузов (Websearch):
Краткий обзор:
{web_summary}
Ключевые факты:
{'; '.join(web_bullets) if web_bullets else '—'}

3) Прогнозные идеи (Future-агент):
{future_text}
"""

    resp = client.chat.completions.create(
        model=_model_name(),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    full_text = (resp.choices[0].message.content or "").strip()

    # Разделяем основной блок и SWOT
    main_text = full_text
    swot_text = ""
    if "<!--SWOT_START-->" in full_text and "<!--SWOT_END-->" in full_text:
        pre, rest = full_text.split("<!--SWOT_START-->", 1)
        swot_part, _post = rest.split("<!--SWOT_END-->", 1)
        main_text = pre.strip()
        swot_text = swot_part.strip()

    return FinalStrategyResult(
        main_text=main_text,
        swot_text=swot_text,
        raw={
            "rag_summary": rag_summary,
            "web_summary": web_summary,
            "web_bullets": web_bullets,
            "future_text": future_text,
        },
    )
