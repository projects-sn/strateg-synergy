"""
Websearch-агент: поиск информации о поведении вузов в России и СНГ.
Аналитика аналогичных ситуаций у других университетов в области.
"""
import os
import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from openai import OpenAI

import config

# =========================
# Конфиг (модель выбираем в runtime, а не при импорте)
# =========================
def _model_name() -> str:
    """Websearch всегда работает через OpenRouter."""
    return config.OPENROUTER_MODEL

# Используем OpenRouter через OpenAI SDK
def _client() -> OpenAI:
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY не задан для Websearch.")
    return OpenAI(
        base_url=config.OPENROUTER_BASE,
        api_key=openrouter_key,
    )

# =========================
# Память диалогов (in-memory)
# =========================
class SessionStore:
    """Простое хранение истории сообщений по session_id."""
    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self._store:
            self._store[session_id] = []
        return self._store[session_id]
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": role, "content": content})

SESSION_STORE = SessionStore()

# =========================
# 1) Перефраз под аналоги/кейсы в РФ
# =========================
REPHRASE_SYSTEM = """Ты помощник-ресерчер. Перефразируй пользовательский запрос в поисковый запрос, 
сфокусированный на поведении ВУЗОВ (частных и государственных) в России и СНГ, ИСКЛЮЧАЯ корпорацию Синергия. 
Запрос должен искать конкретные проекты, партнёрства, инициативы других вузов. 
ВАЖНО: Если в запросе упоминается Синергия, замени её на "другие вузы" или "вузы России и СНГ".
Пример: запрос 'сотрудничество Синергии со Сбером' должен стать 
'сотрудничество Сбера с государственными и частными вузами России и стран СНГ. Рассмотри все варианты и приведи конкретные примеры.'
Пример: запрос 'что нам поделать с оборонкой' должен стать 
'Какие у современных частных и государственных вузов в России и СНГ есть проекты с оборонной промышленностью и министерством обороны. Рассмотри все варианты и приведи конкретные примеры.' 
Верни только одну строку, 10–25 слов, без комментариев."""

def rephrase_query(user_query: str, session_id: str) -> str:
    """Перефразирует запрос для поиска информации о поведении вузов."""
    client = _client()
    history = SESSION_STORE.get_history(session_id)
    
    messages = [
        {"role": "system", "content": REPHRASE_SYSTEM}
    ]
    
    # Добавляем историю (последние 4 сообщения для контекста)
    for msg in history[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_query})
    
    try:
        resp = client.chat.completions.create(
            model=_model_name(),
            messages=messages,
            temperature=0.2,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        
        # Сохраняем в историю
        SESSION_STORE.add_message(session_id, "user", user_query)
        SESSION_STORE.add_message(session_id, "assistant", rewritten)
        
        return rewritten if rewritten else user_query
    except Exception as e:
        print(f"Ошибка при перефразе: {e}")
        return user_query

# =========================
# 2) Вызов web_search через OpenAI API
# =========================
WEB_SEARCH_SYSTEM = """Ты исследователь. Используй инструмент web_search, чтобы найти свежие данные о поведении ВЫСШИХ УЧЕБНЫХ ЗАВЕДЕНИЙ 
(частных и государственных) в России и СНГ. 
ВАЖНО: Исключи из поиска любую информацию о корпорации "Синергия". Фокусируйся ТОЛЬКО на других вузах.

Цель: показать конкретные проекты, партнёрства, инициативы других вузов с указанием дат. 
Для каждого факта указывай дату события/публикации, если она доступна. 
Можешь использовать немного эмодзи для улучшения читаемости (🔍 для поиска, 📅 для дат, ✅ для успешных проектов), но не злоупотребляй - используй их в меру (2-3 эмодзи на ответ). 
Формат ответа — JSON со следующими полями:
{
  "rewritten": "<строка перефраза>",
  "summary": "<4-6 предложений краткого обзора с упоминанием дат>",
  "bullets": ["краткий факт 1 с датой", "краткий факт 2 с датой", "..."],
  "sources": [{"title": "<заголовок>", "url": "<ссылка>", "date": "<дата публикации если есть>"}]
}
Источники должны соответствовать найденным страницам. Не выдумывай ссылки. 
Всегда указывай даты событий/публикаций, если они найдены."""

def _call_web_search(orig_query: str, rewritten: str) -> Dict[str, Any]:
    """
    Выполняет web search и возвращает структурированный JSON.
    
    Returns:
      {
        "rewritten": str,
        "bullets": [str, ...],
        "summary": str,
        "sources": [{"title": str, "url": str, "date": str}]
      }
    """
    client = _client()
    
    user_prompt = f"""Исходный запрос: {orig_query}
Перефраз для поиска: {rewritten}

Найди информацию в интернете о поведении вузов (частных и государственных) в России и СНГ по данному вопросу. 
Верни JSON с полями rewritten, summary, bullets, sources."""

    try:
        # Пытаемся использовать Responses API с web_search (если поддерживается)
        try:
            resp = client.responses.create(
                model=_model_name(),
                tools=[{"type": "web_search"}],
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": WEB_SEARCH_SYSTEM}]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": user_prompt}
                    ]},
                ],
            )
            
            # Обрабатываем ответ от Responses API
            final_text = ""
            citations: List[Dict[str, str]] = []
            for item in resp.output or []:
                if hasattr(item, 'type') and item.type == "message":
                    for c in (getattr(item, 'content', []) or []):
                        if hasattr(c, 'type') and getattr(c, 'type', '') == "output_text":
                            final_text += getattr(c, 'text', '') or ""
                            for ann in (getattr(c, 'annotations', []) or []):
                                if hasattr(ann, 'type') and getattr(ann, 'type', '') == "url_citation":
                                    citations.append({
                                        "title": getattr(ann, 'title', ''),
                                        "url": getattr(ann, 'url', ''),
                                        "date": getattr(ann, 'date', '') if hasattr(ann, 'date') else ""
                                    })
        except (AttributeError, TypeError):
            # Если Responses API не поддерживается, используем стандартный chat completions
            completion = client.chat.completions.create(
                model=_model_name(),
                messages=[
                    {"role": "system", "content": WEB_SEARCH_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            final_text = completion.choices[0].message.content or ""
            citations = []
        
        # Пытаемся распарсить JSON
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(final_text)
        except json.JSONDecodeError:
            # Если вернуло не-JSON, упакуем как summary с источниками из аннотаций
            parsed = {
                "rewritten": rewritten,
                "summary": final_text.strip(),
                "bullets": [],
                "sources": citations
            }
        
        # Если внутри JSON нет sources — подставим из аннотаций
        if isinstance(parsed, dict) and not parsed.get("sources") and citations:
            parsed["sources"] = citations
        
        # Гарантируем минимальные поля
        parsed.setdefault("rewritten", rewritten)
        parsed.setdefault("summary", "")
        parsed.setdefault("bullets", [])
        parsed.setdefault("sources", [])
        
        return parsed
        
    except Exception as e:
        print(f"Ошибка при web search: {e}")
        # Возвращаем минимальный ответ при ошибке
        return {
            "rewritten": rewritten,
            "summary": f"Не удалось выполнить поиск: {str(e)}",
            "bullets": [],
            "sources": []
        }

# =========================
# 3) Форматирование ответа
# =========================
def _format_answer(data: Dict[str, Any]) -> str:
    """Формирует читабельный ответ для клиента + список источников с датами."""
    parts = []
    if data.get("summary"):
        parts.append(data["summary"])
    bullets = data.get("bullets") or []
    if bullets:
        parts.append("\n— " + "\n— ".join(bullets))
    sources = data.get("sources") or []
    if sources:
        src_lines = []
        for i, s in enumerate(sources, 1):
            t = s.get("title") or "Источник"
            u = s.get("url") or ""
            d = s.get("date") or ""
            if d:
                src_lines.append(f"[{i}] {t} ({d}) — {u}")
            else:
                src_lines.append(f"[{i}] {t} — {u}")
        parts.append("\nИсточники:\n" + "\n".join(src_lines))
    return "\n".join([p for p in parts if p]).strip()

# =========================
# 4) Внешний интерфейс
# =========================
@dataclass
class WebSearchResult:
    session_id: str
    rewritten: str
    answer_text: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

def web_search(session_id: str, user_query: str) -> WebSearchResult:
    """
    Основной вход: на входе session_id и текст пользователя.
    Возвращает перефраз, ответ и источники.
    """
    # 1. Перефразируем запрос
    rewritten = rephrase_query(user_query, session_id)
    
    # 2. Выполняем web search
    result = _call_web_search(user_query, rewritten)
    
    # 3. Форматируем ответ
    answer_text = _format_answer(result)
    
    return WebSearchResult(
        session_id=session_id,
        rewritten=rewritten,
        answer_text=answer_text,
        sources=result.get("sources", []),
        raw=result
    )
