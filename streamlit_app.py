"""
Streamlit-интерфейс для аналитического сервиса Синергия.
Два агента: RAG-агент (внутренние документы) и Websearch-агент (аналитика вузов).

Оптимизация «Начать поиск» (по логам):
- Раньше: RAG (search ~29 s + generate ~23 s) + Websearch + Future — всё подряд.
  Websearch при 500/524 от Artemox ждал до 72–645 s, из-за этого долгое ожидание.
- Сделано: после «Начать поиск» выполняется только RAG; ответ показывается сразу.
  Websearch и Future запускаются при открытии соответствующих вкладок (лениво).
- Retriever (BM25 + FAISS + SentenceTransformer) подгружается в фоне при старте приложения,
  чтобы первый поиск не тратил ~20 s на холодную загрузку.
"""
import concurrent.futures
import logging
import os
import threading
import time
import uuid

import streamlit as st

# Таймаут для Websearch-агента (сек); при превышении показываем «Агент пока недоступен»
WEBSEARCH_TIMEOUT = 60
FUTURE_AGENT_TIMEOUT = 90
POLL_INTERVAL = 2  # интервал опроса фоновых агентов (сек)


def _run_rag_task(search_query: str, primary_query: str, original_query: str):
    """Выполняет RAG (поиск + генерация) в потоке. Возвращает (answer, docs, top_sources, error)."""
    try:
        ret = get_retriever()
        docs = ret.search(search_query, primary_query=primary_query)
        if not docs:
            return (None, [], [], None)
        answer = generate(original_query, docs)
        return (answer, docs, ret.get_top_sources(), None)
    except Exception as e:
        log.warning("RAG task failed: %s", e)
        return (None, [], [], str(e))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Предзагрузка retriever в фоне, чтобы первый «Начать поиск» не ждал холодный старт (~20 s)
_preload_started = False
def _preload_retriever():
    global _preload_started
    if _preload_started:
        return
    _preload_started = True
    def _run():
        try:
            from retriever import get_retriever
            get_retriever()
            log.info("Retriever preloaded (BM25 + FAISS + embedding model)")
        except Exception as e:
            log.warning("Retriever preload failed: %s", e)
    threading.Thread(target=_run, daemon=True).start()
_preload_retriever()

from classifier import FIELDS, FIELDS_RU, classify, params_to_keywords
from generator import generate
from query_enricher import enrich_query
from retriever import get_retriever
from websearch_agent import web_search
from future_agent import future_chat
from final_strategy_agent import build_final_strategy

# Подставить ключ из st.secrets, если нет в env
if "OPENROUTER_API_KEY" not in os.environ:
    try:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
if "ARTEMOX_API_KEY" not in os.environ:
    try:
        os.environ["ARTEMOX_API_KEY"] = st.secrets["ARTEMOX_API_KEY"]
    except Exception:
        pass

st.set_page_config(page_title="Цифровой ассистент Синергии", layout="centered")

# --- Styling: Synergy palette (red/white/black) ---
st.markdown(
    """
    <style>
    :root {
        --synergy-red: #d71920;
        --synergy-black: #111111;
        --synergy-gray: #f4f4f4;
    }
    .stApp {
        background-color: #ffffff;
        color: var(--synergy-black);
    }
    .main h1, .main h2, .main h3 {
        color: var(--synergy-black);
    }
    section[data-testid="stSidebar"] {
        width: 360px !important;
        min-width: 360px !important;
    }
    .synergy-title {
        background: var(--synergy-red);
        color: #ffffff;
        padding: 26px 30px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 38px;
        letter-spacing: 0.4px;
        margin-bottom: 14px;
        margin-top: 0;
    }
    .synergy-caption {
        margin-top: 6px;
        margin-bottom: 22px;
        color: #2b2b2b;
        font-size: 17px;
        line-height: 1.45;
    }
    .main .block-container {
        padding-top: 18px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 18px;
    }
    .stButton > button {
        background-color: var(--synergy-red) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        white-space: nowrap !important;
        width: auto !important;
        min-width: 6rem !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #eeeeee !important;
        color: var(--synergy-black) !important;
        border: 1px solid #d7d7d7 !important;
        width: auto !important;
        min-width: 6rem !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #e3e3e3 !important;
        color: var(--synergy-black) !important;
    }
    .stButton > button:hover {
        background-color: #b9151a !important;
        color: #ffffff !important;
    }
    .stTextInput > div > div > input,
    .stTextArea textarea {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 18px;
        letter-spacing: 0.2px;
        padding: 10px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--synergy-red) !important;
        border-bottom: 3px solid var(--synergy-red) !important;
    }
    .stAlert {
        border-left: 4px solid var(--synergy-red);
    }
    .synergy-note {
        background: var(--synergy-gray);
        border: 1px solid #e0e0e0;
        border-left: 4px solid var(--synergy-black);
        padding: 10px 12px;
        border-radius: 8px;
        color: var(--synergy-black);
    }
    .synergy-separator {
        height: 1px;
        background: #e6e6e6;
        margin: 10px 0 18px 0;
        border: 0;
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 10px 0;
    }
    .metric-pill {
        border: 1px solid var(--synergy-red);
        color: var(--synergy-red);
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        background: #fff5f5;
        white-space: nowrap;
    }
    .swot-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        background: #ffffff;
    }
    .swot-table th, .swot-table td {
        padding: 10px 12px;
        vertical-align: top;
        border-bottom: 1px solid #f0f0f0;
    }
    .swot-table td {
        white-space: pre-line;
    }
    .swot-table tr:last-child th, .swot-table tr:last-child td {
        border-bottom: 0;
    }
    .swot-tag {
        font-weight: 800;
        width: 68px;
        white-space: nowrap;
    }
    .swot-s { color: #1a7f37; background: #eef9f1; }
    .swot-w { color: #b54708; background: #fff4e5; }
    .swot-o { color: #0b4aa2; background: #eaf2ff; }
    .swot-t { color: #b42318; background: #ffeceb; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="synergy-title">Цифровой ассистент руководства корпорации Синергия</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="synergy-caption">Единая аналитическая среда, объединяющая внутренние данные, '
    'внешние кейсы вузов и прогнозы для поддержки управленческих решений.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="synergy-separator"></div>', unsafe_allow_html=True)

# --- Sidebar: описание системы и агентов ---
with st.sidebar:
    st.markdown("### О системе")
    st.write(
        "Сервис помогает принимать решения на основе "
        "внутренних документов, анализа рынка вузов и прогнозных сценариев."
    )
    st.markdown("### Агенты")
    st.markdown("**❌ RAG‑агент**")
    st.write("Внутренняя аналитика: документы, стенограммы, брифинги.")
    st.markdown("**❌ Websearch‑агент**")
    st.write("Внешние кейсы других вузов РФ и СНГ.")
    st.markdown("**❌ Future-agent**")
    st.write("Анализ будущих перспектив и трендов.")
    st.markdown("**❌ Итоговый стратег**")
    st.write("Ранжированные стратегии с SWOT‑анализом.")

# Инициализация session_id для websearch
if "websearch_session_id" not in st.session_state:
    st.session_state["websearch_session_id"] = str(uuid.uuid4())
if "future_session_id" not in st.session_state:
    st.session_state["future_session_id"] = str(uuid.uuid4())
if "final_session_id" not in st.session_state:
    st.session_state["final_session_id"] = str(uuid.uuid4())

# Вкладки для переключения между агентами
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 RAG-агент",
    "🔍 Websearch-агент",
    "🚀 Future-agent",
    "🏁 Итоговый стратег",
])

# Инициализация состояния
if "show_params" not in st.session_state:
    st.session_state["show_params"] = False
if "params" not in st.session_state:
    st.session_state["params"] = {}
if "enriched_query" not in st.session_state:
    st.session_state["enriched_query"] = None
if "query_approved" not in st.session_state:
    st.session_state["query_approved"] = False
if "websearch_unavailable" not in st.session_state:
    st.session_state["websearch_unavailable"] = False
if "future_unavailable" not in st.session_state:
    st.session_state["future_unavailable"] = False

# Опрос фоновых агентов (Websearch, Future) — результат выводится по готовности
def _poll_pending_agents():
    rerun_needed = False
    for key, result_key, unavailable_key, timeout in [
        ("_pending_websearch_future", "websearch_result", "websearch_unavailable", WEBSEARCH_TIMEOUT),
        ("_pending_future_future", "future_result", "future_unavailable", FUTURE_AGENT_TIMEOUT),
    ]:
        fut = st.session_state.get(key)
        if fut is None:
            continue
        start = st.session_state.get(key + "_start", 0)
        if time.time() - start > timeout + 5:
            st.session_state[result_key] = None
            st.session_state[unavailable_key] = True
            del st.session_state[key]
            if key + "_start" in st.session_state:
                del st.session_state[key + "_start"]
            log.warning("%s: снято по таймауту", key)
            continue
        try:
            res = fut.result(timeout=0)
            st.session_state[result_key] = res
            st.session_state[unavailable_key] = False
            del st.session_state[key]
            if key + "_start" in st.session_state:
                del st.session_state[key + "_start"]
            log.info("%s: готов", key)
        except concurrent.futures.TimeoutError:
            rerun_needed = True
        except Exception as e:
            st.session_state[result_key] = None
            st.session_state[unavailable_key] = True
            del st.session_state[key]
            if key + "_start" in st.session_state:
                del st.session_state[key + "_start"]
            log.warning("%s failed: %s", key, e)
    if not rerun_needed and "_agent_executor" in st.session_state:
        try:
            st.session_state["_agent_executor"].shutdown(wait=False)
        except Exception:
            pass
        del st.session_state["_agent_executor"]
    return rerun_needed

_poll_rerun = _poll_pending_agents()

# =========================
# ВКЛАДКА 1: RAG-АГЕНТ
# =========================
with tab1:
    st.subheader("Аналитика внутренних процессов Синергии")

    # ---- 1. Ввод запроса ----
    query = st.text_input(
        "Введите запрос:",
        placeholder="Например: сотрудничество со Сбером в 2025?",
        key="rag_query",
    )
    col1, col2, _ = st.columns([2.6, 3.5, 3])
    with col1:
        recognize_btn = st.button("Распознать параметры", key="rag_recognize")
    with col2:
        search_direct_btn = st.button("Искать без распознавания", key="rag_direct")

    if recognize_btn and query:
        with st.spinner("Распознаю параметры…"):
            try:
                t0 = time.perf_counter()
                out = classify(query)
                log.info("Classifier (Распознать параметры): %.2f s", time.perf_counter() - t0)
                for f in FIELDS:
                    st.session_state["p_" + f] = out.get(f) or ""
                st.session_state["show_params"] = True
                st.session_state["params"] = out
                st.session_state["original_query"] = query.strip()
                st.session_state["query_approved"] = False
                st.rerun()
            except Exception as e:
                err = str(e)
                if "401" in err or "User not found" in err or "unauthorized" in err.lower():
                    st.error(
                        "**Неверный или недействительный API ключ.** Проверьте ключ в личном кабинете "
                        "(OpenRouter: openrouter.ai/keys или Artemox). Убедитесь, что ключ скопирован целиком "
                        "и указан в `.streamlit/secrets.toml` как `OPENROUTER_API_KEY` или `ARTEMOX_API_KEY`."
                    )
                else:
                    st.error(f"Ошибка классификатора: {e}")

    if search_direct_btn and query:
        st.session_state["original_query"] = query.strip()
        st.session_state["query_approved"] = False
        with st.spinner("Обогащение запроса…"):
            try:
                t0 = time.perf_counter()
                enriched_query = enrich_query(query.strip())
                log.info("Enrich query (Искать без распознавания): %.2f s", time.perf_counter() - t0)
                st.session_state["enriched_query"] = enriched_query
            except Exception as e:
                st.error(f"Ошибка при обогащении запроса: {e}")
                st.session_state["enriched_query"] = query.strip()

    # ---- 2. Уточнение параметров и обогащение ----
    if st.session_state.get("show_params"):
        st.subheader("Уточните параметры (можно редактировать и дополнять)")
        for f in FIELDS:
            st.text_input(FIELDS_RU[f], key="p_" + f)

        enrich_btn = st.button("🔄 Обогатить запрос", key="rag_enrich")
        if enrich_btn and st.session_state.get("original_query"):
            with st.spinner("Обогащение запроса…"):
                try:
                    params = {
                        f: (st.session_state.get("p_" + f) or "").strip() or None
                        for f in FIELDS
                    }
                    kw = params_to_keywords(params)
                    query_for_enrichment = st.session_state["original_query"]
                    if kw:
                        query_for_enrichment = query_for_enrichment + " " + kw

                    t0 = time.perf_counter()
                    st.session_state["enriched_query"] = enrich_query(query_for_enrichment)
                    log.info("Enrich query (Обогатить запрос): %.2f s", time.perf_counter() - t0)
                    st.session_state["query_approved"] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка при обогащении запроса: {e}")
                    st.session_state["enriched_query"] = st.session_state["original_query"]

        if st.session_state.get("enriched_query"):
            st.markdown("---")
            st.markdown("### 📝 Обогащенный запрос")
            st.caption("Вы можете отредактировать запрос перед утверждением:")

            edited_query = st.text_area(
                "Обогащенный запрос",
                value=st.session_state.get("enriched_query", ""),
                key="edited_enriched_query",
                height=100,
                label_visibility="collapsed",
            )
            if edited_query != st.session_state.get("enriched_query"):
                st.session_state["enriched_query"] = edited_query

            approve_btn = st.button("Начать поиск", type="primary", key="rag_approve")
            if approve_btn:
                st.session_state["enriched_query"] = edited_query.strip()
                st.session_state["query_approved"] = True
                st.session_state["websearch_unavailable"] = False
                st.session_state["future_unavailable"] = False
                st.rerun()

        if st.session_state.get("query_approved") and st.session_state.get("enriched_query"):
            try:
                params = {
                    f: (st.session_state.get("p_" + f) or "").strip() or None
                    for f in FIELDS
                }
                kw = params_to_keywords(params)
                search_query = st.session_state["original_query"]
                if kw:
                    search_query = search_query + " " + kw
                primary_query = st.session_state["original_query"]
                original_query = st.session_state["original_query"]
                eq = st.session_state["enriched_query"]
                sid_web = st.session_state["websearch_session_id"]
                sid_fut = st.session_state["future_session_id"]

                with st.spinner("Начало анализа…"):
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                    f_rag = ex.submit(_run_rag_task, search_query, primary_query, original_query)
                    f_web = ex.submit(web_search, session_id=sid_web, user_query=eq)
                    f_fut = ex.submit(future_chat, session_id=sid_fut, user_query=eq)
                    t0 = time.perf_counter()
                    rag_result = f_rag.result(timeout=120)
                    log.info("RAG (params): готов за %.2f s", time.perf_counter() - t0)
                    # Websearch и Future продолжают в фоне; результат подхватится при опросе

                st.session_state["query_approved"] = False
                if rag_result and rag_result[3]:
                    st.error(f"Ошибка RAG: {rag_result[3]}")
                elif rag_result and rag_result[0]:
                    st.session_state["last_answer"] = rag_result[0]
                    st.session_state["last_docs"] = rag_result[1]
                    st.session_state["top_sources"] = rag_result[2]
                else:
                    st.info("По запросу ничего не найдено.")

                st.session_state["_pending_websearch_future"] = f_web
                st.session_state["_pending_websearch_future_start"] = time.time()
                st.session_state["_pending_future_future"] = f_fut
                st.session_state["_pending_future_future_start"] = time.time()
                st.session_state["_agent_executor"] = ex
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

    # Для прямого поиска без распознавания параметров
    if (
        (search_direct_btn or st.session_state.get("enriched_query"))
        and st.session_state.get("original_query") == (query or "").strip()
        and not st.session_state.get("show_params")
        and not st.session_state.get("query_approved")
    ):
        st.markdown("---")
        st.markdown("### 📝 Обогащенный запрос")
        st.caption("Вы можете отредактировать запрос перед утверждением:")

        edited_query_direct = st.text_area(
            "Обогащенный запрос",
            value=st.session_state.get("enriched_query", ""),
            key="edited_enriched_query_direct",
            height=100,
            label_visibility="collapsed",
        )
        if edited_query_direct != st.session_state.get("enriched_query"):
            st.session_state["enriched_query"] = edited_query_direct

        approve_btn = st.button("Начать поиск", type="primary", key="approve_direct")
        if approve_btn:
            st.session_state["enriched_query"] = edited_query_direct.strip()
            st.session_state["query_approved"] = True
            st.session_state["websearch_unavailable"] = False
            st.session_state["future_unavailable"] = False
            try:
                q = query.strip()
                eq = st.session_state["enriched_query"]
                sid_web = st.session_state["websearch_session_id"]
                sid_fut = st.session_state["future_session_id"]

                with st.spinner("Начало анализа…"):
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                    f_rag = ex.submit(_run_rag_task, q, q, q)
                    f_web = ex.submit(web_search, session_id=sid_web, user_query=eq)
                    f_fut = ex.submit(future_chat, session_id=sid_fut, user_query=eq)
                    t0 = time.perf_counter()
                    rag_result = f_rag.result(timeout=120)
                    log.info("RAG (direct): готов за %.2f s", time.perf_counter() - t0)
                    # Websearch и Future продолжают в фоне; результат подхватится при опросе

                st.session_state["query_approved"] = False
                if rag_result and rag_result[3]:
                    st.error(f"Ошибка RAG: {rag_result[3]}")
                elif rag_result and rag_result[0]:
                    st.session_state["last_answer"] = rag_result[0]
                    st.session_state["last_docs"] = rag_result[1]
                    st.session_state["top_sources"] = rag_result[2]
                else:
                    st.info("По запросу ничего не найдено.")

                st.session_state["_pending_websearch_future"] = f_web
                st.session_state["_pending_websearch_future_start"] = time.time()
                st.session_state["_pending_future_future"] = f_fut
                st.session_state["_pending_future_future_start"] = time.time()
                st.session_state["_agent_executor"] = ex
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

    # ---- 3. Ответ RAG ----
    if st.session_state.get("last_answer"):
        st.subheader("Ответ")
        raw = st.session_state["last_answer"]
        import re
        cleaned = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        st.markdown(cleaned)

    # ---- 4. Источники RAG ----
    if st.session_state.get("top_sources"):
        st.subheader("Источники")
        top_sources = st.session_state["top_sources"]
        if top_sources:
            for src in top_sources:
                source_text = src.get("file", "Неизвестный файл")
                if src.get("date"):
                    source_text += f" ({src.get('date')})"
                st.markdown(f"• {source_text}")
        else:
            st.caption("Нет результатов")

# =========================
# ВКЛАДКА 2: WEBSEARCH-АГЕНТ
# =========================
with tab2:
    st.subheader("Аналитика аналогичных ситуаций у других университетов")
    
    # Если есть результат от автоматического запуска или сохраненный результат
    if st.session_state.get("websearch_result"):
        result = st.session_state["websearch_result"]
        
        # Показываем результаты в читаемом виде
        st.markdown("### 📊 Результаты анализа")
        
        # Получаем данные из raw
        raw_data = result.raw
        summary = ""
        bullets = []
        parsed_payload = None

        # Если raw_data - строка, пытаемся распарсить как JSON
        if isinstance(raw_data, str):
            import json
            try:
                parsed_payload = json.loads(raw_data)
            except Exception:
                parsed_payload = None
        elif isinstance(raw_data, dict):
            parsed_payload = raw_data

        if isinstance(parsed_payload, dict):
            summary = parsed_payload.get("summary", "") or ""
            bullets = parsed_payload.get("bullets", []) or []

        # Если summary выглядит как JSON, пробуем распарсить ещё раз
        if isinstance(summary, str):
            summary_candidate = summary.strip()
            if "```" in summary_candidate:
                summary_candidate = summary_candidate.replace("```json", "").replace("```", "").strip()
            if summary_candidate.startswith("{"):
                import json
                try:
                    nested = json.loads(summary_candidate)
                    summary = nested.get("summary", "") or ""
                    bullets = nested.get("bullets", []) or bullets
                except Exception:
                    pass

        # Если summary и bullets пустые, пробуем использовать answer_text как JSON
        if (not summary and not bullets) and isinstance(result.answer_text, str):
            import json
            try:
                nested = json.loads(result.answer_text)
                summary = nested.get("summary", "") or summary
                bullets = nested.get("bullets", []) or bullets
            except Exception:
                pass
        
        # Показываем summary
        if summary:
            summary_clean = summary.strip()
            if "```" in summary_clean:
                summary_clean = summary_clean.replace("```json", "").replace("```", "").strip()
            if summary_clean.startswith('"') and summary_clean.endswith('"'):
                summary_clean = summary_clean[1:-1]
            st.markdown(summary_clean)
        
        # Показываем bullets
        if bullets:
            if summary:
                st.markdown("")  # Отступ после summary
            st.markdown("**Ключевые факты:**")
            for bullet in bullets:
                bullet_text = str(bullet).strip()
                if bullet_text.startswith('"') and bullet_text.endswith('"'):
                    bullet_text = bullet_text[1:-1]
                st.markdown(f"• {bullet_text}")

        if not summary and not bullets:
            st.info("Не удалось извлечь текстовый ответ. Попробуйте повторить поиск.")
        
        # Источники
        if result.sources:
            st.markdown("---")
            st.markdown("### 📚 Источники")
            for i, src in enumerate(result.sources, 1):
                title = src.get("title", "Источник")
                url = src.get("url", "")
                date = src.get("date", "")
                
                if date:
                    st.markdown(f"**{i}.** {title} *(опубликовано: {date})*")
                else:
                    st.markdown(f"**{i}.** {title}")
                
                if url:
                    st.markdown(f"🔗 [{url}]({url})")
                st.markdown("")
    
    elif st.session_state.get("_pending_websearch_future"):
        st.markdown(
            '<div class="synergy-note">Websearch‑агент выполняется. Результат появится автоматически по готовности.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("websearch_unavailable"):
        st.markdown(
            '<div class="synergy-note">Агент пока недоступен. Websearch не успел ответить за отведённое время. '
            'Попробуйте позже или нажмите «Начать поиск» в RAG-агенте ещё раз.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("enriched_query"):
        st.markdown(
            '<div class="synergy-note">Нажмите «Начать поиск» в RAG-агенте — после этого здесь появится анализ аналогичных ситуаций у других вузов.</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            '<div class="synergy-note">Сначала обогатите запрос в RAG-агенте и нажмите «Начать поиск» — затем здесь появится анализ аналогичных ситуаций у других вузов.</div>',
            unsafe_allow_html=True,
        )

# =========================
# ВКЛАДКА 3: ПРОГНОЗНЫЙ АГЕНТ
# =========================
with tab3:
    st.subheader("Прогнозные предложения на будущее (1–3 года)")

    if st.session_state.get("future_result"):
        result = st.session_state["future_result"]

        st.markdown("### 💡 Варианты развития")
        import re
        raw = result.answer_text or ""
        cleaned = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        st.markdown(cleaned)

    elif st.session_state.get("_pending_future_future"):
        st.markdown(
            '<div class="synergy-note">Future‑агент выполняется. Результат появится автоматически по готовности.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("future_unavailable"):
        st.markdown(
            '<div class="synergy-note">Агент пока недоступен. Future-agent не успел ответить за отведённое время. '
            'Попробуйте позже или нажмите «Начать поиск» в RAG-агенте ещё раз.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("enriched_query"):
        st.markdown(
            '<div class="synergy-note">Нажмите «Начать поиск» в RAG-агенте — после этого здесь появятся прогнозные предложения.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="synergy-note">Сначала обогатите запрос в RAG-агенте и нажмите «Начать поиск» — затем здесь появятся прогнозы.</div>',
            unsafe_allow_html=True,
        )

# =========================
# ВКЛАДКА 4: ИТОГОВЫЙ СТРАТЕГ
# =========================
with tab4:
    st.subheader("Итоговые отранжированные стратегии")

    if st.session_state.get("final_strategy_result"):
        result = st.session_state["final_strategy_result"]
        import re

        text = result.main_text or ""
        swot_all = result.swot_text or ""

        blocks = re.split(r"\n(?=###\s*Стратегия\s*\d+:)", text)
        header = blocks[0].strip() if blocks else ""
        # Убираем блок «Ранжирование по оптимальности» из заголовка (с конца)
        lines = header.splitlines()
        keep = []
        for line in lines:
            s = line.strip()
            if s.startswith("Ранжирование") or s.startswith("1\ufe0f\u20e3") or s.startswith("2\ufe0f\u20e3") or s.startswith("3\ufe0f\u20e3"):
                break
            keep.append(line)
        header = "\n".join(keep).strip()
        if header:
            st.markdown(header)

        # Парсинг SWOT по стратегиям из swot_text
        swot_by_idx: dict[int, dict[str, list[str]]] = {}
        if swot_all:
            parts = re.split(r"\n(?=###\s*Стратегия\s*\d+:)", swot_all)
            for p in parts:
                m_idx = re.match(r"###\s*Стратегия\s*(\d+):", p.strip())
                if not m_idx:
                    continue
                idx = int(m_idx.group(1))
                swot_by_idx[idx] = {"S": [], "W": [], "O": [], "T": []}
                # извлекаем блоки S/W/O/T
                for key in ["S", "W", "O", "T"]:
                    m = re.search(rf"{key}\s*:\s*(.*?)(?=\n[A-Z]\s*:|\Z)", p, flags=re.DOTALL)
                    if m:
                        lines = []
                        for line in m.group(1).splitlines():
                            line = line.strip()
                            if line.startswith("-"):
                                lines.append(line.lstrip("-").strip())
                        swot_by_idx[idx][key] = lines[:5]

        def _extract_scores(block: str) -> dict[str, str]:
            scores = {}
            for label in ["Затратность", "Рисковость", "Время", "Эффект", "Оптимальность"]:
                m = re.search(rf"{label}\s*=\s*(\d+)", block)
                if not m:
                    m = re.search(rf"{label}\s*:\s*(\d+)", block)
                if m:
                    scores[label] = m.group(1)
            return scores

        def _render_pills(scores: dict):
            if not scores:
                return
            pill_html = '<div class="metric-row">'
            for label in ["Затратность", "Рисковость", "Время", "Эффект", "Оптимальность"]:
                if label not in scores:
                    continue
                val = scores[label]
                if label == "Оптимальность":
                    try:
                        v = int(val)
                        pill_html += f'<span class="metric-pill">{label}: {val}/10</span>' if v < 10 else f'<span class="metric-pill">{label}: {val}</span>'
                    except ValueError:
                        pill_html += f'<span class="metric-pill">{label}: {val}/10</span>'
                else:
                    pill_html += f'<span class="metric-pill">{label}: {val}/10</span>'
            pill_html += "</div>"
            st.markdown(pill_html, unsafe_allow_html=True)

        def _render_swot_table(swot: dict[str, list[str]]):
            def _clean(s: str) -> str:
                s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)
                s = re.sub(r"<[^>]+>", "", s)
                s = s.replace("•", "").strip()
                return s.strip() or "—"

            def _escape(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            def _li(items: list[str]) -> str:
                if not items:
                    return "—"
                cleaned = [_escape(_clean(i)) for i in items]
                return "\n".join(cleaned)

            html = f"""
            <table class="swot-table">
              <tr>
                <th class="swot-tag swot-s">🟢 S</th>
                <td>{_li(swot.get("S", []))}</td>
              </tr>
              <tr>
                <th class="swot-tag swot-w">🟠 W</th>
                <td>{_li(swot.get("W", []))}</td>
              </tr>
              <tr>
                <th class="swot-tag swot-o">🔵 O</th>
                <td>{_li(swot.get("O", []))}</td>
              </tr>
              <tr>
                <th class="swot-tag swot-t">🔴 T</th>
                <td>{_li(swot.get("T", []))}</td>
              </tr>
            </table>
            """
            st.markdown(html, unsafe_allow_html=True)

        # Только блоки стратегий (### Стратегия N:), сортируем по оптимальности (выше — выше)
        strategy_blocks = []
        for i, b in enumerate(blocks[1:], 1):
            b = b.strip()
            if not b or not re.match(r"^###\s*Стратегия\s*\d+:", b):
                continue
            opt = _extract_scores(b).get("Оптимальность", "0")
            try:
                opt_int = int(opt)
            except ValueError:
                opt_int = 0
            strategy_blocks.append((opt_int, i, b))

        strategy_blocks.sort(key=lambda x: (-x[0], x[1]))
        cup_chars = ("\U0001f947", "\U0001f948", "\U0001f949")

        def _drop_ranking_block(text: str) -> str:
            lines = text.splitlines()
            keep = []
            for line in lines:
                s = line.strip()
                if s.startswith("Ранжирование") or s.startswith("1\ufe0f\u20e3") or s.startswith("2\ufe0f\u20e3") or s.startswith("3\ufe0f\u20e3"):
                    break
                keep.append(line)
            return "\n".join(keep).strip()

        def _drop_scores_and_rules(text: str) -> str:
            """Убирает строку с оценками (Оценки 0-10: ...) и горизонтальные разделители (---)."""
            lines = text.splitlines()
            keep = []
            for line in lines:
                s = line.strip()
                if "Оценки" in s and ("Затратность" in s or "Оптимальность" in s or re.search(r"\d+\s*;\s*\d+", s)):
                    continue
                if re.match(r"^[-*_]{2,}\s*$", s):
                    continue
                keep.append(line)
            return "\n".join(keep).strip()

        for rank, (opt_int, i, b) in enumerate(strategy_blocks, 1):
            title_line = b.splitlines()[0].strip()
            title_rest = re.sub(r"^#+\s*", "", title_line).strip()
            if rank <= 3:
                title_rest = f"{cup_chars[rank - 1]} {title_rest}"
            st.markdown("### " + title_rest)

            scores = _extract_scores(b)
            _render_pills(scores)

            b_no_scores = re.sub(r"^Оценки.*?$", "", b, flags=re.MULTILINE).strip()
            desc_raw = "\n".join(b_no_scores.splitlines()[1:]).strip()
            desc = _drop_ranking_block(desc_raw)
            desc = _drop_scores_and_rules(desc)
            if desc:
                st.markdown(desc)

            if "show_swot_map" not in st.session_state:
                st.session_state["show_swot_map"] = {}
            shown = bool(st.session_state["show_swot_map"].get(i, False))
            btn = "Показать SWOT" if not shown else "Скрыть SWOT"
            if st.button(btn, type="primary" if not shown else "secondary", key=f"swot_btn_{i}"):
                st.session_state["show_swot_map"][i] = not shown
                st.rerun()

            if st.session_state["show_swot_map"].get(i, False):
                sw = swot_by_idx.get(i, {"S": [], "W": [], "O": [], "T": []})
                _render_swot_table(sw)

            st.markdown("<br>", unsafe_allow_html=True)
    else:
        # Автозапуск при наличии всех данных
        rag_summary = st.session_state.get("last_answer", "")
        web_result = st.session_state.get("websearch_result")
        future_result = st.session_state.get("future_result")

        if rag_summary and web_result and future_result:
            with st.spinner("Формируем итоговые стратегии…"):
                try:
                    raw_web = web_result.raw or {}
                    web_summary = raw_web.get("summary", "") if isinstance(raw_web, dict) else ""
                    web_bullets = raw_web.get("bullets", []) if isinstance(raw_web, dict) else []

                    t0 = time.perf_counter()
                    final_result = build_final_strategy(
                        rag_summary=rag_summary,
                        web_summary=web_summary,
                        web_bullets=web_bullets if isinstance(web_bullets, list) else [],
                        future_text=future_result.answer_text if future_result else "",
                    )
                    log.info("Final-strategy agent (Итоговый стратег): %.2f s", time.perf_counter() - t0)
                    st.session_state["final_strategy_result"] = final_result
                    st.session_state["show_swot_map"] = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка при формировании стратегий: {e}")
        else:
            st.markdown(
                '<div class="synergy-note">Сначала завершите RAG, Websearch и Future агенты. '
                'После этого здесь автоматически появятся итоговые стратегии.</div>',
                unsafe_allow_html=True,
            )

# Подсказка по ключу
if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ARTEMOX_API_KEY")):
    st.sidebar.warning(
        "API ключ не задан. Укажите OPENROUTER_API_KEY или ARTEMOX_API_KEY в окружении "
        "или в `.streamlit/secrets.toml`."
    )

# Опрос фоновых агентов: если Websearch или Future ещё в работе — обновить страницу через POLL_INTERVAL
if _poll_rerun:
    time.sleep(POLL_INTERVAL)
    st.rerun()
