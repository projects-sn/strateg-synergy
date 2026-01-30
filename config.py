"""Конфигурация путей и моделей RAG."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAG_INDEX = BASE / "rag_index"
DOCUMENTS_JSON = RAG_INDEX / "documents.json"
FAISS_INDEX = RAG_INDEX / "index.faiss"
CONFIG_JSON = RAG_INDEX / "config.json"

# Embedding model (должен совпадать с тем, на чём собран index.faiss)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = False  # Временно отключен для быстрого теста (2.27GB скачивается медленно)

# OpenRouter
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o"

# OpenAI-compatible (Artemox)
ARTEMOX_BASE = "https://api.artemox.com/v1"
ARTEMOX_MODEL = "gpt-4o"

# Retrieval
BM25_TOP_K = 10
SEMANTIC_TOP_K = 10
RERANK_TOP_K = 6
COMBINED_CANDIDATES = 20  # сколько кандидатов отдаём в реранкер

# Lightweight scoring / filtering
MIN_QUERY_TOKEN_OVERLAP = 1  # 0 = без фильтра; 1 = хотя бы 1 общий токен
WEIGHT_BM25 = 0.6
WEIGHT_SEMANTIC = 0.3
WEIGHT_OVERLAP = 0.1
PHRASE_BONUS = 0.1  # бонус, если фраза запроса встречается в тексте
PRIMARY_HIT_BONUS = 0.2  # бонус, если документ найден по исходному запросу

# Hard filters for precision
REQUIRE_QUERY_KEYWORD = True  # требовать ключевое слово из запроса (например, "сбер")
REQUIRE_YEAR_MATCH = True  # если в запросе есть год, требовать его в документе
MIN_KEYWORD_LEN = 4
