"""
RAG Retriever: BM25 + семантический поиск (FAISS) + генерация ответа.
На основе рабочего кода пользователя.
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from generator import generate


def tokenize(text: str) -> list[str]:
    """Токенизатор: нижний регистр, убираем всё, кроме букв/цифр/пробелов."""
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9ё\s]", " ", text)
    tokens = text.split()
    return tokens


def _load_documents() -> list[dict]:
    """Загружает документы и проверяет соответствие id позиции."""
    with open(config.DOCUMENTS_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    # Убеждаемся, что id совпадает с позицией
    for i, d in enumerate(docs):
        assert d["id"] == i, f"id {d['id']} != позиция {i}"
    return docs


class Retriever:
    def __init__(self):
        # 1) Документы
        self.docs = _load_documents()
        print(f"Загружено документов (чанков): {len(self.docs)}")

        # 2) BM25
        corpus_tokens = [tokenize(d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(corpus_tokens)
        print("BM25‑корпус построен, документов:", len(corpus_tokens))

        # 3) FAISS-индекс + эмбеддер
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX))
        print("FAISS-индекс загружен, размер:", self.faiss_index.ntotal)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def search_bm25(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Возвращает top_k результатов BM25: (idx, score)."""
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # Индексы документов отсортированы по убыванию score
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def search_vector(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Возвращает top_k результатов векторного поиска: (idx, score)."""
        # Для e5 хорошо использовать префикс "query:"
        q_emb = self.model.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        q_emb = q_emb.reshape(1, -1)
        scores, indices = self.faiss_index.search(q_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def print_results(self, results: list[tuple[int, float]], title: str):
        """Выводит результаты поиска в консоль."""
        print(f"\n=== {title} ===")
        for rank, (idx, score) in enumerate(results, start=1):
            doc = self.docs[idx]
            text_preview = doc["text"][:300].replace("\n", " ")
            meta = {
                "source": doc.get("source"),
                "file": doc.get("file"),
                "date": doc.get("date", ""),
                "page": doc.get("page", None),
                "chunk_id": doc.get("chunk_id", None),
            }
            print(f"\n#{rank}  [id={idx}]  score={score:.4f}")
            print(f"Источник: {meta['source']}, Файл: {meta['file']}, Дата: {meta['date']}")
            if meta["page"] is not None:
                print(f"Страница: {meta['page']}")
            if meta["chunk_id"] is not None:
                print(f"Chunk ID: {meta['chunk_id']}")
            print("Текст:", text_preview, "...")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "both",
        generate_answer: bool = True,
    ) -> dict:
        """
        Поиск: BM25 + семантический поиск → генерация ответа.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов для каждого метода
            mode: 'bm25', 'vector' или 'both'
            generate_answer: Если True, генерирует финальный ответ на основе найденных результатов
        
        Returns:
            dict с ключами:
                - 'bm25_results': список результатов BM25 (если mode включает 'bm25')
                - 'vector_results': список результатов векторного поиска (если mode включает 'vector')
                - 'combined_docs': объединенный список уникальных документов
                - 'answer': сгенерированный ответ (если generate_answer=True)
        """
        bm25_results = []
        vector_results = []
        combined_docs = []
        answer = None

        # BM25 поиск
        if mode in ("bm25", "both"):
            bm25_results = self.search_bm25(query, top_k=top_k)
            self.print_results(bm25_results, "BM25")

        # Векторный поиск
        if mode in ("vector", "both"):
            vector_results = self.search_vector(query, top_k=top_k)
            self.print_results(vector_results, "Векторный поиск (FAISS)")

        # Объединяем результаты (убираем дубликаты, сохраняем порядок)
        seen_ids = set()
        combined_docs_list = []

        # Сначала добавляем BM25 результаты
        for idx, score in bm25_results:
            if idx not in seen_ids:
                seen_ids.add(idx)
                doc = self.docs[idx].copy()
                doc["_score"] = score
                doc["_source"] = "bm25"
                combined_docs_list.append(doc)

        # Затем добавляем векторные результаты
        for idx, score in vector_results:
            if idx not in seen_ids:
                seen_ids.add(idx)
                doc = self.docs[idx].copy()
                doc["_score"] = score
                doc["_source"] = "semantic"
                combined_docs_list.append(doc)
            else:
                # Если документ уже есть, обновляем источник
                for doc in combined_docs_list:
                    if doc["id"] == idx:
                        doc["_source"] = "both"
                        break

        combined_docs = combined_docs_list

        # Генерация ответа на основе найденных результатов
        if generate_answer and combined_docs:
            print("\n=== Генерация ответа ===")
            print(f"Используется {len(combined_docs)} найденных документов...")
            try:
                answer = generate(query, combined_docs, stream=False)
                print("\n--- Ответ ---")
                print(answer)
            except Exception as e:
                print(f"Ошибка при генерации ответа: {e}")
                answer = None

        return {
            "bm25_results": bm25_results,
            "vector_results": vector_results,
            "combined_docs": combined_docs,
            "answer": answer,
        }

    def search(
        self,
        query: str,
        top_k: int = config.RERANK_TOP_K,
        primary_query: str | None = None,
    ) -> list[dict]:
        """
        Выполняет поиск и возвращает список документов (без генерации ответа).
        Топ-2 источника сохраняются в self.last_top_sources для последующего вывода.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов для каждого метода поиска
            primary_query: Исходный запрос (используется как query, если передан)
        
        Returns:
            Список документов, отсортированных по релевантности
        """
        # Используем primary_query, если передан, иначе query
        search_query = primary_query.strip() if primary_query and primary_query.strip() else query.strip()
        
        # Выполняем поиск без генерации ответа
        result = self.retrieve(
            query=search_query,
            top_k=top_k,
            mode="both",
            generate_answer=False,
        )
        
        # Сохраняем топ-3 источника из BM25 и топ-2 из семантического поиска
        top_sources = []
        
        # Топ-3 из BM25
        for idx, score in result["bm25_results"][:3]:
            doc = self.docs[idx]
            source_info = {
                "file": doc.get("file", ""),
                "source": doc.get("source", ""),
                "date": doc.get("date", ""),
                "score": score,
            }
            top_sources.append(source_info)
        
        # Топ-2 из семантического поиска
        for idx, score in result["vector_results"][:2]:
            doc = self.docs[idx]
            source_info = {
                "file": doc.get("file", ""),
                "source": doc.get("source", ""),
                "date": doc.get("date", ""),
                "score": score,
            }
            top_sources.append(source_info)
        
        # Сохраняем в атрибут объекта для доступа из Streamlit
        self.last_top_sources = top_sources
        
        # Возвращаем только объединенные документы
        return result["combined_docs"]
    
    def get_top_sources(self) -> list:
        """
        Возвращает топ-5 источников из последнего поиска (3 из BM25, 2 из семантического).
        
        Returns:
            Список источников (до 5 элементов)
        """
        return getattr(self, "last_top_sources", [])


# Синглтон для Streamlit
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# Пример использования (для тестирования)
if __name__ == "__main__":
    retriever = Retriever()
    query = "сотрудничество со Сбером"
    result = retriever.retrieve(query, top_k=5, mode="both", generate_answer=True)