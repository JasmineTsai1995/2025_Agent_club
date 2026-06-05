"""
Arks Retriever — MLflow PythonModel (warm serving endpoint).

Deployed as the 'arks-retriever' Model Serving endpoint.
MongoDB clients and retrievers are initialised ONCE in load_context() so every
subsequent call is fast — no cold-start per-request overhead.

Input (via ai_query in UC SQL functions):
    named_struct('query', <query_string>, 'type', <'vector'|'hybrid'>)
    → DataFrame rows with columns: query STRING, type STRING

Output: STRING — JSON payload containing retrieved document contents and metadata

Environment variables set on the serving endpoint:
    MONGODB_URI         — MongoDB Atlas connection string
    DATABRICKS_HOST     — Databricks workspace URL
    DATABRICKS_TOKEN    — Databricks access token
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — keep in sync with register_uc_tools.py
# ---------------------------------------------------------------------------
_EMBED_ENDPOINT = "cxi-text-embedding-3-large"
_DB           = "dbarks"
_CHILD_TABLE  = "km_document_child"
_PARENT_TABLE = "km_document_parent"
_TOP_K        = 10
_FTS_INDEX    = "search_index"
_PARENT_ID    = "doc_id"


def _format_docs(docs: list) -> str:
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))


class RetrieverModel(mlflow.pyfunc.PythonModel):
    """Warm MongoDB retriever — initialised once, serves many requests."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Called once when the serving container starts."""
        import certifi
        from langchain_classic.retrievers import ParentDocumentRetriever
        from langchain_core.documents import Document
        from langchain_core.stores import BaseStore
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from langchain_mongodb.retrievers.hybrid_search import (
            MongoDBAtlasHybridSearchRetriever,
        )
        from databricks_langchain import DatabricksEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pymongo import MongoClient

        # Inline parent-store adapter — avoids importing app-level modules
        class _MongoParentStore(BaseStore):
            def __init__(self, col):
                self._col = col

            def mget(self, keys):
                rows = {
                    r["key"]: r
                    for r in self._col.find({"key": {"$in": list(keys)}})
                }
                return [
                    Document(
                        page_content=rows[k]["value"],
                        metadata=rows[k].get("cmetadata", {}),
                    )
                    if k in rows
                    else None
                    for k in keys
                ]

            def mset(self, kvs):
                pass

            def mdelete(self, keys):
                pass

            def yield_keys(self, *, prefix=None):
                return iter([])

        mongodb_uri = os.environ["MONGODB_URI"]

        embeddings = DatabricksEmbeddings(endpoint=_EMBED_ENDPOINT)

        mongo_client = MongoClient(mongodb_uri, tlsCAFile=certifi.where())
        db = mongo_client[_DB]

        child_col = db[_CHILD_TABLE]
        parent_col = db[_PARENT_TABLE]

        child_store = MongoDBAtlasVectorSearch(
            collection=child_col,
            embedding=embeddings,
            text_key="content",
            metadata_key="metadata",
        )

        # Vector retriever (ParentDocumentRetriever)
        self._vector_retriever = ParentDocumentRetriever(
            vectorstore=child_store,
            docstore=_MongoParentStore(parent_col),
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=128, chunk_overlap=10
            ),
            id_key=_PARENT_ID,
            search_kwargs={"k": _TOP_K},
        )

        # Hybrid retriever
        self._hybrid_retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=child_store,
            search_index_name=_FTS_INDEX,
            k=_TOP_K,
        )
        self._hybrid_parent_col = parent_col

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _run_hybrid(self, query: str) -> str:
        from langchain_core.documents import Document

        docs = self._hybrid_retriever.invoke(query)
        parent_ids = [doc.metadata.get(_PARENT_ID) for doc in docs]

        if any(pid is not None for pid in parent_ids):
            unique_ids = list(
                dict.fromkeys(pid for pid in parent_ids if pid is not None)
            )
            rows = {
                r["key"]: r
                for r in self._hybrid_parent_col.find({"key": {"$in": unique_ids}})
            }
            result: list = []
            seen: set = set()
            for doc, pid in zip(docs, parent_ids):
                if pid and pid in rows and pid not in seen:
                    result.append(
                        Document(
                            page_content=rows[pid]["value"],
                            metadata=rows[pid].get("cmetadata", {}),
                        )
                    )
                    seen.add(pid)
                elif pid is None:
                    result.append(doc)
            docs = result

        return _format_docs(docs)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.Series:
        results = []
        for _, row in model_input.iterrows():
            # Primary path: ai_query sends named_struct → columns 'query' and 'type'
            if "query" in row.index:
                query = str(row["query"])
                query_type = str(row.get("type", "vector")).lower()
            else:
                # Fallback: single-column (plain string) input
                raw = row.iloc[0]
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                        query = data.get("query", raw)
                        query_type = str(data.get("type", "vector")).lower()
                    except (json.JSONDecodeError, AttributeError):
                        query, query_type = raw, "vector"
                else:
                    query, query_type = str(raw), "vector"

            if query_type == "hybrid":
                results.append(self._run_hybrid(query))
            else:
                docs = self._vector_retriever.invoke(query)
                results.append(_format_docs(docs))

        return pd.Series(results)


mlflow.models.set_model(RetrieverModel())
