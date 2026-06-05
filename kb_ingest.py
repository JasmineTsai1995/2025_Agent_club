from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BASE_DIR))

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from app.kb_support.preprocess import MarkdownContentProcessor
from config.prompts import mermaid_summary_prompt, summary_prompt, html_table_summary_prompt
from app.kb_support.mongo_index import create_chinese_fts_index
from app.kb_support.schema import MongoParentDocumentStore
from app.kb_support.splitter import build_parent_splitter, build_splitter
from app.kb_support.config import (
    CHILD_TABLE,
    EMBED_DIM,
    EMBED_ENDPOINT,
    EMBED_MODEL_NAME,
    HTML_TABLE_SUMMARIZE,
    KB_INGEST_FILE_PATH,
    LLM_ENDPOINT,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    MONGODB_DB,
    MONGODB_URI,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBED_MODEL,
    PARENT_TABLE,
    PROPOSITION_MAX,
    PROPOSITION_MIN_CHARS,
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_CHUNK_SIZE,
    SEMANTIC_THRESHOLD_AMOUNT,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Splitter settings (no loop)
CHILD_SPLITTER = "recursive"  # recursive | semantic | propositions

# Mongo settings
CLEAR_COLLECTION_BEFORE_INGEST = False


def _build_models(provider: str):
    """Build LLM and Embedding instances based on provider config."""
    import os

    if provider == "databricks":
        from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
        emb = DatabricksEmbeddings(endpoint=EMBED_ENDPOINT)

    elif provider == "mixed":
        # LLM: Databricks, Embedding: OpenRouter
        from databricks_langchain import ChatDatabricks
        from langchain_openai import OpenAIEmbeddings
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
        emb = OpenAIEmbeddings(
            model=OPENROUTER_EMBED_MODEL,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base=OPENROUTER_BASE_URL,
            check_embedding_ctx_length=False,
        )

    else:  # google (default)
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)
        emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_NAME)

    logger.info("Models initialized | provider=%s llm=%s embed=%s", provider,
                LLM_ENDPOINT if provider in ("databricks", "mixed") else LLM_MODEL_NAME,
                OPENROUTER_EMBED_MODEL if provider == "mixed" else EMBED_MODEL_NAME if provider == "google" else EMBED_ENDPOINT)
    return llm, emb


def load_md_from_file(md_path: Path) -> tuple[str, dict]:
    with md_path.open("r", encoding="utf-8") as file:
        markdown = file.read()
    file_id = md_path.stem
    metadata = {"filename": file_id, "file_id": file_id}
    return markdown, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a single Markdown file into MongoDB.")
    parser.add_argument(
        "markdown_path",
        nargs="?",
        default=str(KB_INGEST_FILE_PATH),
        help="Path to the markdown file to ingest. Defaults to config.paths.kb_ingest_file_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_file_path = Path(args.markdown_path).expanduser()
    if not md_file_path.is_absolute():
        md_file_path = (Path.cwd() / md_file_path).resolve()

    if not md_file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file_path}")

    logger.info("Start single-file ingest | file=%s splitter=%s", md_file_path, CHILD_SPLITTER)

    chat_llm, embeddings = _build_models(LLM_PROVIDER)

    parent_splitter = build_parent_splitter()
    child_splitter = build_splitter(
        CHILD_SPLITTER,
        embeddings,
        chat_llm,
        recursive_chunk_size=RECURSIVE_CHUNK_SIZE,
        recursive_chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        semantic_threshold_amount=SEMANTIC_THRESHOLD_AMOUNT,
        proposition_max=PROPOSITION_MAX,
        proposition_min_chars=PROPOSITION_MIN_CHARS,
    )

    mongo_client = MongoClient(MONGODB_URI)
    mongo_db = mongo_client[MONGODB_DB]
    mongo_client.admin.command("ping")

    parent_collection = mongo_db[PARENT_TABLE]
    child_collection = mongo_db[CHILD_TABLE]

    if CLEAR_COLLECTION_BEFORE_INGEST:
        parent_collection.delete_many({})
        child_collection.delete_many({})
        logger.info("Collections cleared before ingest.")

    parent_store = MongoParentDocumentStore(collection=parent_collection)
    child_store = MongoDBAtlasVectorSearch(
        collection=child_collection,
        embedding=embeddings,
        text_key="content",
        metadata_key="metadata",
    )
    try:
        child_store.create_vector_search_index(dimensions=EMBED_DIM)
    except Exception:
        pass
    create_chinese_fts_index(
        collection=child_collection,
        index_name="search_index",
        text_field="content",
    )

    retriever = ParentDocumentRetriever(
        vectorstore=child_store,
        docstore=parent_store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    markdown, metadata = load_md_from_file(md_file_path)
    summary_chain = summary_prompt | chat_llm
    mermaid_chain = mermaid_summary_prompt | chat_llm
    kwargs = {
        "summary_chain": summary_chain,
        "mermaid_summary_chain": mermaid_chain,
    }
    if HTML_TABLE_SUMMARIZE:
        html_table_chain = html_table_summary_prompt | chat_llm
        kwargs["html_table_summary_chain"] = html_table_chain

    document = MarkdownContentProcessor().process(markdown, metadata, **kwargs)
    retriever.add_documents([document])

    mongo_client.close()
    logger.info("Ingest completed | file=%s splitter=%s", md_file_path.name, CHILD_SPLITTER)


if __name__ == "__main__":
    main()
