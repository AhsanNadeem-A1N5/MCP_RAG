from fastmcp import FastMCP
import os
import tempfile
import logging
import time
import shutil
import json
from typing import Optional, List
from collections import OrderedDict
from threading import RLock
from contextlib import contextmanager
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from filelock import FileLock
from sentence_transformers import CrossEncoder


mcp = FastMCP("My MCP Server")
logger = logging.getLogger(__name__)


def _load_faiss_index(index_path: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """Load FAISS index from local directory."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    return FAISS.load_local(
        index_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )


def _build_context(chunks: List[str], max_chars: int = 3000) -> str:
    context = []
    used = 0
    for ch in chunks:
        if used + len(ch) + 2 > max_chars:
            break
        context.append(ch)
        used += len(ch) + 2
    return "\n\n".join(context)


def _rerank_with_openai(client: OpenAI, chat_model: str, query: str, docs: List[str], top_k: int) -> List[int]:
    if not docs:
        return []
    # Request the model to rank doc indices by relevance
    enumerated = "\n".join([f"[{i}] {docs[i][:1000]}" for i in range(len(docs))])
    prompt = (
        "Rank the following passages by relevance to the query. "
        "Return ONLY a JSON array of top indices in descending relevance, no prose.\n\n"
        f"Query: {query}\n\nPassages:\n{enumerated}\n\nOutput JSON array of indices only."
    )
    try:
        resp = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "system", "content": "You return only JSON arrays."},
                      {"role": "user", "content": prompt}],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        import json as _json
        idxs = _json.loads(content)
        if isinstance(idxs, list):
            return [int(i) for i in idxs[:top_k] if isinstance(i, int) and 0 <= i < len(docs)]
    except Exception:
        pass
    # Fallback to original order if parsing fails
    return list(range(min(top_k, len(docs))))


# -------------------- Medium/Large index caching helpers --------------------
_VS_CACHE: "OrderedDict[str, dict]" = OrderedDict()  # agent_id -> {"vs": FAISS, "ts": float}
_EMB_CACHE: dict = {}
_AGENT_LOCKS: dict = {}
_CACHE_LOCK = RLock()

def _cache_limits():
    max_items = int(os.getenv("RAG_CACHE_MAX_ITEMS", "8"))
    ttl_seconds = float(os.getenv("RAG_CACHE_TTL_SECONDS", "900"))
    return max_items, ttl_seconds

def _get_embeddings(model: str, api_key: str):
    with _CACHE_LOCK:
        emb = _EMB_CACHE.get(model)
        if emb is None:
            _EMB_CACHE[model] = OpenAIEmbeddings(model=model, api_key=api_key)
            emb = _EMB_CACHE[model]
        return emb

def _get_cached_vs(agent_id: str, ttl: float):
    with _CACHE_LOCK:
        item = _VS_CACHE.get(agent_id)
        if not item:
            return None
        if (time.time() - item["ts"]) > ttl:
            _VS_CACHE.pop(agent_id, None)
            return None
        _VS_CACHE.move_to_end(agent_id)
        return item["vs"]

def _set_cached_vs(agent_id: str, vs):
    max_items, _ = _cache_limits()
    with _CACHE_LOCK:
        _VS_CACHE[agent_id] = {"vs": vs, "ts": time.time()}
        _VS_CACHE.move_to_end(agent_id)
        while len(_VS_CACHE) > max_items:
            _VS_CACHE.popitem(last=False)

def _get_agent_lock(agent_id: str) -> RLock:
    with _CACHE_LOCK:
        lk = _AGENT_LOCKS.get(agent_id)
        if lk is None:
            lk = RLock()
            _AGENT_LOCKS[agent_id] = lk
        return lk

def _agent_cache_base() -> str:
    base = os.getenv("RAG_CACHE_DIR", os.path.join(os.getcwd(), ".rag_cache"))
    Path(base).mkdir(parents=True, exist_ok=True)
    return base

def _agent_cache_dir(agent_id: str) -> str:
    base = _agent_cache_base()
    d = os.path.join(base, agent_id)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d

def _faiss_files_exist(d: str) -> bool:
    """Check if FAISS index files exist in directory."""
    return (os.path.exists(os.path.join(d, "index.faiss")) and 
            os.path.exists(os.path.join(d, "index.pkl")))

def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total

def _touch_last_access(dir_path: str) -> None:
    Path(os.path.join(dir_path, ".last_access")).write_text(str(time.time()))

def _last_access(dir_path: str) -> float:
    p = os.path.join(dir_path, ".last_access")
    try:
        return float(Path(p).read_text().strip())
    except Exception:
        return os.path.getmtime(dir_path)

@contextmanager
def _maybe_lock(path: str, timeout: int = 300):
    try:
        lock = FileLock(path, timeout=timeout)
        with lock:
            yield
    except Exception:
        # If file lock fails for any reason, proceed without locking to avoid deadlock
        yield

def _enforce_disk_budget():
    base = _agent_cache_base()
    max_mb = int(os.getenv("RAG_DISK_CACHE_MAX_MB", "1024"))
    budget = max_mb * 1024 * 1024
    subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    sizes = {d: _dir_size_bytes(d) for d in subdirs}
    total = sum(sizes.values())
    if total <= budget:
        return
    for d in sorted(subdirs, key=_last_access):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        total -= sizes.get(d, 0)
        if total <= budget:
            break


def _local_version_path(d: str) -> str:
    return os.path.join(d, ".version")

def _read_local_version(d: str) -> float:
    p = _local_version_path(d)
    try:
        return float(Path(p).read_text().strip())
    except Exception:
        return 0.0

def _write_local_version(d: str, ts: float) -> None:
    Path(_local_version_path(d)).write_text(str(ts))


@mcp.tool
def get_response_from_rag(
    agent_id: str, 
    query: str, 
    k: int = 5, 
    force_refresh: bool = False,
    metadata_filter: Optional[str] = None
) -> str:
    """Answer a query using FAISS Vector Search, where agent_id is the FAISS index directory name.

    Environment variables expected:
    - OPENAI_API_KEY: OpenAI API key
    - FAISS_INDEX_BASE_PATH (optional): Base path for FAISS indices (default: ./faiss_indices)
    - RAG_EMBEDDING_MODEL (optional): OpenAI embedding model (default: text-embedding-3-small)
    - RAG_CHAT_MODEL (optional): OpenAI chat model (default: gpt-4o-mini)
    - RAG_CACHE_MAX_ITEMS (optional): max cached VectorStore clients (default: 8)
    - RAG_CACHE_TTL_SECONDS (optional): TTL for in-memory cache (default: 900)
    - RAG_RERANKER_MODEL (optional): cross-encoder model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
    
    Args:
        agent_id: FAISS index directory name/identifier
        query: Query string
        k: Number of results
        force_refresh: Force cache refresh
        metadata_filter: JSON string for metadata filter, e.g., '{"source": "document.pdf"}'
    """

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "RAG error: OPENAI_API_KEY is not set."

        embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        reranker_model = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        index_base_path = os.getenv("FAISS_INDEX_BASE_PATH", "./faiss_indices")
        
        # Construct full index path
        index_path = os.path.join(index_base_path, agent_id)
        
        if not os.path.exists(index_path):
            return f"RAG error: FAISS index not found at {index_path}"

        max_items, ttl = _cache_limits()
        
        # Parse metadata filter if provided
        filter_dict = None
        if metadata_filter:
            try:
                filter_dict = json.loads(metadata_filter)
            except Exception as e:
                logger.warning(f"Failed to parse metadata_filter: {e}")

        # Try in-memory cache fast path
        if not force_refresh:
            vs = _get_cached_vs(agent_id, ttl)
            if vs is not None:
                docs = vs.similarity_search(query, k=max(1, k * 3), filter=filter_dict)  # overfetch for reranking
                # Try Cross-Encoder; if OOM or unavailable, fall back to OpenAI reranker
                top_docs = []
                if docs:
                    try:
                        cross_encoder = CrossEncoder(reranker_model)
                        pairs = [(query, d.page_content) for d in docs]
                        scores = cross_encoder.predict(pairs)
                        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                        top_docs = [d for d, _ in ranked[:k]]
                    except Exception:
                        client_tmp = OpenAI(api_key=openai_api_key)
                        idxs = _rerank_with_openai(client_tmp, chat_model, query, [d.page_content for d in docs], k)
                        top_docs = [docs[i] for i in idxs]
                context_text = _build_context([d.page_content for d in top_docs])

                client = OpenAI(api_key=openai_api_key)
                system_msg = (
                    "You are a helpful assistant that answers strictly using the provided context. "
                    "If the answer cannot be derived from context, say you don't have enough information."
                )
                user_msg = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer succinctly."
                completion = client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                )
                return (completion.choices[0].message.content or "").strip()

        # Single-flight per agent: load FAISS index and cache it
        lock = _get_agent_lock(agent_id)
        with lock:
            # Re-check after acquiring the lock
            if not force_refresh:
                vs = _get_cached_vs(agent_id, ttl)
                if vs is not None:
                    docs = vs.similarity_search(query, k=max(1, k * 3), filter=filter_dict)
                    top_docs = []
                    if docs:
                        try:
                            cross_encoder = CrossEncoder(reranker_model)
                            pairs = [(query, d.page_content) for d in docs]
                            scores = cross_encoder.predict(pairs)
                            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                            top_docs = [d for d, _ in ranked[:k]]
                        except Exception:
                            client_tmp = OpenAI(api_key=openai_api_key)
                            idxs = _rerank_with_openai(client_tmp, chat_model, query, [d.page_content for d in docs], k)
                            top_docs = [docs[i] for i in idxs]
                    context_text = _build_context([d.page_content for d in top_docs])

                    client = OpenAI(api_key=openai_api_key)
                    system_msg = (
                        "You are a helpful assistant that answers strictly using the provided context. "
                        "If the answer cannot be derived from context, say you don't have enough information."
                    )
                    user_msg = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer succinctly."
                    completion = client.chat.completions.create(
                        model=chat_model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.2,
                    )
                    return (completion.choices[0].message.content or "").strip()

            embeddings = _get_embeddings(embedding_model, openai_api_key)
            vs = _load_faiss_index(index_path, embeddings)
            _set_cached_vs(agent_id, vs)

        # Query after client creation with overfetch + rerank
        docs = vs.similarity_search(query, k=max(1, k * 3), filter=filter_dict)
        top_docs = []
        if docs:
            try:
                cross_encoder = CrossEncoder(reranker_model)
                pairs = [(query, d.page_content) for d in docs]
                scores = cross_encoder.predict(pairs)
                ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                top_docs = [d for d, _ in ranked[:k]]
            except Exception:
                client_tmp = OpenAI(api_key=openai_api_key)
                idxs = _rerank_with_openai(client_tmp, chat_model, query, [d.page_content for d in docs], k)
                top_docs = [docs[i] for i in idxs]
        context_text = _build_context([d.page_content for d in top_docs])

        client = OpenAI(api_key=openai_api_key)
        system_msg = (
            "You are a helpful assistant that answers strictly using the provided context. "
            "If the answer cannot be derived from context, say you don't have enough information."
        )
        user_msg = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer succinctly."

        completion = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )

        return (completion.choices[0].message.content or "").strip()

    except Exception as e:
        logging.exception("RAG tool failed")
        return f"RAG error: {e}"


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=5000)
