from __future__ import annotations

import ast
import hashlib
from html import unescape
from html.parser import HTMLParser
import io
import json
import operator
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph


BACKEND_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = BACKEND_DIR / "data" / "Stock_Market_Performance_2024.pdf"
UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"
CACHE_DIR = BACKEND_DIR / "data" / "cache"
DRAFT_DIR = BACKEND_DIR / "storage" / "drafts"
HISTORY_DIR = BACKEND_DIR / "storage" / "history"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("AGENTIC_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("AGENTIC_EMBED_MODEL", "nomic-embed-text")


@dataclass
class SessionRecord:
    history: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    draft: str = ""


SESSIONS: dict[str, SessionRecord] = {}
SESSION_LOCK = threading.Lock()


def get_session(session_id: str) -> SessionRecord:
    safe_id = session_id.strip()[:80] or "default"
    with SESSION_LOCK:
        return SESSIONS.setdefault(safe_id, SessionRecord())


def reset_session(session_id: str) -> None:
    with SESSION_LOCK:
        SESSIONS.pop(session_id.strip()[:80] or "default", None)


class KnowledgeBase:
    def __init__(self, path: Path, document_id: str, display_name: str, uploaded: bool = False) -> None:
        self.path = path
        self.document_id = document_id
        self.display_name = display_name
        self.uploaded = uploaded
        self._chunks: list[dict[str, Any]] | None = None
        self.error: str | None = None

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / f"{self.document_id}.json"

    @property
    def embedding_cache_path(self) -> Path:
        return CACHE_DIR / f"{self.document_id}-embeddings.json"

    def _read_cache(self) -> list[dict[str, Any]] | None:
        try:
            source = self.path.stat()
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if (
                payload.get("version") == 1
                and payload.get("source_size") == source.st_size
                and payload.get("source_mtime_ns") == source.st_mtime_ns
                and isinstance(payload.get("chunks"), list)
                and payload["chunks"]
            ):
                return payload["chunks"]
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return None

    def _write_cache(self, chunks: list[dict[str, Any]]) -> None:
        try:
            source = self.path.stat()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "source_size": source.st_size,
                "source_mtime_ns": source.st_mtime_ns,
                "chunks": chunks,
            }
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _load(self) -> list[dict[str, Any]]:
        if self._chunks is not None:
            return self._chunks
        cached = self._read_cache()
        if cached is not None:
            self._chunks = cached
            return cached
        chunks: list[dict[str, Any]] = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(self.path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = " ".join((page.extract_text() or "").split())
                if not text:
                    continue
                start = 0
                while start < len(text):
                    end = min(len(text), start + 1200)
                    if end < len(text):
                        boundary = text.rfind(" ", start + 700, end)
                        if boundary > start:
                            end = boundary
                    chunks.append({"page": page_number, "text": text[start:end].strip()})
                    if end >= len(text):
                        break
                    start = max(end - 180, start + 1)
            if not chunks:
                raise ValueError("No extractable text was found in the PDF")
            self._write_cache(chunks)
        except Exception as exc:
            self.error = str(exc)
            chunks = [{
                "page": 0,
                "text": (
                    "The bundled source is Stock_Market_Performance_2024.pdf. "
                    "Install the backend requirements to enable page-level PDF extraction and retrieval."
                ),
            }]
        self._chunks = chunks
        return chunks

    @property
    def chunk_count(self) -> int:
        return len(self._load())

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        chunks = self._load()
        tokens = {
            token
            for token in re.findall(r"[a-z0-9%$]+", query.lower())
            if len(token) > 2 and token not in {"what", "were", "with", "from", "that", "this", "about", "stock", "market"}
        }
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, chunk in enumerate(chunks):
            lowered = chunk["text"].lower()
            score = sum(lowered.count(token) for token in tokens)
            scored.append((score, -index, chunk))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        selected = [item[2] for item in scored[:limit] if item[0] > 0]
        return selected or chunks[: min(limit, len(chunks))]

    def _read_embedding_cache(self, chunk_count: int) -> list[list[float]] | None:
        try:
            source = self.path.stat()
            payload = json.loads(self.embedding_cache_path.read_text(encoding="utf-8"))
            vectors = payload.get("vectors")
            if (
                payload.get("version") == 1
                and payload.get("model") == EMBEDDING_MODEL
                and payload.get("source_size") == source.st_size
                and payload.get("source_mtime_ns") == source.st_mtime_ns
                and isinstance(vectors, list)
                and len(vectors) == chunk_count
            ):
                return vectors
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return None

    def _write_embedding_cache(self, vectors: list[list[float]]) -> None:
        try:
            source = self.path.stat()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.embedding_cache_path.write_text(
                json.dumps({
                    "version": 1,
                    "model": EMBEDDING_MODEL,
                    "source_size": source.st_size,
                    "source_mtime_ns": source.st_mtime_ns,
                    "vectors": vectors,
                }),
                encoding="utf-8",
            )
        except OSError:
            pass

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def semantic_search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Retrieve chunks by Ollama embedding similarity, caching document vectors."""
        chunks = self._load()
        vectors = self._read_embedding_cache(len(chunks))
        if vectors is None:
            vectors = []
            for start in range(0, len(chunks), 32):
                vectors.extend(ollama_embeddings([chunk["text"] for chunk in chunks[start:start + 32]]))
            if len(vectors) != len(chunks):
                raise RuntimeError("Ollama returned an incomplete embedding batch")
            self._write_embedding_cache(vectors)
        query_vector = ollama_embeddings([query])[0]
        ranked = sorted(
            zip((self._cosine(query_vector, vector) for vector in vectors), chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        return [chunk for score, chunk in ranked[:limit] if score > 0] or chunks[: min(limit, len(chunks))]

    def search_with_method(self, query: str, limit: int = 3) -> tuple[list[dict[str, Any]], str]:
        status = ollama_status()
        if status.get("embedding_ready"):
            try:
                return self.semantic_search(query, limit), "ollama-semantic"
            except Exception:
                pass
        return self.search(query, limit), "lexical-fallback"


BUNDLED_DOCUMENT_ID = "stock-market-performance-2024"
KNOWLEDGE = KnowledgeBase(PDF_PATH, BUNDLED_DOCUMENT_ID, PDF_PATH.name)
KNOWLEDGE_REGISTRY: dict[str, KnowledgeBase] = {BUNDLED_DOCUMENT_ID: KNOWLEDGE}
KNOWLEDGE_LOCK = threading.Lock()
MAX_PDF_BYTES = 25 * 1024 * 1024


def uploaded_document_id(path: Path) -> str:
    safe_stem = re.sub(r"[^a-z0-9-]+", "-", path.stem.lower().replace("_", "-")).strip("-")
    return f"upload-{safe_stem[:70]}"


def load_existing_uploads() -> None:
    if not UPLOAD_DIR.exists():
        return
    for path in sorted(UPLOAD_DIR.glob("*.pdf")):
        document_id = uploaded_document_id(path)
        clean_stem = re.sub(r"-[0-9a-f]{12}$", "", path.stem, flags=re.IGNORECASE)
        KNOWLEDGE_REGISTRY.setdefault(
            document_id,
            KnowledgeBase(path, document_id, f"{clean_stem}.pdf", uploaded=True),
        )


def document_metadata(knowledge: KnowledgeBase) -> dict[str, Any]:
    return {
        "id": knowledge.document_id,
        "name": knowledge.display_name,
        "stored_name": knowledge.path.name,
        "uploaded": knowledge.uploaded,
        "chunks": knowledge.chunk_count,
        "error": knowledge.error,
    }


def list_knowledge_documents() -> list[dict[str, Any]]:
    with KNOWLEDGE_LOCK:
        documents = list(KNOWLEDGE_REGISTRY.values())
    return [document_metadata(item) for item in documents]


def knowledge_document_count() -> int:
    with KNOWLEDGE_LOCK:
        return len(KNOWLEDGE_REGISTRY)


def register_uploaded_pdf(filename: str, content: bytes) -> dict[str, Any]:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only .pdf files are accepted")
    if not content.startswith(b"%PDF-"):
        raise ValueError("The selected file does not have a valid PDF signature")
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 25 MB upload limit")
    if not content:
        raise ValueError("The selected PDF is empty")

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if not reader.pages:
            raise ValueError("PDF contains no pages")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF could not be parsed: {exc}") from exc

    original_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(filename).stem).strip("-") or "document"
    digest = hashlib.sha256(content).hexdigest()[:12]
    stored_name = f"{original_stem[:55]}-{digest}.pdf"
    path = UPLOAD_DIR / stored_name
    document_id = uploaded_document_id(path)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    knowledge = KnowledgeBase(path, document_id, Path(filename).name, uploaded=True)
    knowledge.chunk_count
    if knowledge.error:
        path.unlink(missing_ok=True)
        raise ValueError(f"PDF text could not be extracted: {knowledge.error}")
    with KNOWLEDGE_LOCK:
        KNOWLEDGE_REGISTRY[document_id] = knowledge
    return document_metadata(knowledge)


load_existing_uploads()


def ollama_status() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=0.6) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [item.get("name", "") for item in payload.get("models", [])]
        return {
            "available": True,
            "base_url": OLLAMA_BASE_URL,
            "configured_model": DEFAULT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "models": models,
            "model_ready": any(name.split(":")[0] == DEFAULT_MODEL.split(":")[0] for name in models),
            "embedding_ready": any(name.split(":")[0] == EMBEDDING_MODEL.split(":")[0] for name in models),
        }
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return {
            "available": False,
            "base_url": OLLAMA_BASE_URL,
            "configured_model": DEFAULT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "models": [],
            "model_ready": False,
            "embedding_ready": False,
        }


def ollama_embeddings(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or not all(isinstance(vector, list) for vector in vectors):
        raise RuntimeError("Ollama did not return embeddings")
    return vectors


def invoke_ollama_history(system: str, history: list[dict[str, str]]) -> str:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        model=DEFAULT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=450,
        client_kwargs={"timeout": 120.0},
    )
    messages: list[Any] = [SystemMessage(content=system)]
    for item in history:
        content = str(item.get("content", ""))
        if item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    response = model.invoke(messages)
    return str(response.content)


def invoke_ollama(system: str, prompt: str) -> str:
    return invoke_ollama_history(system, [{"role": "user", "content": prompt}])


class DuckDuckGoParser(HTMLParser):
    """Extract the small result set returned by DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capture: str | None = None
        self.buffer: list[str] = []

    @staticmethod
    def _has_class(attrs: list[tuple[str, str | None]], value: str) -> bool:
        classes = dict(attrs).get("class") or ""
        return value in classes.split()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and (self._has_class(attrs, "result__a") or self._has_class(attrs, "result-link")):
            self.current = {"title": "", "url": unescape(dict(attrs).get("href") or ""), "snippet": ""}
            self.capture = "title"
            self.buffer = []
        elif self.current and tag in {"a", "div", "td"} and (self._has_class(attrs, "result__snippet") or self._has_class(attrs, "result-snippet")):
            self.capture = "snippet"
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.current and self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.current or not self.capture:
            return
        if (self.capture == "title" and tag == "a") or (self.capture == "snippet" and tag in {"a", "div", "td"}):
            self.current[self.capture] = " ".join("".join(self.buffer).split())
            if self.capture == "snippet":
                if self.current["title"] and self.current["url"]:
                    self.results.append(self.current)
                self.current = None
            self.capture = None
            self.buffer = []


def search_web(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search the public web without requiring a search API key."""
    encoded_query = urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        f"https://lite.duckduckgo.com/lite/?{encoded_query}",
        headers={"User-Agent": "Agentic-Lang/1.0 (+local research tool)"},
    )
    with urllib.request.urlopen(request, timeout=12.0) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = DuckDuckGoParser()
    parser.feed(html)
    results: list[dict[str, str]] = []
    for item in parser.results:
        url = urllib.parse.unquote(unescape(item["url"]))
        redirect_query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if redirect_query.get("uddg"):
            url = redirect_query["uddg"][0]
        if url.startswith("//"):
            url = f"https:{url}"
        results.append({"title": item["title"], "url": url, "snippet": item["snippet"]})
        if len(results) >= limit:
            break
    return results


ALLOWED_OPERATORS: dict[type[ast.operator], tuple[str, Any]] = {
    ast.Add: ("add", operator.add),
    ast.Sub: ("subtract", operator.sub),
    ast.Mult: ("multiply", operator.mul),
}


def evaluate_expression(expression: str) -> tuple[int | float, list[dict[str, Any]]]:
    tree = ast.parse(expression, mode="eval")
    calls: list[dict[str, Any]] = []

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            name, operation = ALLOWED_OPERATORS[type(node.op)]
            result = operation(left, right)
            calls.append({"name": name, "arguments": {"a": left, "b": right}, "result": result})
            return result
        raise ValueError("Only numbers, parentheses, addition, subtraction, and multiplication are supported")

    return evaluate(tree), calls


def calculate(message: str) -> tuple[str, list[dict[str, Any]]]:
    lowered = message.lower().replace("×", "*")
    numbers = [int(value) for value in re.findall(r"-?\d+", lowered)]
    calls: list[dict[str, Any]] = []
    result: int | float

    if "add" in lowered and "multiply" in lowered and len(numbers) >= 3:
        first = numbers[0] + numbers[1]
        calls.append({"name": "add", "arguments": {"a": numbers[0], "b": numbers[1]}, "result": first})
        result = first * numbers[2]
        calls.append({"name": "multiply", "arguments": {"a": first, "b": numbers[2]}, "result": result})
    elif "subtract" in lowered and " from " in lowered and len(numbers) >= 2:
        result = numbers[1] - numbers[0]
        calls.append({"name": "subtract", "arguments": {"a": numbers[1], "b": numbers[0]}, "result": result})
    else:
        matches = re.findall(r"(?<!\w)(-?\d+(?:\s*[+\-*]\s*-?\d+)+(?:\s*[+\-*]\s*-?\d+)*)(?!\w)", lowered)
        if matches:
            result, calls = evaluate_expression(matches[0])
        elif "add" in lowered and len(numbers) >= 2:
            result = numbers[0] + numbers[1]
            calls.append({"name": "add", "arguments": {"a": numbers[0], "b": numbers[1]}, "result": result})
        elif "multiply" in lowered and len(numbers) >= 2:
            result = numbers[0] * numbers[1]
            calls.append({"name": "multiply", "arguments": {"a": numbers[0], "b": numbers[1]}, "result": result})
        elif "subtract" in lowered and len(numbers) >= 2:
            result = numbers[0] - numbers[1]
            calls.append({"name": "subtract", "arguments": {"a": numbers[0], "b": numbers[1]}, "result": result})
        else:
            return (
                "I can use add, subtract, and multiply tools. Try: “Add 40 + 12 and then multiply the result by 6.”",
                [],
            )
    answer = f"The result is {result}."
    if "joke" in lowered:
        answer += " Bonus joke: Why did the graph cross the road? It followed the edge to the other node."
    return answer, calls


def extract_facts(message: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    patterns = {
        "name": r"(?:my name is|call me)\s+([a-z][a-z .'-]{0,40})",
        "location": r"(?:i live in|i am from)\s+([a-z][a-z ,.'-]{0,60})",
        "preference": r"(?:i like|i love)\s+([^.!?]{1,80})",
        "note": r"remember that\s+([^.!?]{1,120})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            facts[key] = match.group(1).strip().title() if key in {"name", "location"} else match.group(1).strip()
    return facts


class AgentState(TypedDict, total=False):
    request: dict[str, Any]
    route: str
    answer: str
    citations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    memory: dict[str, str]
    document: str
    saved_file: str | None
    model_used: bool
    warning: str | None
    knowledge_documents: list[str]
    web_results: list[dict[str, str]]
    trace: Annotated[list[str], operator.add]


def supervisor(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    requested = str(request.get("mode", "auto")).lower()
    message = str(request.get("message", "")).lower()
    if requested != "auto":
        route = requested
    elif any(word in message for word in ["search the web", "web search", "on the web", "latest", "current news", "look up online"]):
        route = "web"
    elif any(word in message for word in ["stock", "market", "2024", "document", "pdf"]):
        route = "rag"
    elif any(word in message for word in ["add", "subtract", "multiply", "calculate", "+", "*", "-"]):
        route = "react"
    elif any(word in message for word in ["remember", "my name", "i live", "i like", "what do you know"]):
        route = "memory"
    elif any(word in message for word in ["draft", "write", "save document", "revise"]):
        route = "drafter"
    else:
        route = "chat"
    if route == "assistant":
        route = "chat"
    if route not in {"chat", "react", "rag", "web", "memory", "drafter"}:
        route = "chat"
    return {"route": route, "trace": [f"supervisor: routed request to {route}"]}


def route_agent(state: AgentState) -> str:
    return state["route"]


def react_agent(state: AgentState) -> dict[str, Any]:
    answer, calls = calculate(str(state["request"].get("message", "")))
    tool_names = ", ".join(call["name"] for call in calls) or "none"
    return {
        "answer": answer,
        "tool_calls": calls,
        "model_used": False,
        "trace": [f"react: executed tools [{tool_names}]", "react: returned tool-grounded answer"],
    }


def rag_agent(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    query = str(request.get("message", ""))
    requested_ids = [str(item) for item in request.get("document_ids", []) if str(item)]
    if not requested_ids and request.get("document_id"):
        requested_ids = [str(request["document_id"])]
    if not requested_ids:
        requested_ids = list(KNOWLEDGE_REGISTRY)
    requested_ids = list(dict.fromkeys(requested_ids))

    knowledge_sources: list[KnowledgeBase] = []
    missing_ids: list[str] = []
    for document_id in requested_ids:
        knowledge = KNOWLEDGE_REGISTRY.get(document_id)
        if knowledge is None:
            missing_ids.append(document_id)
        else:
            knowledge_sources.append(knowledge)
    if not knowledge_sources:
        knowledge_sources = [KNOWLEDGE]

    evidence: list[dict[str, Any]] = []
    retrieval_methods: list[str] = []
    for knowledge in knowledge_sources:
        selected_chunks, retrieval_method = knowledge.search_with_method(query, limit=3)
        retrieval_methods.append(retrieval_method)
        for chunk in selected_chunks:
            evidence.append({"knowledge": knowledge, **chunk})
    citations = [
        {
            "id": f"S{index}",
            "source": item["knowledge"].display_name,
            "page": item["page"],
            "excerpt": item["text"][:420],
        }
        for index, item in enumerate(evidence, start=1)
    ]
    context = "\n\n".join(
        f"[S{index}] Source: {item['knowledge'].display_name} | Page {item['page']}\n{item['text']}"
        for index, item in enumerate(evidence, start=1)
    )
    source_names = [knowledge.display_name for knowledge in knowledge_sources]
    model_used = False
    warnings = [knowledge.error for knowledge in knowledge_sources if knowledge.error]
    if missing_ids:
        warnings.append(f"Missing documents were skipped: {', '.join(missing_ids)}")
    warning = "; ".join(item for item in warnings if item) or None
    if request.get("use_model"):
        try:
            answer = invoke_ollama(
                (
                    "You are a multi-document RAG analyst. Answer only from the supplied excerpts. "
                    "Synthesize evidence across all relevant sources, distinguish their claims, and cite every material claim "
                    "with one or more evidence IDs such as [S1] or [S2][S5]. Never invent an ID. "
                    "If the evidence is insufficient, say so explicitly."
                ),
                f"Question: {query}\n\nSelected sources: {', '.join(source_names)}\n\nRetrieved evidence:\n{context}",
            )
            model_used = True
        except Exception as exc:
            warning = f"Ollama unavailable; used extractive answer instead: {exc}"
            answer = ""
    else:
        answer = ""
    if not answer:
        excerpts = "\n\n".join(
            f"[{item['knowledge'].display_name}, p. {item['page']}] {item['text'][:520]}"
            for item in evidence
        )
        answer = f"The most relevant evidence across {len(source_names)} selected PDF(s) is:\n\n{excerpts}"
    source_references: dict[str, list[tuple[str, int]]] = {}
    for citation in citations:
        source_references.setdefault(citation["source"], []).append((citation["id"], int(citation["page"])))
    source_appendix = "\n".join(
        f"- {source}: {', '.join(f'[{citation_id}] p. {page}' for citation_id, page in references)}"
        for source, references in source_references.items()
    )
    answer = f"{answer.rstrip()}\n\nSources used:\n{source_appendix}"
    return {
        "answer": answer,
        "citations": citations,
        "knowledge_documents": source_names,
        "retrieval_method": "ollama-semantic" if retrieval_methods and all(item == "ollama-semantic" for item in retrieval_methods) else "lexical-fallback",
        "model_used": model_used,
        "warning": warning,
        "trace": [
            f"rag: searched {len(source_names)} PDF sources",
            f"rag: retrieved {len(evidence)} source-labelled chunks",
            f"rag: retrieval method={retrieval_methods[0] if retrieval_methods else 'none'}",
            "rag: synthesized evidence with Ollama" if model_used else "rag: returned extractive evidence without Ollama",
        ],
    }


def web_agent(state: AgentState) -> dict[str, Any]:
    """Search current public web results, then optionally synthesize them with Ollama."""
    request = state["request"]
    query = str(request.get("message", "")).strip()
    warning: str | None = None
    model_used = False
    try:
        results = search_web(query, limit=5)
    except Exception as exc:
        results = []
        warning = f"Web search is unavailable: {exc}"

    if results:
        context = "\n\n".join(
            f"[W{index}] {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
            for index, item in enumerate(results, start=1)
        )
        answer = ""
        if request.get("use_model"):
            try:
                answer = invoke_ollama(
                    "You are a careful web research assistant. Answer only from the supplied search results. "
                    "Cite every material claim with result IDs like [W1]. Say when the results are insufficient. "
                    "Do not invent facts, sources, or URLs.",
                    f"Question: {query}\n\nSearch results:\n{context}",
                )
                model_used = True
            except Exception as exc:
                warning = f"Ollama unavailable; showing the search results directly: {exc}"
        if not answer:
            answer = f"I found {len(results)} web results for “{query}”. Review the sources below for the current details."
    else:
        answer = "I could not find web results for that request. Try a more specific search."

    return {
        "answer": answer,
        "web_results": results,
        "model_used": model_used,
        "warning": warning,
        "trace": [f"web: searched public results for {query!r}", f"web: received {len(results)} results"],
    }


def memory_agent(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    message = str(request.get("message", ""))
    session = get_session(str(request.get("session_id", "default")))
    new_facts = extract_facts(message)
    session.facts.update(new_facts)
    model_used = False
    warning: str | None = None
    answer = ""
    if request.get("use_model"):
        try:
            answer = invoke_ollama_history(
                (
                    "You are a concise conversational assistant with memory. Use the supplied conversation history "
                    "to answer the latest user message. Correctly recall details from earlier turns and do not claim "
                    "to remember information that is not present."
                ),
                session.history,
            )
            model_used = True
        except Exception as exc:
            warning = f"Ollama unavailable; used local memory fallback instead: {exc}"

    if not answer:
        lowered = message.lower()
        if "what is my name" in lowered:
            answer = f"Your name is {session.facts['name']}." if "name" in session.facts else "You have not told me your name yet."
        elif "what do you remember" in lowered or "what do you know" in lowered:
            answer = "I remember: " + "; ".join(f"{key}: {value}" for key, value in session.facts.items()) if session.facts else "I do not have any saved facts for this session yet."
        elif new_facts:
            answer = "Saved to this session: " + "; ".join(f"{key}: {value}" for key, value in new_facts.items())
        else:
            answer = "Memory is active. Tell me a fact, then ask me about it in a later message."
    return {
        "answer": answer,
        "memory": dict(session.facts),
        "model_used": model_used,
        "warning": warning,
        "trace": [f"memory: extracted {len(new_facts)} new facts", "memory: read full conversation history"],
    }


def safe_filename(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(value or "agentic-draft").stem).strip("-")
    return f"{(stem or 'agentic-draft')[:60]}.txt"


def save_conversation_history(session_id: str, filename: str) -> str:
    session = get_session(session_id)
    safe_name = safe_filename(filename or "conversation-log.txt")
    lines = ["Your Conversation Log:"]
    for message in session.history:
        speaker = "You" if message.get("role") == "user" else "AI"
        lines.append(f"{speaker}: {message.get('content', '')}")
    lines.append("End of Conversation")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    (HISTORY_DIR / safe_name).write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return safe_name


def save_text_file(content: str, filename: str) -> str:
    """Save a user-selected piece of text in the same local export area as chat logs."""
    if not str(content).strip():
        raise ValueError("Text to save must not be empty")
    safe_name = safe_filename(filename or "agentic-note.txt")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    (HISTORY_DIR / safe_name).write_text(str(content), encoding="utf-8")
    return safe_name


def drafter_agent(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    session = get_session(str(request.get("session_id", "default")))
    action = str(request.get("draft_action", "update"))
    message = str(request.get("message", "")).strip()
    supplied = request.get("draft_content")
    model_used = False
    warning: str | None = None
    saved_file: str | None = None

    if supplied is not None:
        session.draft = str(supplied)
    if action == "revise" and message:
        if request.get("use_model"):
            try:
                session.draft = invoke_ollama(
                    "You are Drafter. Return only the complete revised document, without commentary.",
                    f"Current document:\n{session.draft}\n\nRevision request:\n{message}",
                )
                model_used = True
            except Exception as exc:
                warning = f"Ollama could not revise the draft: {exc}"
        elif not session.draft:
            session.draft = message
            warning = "Ollama is off, so the instruction was stored as the initial draft."
        else:
            warning = "Turn on Ollama for instruction-based rewriting, or edit the document directly."
    elif not session.draft and message:
        session.draft = message

    if action == "save":
        DRAFT_DIR.mkdir(parents=True, exist_ok=True)
        saved_file = safe_filename(str(request.get("draft_filename", "agentic-draft.txt")))
        (DRAFT_DIR / saved_file).write_text(session.draft, encoding="utf-8")
        answer = f"Document saved as {saved_file}."
    elif action == "revise" and session.draft:
        answer = session.draft
    else:
        answer = "Document state updated." if session.draft else "The draft is empty. Add content to begin."
    return {
        "answer": answer,
        "document": session.draft,
        "saved_file": saved_file,
        "model_used": model_used,
        "warning": warning,
        "trace": [f"drafter: action={action}", f"drafter: document has {len(session.draft)} characters"],
    }


def chat_agent(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    message = str(request.get("message", ""))
    if request.get("use_model"):
        try:
            session = get_session(str(request.get("session_id", "default")))
            answer = invoke_ollama_history(
                "You are the concise general assistant inside Agentic Lang Studio. Use earlier turns when helpful.",
                session.history,
            )
            return {"answer": answer, "model_used": True, "trace": ["chat: invoked Ollama with conversation context"]}
        except Exception as exc:
            warning = f"Ollama unavailable: {exc}"
    else:
        warning = "Start Ollama with the configured model to use the stateless chat agent."
    return {
        "answer": "The Chat agent is ready, but local model generation is currently unavailable.",
        "model_used": False,
        "warning": warning,
        "trace": ["chat: returned model availability guidance"],
    }


agent_builder = StateGraph(AgentState)
agent_builder.add_node("supervisor", supervisor)
agent_builder.add_node("chat", chat_agent)
agent_builder.add_node("react", react_agent)
agent_builder.add_node("rag", rag_agent)
agent_builder.add_node("web", web_agent)
agent_builder.add_node("memory", memory_agent)
agent_builder.add_node("drafter", drafter_agent)
agent_builder.add_edge(START, "supervisor")
agent_builder.add_conditional_edges(
    "supervisor",
    route_agent,
    {"chat": "chat", "react": "react", "rag": "rag", "web": "web", "memory": "memory", "drafter": "drafter"},
)
for node_name in ["chat", "react", "rag", "web", "memory", "drafter"]:
    agent_builder.add_edge(node_name, END)
AGENT_GRAPH = agent_builder.compile()


def run_agent(request: dict[str, Any]) -> dict[str, Any]:
    session_id = str(request.get("session_id", "default"))
    session = get_session(session_id)
    message = str(request.get("message", ""))
    if message:
        session.history.append({"role": "user", "content": message})
    result = dict(AGENT_GRAPH.invoke({"request": request, "trace": []}))
    result.pop("request", None)
    if result.get("answer"):
        session.history.append({"role": "assistant", "content": str(result["answer"])})
    result["history"] = list(session.history)
    result.setdefault("citations", [])
    result.setdefault("tool_calls", [])
    result.setdefault("memory", dict(session.facts))
    result.setdefault("document", session.draft)
    result.setdefault("saved_file", None)
    result.setdefault("warning", None)
    result.setdefault("knowledge_documents", [])
    result.setdefault("web_results", [])
    return result


CONCEPTS = [
    {"name": "Stateless chat", "source": "Agents/Agent_Bot.py", "status": "live"},
    {"name": "Conversation memory", "source": "Agents/Memory_Agent.py", "status": "live"},
    {"name": "ReAct calculator tools", "source": "Agents/ReAct.py", "status": "live"},
    {"name": "Stock report RAG", "source": "Agents/RAG_Agent.py", "status": "live"},
    {"name": "Document drafting", "source": "Agents/Drafter.py", "status": "live"},
    {"name": "Typed graph state", "source": "exercise.ipynb", "status": "live"},
    {"name": "Sequential graph steps", "source": "exercise.ipynb", "status": "live"},
    {"name": "Conditional routing", "source": "exercise.ipynb", "status": "live"},
    {"name": "Cyclic graph loops", "source": "exercise.ipynb", "status": "live"},
    {"name": "Checkpoint persistence", "source": "exercise.ipynb", "status": "live"},
    {"name": "Supervisor routing", "source": "sequence.ipynb", "status": "live"},
    {"name": "Public web research", "source": "web search", "status": "live"},
]
