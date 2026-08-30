from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent_engine import (
    CONCEPTS,
    DRAFT_DIR,
    HISTORY_DIR,
    KNOWLEDGE,
    MAX_PDF_BYTES,
    PDF_PATH,
    get_session,
    knowledge_document_count,
    list_knowledge_documents,
    ollama_status,
    register_uploaded_pdf,
    reset_session,
    run_agent,
    save_conversation_history,
    save_text_file,
)
from .workflows import WORKFLOW_CATALOG, run_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(
    title="Agentic Lang Studio",
    version="1.0.0",
    description="One runnable project for the LangGraph notebooks and agent scripts.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WorkflowRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    session_id: str = Field(default="studio-session", max_length=80)
    mode: Literal["auto", "chat", "react", "rag", "web", "memory", "drafter", "assistant"] = "auto"
    message: str = Field(default="", max_length=8000)
    use_model: bool = False
    draft_action: Literal["update", "revise", "save", "read"] = "update"
    draft_content: str | None = Field(default=None, max_length=100_000)
    draft_filename: str = Field(default="agentic-draft.txt", max_length=100)
    document_id: str | None = Field(default=None, max_length=100)
    document_ids: list[str] = Field(default_factory=list, max_length=8)


class SaveHistoryRequest(BaseModel):
    filename: str = Field(default="conversation-log.txt", max_length=100)


class SaveTextRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    filename: str = Field(default="agentic-note.txt", max_length=100)


@app.get("/api/health")
def health() -> dict[str, Any]:
    model = ollama_status()
    return {
        "status": "ok",
        "runtime": "langgraph",
        "site": "Agentic Lang Studio",
        "concept_count": len(CONCEPTS),
        "workflow_count": len(WORKFLOW_CATALOG),
        "agent_count": len(CONCEPTS),
        "pdf": {
            "name": PDF_PATH.name,
            "available": PDF_PATH.exists(),
            "chunks": KNOWLEDGE.chunk_count,
            "error": KNOWLEDGE.error,
            "document_count": knowledge_document_count(),
        },
        "ollama": model,
    }


@app.get("/api/concepts")
def concepts() -> dict[str, Any]:
    return {"concepts": CONCEPTS}


@app.get("/api/workflows")
def workflows() -> dict[str, Any]:
    return {"workflows": WORKFLOW_CATALOG}


@app.get("/api/knowledge/documents")
def knowledge_documents() -> dict[str, Any]:
    return {"documents": list_knowledge_documents(), "max_upload_bytes": MAX_PDF_BYTES}


@app.post("/api/knowledge/upload")
async def upload_knowledge_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    content = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    try:
        document = register_uploaded_pdf(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "uploaded", "document": document}


@app.post("/api/workflows/{workflow_id}/run")
def execute_workflow(workflow_id: str, request: WorkflowRequest) -> dict[str, Any]:
    try:
        result = run_workflow(workflow_id, request.inputs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"workflow_id": workflow_id, "result": result}


@app.post("/api/agents/run")
def execute_agent(request: AgentRequest) -> dict[str, Any]:
    if not request.message.strip() and request.mode != "drafter":
        raise HTTPException(status_code=422, detail="message must not be empty")
    return run_agent(request.model_dump())


@app.get("/api/sessions/{session_id}")
def session_state(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    return {"history": session.history, "memory": session.facts, "document": session.draft}


@app.post("/api/sessions/{session_id}/reset")
def clear_session(session_id: str) -> dict[str, str]:
    reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/api/sessions/{session_id}/history/save")
def save_session_history(session_id: str, request: SaveHistoryRequest) -> dict[str, str]:
    filename = save_conversation_history(session_id, request.filename)
    return {"status": "saved", "saved_file": filename}


@app.post("/api/sessions/{session_id}/text/save")
def save_selected_text(session_id: str, request: SaveTextRequest) -> dict[str, str]:
    # Keep the session in the route for a consistent client contract and future per-session export folders.
    del session_id
    try:
        filename = save_text_file(request.content, request.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "saved", "saved_file": filename}


@app.get("/api/history/{filename}")
def download_history(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = HISTORY_DIR / safe_name
    if safe_name != filename or path.suffix.lower() != ".txt" or not path.exists():
        raise HTTPException(status_code=404, detail="Conversation log not found")
    return FileResponse(path, media_type="text/plain", filename=safe_name)


@app.get("/api/drafts/{filename}")
def download_draft(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = DRAFT_DIR / safe_name
    if safe_name != filename or path.suffix.lower() != ".txt" or not path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")
    return FileResponse(path, media_type="text/plain", filename=safe_name)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
