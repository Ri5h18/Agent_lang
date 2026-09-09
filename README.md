# Agentic Lang

Agentic Lang is a single local-first React + FastAPI site for conversation, grounded PDF research, public web search, and document drafting. LangGraph coordinates the backend agents, but the workflow stays behind the product UI.

## What the site does

- **Assistant** — conversational Ollama chat with session context and a deterministic fallback when Ollama is offline.
- **Knowledge** — page-aware PDF retrieval over the bundled stock report and uploaded PDFs. When `nomic-embed-text` is available, chunks are ranked by Ollama cosine similarity; otherwise the app uses a deterministic lexical fallback. Retrieved excerpts are passed into Ollama for a cited answer when the model is enabled.
- **Web search** — searches current public results through DuckDuckGo Lite, shows source links, and optionally asks Ollama to synthesize them with `[W#]` citations.
- **Drafter** — revise a document through chat, edit the live text directly, and save it as a `.txt` file.
- **Exports** — save the full chat history or save any individual assistant response as its own text file, with a download link in the UI.

The bundled report is in `backend\data\Stock_Market_Performance_2024.pdf`. Uploaded PDFs are indexed into local cached passages under `backend\data\cache` and stored under `backend\data\uploads`.

## Setup and run

Open PowerShell:

```powershell
cd C:\Users\admin\Desktop\AGENTIC_LANG
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run-site.ps1
```

Open `http://127.0.0.1:8000`. The production build and API are served by one process. For development, run `.\run-backend.ps1` and `.\run-frontend.ps1` in separate terminals, then open `http://127.0.0.1:5173`.

## Project map

```text
AGENTIC_LANG/
├── backend/app/main.py          # FastAPI API and production static-file hosting
├── backend/app/agent_engine.py # LangGraph supervisor, RAG, web search, memory, chat, and Drafter
├── backend/app/workflows.py     # Internal graph examples; not rendered in the site
├── backend/data/                 # Bundled PDF, uploads, and retrieval cache
├── backend/storage/              # Saved drafts and conversation/text exports
├── backend/tests/                # Offline API and behavior checks
├── frontend/src/App.jsx          # Focused four-mode workspace UI
├── frontend/src/styles.css       # Responsive visual system
├── setup.ps1                     # Python/npm installation and production build
└── run-site.ps1                  # One-process production launch
```
