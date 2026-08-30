import { useEffect, useRef, useState } from 'react'

const STOCK_REPORT_ID = 'stock-market-performance-2024'
const SESSION_KEY = 'agentic-lang-session'
const MODES = [
  { id: 'assistant', label: 'Assistant', icon: 'chat', hint: 'Ask anything' },
  { id: 'rag', label: 'Knowledge', icon: 'book', hint: 'Search your PDFs' },
  { id: 'web', label: 'Web search', icon: 'globe', hint: 'Find current sources' },
  { id: 'drafter', label: 'Drafter', icon: 'file', hint: 'Write and export' },
]

async function api(path, options = {}) {
  const isForm = options.body instanceof FormData
  const response = await fetch(path, { ...options, headers: isForm ? { ...options.headers } : { 'Content-Type': 'application/json', ...options.headers } })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`)
  return payload
}

function Icon({ name, size = 18 }) {
  const paths = {
    chat: <><path d="M4 5h16v11H9l-5 4Z"/><path d="M8 9h8M8 12h5"/></>,
    book: <><path d="M4 4h6a3 3 0 0 1 3 3v13a3 3 0 0 0-3-3H4Z"/><path d="M20 4h-4a3 3 0 0 0-3 3v13a3 3 0 0 1 3-3h4Z"/></>,
    globe: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></>,
    file: <><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5M9 12h6M9 16h6"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    reset: <><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/></>,
    upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    alert: <><path d="M12 3 2.5 20h19Z"/><path d="M12 9v4M12 17h.01"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function MessageRow({ message, index, onSave, savedKey }) {
  const isUser = message.role === 'user'
  const key = `${message.role}-${index}`
  return <article className={`message-row ${isUser ? 'user' : 'assistant'}`}><div className="message-avatar">{isUser ? 'Y' : 'AL'}</div><div className="message-body"><div className="message-meta"><strong>{isUser ? 'You' : 'Agentic Lang'}</strong><span>{isUser ? 'request' : 'response'}</span></div><div className="message-content">{message.content}</div>{!isUser && <button className="message-save" onClick={() => onSave(message.content, key)}><Icon name={savedKey === key ? 'check' : 'download'} size={13}/>{savedKey === key ? 'Saved' : 'Save text'}</button>}</div></article>
}

function SourcePicker({ documents, selected, onToggle, onUpload, uploading }) {
  return <section className="source-picker"><div className="source-picker-head"><div><span className="label">Knowledge sources</span><strong>{selected.length} selected</strong></div><label className={uploading ? 'upload-button disabled' : 'upload-button'}><Icon name="upload" size={14}/>{uploading ? 'Indexing...' : 'Add PDFs'}<input type="file" accept="application/pdf" multiple onChange={onUpload} disabled={uploading}/></label></div><div className="source-list">{documents.map((document) => <label className={selected.includes(document.id) ? 'source-item selected' : 'source-item'} key={document.id}><input type="checkbox" checked={selected.includes(document.id)} onChange={() => onToggle(document.id)}/><span className="checkmark">✓</span><span className="source-name"><strong>{document.name}</strong><small>{document.chunks} indexed passages</small></span></label>)}{!documents.length && <span className="muted">No PDF sources yet.</span>}</div></section>
}

function WebResults({ results }) {
  if (!results?.length) return null
  return <section className="web-results"><div className="section-label">Sources found</div>{results.map((item, index) => <a className="web-result" href={item.url} target="_blank" rel="noreferrer" key={`${item.url}-${index}`}><span className="result-number">W{index + 1}</span><span><strong>{item.title}</strong><small>{item.snippet}</small><em>{item.url}</em></span></a>)}</section>
}

export default function App() {
  const [activeMode, setActiveMode] = useState('assistant')
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [selectedDocuments, setSelectedDocuments] = useState([STOCK_REPORT_ID])
  const [history, setHistory] = useState([])
  const [draft, setDraft] = useState('')
  const [prompt, setPrompt] = useState('')
  const [useModel, setUseModel] = useState(false)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [result, setResult] = useState(null)
  const [historyName, setHistoryName] = useState('conversation-log.txt')
  const [draftName, setDraftName] = useState('agentic-draft.txt')
  const [savedKey, setSavedKey] = useState('')
  const [savedUrl, setSavedUrl] = useState('')
  const [sessionId] = useState(() => { const existing = window.localStorage.getItem(SESSION_KEY); const value = existing || `workspace-${Math.random().toString(36).slice(2, 10)}`; window.localStorage.setItem(SESSION_KEY, value); return value })
  const endRef = useRef(null)
  const mode = MODES.find((item) => item.id === activeMode) || MODES[0]
  const modelReady = Boolean(health?.ollama?.model_ready)
  const semanticReady = Boolean(health?.ollama?.embedding_ready)

  useEffect(() => { Promise.all([api('/api/health'), api('/api/knowledge/documents'), api(`/api/sessions/${sessionId}`)]).then(([status, knowledge, session]) => { setHealth(status); setUseModel(Boolean(status.ollama?.model_ready)); setDocuments(knowledge.documents || []); setHistory(session.history || []); setDraft(session.document || '') }).catch((err) => setError(err.message)) }, [sessionId])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [history, loading])

  function toggleDocument(id) { setSelectedDocuments((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id]) }

  async function uploadPdf(event) {
    const files = Array.from(event.target.files || []); event.target.value = ''; if (!files.length) return
    setUploading(true); setError(''); setNotice(''); setSavedUrl('')
    const added = []; const failed = []
    for (const file of files) {
      try { const form = new FormData(); form.append('file', file); const data = await api('/api/knowledge/upload', { method: 'POST', body: form }); added.push(data.document) } catch (err) { failed.push(`${file.name}: ${err.message}`) }
    }
    if (added.length) { setDocuments((items) => [...items.filter((item) => !added.some((document) => document.id === item.id)), ...added]); setSelectedDocuments((items) => [...new Set([...items, ...added.map((document) => document.id)])]); setNotice(`Indexed ${added.length} PDF${added.length === 1 ? '' : 's'} for retrieval.`) }
    if (failed.length) setError(failed.join(' · '))
    setUploading(false)
  }

  async function run(event) {
    event?.preventDefault(); const message = prompt.trim(); if (!message || loading) return
    setLoading(true); setError(''); setNotice(''); setSavedKey(''); setSavedUrl(''); setPrompt(''); setHistory((items) => [...items, { role: 'user', content: message }])
    const request = { session_id: sessionId, mode: activeMode === 'assistant' ? 'auto' : activeMode, message, use_model: useModel && modelReady }
    if (activeMode === 'rag') request.document_ids = selectedDocuments
    if (activeMode === 'drafter') { request.draft_action = 'revise'; request.draft_content = draft }
    try { const data = await api('/api/agents/run', { method: 'POST', body: JSON.stringify(request) }); setHistory(data.history || []); setDraft(data.document || draft); setResult(data); if (data.warning) setNotice(data.warning) } catch (err) { setError(err.message); setHistory((items) => items.slice(0, -1)) } finally { setLoading(false) }
  }

  async function reset() { setError(''); setNotice(''); setSavedUrl(''); try { await api(`/api/sessions/${sessionId}/reset`, { method: 'POST' }); setHistory([]); setDraft(''); setResult(null); setSavedKey('') } catch (err) { setError(err.message) } }
  async function saveHistory() { if (!history.length) return; try { const data = await api(`/api/sessions/${sessionId}/history/save`, { method: 'POST', body: JSON.stringify({ filename: historyName }) }); setNotice(`Saved ${data.saved_file}`); setSavedUrl(`/api/history/${encodeURIComponent(data.saved_file)}`) } catch (err) { setError(err.message) } }
  async function saveMessage(content, key) { try { const data = await api(`/api/sessions/${sessionId}/text/save`, { method: 'POST', body: JSON.stringify({ content, filename: `message-${Date.now()}.txt` }) }); setSavedKey(key); setNotice(`Saved ${data.saved_file}`); setSavedUrl(`/api/history/${encodeURIComponent(data.saved_file)}`) } catch (err) { setError(err.message) } }
  async function saveDraft() { try { const data = await api('/api/agents/run', { method: 'POST', body: JSON.stringify({ session_id: sessionId, mode: 'drafter', draft_action: 'save', draft_content: draft, draft_filename: draftName }) }); setResult(data); setDraft(data.document || draft); setNotice(`Saved ${data.saved_file}`); setSavedUrl(`/api/drafts/${encodeURIComponent(data.saved_file)}`) } catch (err) { setError(err.message) } }

  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><span className="brand-mark">AL</span><span><strong>Agentic Lang</strong><small>Local research workspace</small></span></div><div className="nav-label">Workspace</div><nav className="mode-nav" aria-label="Workspace modes">{MODES.map((item) => <button key={item.id} className={activeMode === item.id ? 'mode-link active' : 'mode-link'} onClick={() => setActiveMode(item.id)}><Icon name={item.icon}/><span><strong>{item.label}</strong><small>{item.hint}</small></span></button>)}</nav><div className="sidebar-foot"><div className="runtime"><span className={modelReady ? 'status-dot ready' : 'status-dot'}/><span><strong>{modelReady ? 'Ollama connected' : 'Local fallback active'}</strong><small>{health?.ollama?.configured_model || 'llama3.1'}</small></span></div><p className="version">FASTAPI · LANGGRAPH · OLLAMA</p></div></aside>
    <main className="main-frame"><header className="topbar"><div className="mobile-brand">AL</div><div className="breadcrumb"><span>Workspace</span><b>/</b><strong>{mode.label}</strong></div><div className="top-actions"><label className="ollama-toggle"><input type="checkbox" checked={useModel} onChange={(event) => setUseModel(event.target.checked)} disabled={!modelReady}/><span className="toggle-track"/><span>Ollama</span></label><button className="reset-link" onClick={reset}><Icon name="reset" size={14}/> New session</button></div></header>
      <div className="page"><header className="page-heading"><div><span className="eyebrow">{activeMode === 'rag' ? 'Grounded retrieval' : activeMode === 'web' ? 'Live research' : activeMode === 'drafter' ? 'Writing room' : 'Private AI workspace'}</span><h1>{mode.label}</h1><p>{activeMode === 'rag' ? 'Ask questions against selected PDF sources. Retrieved passages are passed into Ollama before it writes an answer.' : activeMode === 'web' ? 'Search the public web, review the sources, and ask Ollama to synthesize the result.' : activeMode === 'drafter' ? 'Shape a document through conversation, edit it directly, and export the text when it is ready.' : 'One calm place for local chat, grounded research, and practical writing.'}</p></div><div className="heading-status"><span className={modelReady ? 'status-dot ready' : 'status-dot'}/>{modelReady ? 'Model ready' : 'Fallback mode'}</div></header>
        {activeMode === 'rag' && <SourcePicker documents={documents} selected={selectedDocuments} onToggle={toggleDocument} onUpload={uploadPdf} uploading={uploading}/>}<section className={activeMode === 'drafter' ? 'workspace-grid with-draft' : 'workspace-grid'}><section className="chat-card"><div className="card-header"><div><span className="section-label">Conversation</span><strong>{history.length} message{history.length === 1 ? '' : 's'}</strong></div>{activeMode !== 'drafter' && <div className="header-note">{activeMode === 'rag' ? (result?.retrieval_method === 'ollama-semantic' ? 'Ollama semantic retrieval' : semanticReady ? 'Ollama embeddings ready' : 'Lexical fallback') : useModel && modelReady ? 'Ollama generation on' : 'Deterministic fallback'}</div>}</div><div className="messages">{!history.length && !loading && <div className="welcome"><div className="welcome-mark"><Icon name={mode.icon} size={25}/></div><h2>{activeMode === 'rag' ? 'Ask your documents' : activeMode === 'web' ? 'Find what is current' : activeMode === 'drafter' ? 'Start a document' : 'What are we working on?'}</h2><p>{activeMode === 'rag' ? 'Select a source above, then ask a precise question.' : activeMode === 'web' ? 'Try a topic, person, product, or current event.' : activeMode === 'drafter' ? 'Describe the first version or paste content into the editor.' : 'Ask a question or choose a focused mode from the left.'}</p><button onClick={() => setPrompt(activeMode === 'rag' ? 'Summarize the key points in this document.' : activeMode === 'web' ? 'Search the web for the latest developments in artificial intelligence.' : activeMode === 'drafter' ? 'Write a concise project update for a local AI research workspace.' : 'Explain how this workspace can help me.')}>Try an example <span>→</span></button></div>}{history.map((message, index) => <MessageRow key={`${message.role}-${index}`} message={message} index={index} onSave={saveMessage} savedKey={savedKey}/>)}{loading && <div className="thinking"><span/><span/><span/> Working with {mode.label.toLowerCase()}...</div>}{error && <div className="inline-alert error"><Icon name="alert"/><span>{error}</span></div>}{notice && <div className="inline-alert"><Icon name="check"/><span>{notice}{savedUrl && <a className="download-link" href={savedUrl}>Download</a>}</span></div>}<div ref={endRef}/></div><form className="composer" onSubmit={run}><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder={activeMode === 'drafter' ? 'Describe a change to the document...' : `Message ${mode.label.toLowerCase()}...`} rows="3" onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); run() } }}/><div className="composer-foot"><span>Enter to send · Shift + Enter for a new line</span><button className="send-button" disabled={loading || !prompt.trim() || (activeMode === 'rag' && !selectedDocuments.length)} aria-label="Send message"><Icon name="send" size={17}/></button></div></form></section>{activeMode === 'drafter' && <aside className="draft-card"><div className="card-header"><div><span className="section-label">Document</span><strong>Live draft</strong></div><span className="char-count">{draft.length} chars</span></div><textarea className="draft-editor" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Your document will appear here..."/><div className="export-controls"><label>Filename<input value={draftName} onChange={(event) => setDraftName(event.target.value)}/></label><button className="primary-button" onClick={saveDraft} disabled={!draft.trim()}><Icon name="download" size={14}/> Save draft</button></div></aside>}</section>{activeMode === 'web' && <WebResults results={result?.web_results}/>}<footer className="export-bar"><div><span className="section-label">Export</span><strong>Keep the useful parts</strong><small>Save the complete conversation or save any individual agent response above.</small></div><div className="export-action"><input value={historyName} onChange={(event) => setHistoryName(event.target.value)} aria-label="Chat history filename"/><button className="secondary-button" onClick={saveHistory} disabled={!history.length}><Icon name="download" size={14}/> Save chat history</button></div></footer></div></main>
  </div>
}
