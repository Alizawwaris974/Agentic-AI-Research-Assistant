# Agentic AI Research Assistant — Full Implementation Plan

## Goal
Build a **fully functional Kaggle Python notebook** implementing a multi-agent AI research assistant with LangGraph ReAct loop, deploy it via **FastAPI + ngrok tunnel**, connect to **n8n** for a chat-based UI, and include a **standalone interactive HTML chat UI** as a creative fallback.

---

## Key Design Decisions (from your answers)

| Decision | Choice |
|---|---|
| **LLM Provider** | Groq (free tier, LLaMA-3.1-70B) |
| **RAG Domain** | AI/ML research papers (arXiv abstracts) |
| **Web Search** | DuckDuckGo (free, no API key) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (free, runs on CPU) |
| **n8n Integration** | Chat Trigger → HTTP Request → FastAPI backend |
| **Kaggle Setup** | API keys via Kaggle Secrets |

---

## Gap Analysis: Original Spec vs. Implementation

> [!WARNING]
> ### Critical gaps identified in the original spec that this plan addresses:

| Gap | Issue | Fix |
|---|---|---|
| **Paid APIs everywhere** | Spec uses OpenAI embeddings ($), Tavily search ($), GPT-4o ($) | → Replace with HuggingFace embeddings, DuckDuckGo search, Groq (all free) |
| **No dataset provided** | RAG tool references "uploaded PDFs, papers" but provides no data | → Embed 500 arXiv CS/AI abstracts from HuggingFace `ccdv/arxiv-summarization` |
| **Code executor is unsafe** | `exec()` with basic string blacklist is trivially bypassed | → Add AST-level safety checking + timeout + restricted globals |
| **No CORS** | FastAPI has no CORS middleware — n8n chat widget will fail | → Add `CORSMiddleware` |
| **Session memory won't work with `create_react_agent`** | Manual `SESSION_STORE` dict is fragile | → Use LangGraph's built-in `MemorySaver` checkpointer with `thread_id` |
| **Streaming is broken** | SSE endpoint collects tokens wrong; misses final answer | → Fix token collection + proper SSE formatting |
| **No n8n workflow** | Spec mentions Streamlit but no n8n integration | → Generate importable n8n workflow JSON |
| **No UI** | Spec has no frontend | → Build a beautiful standalone HTML chat UI with glassmorphism design |
| **Evaluation is impractical** | 200-question benchmark requires expensive API calls | → Create a 20-question mini-benchmark that runs quickly |
| **`iteration_count` not in prebuilt state** | Custom state field breaks `create_react_agent` | → Use `recursion_limit` config parameter instead (built-in) |

---

## Proposed Changes

### Deliverable 1: Kaggle Notebook

#### [NEW] [agentic_ai_research_assistant.ipynb](file:///c:/Users/USER/Desktop/Portfolio%20Projects/agentic%20AI%20RA/agentic_ai_research_assistant.py)

> [!NOTE]
> Created as a `.py` file with Kaggle notebook cell markers (`# %%`) for easy conversion. Can be uploaded directly to Kaggle as a script or converted to `.ipynb`.

**Cell structure:**

| Cell | Content |
|---|---|
| **Cell 1: Setup** | `pip install` all dependencies (langchain, langgraph, langchain-groq, langchain-community, chromadb, sentence-transformers, duckduckgo-search, fastapi, uvicorn, pyngrok, nest_asyncio) |
| **Cell 2: Configuration** | API key loading from Kaggle Secrets / environment, model configuration |
| **Cell 3: Dataset Loading** | Load 500 arXiv AI/ML abstracts from HuggingFace `ccdv/arxiv-summarization` dataset |
| **Cell 4: RAG Setup** | HuggingFace embeddings → ChromaDB vectorstore → populate with arXiv abstracts |
| **Cell 5: Tool Definitions** | 4 tools: DuckDuckGo search, RAG retrieval, code executor (with AST safety), summariser (Groq) |
| **Cell 6: LangGraph Agent** | `create_react_agent` with Groq LLM, all 4 tools, system prompt, `MemorySaver` checkpointer |
| **Cell 7: Test the Agent** | Run 3 sample research queries, display results with tool call traces |
| **Cell 8: Mini Evaluation** | 20-question benchmark with LLM-judge scoring |
| **Cell 9: FastAPI App** | Full FastAPI with `/research` (POST), `/research/stream` (SSE), `/health`, `/session/{id}/history`, CORS enabled |
| **Cell 10: HTML Chat UI** | Embedded interactive HTML UI served at `/` — glassmorphism design, streaming support, session management |
| **Cell 11: Launch Server** | ngrok tunnel + uvicorn launch, print public URL for n8n webhook |

---

### Deliverable 2: n8n Workflow

#### [NEW] [n8n_workflow.json](file:///c:/Users/USER/Desktop/Portfolio%20Projects/agentic%20AI%20RA/n8n_workflow.json)

Importable n8n workflow with:
- **Chat Trigger** node (provides built-in chat widget UI)
- **HTTP Request** node (calls FastAPI `/research` endpoint)
- **Set** node (maps `answer` → `output` for chat display)
- User just needs to:
  1. Import the JSON into n8n
  2. Replace the URL with their ngrok/server URL
  3. Activate the workflow

---

### Deliverable 3: Standalone HTML Chat UI

#### Embedded within FastAPI (Cell 10)

A premium, interactive chat interface with:
- 🎨 **Glassmorphism design** with dark mode, gradient backgrounds, blur effects
- ⚡ **Real-time SSE streaming** — tokens appear as they're generated
- 🔧 **Tool call indicators** — shows which tools the agent is using (search, RAG, code, summarise)
- 💬 **Session memory** — conversation persists across messages
- 📱 **Fully responsive** — works on mobile and desktop
- ✨ **Micro-animations** — typing indicators, message fade-ins, tool call pulses
- 🎯 **Suggested queries** — clickable starter questions

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    KAGGLE NOTEBOOK                                │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐        │
│  │ arXiv Data  │──→│ ChromaDB     │   │ Groq LLM      │        │
│  │ 500 papers  │   │ (embeddings) │   │ LLaMA-3.1-70B │        │
│  └─────────────┘   └──────┬───────┘   └───────┬───────┘        │
│                           │                   │                  │
│  ┌────────────────────────┴───────────────────┴──────────┐      │
│  │              LangGraph ReAct Agent                     │      │
│  │  ┌──────────┐ ┌─────┐ ┌──────┐ ┌───────────┐        │      │
│  │  │DuckDuckGo│ │ RAG │ │ Code │ │ Summariser│        │      │
│  │  │ Search   │ │ Tool│ │ Exec │ │    Tool   │        │      │
│  │  └──────────┘ └─────┘ └──────┘ └───────────┘        │      │
│  │              + MemorySaver (session memory)            │      │
│  └────────────────────────┬──────────────────────────────┘      │
│                           │                                      │
│  ┌────────────────────────┴──────────────────────────────┐      │
│  │              FastAPI Server (:8000)                     │      │
│  │  POST /research  │  GET /research/stream (SSE)         │      │
│  │  GET /  (HTML UI) │  GET /session/{id}/history          │      │
│  └────────────────────────┬──────────────────────────────┘      │
│                           │                                      │
│  ┌────────────────────────┴──────────┐                          │
│  │         ngrok tunnel              │                          │
│  │  https://xxxx.ngrok-free.app      │                          │
│  └────────────────────────┬──────────┘                          │
└───────────────────────────┼──────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
     ┌─────────────┐ ┌───────────┐  ┌──────────────┐
     │  n8n Chat   │ │ Standalone│  │  Direct API  │
     │  Widget     │ │ HTML UI   │  │  Calls       │
     │  (workflow) │ │ (browser) │  │  (Postman)   │
     └─────────────┘ └───────────┘  └──────────────┘
```

---

## Open Questions

> [!IMPORTANT]
> ### Do you have an ngrok auth token?
> The notebook uses ngrok to create a public tunnel so n8n can reach the FastAPI server running on Kaggle. You can get a **free** ngrok token at [ngrok.com](https://ngrok.com). If you don't want to use ngrok, I can use Cloudflare's free tunnel instead (no account needed, but less reliable).

> [!IMPORTANT]
> ### Where is your n8n instance hosted?
> The n8n workflow needs to know the FastAPI URL. Options:
> - **n8n Cloud** (hosted by n8n.io) — just import the workflow and update the URL
> - **Self-hosted n8n** (Docker/local) — same, but the ngrok URL changes each notebook restart
> - For production, you'd deploy FastAPI on a persistent server (Railway, Render, etc.)

---

## Verification Plan

### Automated Tests (within notebook)
1. **Tool unit tests**: Each tool tested individually with sample inputs
2. **Agent integration test**: 3 sample queries run through full agent pipeline, verify tool calls happen and answers are coherent
3. **Mini-benchmark**: 20 multi-hop questions scored by LLM judge, target ≥ 75% accuracy
4. **API endpoint test**: Health check + POST to `/research` + verify JSON response

### Manual Verification
1. Open the standalone HTML UI at the ngrok URL → send queries → verify streaming works
2. Import n8n workflow → activate → test chat widget → verify responses appear
3. Test session memory: ask follow-up questions that reference prior answers
