#  Agentic AI Research Assistant

**A powerful multi-agent research system built with LangGraph (ReAct), Gemini + Groq fallback, local RAG, safe code execution, and streaming UI.**

Designed for deep AI/ML research — it can search the web, query a curated knowledge base, run safe Python code, and summarize content. Fully deployable with **n8n** workflows.

---

##  Features

- **ReAct Agent** powered by LangGraph (`create_react_agent`)
- **Smart LLM Switching**: Gemini 2.5 Flash (default) with automatic fallback to Groq
- **Tools**:
  - Web Search (DuckDuckGo)
  - Local RAG (96 curated + real arXiv abstracts)
  - Safe sandboxed Code Execution (AST + timeout)
  - Intelligent Summarizer
- **Beautiful "Field Notes" UI** — research notebook aesthetic with live citation rail
- **Streaming responses** via Server-Sent Events (SSE)
- **Persistent sessions** via LangGraph MemorySaver
- **n8n Ready** — includes ready-to-import workflow

---

## 📁 Project Structure
```
agentic-research-assistant/
├── main.py                    # FastAPI backend (for Render/Railway)
├── requirements.txt
├── chat_ui.html
├── n8n_workflow.json
├── README.md
└── src/                       # Modular code from notebook
    ├── __init__.py
    ├── config.py
    ├── corpus.py
    ├── tools.py
    ├── agent.py
    └── utils.py
```

## 🚀 Quick Start (Local / Kaggle)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/agentic-research-assistant.git
cd agentic-research-assistant
2. Install Dependencies
Bashpip install -r requirements.txt
3. Set API Keys
Create a .env file:
envGOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key          # Optional but recommended as fallback
NGROK_AUTH_TOKEN=your_ngrok_token   # Optional
4. Run the Server
Bashpython main.py
Open the printed public URL in your browser to access the Field Notes UI.
```

## Production Deployment (Recommended)

For persistent use with n8n, deploy the backend on a real hosting platform:
Recommended Platforms (Free Tier Available)

Render.com (easiest)
Railway.app
Fly.io
Hugging Face Spaces (with Docker)

Deploy on Render.com (Step-by-step)

Push this repo to GitHub
Go to render.com → New Web Service
Connect your GitHub repo
Set these settings:
Name: agentic-research-assistant
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

Add Environment Variables (from your .env)
Deploy

After deployment, you will get a stable URL like:
https://agentic-research-assistant.onrender.com

## 🔗 Integration with n8n (Step-by-Step)
Method 1: Import Ready Workflow (Recommended)

Download n8n_workflow.json from this repo
Open your n8n instance → Import from File
Import the workflow
Edit the "Call FastAPI /research" node:
Replace the URL with your deployed backend URL:texthttps://your-app.onrender.com/research

Activate the workflow
Use n8n’s Chat Trigger to talk to your agent

Method 2: Manual Setup

Create a new workflow
Add Chat Trigger node
Add HTTP Request node:
Method: POST
URL: https://your-app.onrender.com/research
Body Content Type: JSON
Body Parameters:JSON{
  "query": "={{ $json.chatInput }}",
  "session_id": "={{ $json.sessionId || $now }}"
}

Add Set node to map $json.answer → output
Connect nodes and activate



## Tech Stack

LLM: Gemini 2.5 Flash + Groq fallback
Framework: LangGraph + LangChain
Vector Store: ChromaDB
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Backend: FastAPI + Uvicorn
UI: Custom HTML/CSS/JS (Field Notes design)
Workflow: n8n
Tunneling: ngrok / Cloudflare Tunnel


## Configuration Tips

Use Groq as primary if you hit Gemini quota limits frequently
Add NGROK_AUTH_TOKEN for more stable tunnels on Kaggle
The agent has automatic fallback between Gemini and Groq


## License
MIT License
