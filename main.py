# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import uuid
from langchain_core.messages import HumanMessage

from src.config import LLM_PROVIDER, AGENT_MODEL
from src.agent import agent_graph, get_message_text
from src.tools import TOOLS

app = FastAPI(title="Agentic AI Research Assistant")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": AGENT_MODEL,
        "tools": [t.name for t in TOOLS]
    }

@app.post("/research")
async def research_endpoint(req: QueryRequest):
    sid = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": sid}, "recursion_limit": 25}
    
    try:
        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=req.query)]}, 
            config=config
        )
        answer = get_message_text(result["messages"][-1].content)
        return {"session_id": sid, "answer": answer}
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"session_id": sid, "error": str(e), "answer": None}
        )

@app.get("/research/stream")
async def research_stream(query: str, session_id: str | None = None):
    sid = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": sid}, "recursion_limit": 25}

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'session_id': sid})}\\n\\n"
        final_chunks = []
        try:
            async for event in agent_graph.astream_events(
                {"messages": [HumanMessage(content=query)]},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    content = get_message_text(event["data"]["chunk"].content)
                    if content:
                        final_chunks.append(content)
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\\n\\n"
                elif event["event"] == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': event['name']})}\\n\\n"
                elif event["event"] == "on_tool_end":
                    preview = get_message_text(event.get("data", {}).get("output", ""))[:200]
                    yield f"data: {json.dumps({'type': 'tool_done', 'tool': event['name'], 'preview': preview})}\\n\\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"

        yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'answer': ''.join(final_chunks)})}\\n\\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Serve beautiful UI
with open("chat_ui.html", "r") as f:
    CHAT_UI_HTML = f.read()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return CHAT_UI_HTML




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
