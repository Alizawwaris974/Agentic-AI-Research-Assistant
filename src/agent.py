from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from src.config import AGENT_MODEL
from src.tools import TOOLS, build_llm

llm = build_llm(streaming=True, max_tokens=1024, with_retry=False)

SYSTEM_PROMPT = """You are an expert AI research assistant..."""  # full prompt

memory = MemorySaver()
agent_graph = create_react_agent(llm, tools=TOOLS, prompt=SYSTEM_PROMPT, checkpointer=memory)

def get_message_text(content):
    # Your helper function from notebook
    if isinstance(content, str):
        return content
    # handle list case etc.
    return str(content)