CURATED_CORPUS = [
    {"id": "doc01", "title": "Transformers & Self-Attention",
     "text": "The Transformer architecture replaced recurrence with self-attention, letting every token attend to every other token in a sequence directly. This parallelism made training on much larger datasets practical and is the backbone of nearly all modern large language models."},
    {"id": "doc02", "title": "Attention Is All You Need (concept)",
     "text": "Multi-head attention splits the attention computation into several parallel heads, each learning to focus on different relationships in the input (e.g. syntax vs. coreference), then concatenates the results. This gives the model multiple complementary 'views' of the same sequence."},
    {"id": "doc03", "title": "Retrieval-Augmented Generation (RAG)",
     "text": "RAG combines a retriever (typically a vector search over an embedding index) with a generator LLM. Instead of relying purely on parametric memory, the model is given retrieved passages as context, which reduces hallucination and lets it cite up-to-date or domain-specific sources."},
    {"id": "doc04", "title": "Dense Embeddings for Retrieval",
     "text": "Dense retrieval encodes text into fixed-length vectors such that semantically similar passages land close together in vector space. Compact sentence-embedding models like MiniLM can run on CPU and still give strong retrieval quality for moderate-sized corpora."},
    {"id": "doc05", "title": "ReAct: Reason + Act",
     "text": "The ReAct pattern interleaves chain-of-thought reasoning steps with discrete tool-calling actions. The model reasons about what it needs, calls a tool to get it, observes the result, and repeats — which grounds reasoning in real, verifiable information rather than pure recall."},
    {"id": "doc06", "title": "Chain-of-Thought Prompting",
     "text": "Chain-of-thought prompting asks a model to produce intermediate reasoning steps before its final answer. This measurably improves performance on multi-step arithmetic, logic, and planning tasks compared to asking for a direct answer."},
    {"id": "doc07", "title": "Tool Use / Function Calling in LLMs",
     "text": "Modern LLMs can be given structured tool schemas (name, description, JSON parameters) and choose to emit a structured call instead of free text. The host application executes the real function and feeds the result back, letting the model interact with live systems and data."},
    {"id": "doc08", "title": "LangGraph State Machines",
     "text": "LangGraph models an agent as a directed graph of nodes (each a function or LLM call) connected by conditional edges over a shared, typed state object. This makes branching, looping, and multi-agent hand-offs explicit and debuggable compared to a single monolithic prompt."},
    {"id": "doc09", "title": "Supervisor / Multi-Agent Patterns",
     "text": "In a supervisor pattern, one routing agent decomposes an incoming task and delegates sub-tasks to specialised worker agents (e.g. search, coding, summarising), then synthesises their outputs into one coherent final answer."},
    {"id": "doc10", "title": "Vector Databases",
     "text": "Vector databases such as Chroma, FAISS, and Pinecone index high-dimensional embeddings and support fast approximate nearest-neighbour search, which is the core primitive behind semantic search and RAG retrieval at scale."},
    {"id": "doc11", "title": "Maximal Marginal Relevance (MMR) Retrieval",
     "text": "MMR search re-ranks retrieved candidates to balance relevance against diversity, penalising results that are too similar to ones already selected. This avoids returning five near-duplicate passages and surfaces more varied supporting evidence."},
    {"id": "doc12", "title": "Hallucination in LLMs",
     "text": "Hallucination refers to a language model generating fluent but factually incorrect or unsupported statements. Grounding generation in retrieved documents, citing sources, and instructing the model to say 'I don't know' are common mitigation strategies."},
    {"id": "doc13", "title": "Reinforcement Learning from Human Feedback (RLHF)",
     "text": "RLHF fine-tunes a base language model using a reward model trained on human preference comparisons between candidate outputs, then optimises the policy (often via PPO) to produce responses humans rate as more helpful and safe."},
    {"id": "doc14", "title": "Context Window and Long-Context Models",
     "text": "The context window is the maximum number of tokens a model can attend to in a single call. Larger context windows let an agent keep more conversation history or retrieved evidence in view, but very long contexts can still suffer from 'lost in the middle' attention degradation."},
    {"id": "doc15", "title": "Streaming Token Generation",
     "text": "Streaming returns generated tokens incrementally as they are produced rather than waiting for the full response, dramatically lowering perceived latency for interactive applications and enabling UIs to render partial answers in real time."},
    {"id": "doc16", "title": "Server-Sent Events (SSE)",
     "text": "SSE is a simple one-directional protocol over plain HTTP where a server keeps a connection open and pushes 'data:' formatted events to the client. It is lighter-weight than WebSockets and is the standard transport for streaming LLM token output to a browser."},
    {"id": "doc17", "title": "Sentence-Transformers / MiniLM",
     "text": "MiniLM is a distilled, compact transformer that preserves much of the quality of larger encoder models while being far smaller and faster, making it practical to embed thousands of documents on CPU within seconds."},
    {"id": "doc18", "title": "Tokenisation",
     "text": "Tokenisation breaks raw text into sub-word units (e.g. via byte-pair encoding) that a model's vocabulary can represent. Tokenisation choices affect both context-window usage and how well a model handles rare words, code, or non-English text."},
    {"id": "doc19", "title": "Knowledge Cutoffs and Web Search Tools",
     "text": "Because a language model's parametric knowledge is frozen at training time, agents are commonly given a live web-search tool so they can answer questions about events, prices, or releases that postdate training."},
    {"id": "doc20", "title": "Prompt Injection",
     "text": "Prompt injection occurs when untrusted content (a web page, document, or tool output) contains text crafted to hijack an agent's instructions. Agentic systems should treat retrieved or tool-returned content as data, not as commands to obey."},
    {"id": "doc21", "title": "Sandboxed Code Execution",
     "text": "Letting an LLM agent run arbitrary code is powerful but risky; safe designs restrict available modules and builtins, disallow filesystem and network access, and enforce a hard execution timeout so a runaway or malicious snippet cannot hang or compromise the host."},
    {"id": "doc22", "title": "Few-Shot Learning",
     "text": "Few-shot prompting provides a handful of input-output examples directly in the prompt, letting a model infer the desired task format without any gradient updates — useful for steering style or output structure on the fly."},
    {"id": "doc23", "title": "Fine-Tuning vs. In-Context Learning",
     "text": "Fine-tuning updates a model's weights on task-specific data, while in-context learning (prompting, RAG, few-shot examples) adapts behaviour purely through input text at inference time. The latter is cheaper to iterate on and is the dominant approach for most agentic applications."},
    {"id": "doc24", "title": "Multi-Hop Question Answering",
     "text": "Multi-hop questions require chaining several individually retrievable facts together (e.g. 'who directed the film that won the award the year X happened') rather than answering from a single passage, which is exactly what tool-using agentic systems are designed to handle well."},
    {"id": "doc25", "title": "Groq LPU Inference",
     "text": "Groq's Language Processing Unit (LPU) is purpose-built hardware for deterministic, low-latency transformer inference, offering substantially higher tokens-per-second throughput than typical GPU serving for many open-weight chat models."},
    {"id": "doc26", "title": "Open-Weight Model Ecosystem",
     "text": "Open-weight model families (Llama, Mistral, Qwen, GPT-OSS, Gemma) can be self-hosted or accessed through low-cost inference providers, removing the per-token cost and vendor lock-in of closed proprietary APIs for many production use cases."},
    {"id": "doc27", "title": "Evaluation via LLM-as-Judge",
     "text": "LLM-as-judge evaluation uses a separate (often stronger) model to score a system's outputs against a reference answer or rubric, trading some precision against ground-truth human grading for much greater scalability and speed. Reasoning models can spend their output budget on hidden chain-of-thought, so a judge given too small a token budget can fail silently rather than erroring."},
     (frosted-glass panels, blur, soft gradients) is one stylistic choice among many for a chat interface; for a research-tool product, a design language that visually echoes the subject matter — citations, margins, footnotes — can communicate the tool's purpose more directly than a generic dark-glass aesthetic."},
]

print(f"📚 Curated corpus: {len(CURATED_CORPUS)} documents (always available, zero network dependency).")

# --- Best-effort: add a real external dataset on top ----------------------
import concurrent.futures as _cf

def _load_real_arxiv_abstracts(n: int = 60):
    from datasets import load_dataset
    ds = load_dataset("ccdv/arxiv-summarization", split="test", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        abstract = (row.get("abstract") or "").strip()
        if len(abstract) < 200:  # skip near-empty/degenerate rows
            continue
        out.append({
            "id": f"arxiv_{i}",
            "title": f"arXiv paper {i} (ccdv/arxiv-summarization test split)",
            "text": abstract[:1200],
        })
    return out

REAL_CORPUS = []
try:
    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
        REAL_CORPUS = _pool.submit(_load_real_arxiv_abstracts, 60).result(timeout=25)
    print(f" Loaded {len(REAL_CORPUS)} real abstracts from HuggingFace `ccdv/arxiv-summarization`.")
except _cf.TimeoutError:
    print(" Real-dataset load timed out after 25s (slow/no internet?) — continuing on the curated corpus only.")
except Exception as e:
    print(f" Real-dataset load failed ({type(e).__name__}: {e}) — continuing on the curated corpus only.")

RESEARCH_CORPUS = CURATED_CORPUS + REAL_CORPUS
print(f" Final research corpus: {len(RESEARCH_CORPUS)} documents "
      f"({len(CURATED_CORPUS)} curated + {len(REAL_CORPUS)} real).")
