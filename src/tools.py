import ast
import io
import contextlib
import concurrent.futures
import math
import statistics
import json as json_module
import re as re_module
from langchain_core.tools import tool
#from duckduckgo_search import DDGS

# ── Tool 1: Web Search (DuckDuckGo, free, no API key) ──────────────────
import datetime as _dt

try:
    from ddgs import DDGS  # the package was renamed from duckduckgo_search -> ddgs
except ImportError:
    from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    '''Search the live web for recent facts, news, papers, or events.
    Returns up to 4 results with title, a short snippet, and URL.'''
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        lines = [f"(search performed on {_dt.date.today().isoformat()} — use this for any 'latest'/'current' framing)"]
        for i, r in enumerate(results, 1):
            title = r.get('title', '')[:120]
            body = r.get('body', '')[:280]
            lines.append(f"[{i}] {title}\n{body}\nURL: {r.get('href', '')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Web search error: {e}"

# ── Tool 2: RAG Retrieval (local ChromaDB corpus) ───────────────────────
@tool
def rag_retrieval(query: str) -> str:
    '''Retrieve relevant passages from the local research knowledge base
    (curated notes on agentic-AI, RAG, transformers, evaluation, etc.).'''
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in local corpus."
    out = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        out.append(f"[{i}] Source: {source}\n{doc.page_content[:600]}")
    return "\n\n".join(out)

# ── Tool 3: Code Execution (AST allow-list + restricted globals + timeout)
_ALLOWED_NODE_TYPES = (
    ast.Module, ast.Expr, ast.Load, ast.Store, ast.Name, ast.Constant,
    ast.Assign, ast.AugAssign, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Index,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Slice,
    ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass,
    ast.FunctionDef, ast.Return, ast.Lambda, ast.arguments, ast.arg,
    ast.comprehension, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.USub, ast.UAdd, ast.Is, ast.IsNot,
    ast.JoinedStr, ast.FormattedValue, ast.IfExp, ast.Starred,
)
_FORBIDDEN_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "input", "globals",
    "locals", "vars", "dir", "getattr", "setattr", "delattr", "__builtins__",
    "breakpoint", "help",
}

def _ast_is_safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Imports are not allowed inside code_executor."
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return False, f"Disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"Use of '{node.id}' is not allowed."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "Access to dunder attributes is not allowed."
    return True, ""

import builtins as _builtins_module
_ALLOWED_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "float",
    "int", "len", "list", "max", "min", "print", "range",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "type",
]
_SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _ALLOWED_BUILTIN_NAMES}

def _run_code(code: str) -> str:
    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math, "statistics": statistics,
        "json": json_module, "re": re_module,
    }
    safe_locals: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<code_executor>", "exec"), safe_globals, safe_locals)
    out = buf.getvalue()
    if "result" in safe_locals:
        out += f"\nresult = {safe_locals['result']!r}"
    return out or "(code ran with no output — assign to a variable named `result` to return a value)"

@tool
def code_executor(code: str) -> str:
    '''Execute a restricted subset of Python (no imports, no file/network/
    OS access) for calculations, data transforms, or quick analysis.
    Assign your answer to a variable named `result` to have it returned.
    Times out after 5 seconds.'''
    safe, reason = _ast_is_safe(code)
    if not safe:
        return f"Rejected: {reason}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_code, code)
        try:
            return future.result(timeout=5)
        except concurrent.futures.TimeoutError:
            return "Execution error: timed out after 5 seconds."
        except Exception as e:
            return f"Execution error: {e}"

# ── Tool 4: Summariser (provider-agnostic LLM via build_llm()) ─────────
_summary_llm = build_llm(max_tokens=512, disable_thinking=True)

@tool
def summarise_text(text: str, style: str = "bullet") -> str:
    '''Summarise a long piece of text. style: 'bullet' | 'paragraph' | 'table'.'''
    prompt = (
        f"Summarise the following text in {style} format. "
        f"Be concise and accurate:\n\n{text[:4000]}"
    )
    return _summary_llm.invoke(prompt).content

TOOLS = [web_search, rag_retrieval, code_executor, summarise_text]
print(f" {len(TOOLS)} tools ready: {[t.name for t in TOOLS]}")
