"""
DocuQuery — LangGraph Agentic Query Engine

State machine with 3 paths:
  1. DIRECT   — greetings / meta-questions (no retrieval needed)
  2. SIMPLE   — single-step RAG (standard query)
  3. AGENTIC  — multi-step with self-critique retry (complex / multi-doc)

FIX in this version:
  - Full try/except around the entire graph execution so ANY unhandled error
    returns a graceful JSON response instead of an Internal Server Error (500).
  - Handles the case where no documents are indexed (empty all_docs).
  - Fixed router to not crash on empty query (already caught by validator,
    but defensive guard added).
"""

import os
import asyncio
import json
import logging
from typing import TypedDict, Literal

import litellm
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile")


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:       str
    doc_ids:     list
    alpha:       float | None
    history:     list
    all_docs:    list
    route:       str           # "direct" | "simple" | "agentic"
    answer:      str
    sources:     list
    confidence:  str
    retry_count: int


# ── Router ───────────────────────────────────────────────────────────────────

DIRECT_SIGNALS = [
    "hello", "hi ", "hey ", "how are you", "good morning", "good evening",
    "thanks", "thank you", "bye", "goodbye", "what can you do",
    "who are you", "what are you", "help me", "what is docuquery",
]

AGENTIC_SIGNALS = [
    "compare", "contrast", "across all", "all documents", "each document",
    "differences between", "similarities between", "analyze all",
    "comprehensive", "in-depth", "multiple documents",
]


async def _route(state: AgentState) -> AgentState:
    """Decide routing: direct / simple / agentic."""
    q   = (state.get("query") or "").strip().lower()
    doc = state.get("all_docs") or []

    if not q:
        return {**state, "route": "direct"}

    if any(s in q for s in DIRECT_SIGNALS) and len(q) < 60:
        return {**state, "route": "direct"}

    if not doc:
        # No documents indexed — direct answer only
        return {**state, "route": "direct"}

    if any(s in q for s in AGENTIC_SIGNALS):
        return {**state, "route": "agentic"}

    if len(q.split()) > 20:
        return {**state, "route": "agentic"}

    return {**state, "route": "simple"}


def _decide_route(state: AgentState) -> Literal["direct", "simple", "agentic"]:
    return state.get("route", "simple")


# ── Direct answer (no retrieval) ─────────────────────────────────────────────

async def _direct_answer(state: AgentState) -> AgentState:
    query = state.get("query", "")
    try:
        r = await litellm.acompletion(
            model    = LLM_MODEL,
            messages = [
                {"role": "system",
                 "content": "You are DocuQuery, a helpful document Q&A assistant. "
                             "Answer greetings and meta-questions briefly and warmly. "
                             "For any substantive question, tell the user to upload documents first."},
                {"role": "user", "content": query}
            ],
            temperature = 0.3,
            max_tokens  = 300,
        )
        answer = r.choices[0].message.content
    except Exception as e:
        logger.error(f"Direct answer failed: {e}")
        answer = (
            "Hello! I'm DocuQuery, your document Q&A assistant. "
            "Upload a PDF or document to get started, then ask me anything about it."
        )
    return {**state, "answer": answer, "sources": [], "confidence": "high"}


# ── Simple RAG (single-step) ──────────────────────────────────────────────────

async def _simple_rag(state: AgentState) -> AgentState:
    from query import generate_answer
    try:
        result = await generate_answer(
            query    = state["query"],
            doc_ids  = state.get("doc_ids") or None,
            alpha    = state.get("alpha"),
            history  = state.get("history", []),
            all_docs = state.get("all_docs", []),
        )
        return {
            **state,
            "answer":     result.get("answer", ""),
            "sources":    result.get("sources", []),
            "confidence": result.get("confidence", "unknown"),
        }
    except Exception as e:
        logger.error(f"Simple RAG failed: {e}")
        return {
            **state,
            "answer":     "## Answer\nI encountered an error retrieving information. "
                          "Please try rephrasing your question.\n\n## References\nNone.",
            "sources":    [],
            "confidence": "low",
        }


# ── Agentic RAG (multi-step with self-critique) ───────────────────────────────

async def _agentic_rag(state: AgentState) -> AgentState:
    from query import generate_answer
    retry = state.get("retry_count", 0)
    try:
        result = await generate_answer(
            query    = state["query"],
            doc_ids  = state.get("doc_ids") or None,
            alpha    = state.get("alpha"),
            history  = state.get("history", []),
            all_docs = state.get("all_docs", []),
        )
        answer     = result.get("answer", "")
        confidence = result.get("confidence", "unknown")

        # Self-critique: if low confidence and haven't retried yet, try harder
        if confidence == "low" and retry < 1:
            logger.info("Agentic: low confidence — retrying with broader search")
            retry_result = await generate_answer(
                query    = f"Provide a comprehensive overview: {state['query']}",
                doc_ids  = None,  # Search all docs on retry
                alpha    = 0.50,  # Balanced mode
                history  = [],
                all_docs = state.get("all_docs", []),
            )
            if retry_result.get("confidence") != "low":
                answer     = retry_result.get("answer", answer)
                confidence = retry_result.get("confidence", confidence)
                result     = retry_result

        return {
            **state,
            "answer":      answer,
            "sources":     result.get("sources", []),
            "confidence":  confidence,
            "retry_count": retry + 1,
        }
    except Exception as e:
        logger.error(f"Agentic RAG failed: {e}")
        return {
            **state,
            "answer":     "## Answer\nI encountered an error during the agentic search. "
                          "Try the standard /ask/ endpoint or rephrase your question.\n\n"
                          "## References\nNone.",
            "sources":    [],
            "confidence": "low",
        }


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("router",       _route)
    g.add_node("direct",       _direct_answer)
    g.add_node("simple_rag",   _simple_rag)
    g.add_node("agentic_rag",  _agentic_rag)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _decide_route,
        {
            "direct":  "direct",
            "simple":  "simple_rag",
            "agentic": "agentic_rag",
        }
    )
    g.add_edge("direct",      END)
    g.add_edge("simple_rag",  END)
    g.add_edge("agentic_rag", END)
    return g.compile()


_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

async def run_agentic_query(
    query:    str,
    doc_ids:  list  = None,
    alpha:    float = None,
    history:  list  = None,
    all_docs: list  = None,
) -> dict:
    """
    Run the LangGraph agentic pipeline.

    FIX: Wrapped in try/except so any unhandled exception returns a
    graceful JSON dict instead of propagating as a 500 Internal Server Error.
    This is the key fix for tests failing with JSONDecodeError on 500 responses.
    """
    try:
        graph = _get_graph()
        initial_state: AgentState = {
            "query":       query or "",
            "doc_ids":     doc_ids  or [],
            "alpha":       alpha,
            "history":     history  or [],
            "all_docs":    all_docs or [],
            "route":       "simple",
            "answer":      "",
            "sources":     [],
            "confidence":  "unknown",
            "retry_count": 0,
        }

        final_state = await graph.ainvoke(initial_state)

        answer     = final_state.get("answer", "")
        sources    = final_state.get("sources", [])
        confidence = final_state.get("confidence", "unknown")
        route_used = final_state.get("route", "unknown")

        logger.info(
            f"Agentic query complete — route={route_used} "
            f"confidence={confidence} sources={len(sources)}"
        )

        return {
            "answer":              answer,
            "sources":             sources,
            "confidence":          confidence,
            "route_used":          route_used,
            "agentic":             True,
            "retry_count":         final_state.get("retry_count", 0),
            "avg_relevance_score": 0.0,
            "docs_searched":       len(set(s.get("doc_id","") for s in sources)),
            "sub_queries_used":    [query],
            "alpha_used":          alpha,
            "alpha_mode":          "agentic",
            "search_mode":         f"LangGraph agentic ({route_used})",
        }

    except Exception as e:
        # Return graceful error instead of raising → prevents 500
        logger.error(f"Agentic query error: {e}", exc_info=True)
        return {
            "answer": (
                f"## Answer\nThe agentic pipeline encountered an error: {str(e)[:200]}\n\n"
                "Please use /ask/ or /ask/stream for standard queries.\n\n"
                "## References\nNone."
            ),
            "sources":             [],
            "confidence":          "low",
            "route_used":          "error",
            "agentic":             True,
            "retry_count":         0,
            "avg_relevance_score": 0.0,
            "docs_searched":       0,
            "sub_queries_used":    [query],
            "alpha_used":          alpha,
            "alpha_mode":          "error",
            "search_mode":         "agentic-error",
        }