"""
pytest feature verification suite for DocuQuery v2 additions.

IMPORTANT: Run with pytest, not directly with python:
    pytest tests/test_features_v2.py -v

Or use the project runner:
    python run_tests.py features
"""
import sys
import os

# Fix Windows import paths — ensures project modules (query, utils, etc.)
# are importable regardless of which directory pytest is called from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import httpx
import json
import asyncio

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session", autouse=True)
def check_server():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("API server not healthy")
    except Exception:
        pytest.skip(
            "API server not reachable. Start it first:\n"
            "  uvicorn main:app --reload --port 8000"
        )


async def _stream(payload, timeout=45):
    answer = ""
    metadata = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{BASE_URL}/ask/stream", json=payload) as r:
            status = r.status_code
            if status == 200:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        token = data.get("token", "")
                        if "__METADATA__" in token:
                            metadata = json.loads(token.split("__METADATA__", 1)[1])
                        else:
                            answer += token
                    except json.JSONDecodeError:
                        pass
            return {"status": status, "answer": answer, "metadata": metadata}


def _get_any_doc_id():
    r = httpx.get(f"{BASE_URL}/documents/", timeout=15)
    docs = r.json().get("documents", [])
    return docs[0]["doc_id"] if docs else None


# ── Image pipeline ────────────────────────────────────────────────────────────

def test_image_extractor_module_loads():
    """image_extractor.py must be importable with all expected functions."""
    import image_extractor
    assert hasattr(image_extractor, "extract_images_from_pdf")
    assert hasattr(image_extractor, "get_images_for_doc")
    assert hasattr(image_extractor, "delete_images_for_doc")


def test_documents_images_endpoint_exists():
    """GET /documents/{id}/images must return 200 with an 'images' key."""
    doc_id = _get_any_doc_id()
    if not doc_id:
        pytest.skip("No documents indexed — upload a PDF first")
    r = httpx.get(f"{BASE_URL}/documents/{doc_id}/images", timeout=15)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert "images" in body, f"Response missing 'images' key: {body}"


def test_show_images_intent_routes_to_image_handler():
    """
    Core image bug fix: 'show me the images present' must NOT fall through
    to text-only hybrid RAG (which can never match images). It must route
    to the dedicated image-index handler (search_mode = 'direct image index
    lookup') or return high confidence.
    """
    doc_id = _get_any_doc_id()
    if not doc_id:
        pytest.skip("No documents indexed")
    res = asyncio.run(_stream({
        "query": "show me the images present in this document",
        "doc_ids": [doc_id], "history": []
    }))
    if res["status"] == 429:
        pytest.skip("Rate limited")
    assert res["status"] == 200
    alpha_mode = res["metadata"].get("alpha_mode", "")
    confidence = res["metadata"].get("confidence", "")
    assert "image" in alpha_mode.lower() or confidence == "high", (
        f"Expected image-lookup routing. alpha_mode={alpha_mode!r}, "
        f"confidence={confidence!r}. This indicates the query fell through "
        f"to text RAG instead of the dedicated image handler."
    )


# ── Document filtering still works after toggle UX change ─────────────────────

def test_doc_id_filter_restricts_retrieval():
    """
    The toggle UX only changed button labels/callbacks in app.py.
    The backend doc_ids filtering logic in query.py must still work.
    """
    doc_id = _get_any_doc_id()
    if not doc_id:
        pytest.skip("No documents indexed")
    res = asyncio.run(_stream({
        "query": "what is this document about",
        "doc_ids": [doc_id], "history": []
    }))
    if res["status"] == 429:
        pytest.skip("Rate limited")
    assert res["status"] == 200
    sources = res["metadata"].get("sources", [])
    if sources:
        bad = [s for s in sources if s.get("doc_id") != doc_id]
        assert not bad, (
            f"Doc filter broken — sources from other docs: "
            f"{[s.get('doc_id') for s in bad]}"
        )


# ── Agentic endpoint ──────────────────────────────────────────────────────────

def test_agentic_endpoint_returns_valid_shape():
    """POST /ask/agentic must return a valid RAG response dict."""
    r = httpx.post(
        f"{BASE_URL}/ask/agentic",
        json={"query": "What is LangChain?", "doc_ids": [], "history": []},
        timeout=60
    )
    if r.status_code == 429:
        pytest.skip("Rate limited")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert "answer" in body, f"Missing 'answer' key: {body.keys()}"
    assert "confidence" in body, f"Missing 'confidence' key: {body.keys()}"
    assert "search_mode" in body, f"Missing 'search_mode' key: {body.keys()}"
    assert "agentic" in body["search_mode"].lower(), (
        f"Expected agentic search_mode, got: {body['search_mode']!r}"
    )


# ── Streaming sanity ──────────────────────────────────────────────────────────

def test_streaming_produces_non_empty_answer():
    """POST /ask/stream must produce at least one token for a known query."""
    res = asyncio.run(_stream({
        "query": "What is LangChain?",
        "doc_ids": [], "history": []
    }))
    if res["status"] == 429:
        pytest.skip("Rate limited")
    assert res["status"] == 200
    assert len(res["answer"].strip()) > 0, "Streaming produced empty answer"


# ── Safety net: run instructions when executed directly ──────────────────────

if __name__ == "__main__":
    print(
        "\nThis is a pytest file — run it with:\n"
        "  pytest tests/test_features_v2.py -v\n\n"
        "Or use the project runner:\n"
        "  python run_tests.py features\n"
    )
    sys.exit(0)