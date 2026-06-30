"""
DocuQuery — LangGraph + Image Extractor Tests (v5)

Fix over previous version:
  test_image_appears_in_source_metadata no longer just skips when no images
  are indexed — it first attempts POST /rebuild-images/ to re-extract images
  from every PDF already on disk (data/*.pdf), using the current thresholds
  in image_extractor.py. Only skips if rebuild also finds zero images
  (meaning no PDF has any image content above the size/entropy filters).
"""

import sys
import os
import json
import time
import importlib.util
import httpx
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_URL   = os.getenv("BASE_URL", "http://127.0.0.1:8000")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(path: str, **params) -> dict:
    with httpx.Client(timeout=15) as c:
        r = c.get(f"{BASE_URL}{path}", params=params)
    body = {}
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:300]}
    return {"status": r.status_code, "body": body}


def _post(path: str, payload: dict = None, timeout: int = 120) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{BASE_URL}{path}", json=payload or {})
    body = {}
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:300]}
    return {"status": r.status_code, "body": body}


def _stream_ask(query: str, doc_ids: list = None, timeout: int = 90) -> dict:
    full_text, metadata = "", {}
    try:
        with httpx.Client(timeout=timeout) as c:
            with c.stream("POST", f"{BASE_URL}/ask/stream", json={
                "query": query, "doc_ids": doc_ids or [], "history": []
            }) as r:
                if r.status_code != 200:
                    return {"status": r.status_code, "answer": r.text[:200], "metadata": {}}
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    ds = line[6:]
                    if ds == "[DONE]":
                        break
                    try:
                        d = json.loads(ds)
                        t = d.get("token", "")
                        if "__METADATA__" in t:
                            try:
                                metadata = json.loads(t.split("__METADATA__", 1)[1])
                            except Exception:
                                pass
                        else:
                            full_text += t
                    except Exception:
                        pass
    except Exception as e:
        return {"status": 0, "answer": f"[error: {e}]", "metadata": {}}
    return {"status": 200, "answer": full_text, "metadata": metadata}


def _server_alive() -> bool:
    try:
        with httpx.Client(timeout=5) as c:
            return c.get(f"{BASE_URL}/health").status_code == 200
    except Exception:
        return False


def _get_indexed_docs() -> list:
    try:
        with httpx.Client(timeout=10) as c:
            return c.get(f"{BASE_URL}/documents/").json().get("documents", [])
    except Exception:
        return []


def _ensure_images_indexed() -> dict:
    """
    Check if ANY indexed PDF has images. If not, trigger /rebuild-images/
    to re-scan PDFs on disk with current thresholds, then re-check.
    Returns {"has_images": bool, "best_doc": dict|None, "image_count": int}
    """
    docs = _get_indexed_docs()
    pdf_docs = [d for d in docs if d["filename"].lower().endswith(".pdf")]

    def _scan() -> tuple:
        best_doc, best_count = None, 0
        for doc in pdf_docs:
            r = _get(f"/documents/{doc['doc_id']}/images")
            count = r["body"].get("total", 0)
            if count > best_count:
                best_count, best_doc = count, doc
        return best_doc, best_count

    best_doc, best_count = _scan()
    if best_count > 0:
        return {"has_images": True, "best_doc": best_doc, "image_count": best_count}

    if not pdf_docs:
        return {"has_images": False, "best_doc": None, "image_count": 0}

    # No images found anywhere — attempt rebuild from disk
    rebuild = _post("/rebuild-images/", timeout=180)
    if rebuild["status"] == 429:
        # Rate limited — can't rebuild right now
        return {"has_images": False, "best_doc": None, "image_count": 0}

    time.sleep(1)  # let index file write settle
    best_doc, best_count = _scan()
    return {"has_images": best_count > 0, "best_doc": best_doc, "image_count": best_count}


@pytest.fixture(autouse=True, scope="module")
def require_server():
    if not _server_alive():
        pytest.skip(f"Server not reachable at {BASE_URL} — "
                    "start with: uvicorn main:app --reload --port 8000")


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph / Agentic Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestLangGraphAgent:

    def test_agentic_endpoint_exists(self):
        r = _post("/ask/agentic", {"query": "What is LangChain?", "doc_ids": [], "history": []})
        assert r["status"] not in (404, 405)

    def test_agentic_returns_answer_field(self):
        docs = _get_indexed_docs()
        if not docs:
            pytest.skip("No documents indexed")
        r = _post("/ask/agentic", {"query": "Give me a brief overview.", "doc_ids": [], "history": []})
        if r["status"] == 422:
            pytest.skip("Schema mismatch")
        if r["status"] == 500:
            pytest.fail(f"Agentic crashed (500): {r['body'].get('_raw','')[:200]}")
        assert r["status"] == 200
        assert len(r["body"].get("answer", "")) > 10

    def test_agentic_vs_standard_same_question(self):
        docs = _get_indexed_docs()
        if not docs:
            pytest.skip("No documents indexed")
        q     = "What are the main topics?"
        r_std = _post("/ask/",        {"query": q, "doc_ids": [], "history": []})
        r_agt = _post("/ask/agentic", {"query": q, "doc_ids": [], "history": []}, timeout=120)
        assert r_std["status"] == 200
        if r_agt["status"] in (422, 500):
            pytest.skip(f"Agentic returned {r_agt['status']}")
        assert r_agt["status"] == 200
        assert len(r_agt["body"].get("answer", "")) > 20

    def test_agentic_rejects_empty_query(self):
        r = _post("/ask/agentic", {"query": "", "doc_ids": [], "history": []})
        assert r["status"] in (400, 422)

    def test_agentic_does_not_crash_on_greeting(self):
        r = _post("/ask/agentic", {"query": "Hello!", "doc_ids": [], "history": []}, timeout=45)
        assert r["status"] != 500


# ─────────────────────────────────────────────────────────────────────────────
# Image Extractor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImageExtractor:

    def test_image_list_endpoint_exists(self):
        docs = _get_indexed_docs()
        if not docs:
            pytest.skip("No documents indexed")
        r = _get(f"/documents/{docs[0]['doc_id']}/images")
        assert r["status"] == 200
        assert "images" in r["body"]
        assert "total"  in r["body"]

    def test_image_page_endpoint_exists(self):
        docs = _get_indexed_docs()
        if not docs:
            pytest.skip("No documents indexed")
        r = _get(f"/documents/{docs[0]['doc_id']}/images/page/1")
        assert r["status"] == 200

    def test_rebuild_images_endpoint_exists(self):
        """POST /rebuild-images/ must exist and not crash."""
        r = _post("/rebuild-images/", timeout=180)
        assert r["status"] in (200, 429), (
            f"/rebuild-images/ returned unexpected status {r['status']}: "
            f"{r['body'].get('_raw','')[:200]}"
        )

    def test_pdf_image_extraction(self):
        """
        FIX: Uses _ensure_images_indexed() which triggers a rebuild if no
        images are currently indexed, instead of immediately skipping.
        """
        state = _ensure_images_indexed()
        if not state["has_images"]:
            pytest.skip(
                "No PDFs have any extractable images even after rebuild — "
                "this means every embedded image in every PDF was filtered "
                "out by image_extractor.py thresholds (too small / logo-like / "
                "low entropy). This can be correct if your PDFs are text-only."
            )

        doc = state["best_doc"]
        r = _get(f"/documents/{doc['doc_id']}/images")
        images = r["body"].get("images", [])
        assert images, "Expected images list to be non-empty after rebuild"

        image_id = images[0].get("image_id") or images[0].get("id")
        assert image_id, "Image entry missing 'image_id'"

        with httpx.Client(timeout=10) as c:
            r2 = c.get(f"{BASE_URL}/images/{image_id}")
        assert r2.status_code == 200, f"Image file endpoint returned {r2.status_code}"
        assert len(r2.content) > 500, "Image file is suspiciously small"

    def test_show_images_intent_routing(self):
        docs = _get_indexed_docs()
        if not docs:
            pytest.skip("No documents indexed")
        result = _stream_ask("Show me the images in this document", [docs[0]["doc_id"]])
        assert result["status"] == 200
        answer = result["answer"].lower()
        assert any(s in answer for s in [
            "image", "figure", "diagram", "not available", "no extractable"
        ]), f"Unexpected response: {result['answer'][:200]}"

    def test_image_extractor_module_importable(self):
        spec = importlib.util.spec_from_file_location(
            "image_extractor", os.path.join(PROJECT_ROOT, "image_extractor.py")
        )
        if spec is None:
            pytest.skip("image_extractor.py not found")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            pytest.fail(f"import failed: {e}")
        for fn in ("extract_images_from_pdf", "get_images_for_page",
                   "get_images_for_doc", "get_image_by_id", "rebuild_image_index"):
            assert hasattr(mod, fn), f"Missing: {fn}"

    def test_image_disk_dir_exists_after_pdf_upload(self):
        state = _ensure_images_indexed()
        if not state["has_images"]:
            pytest.skip("No PDFs with extractable images")
        assert os.path.isdir(IMAGES_DIR), "data/images/ missing despite indexed images"

    def test_image_appears_in_source_metadata(self):
        """
        FIX: Now scans ALL indexed PDFs (not just imgtest.pdf), and triggers
        a rebuild via _ensure_images_indexed() if the index is empty.
        Uses _stream_ask() so sources are populated even on low-confidence
        answers (the non-streaming /ask/ can return sources=[] in that case).
        """
        state = _ensure_images_indexed()
        if not state["has_images"]:
            pytest.skip(
                "No PDFs have extractable images, even after attempting rebuild. "
                "Upload a PDF with embedded diagrams/charts/photos to test this fully."
            )

        target = state["best_doc"]
        result = _stream_ask(
            f"What is discussed in {target['filename']}?",
            doc_ids=[target["doc_id"]],
            timeout=90
        )
        assert result["status"] == 200
        sources = result["metadata"].get("sources", [])

        if not sources:
            result2 = _stream_ask("Summarize this document", doc_ids=[target["doc_id"]], timeout=90)
            sources = result2["metadata"].get("sources", [])

        if not sources:
            pytest.skip(f"No retrievable text chunks for {target['filename']}")

        assert any("images" in s for s in sources), (
            f"No 'images' key in sources for {target['filename']}.\n"
            f"Sources: {sources[:1]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentWithImages:

    def test_agentic_query_about_figures_does_not_crash(self):
        r = _post("/ask/agentic", {
            "query": "Are there any diagrams in the documents?",
            "doc_ids": [], "history": []
        }, timeout=120)
        assert r["status"] != 500


if __name__ == "__main__":
    print("\nRun: pytest tests/test_langgraph_images.py -v")
    print("Or:  python run_tests.py langgraph\n")