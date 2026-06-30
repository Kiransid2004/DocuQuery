import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
pytest contract tests -- verify API shape stays stable across changes.
Fast, no LLM calls needed for most of these.
"""
import pytest
import httpx

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session", autouse=True)
def check_server():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("API server not healthy")
    except Exception:
        pytest.skip("API server not reachable")


def test_health_endpoint_shape():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "indexed_documents" in body
    assert "feedback" in body
    assert "eval_quality" in body

def test_documents_list_shape():
    r = httpx.get(f"{BASE_URL}/documents/")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body
    assert "documents" in body
    assert isinstance(body["documents"], list)

def test_feedback_get_shape():
    r = httpx.get(f"{BASE_URL}/feedback/")
    assert r.status_code == 200
    body = r.json()
    assert "thumbs_up" in body
    assert "thumbs_down" in body

def test_eval_log_shape():
    r = httpx.get(f"{BASE_URL}/eval-log/")
    assert r.status_code == 200
    body = r.json()
    assert "averages" in body
    assert "recent" in body

def test_root_endpoint():
    r = httpx.get(f"{BASE_URL}/")
    assert r.status_code == 200
    assert "version" in r.json()

if __name__ == "__main__":
    print("\nRun with: pytest " + __file__ + " -v\nOr: python run_tests.py\n")
    import sys; sys.exit(0)