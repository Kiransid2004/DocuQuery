from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import os

load_dotenv()

BM25_PATH = "data/bm25_params.json"
_bm25: BM25Encoder = None

def get_bm25() -> BM25Encoder:
    global _bm25
    if _bm25 is not None:
        return _bm25
    _bm25 = BM25Encoder()
    if os.path.exists(BM25_PATH):
        _bm25.load(BM25_PATH)
        print("BM25 params loaded from disk.")
    else:
        print("No BM25 params — will fit on first upload.")
    return _bm25

def fit_bm25(corpus: list[str]):
    global _bm25
    bm25 = get_bm25()
    print(f"Fitting BM25 on {len(corpus)} chunks...")
    bm25.fit(corpus)
    os.makedirs("data", exist_ok=True)
    bm25.dump(BM25_PATH)
    print("BM25 params saved.")
    _bm25 = bm25

def encode_sparse(texts: list[str]) -> list[dict]:
    return get_bm25().encode_documents(texts)

def encode_sparse_query(query: str) -> dict:
    results = get_bm25().encode_queries([query])
    return results[0]

def hybrid_score_norm(dense: list, sparse: dict, alpha: float) -> tuple:
    dense_scaled  = [v * alpha for v in dense]
    sparse_scaled = {
        "indices": sparse["indices"],
        "values":  [v * (1 - alpha) for v in sparse["values"]]
    }
    return dense_scaled, sparse_scaled