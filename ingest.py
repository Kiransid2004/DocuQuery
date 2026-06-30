from chonkie import SemanticChunker
from dotenv import load_dotenv
from collections import defaultdict
import os

load_dotenv()

# ── Chunker ───────────────────────────────────────────────────────────────────

chunker = SemanticChunker(
    embedding_model = "minishlab/potion-base-8M",
    chunk_size      = 512,
    threshold       = 0.5,
    min_sentences   = 2
)

# ── PDF extraction — pypdf (fast, no OCR, no memory errors) ───────────────────

def extract_with_pypdf(path: str) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text.strip(), "page": i})
    print(f"pypdf extracted {len(pages)} pages")
    return pages


# ── Non-PDF extraction — docling (DOCX, PPTX etc.) ────────────────────────────

def extract_with_docling(path: str) -> list[dict]:
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat

    pipeline_options                    = PdfPipelineOptions()
    pipeline_options.do_ocr             = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result     = converter.convert(path)
    doc        = result.document
    page_texts = defaultdict(list)

    for item, _ in doc.iterate_items():
        text = ""
        if hasattr(item, "text") and item.text:
            text = item.text.strip()
        elif hasattr(item, "export_to_markdown"):
            try:
                text = item.export_to_markdown().strip()
            except Exception:
                pass

        if not text:
            continue

        page_no = 1
        if hasattr(item, "prov") and item.prov:
            try:
                page_no = item.prov[0].page_no
            except Exception:
                pass

        page_texts[page_no].append(text)

    pages = [
        {"text": "\n\n".join(texts).strip(), "page": page_no}
        for page_no, texts in sorted(page_texts.items())
        if "\n\n".join(texts).strip()
    ]

    if not pages:
        full_text = doc.export_to_markdown()
        pages     = [{"text": full_text, "page": 1}]

    print(f"docling extracted {len(pages)} pages")
    return pages


# ── Smart loader ──────────────────────────────────────────────────────────────

def load_document_with_pages(path: str) -> list[dict]:
    """
    PDF  → always pypdf (fast, no memory errors)
    Other → docling
    """
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    ext          = path.lower().split(".")[-1]

    if ext == "pdf":
        print(f"PDF ({file_size_mb:.1f}MB) — using pypdf")
        pages = extract_with_pypdf(path)
        if pages:
            return pages
        print("pypdf returned empty — falling back to docling")

    print(f"{ext.upper()} ({file_size_mb:.1f}MB) — using docling")
    return extract_with_docling(path)


def load_document(path: str) -> str:
    pages = load_document_with_pages(path)
    return "\n\n".join(p["text"] for p in pages)


# ── Contextual chunk enrichment ───────────────────────────────────────────────

def enrich_chunk(
    text:     str,
    filename: str,
    page:     int
) -> str:
    """
    Prepend document title and page reference to every chunk.
    Gives the reranker and LLM more context per chunk.
    Fixes most 'answer not in context' failures.

    Example output:
    [Source: Introduction to Agents.pdf | Page 8]
    An AI Agent can be defined as the combination of models, tools...
    """
    name = filename.replace(".pdf", "").replace(".docx", "").replace("_", " ")
    return f"[Source: {name} | Page {page}]\n{text}"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    chunks = chunker.chunk(text)
    return [c.text for c in chunks if len(c.text.strip()) > 50]


def chunk_pages(pages: list[dict], filename: str = "") -> list[dict]:
    """
    Chunk each page semantically and enrich with source context.
    filename is used for contextual enrichment prefix.
    """
    chunked = []
    for page_data in pages:
        try:
            raw_chunks = chunker.chunk(page_data["text"])
            for i, c in enumerate(raw_chunks):
                if len(c.text.strip()) > 50:
                    raw_text     = c.text.strip()
                    enriched     = enrich_chunk(raw_text, filename, page_data["page"]) if filename else raw_text
                    chunked.append({
                        "text":         enriched,    # enriched version stored
                        "raw_text":     raw_text,    # original text preserved
                        "page":         page_data["page"],
                        "chunk_index":  i
                    })
        except Exception as e:
            print(f"Chunking failed for page {page_data['page']}: {e}")
            if len(page_data["text"].strip()) > 50:
                raw_text = page_data["text"].strip()
                enriched = enrich_chunk(raw_text, filename, page_data["page"]) if filename else raw_text
                chunked.append({
                    "text":        enriched,
                    "raw_text":    raw_text,
                    "page":        page_data["page"],
                    "chunk_index": 0
                })
    return chunked


# ── Document health ───────────────────────────────────────────────────────────

def analyze_document_health(chunks: list) -> dict:
    if not chunks:
        return {
            "health": "poor",
            "note":   "No content could be extracted from this document."
        }

    texts       = [c.get("raw_text", c["text"]) if isinstance(c, dict) else c for c in chunks]
    avg_len     = sum(len(t) for t in texts) / len(texts)
    short_ratio = sum(1 for t in texts if len(t) < 100) / len(texts)

    if avg_len > 300 and short_ratio < 0.2:
        return {"health": "excellent", "note": "Document is well-structured and highly queryable."}
    elif avg_len > 150 and short_ratio < 0.4:
        return {"health": "good",      "note": "Document is readable. Answers should be accurate."}
    elif avg_len > 80:
        return {"health": "fair",      "note": "Some sections may have extraction issues."}
    else:
        return {"health": "poor",      "note": "Short chunks detected — answers may be less accurate."}