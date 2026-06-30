"""
DocuQuery — Image Extractor v3

Fixes over v2:
  - v2's thresholds (150x150, 10KB, entropy 3.5) were too aggressive and
    filtered out ALL images from imgtest.pdf, leaving image_index.json
    empty — which is why images stopped appearing in chat entirely.
  - Relaxed back to sensible values that still filter the deeplearning.ai
    logo specifically (it's ~100x100 and near-solid color) without
    rejecting legitimate content images.
  - Added rebuild_image_index() — re-scans every PDF already on disk
    (data/*.pdf) and re-extracts images. Lets you recover from a wiped
    index without re-uploading every document through the UI.
"""

import os
import json
import math
import hashlib
import logging

logger = logging.getLogger(__name__)

IMAGES_BASE = "data/images"
IMAGE_INDEX = "data/image_index.json"

# ── Filtering thresholds (relaxed from v2) ──────────────────────────────────
MIN_WIDTH        = 100    # px — was 150, too aggressive
MIN_HEIGHT       = 100    # px
MIN_FILE_BYTES   = 5_000  # 5 KB — was 10KB, too aggressive
MAX_LOGO_RATIO   = 1.3    # tighter ratio — only near-perfect squares flagged
LOGO_MAX_PX      = 160    # px — only SMALL squares are logos (was 250, too broad)
MIN_ENTROPY_BITS = 2.5    # was 3.5 — only reject truly flat/solid images


def _compute_entropy(img_bytes: bytes) -> float:
    """Shannon entropy — solid-color images score low, photos/diagrams score high."""
    freq = [0] * 256
    for b in img_bytes:
        freq[b] += 1
    n = len(img_bytes)
    entropy = 0.0
    for f in freq:
        if f > 0:
            p = f / n
            entropy -= p * math.log2(p)
    return entropy


def _is_logo_or_icon(width: int, height: int) -> bool:
    """
    Only flags SMALL near-square images as logos.
    A 100x100 deeplearning.ai logo → caught.
    A 600x400 diagram or chart → NOT caught (aspect ratio or size excludes it).
    """
    ratio = max(width, height) / max(min(width, height), 1)
    return ratio <= MAX_LOGO_RATIO and max(width, height) <= LOGO_MAX_PX


def _load_index() -> dict:
    if os.path.exists(IMAGE_INDEX):
        with open(IMAGE_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(index: dict):
    os.makedirs("data", exist_ok=True)
    with open(IMAGE_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def extract_images_from_pdf(pdf_path: str, doc_id: str, filename: str) -> list[dict]:
    """Extract meaningful images from a PDF, filtering logos/decorative elements."""
    if not pdf_path.lower().endswith(".pdf"):
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed — image extraction disabled")
        return []

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return []

    out_dir = os.path.join(IMAGES_BASE, doc_id)
    os.makedirs(out_dir, exist_ok=True)

    index   = _load_index()
    results = []
    skipped = {"too_small": 0, "logo": 0, "low_entropy": 0, "file_size": 0}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Cannot open PDF {pdf_path}: {e}")
        return []

    for page_num in range(len(doc)):
        page    = doc[page_num]
        page_no = page_num + 1

        try:
            img_list = page.get_images(full=True)
        except Exception as e:
            logger.warning(f"Page {page_no} image list failed: {e}")
            continue

        for img_info in img_list:
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
            except Exception:
                continue

            width  = base_img.get("width",  0)
            height = base_img.get("height", 0)
            data   = base_img.get("image",  b"")
            if not data:
                continue

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                skipped["too_small"] += 1
                continue

            if _is_logo_or_icon(width, height):
                skipped["logo"] += 1
                logger.debug(f"Skipped logo/icon: {width}x{height} xref={xref}")
                continue

            entropy = _compute_entropy(data)
            if entropy < MIN_ENTROPY_BITS:
                skipped["low_entropy"] += 1
                logger.debug(f"Skipped low-entropy image: {entropy:.2f} bits, {width}x{height}")
                continue

            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png_bytes = pix.tobytes("png")
            except Exception as e:
                logger.debug(f"Pixmap failed for xref {xref}: {e}")
                continue

            if len(png_bytes) < MIN_FILE_BYTES:
                skipped["file_size"] += 1
                continue

            image_id  = hashlib.sha256(png_bytes).hexdigest()[:10]
            file_path = os.path.join(out_dir, f"{image_id}.png")

            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(png_bytes)

            meta = {
                "image_id": image_id, "id": image_id, "doc_id": doc_id,
                "filename": filename, "page": page_no,
                "width": width, "height": height,
                "size": len(png_bytes), "entropy": round(entropy, 2),
                "path": file_path, "ext": "png",
            }

            index[image_id] = meta
            page_key = f"{doc_id}::page::{page_no}"
            if page_key not in index:
                index[page_key] = []
            existing_ids = [i["image_id"] for i in index[page_key] if isinstance(i, dict)]
            if image_id not in existing_ids:
                index[page_key].append(meta)

            results.append(meta)

    doc.close()
    _save_index(index)

    logger.info(
        f"Extracted {len(results)} image(s) from {filename} "
        f"(skipped: too_small={skipped['too_small']}, logo={skipped['logo']}, "
        f"low_entropy={skipped['low_entropy']}, file_size={skipped['file_size']})"
    )
    return results


def rebuild_image_index(data_dir: str = "data") -> dict:
    """
    Re-scan every PDF already on disk and re-extract images.
    Use this to recover when image_index.json was wiped or when
    thresholds changed and you want to re-apply them to existing PDFs,
    without needing to re-upload through the UI.

    Matches each PDF filename to its doc_id via the registry.
    """
    from utils import load_registry  # local import to avoid circularity

    registry = load_registry()
    if not registry:
        logger.warning("Registry is empty — nothing to rebuild")
        return {"rebuilt": 0, "documents": []}

    rebuilt = []
    for doc_id, info in registry.items():
        filename = info.get("filename", "")
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(data_dir, filename)
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF on disk missing for rebuild: {pdf_path}")
            continue
        images = extract_images_from_pdf(pdf_path, doc_id, filename)
        rebuilt.append({"doc_id": doc_id, "filename": filename, "images_found": len(images)})
        logger.info(f"Rebuilt images for {filename}: {len(images)} image(s)")

    return {"rebuilt": len(rebuilt), "documents": rebuilt}


def get_images_for_page(doc_id: str, page: int) -> list[dict]:
    index    = _load_index()
    page_key = f"{doc_id}::page::{page}"
    items    = index.get(page_key, [])
    return [i for i in items if isinstance(i, dict)]


def get_images_for_doc(doc_id: str) -> list[dict]:
    index   = _load_index()
    results = []
    seen    = set()
    for key, val in index.items():
        if not key.startswith(f"{doc_id}::page::"):
            continue
        if isinstance(val, list):
            for img in val:
                if isinstance(img, dict) and img.get("image_id") not in seen:
                    seen.add(img["image_id"])
                    results.append(img)
    return results


def get_image_by_id(image_id: str) -> dict | None:
    index = _load_index()
    return index.get(image_id)


def delete_images_for_doc(doc_id: str):
    import shutil
    doc_dir = os.path.join(IMAGES_BASE, doc_id)
    if os.path.isdir(doc_dir):
        shutil.rmtree(doc_dir)
        logger.info(f"Deleted image directory: {doc_dir}")

    index = _load_index()
    to_delete = [k for k in index
                 if k.startswith(f"{doc_id}::") or
                 (isinstance(index[k], dict) and index[k].get("doc_id") == doc_id)]
    for k in to_delete:
        del index[k]
    _save_index(index)
    logger.info(f"Removed {len(to_delete)} image index entries for doc_id={doc_id}")