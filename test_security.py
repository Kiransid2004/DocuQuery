"""
DocuQuery — Image Extractor v2

Improvements over v1:
  ✅ Stricter filtering to remove logos, icons, decorative images
  ✅ Aspect ratio filter — near-square small images are likely logos
  ✅ Minimum entropy filter — solid color/gradient images skipped
  ✅ Higher min-size threshold: 150x150px (was 80x80)
  ✅ Higher min-file-size: 10KB (was 3KB)
  ✅ Images stored in data/images/{doc_id}/{image_id}.png
  ✅ Indexed by (doc_id, page) for fast lookup by build_sources()
"""

import os
import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGES_BASE   = "data/images"
IMAGE_INDEX   = "data/image_index.json"

# ── Filtering thresholds ─────────────────────────────────────────────────────
MIN_WIDTH        = 150    # px — raised from 80 to skip small logos
MIN_HEIGHT       = 150    # px
MIN_FILE_BYTES   = 10_000 # 10 KB — raised from 3 KB
MAX_LOGO_RATIO   = 1.5    # images with w/h ratio < 1.5 AND small = likely logo
LOGO_MAX_PX      = 250    # px — if square-ish AND smaller than this = skip
MIN_ENTROPY_BITS = 3.5    # low entropy = solid color / gradient = decorative


def _compute_entropy(img_bytes: bytes) -> float:
    """
    Shannon entropy of the image byte distribution.
    Solid-color or near-solid images have entropy < 3 bits.
    Real photographs/diagrams have entropy > 5 bits.
    """
    import math
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


def _is_logo_or_icon(width: int, height: int, file_size: int) -> bool:
    """
    Heuristic: images that are near-square AND smaller than LOGO_MAX_PX
    are very likely logos, icons, or decorative elements.
    """
    ratio = max(width, height) / max(min(width, height), 1)
    if ratio <= MAX_LOGO_RATIO and max(width, height) <= LOGO_MAX_PX:
        return True
    return False


def _load_index() -> dict:
    if os.path.exists(IMAGE_INDEX):
        with open(IMAGE_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(index: dict):
    os.makedirs("data", exist_ok=True)
    with open(IMAGE_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def extract_images_from_pdf(
    pdf_path: str,
    doc_id:   str,
    filename: str,
) -> list[dict]:
    """
    Extract meaningful images from a PDF.

    Filtering pipeline (image rejected if ANY condition is true):
      1. Width < MIN_WIDTH or Height < MIN_HEIGHT
      2. File size after PNG export < MIN_FILE_BYTES
      3. Near-square AND small (logo/icon heuristic)
      4. Low entropy (solid color / gradient background)

    Returns list of image metadata dicts.
    """
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

            # Filter 1: minimum dimensions
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                skipped["too_small"] += 1
                continue

            # Filter 2: logo/icon heuristic
            if _is_logo_or_icon(width, height, len(data)):
                skipped["logo"] += 1
                logger.debug(f"Skipped logo/icon: {width}x{height} xref={xref}")
                continue

            # Filter 3: entropy (solid color / gradient)
            entropy = _compute_entropy(data)
            if entropy < MIN_ENTROPY_BITS:
                skipped["low_entropy"] += 1
                logger.debug(f"Skipped low-entropy image: {entropy:.2f} bits, {width}x{height}")
                continue

            # Render to PNG for consistent format + get accurate file size
            try:
                pix     = fitz.Pixmap(doc, xref)
                if pix.n > 4:           # CMYK or other exotic — convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png_bytes = pix.tobytes("png")
            except Exception as e:
                logger.debug(f"Pixmap failed for xref {xref}: {e}")
                continue

            # Filter 4: file size after PNG conversion
            if len(png_bytes) < MIN_FILE_BYTES:
                skipped["file_size"] += 1
                continue

            # Generate stable ID from content hash
            image_id  = hashlib.sha256(png_bytes).hexdigest()[:10]
            file_path = os.path.join(out_dir, f"{image_id}.png")

            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(png_bytes)

            meta = {
                "image_id": image_id,
                "id":       image_id,          # alias
                "doc_id":   doc_id,
                "filename": filename,
                "page":     page_no,
                "width":    width,
                "height":   height,
                "size":     len(png_bytes),
                "entropy":  round(entropy, 2),
                "path":     file_path,
                "ext":      "png",
            }

            # Store in index keyed by image_id and by (doc_id, page)
            index[image_id] = meta
            page_key = f"{doc_id}::page::{page_no}"
            if page_key not in index:
                index[page_key] = []
            # Avoid duplicate entries
            existing_ids = [i["image_id"] for i in index[page_key]
                            if isinstance(i, dict)]
            if image_id not in existing_ids:
                index[page_key].append(meta)

            results.append(meta)

    doc.close()
    _save_index(index)

    logger.info(
        f"Extracted {len(results)} image(s) from {filename} "
        f"(skipped: too_small={skipped['too_small']}, "
        f"logo={skipped['logo']}, "
        f"low_entropy={skipped['low_entropy']}, "
        f"file_size={skipped['file_size']})"
    )
    return results


def get_images_for_page(doc_id: str, page: int) -> list[dict]:
    """Return images on a specific page of a document."""
    index    = _load_index()
    page_key = f"{doc_id}::page::{page}"
    items    = index.get(page_key, [])
    return [i for i in items if isinstance(i, dict)]


def get_images_for_doc(doc_id: str) -> list[dict]:
    """Return all images extracted from a document."""
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
    """Return metadata for a single image by its ID."""
    index = _load_index()
    return index.get(image_id)


def delete_images_for_doc(doc_id: str):
    """Delete all images and index entries for a document."""
    import shutil
    doc_dir = os.path.join(IMAGES_BASE, doc_id)
    if os.path.isdir(doc_dir):
        shutil.rmtree(doc_dir)
        logger.info(f"Deleted image directory: {doc_dir}")

    index    = _load_index()
    to_delete = [k for k in index
                 if k.startswith(f"{doc_id}::") or
                 (isinstance(index[k], dict) and index[k].get("doc_id") == doc_id)]
    for k in to_delete:
        del index[k]
    _save_index(index)
    logger.info(f"Removed {len(to_delete)} image index entries for doc_id={doc_id}")