"""Paper processing: PDF text extraction, page rendering, chunking,
region proposals, structured paper building, and dense embedding index."""
from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore[assignment]

from .config import SessionRecord, env_int, run_cmd, sanitize_json_obj
from .slides import image_dimensions

# ---------------------------------------------------------------------------
# PDF page-count estimation
# ---------------------------------------------------------------------------


def estimate_pdf_page_count(pdf_path: Path) -> int:
    """Estimate the number of pages in a PDF without a full parser.

    Scans the raw bytes for ``/Type /Page`` patterns.
    """
    try:
        raw = pdf_path.read_bytes()
    except Exception:
        return 0
    c = len(re.findall(rb"/Type\s*/Page(?!s)", raw))
    if c <= 0:
        c = max(1, raw.count(b"/Page"))
    return min(c, 5000)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def extract_pdf_pages_text(pdf_path: Path) -> Tuple[List[str], str]:
    """Extract text from each page of a PDF.

    Tries, in order: ``pypdf``, ``PyPDF2``, ``pdftotext`` (external).

    Returns
    -------
    pages : list[str]
        One string per page (empty string if extraction failed).
    extractor : str
        Name of the extractor that succeeded (or an error tag).
    """
    # Preferred parser: pypdf
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        out = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            out.append(txt.strip())
        return out, "pypdf"
    except Exception:
        pass

    # Fallback parser: PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        out = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            out.append(txt.strip())
        return out, "PyPDF2"
    except Exception:
        pass

    # Optional external tool fallback if available.
    if shutil.which("pdftotext"):
        code, out_text, err = run_cmd(
            ["pdftotext", "-layout", str(pdf_path), "-"], timeout=300
        )
        if code == 0 and out_text.strip():
            pages = [x.strip() for x in out_text.split("\f")]
            if pages and pages[-1] == "":
                pages = pages[:-1]
            return pages, "pdftotext"
        return [], f"pdftotext_error:{err[:120]}"

    return [], "no_pdf_text_extractor_available"


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def split_text_chunks(text: str, max_chars: int = 900) -> List[str]:
    """Split *text* into sentence-bounded chunks of at most *max_chars*."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts: List[str] = []
    cur: List[str] = []
    size = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if not sent:
            continue
        if size + len(sent) + 1 > max_chars and cur:
            parts.append(" ".join(cur))
            cur = [sent]
            size = len(sent)
        else:
            cur.append(sent)
            size += len(sent) + 1
    if cur:
        parts.append(" ".join(cur))
    return parts


# ---------------------------------------------------------------------------
# PDF page rendering to PNG
# ---------------------------------------------------------------------------


def render_pdf_pages(pdf_path: Path, out_dir: Path) -> Tuple[List[Path], str]:
    """Render every page of a PDF to PNG images in *out_dir*.

    Tries, in order: ``pdftoppm`` (external), ``PyMuPDF``, ``pypdfium2``.

    Returns
    -------
    pages : list[Path]
        Sorted list of rendered PNG paths.
    renderer : str
        Name of the renderer that succeeded (or an error tag).
    """
    if not pdf_path.exists():
        return [], "pdf_missing"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preferred external renderer: pdftoppm.
    if shutil.which("pdftoppm"):
        prefix = out_dir / "page"
        code, _out, err = run_cmd(
            ["pdftoppm", "-png", str(pdf_path), str(prefix)], timeout=600
        )
        if code == 0:
            pages = sorted(out_dir.glob("page-*.png"))
            if pages:
                return pages, "pdftoppm"
        return [], f"pdftoppm_error:{err[:160]}"

    # Optional fallback: PyMuPDF.
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        pages: List[Path] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            p = out_dir / f"page-{i + 1}.png"
            pix.save(str(p))
            pages.append(p)
        if pages:
            return pages, "pymupdf"
    except Exception:
        pass

    # Optional fallback: pypdfium2 (pure Python wheel, no system poppler needed).
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(pdf_path))
        pages: List[Path] = []
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            p = out_dir / f"page-{i + 1}.png"
            pil_image.save(str(p), format="PNG")
            pages.append(p)
            try:
                page.close()
            except Exception:
                pass
        try:
            doc.close()
        except Exception:
            pass
        if pages:
            return pages, "pypdfium2"
    except Exception:
        pass
    return [], "no_pdf_page_renderer"


# ---------------------------------------------------------------------------
# Paper region proposals
# ---------------------------------------------------------------------------


def make_paper_region_proposals(
    width: int,
    height: int,
    page_text: str,
    per_page: int,
) -> List[Tuple[str, List[int], str]]:
    """Generate candidate region proposals for a single paper page.

    Returns a list of ``(region_type, bbox, semantic_hint)`` tuples where
    *semantic_hint* is one of ``"Figure"``, ``"Table"``, ``"Mixed"``.
    """
    # (region_type, bbox, semantic_hint)
    candidates: List[Tuple[int, str, List[int], str]] = []
    full = [0, 0, width, height]
    upper = [int(0.05 * width), int(0.08 * height), int(0.95 * width), int(0.58 * height)]
    lower = [int(0.05 * width), int(0.42 * height), int(0.95 * width), int(0.95 * height)]
    center = [int(0.10 * width), int(0.20 * height), int(0.90 * width), int(0.85 * height)]
    left_mid = [int(0.05 * width), int(0.20 * height), int(0.52 * width), int(0.88 * height)]
    right_mid = [int(0.48 * width), int(0.20 * height), int(0.95 * width), int(0.88 * height)]

    low = (page_text or "").lower()
    has_figure = ("figure" in low) or ("fig." in low)
    has_table = ("table" in low) or ("tab." in low)

    # Always keep one broad context region.
    candidates.append((100, "page_full", full, "Mixed"))
    candidates.append((85, "center_band", center, "Mixed"))

    if has_figure:
        candidates.append((95, "figure_main", upper, "Figure"))
        candidates.append((80, "figure_left", left_mid, "Figure"))
    if has_table:
        candidates.append((94, "table_main", lower, "Table"))
        candidates.append((79, "table_right", right_mid, "Table"))
    if not has_figure and not has_table:
        candidates.append((82, "upper_large", upper, "Mixed"))
        candidates.append((81, "lower_large", lower, "Mixed"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out: List[Tuple[str, List[int], str]] = []
    seen: set[Tuple[int, ...]] = set()
    for _score, typ, bbox, hint in candidates:
        k = (bbox[0], bbox[1], bbox[2], bbox[3])
        if k in seen:
            continue
        seen.add(k)
        out.append((typ, bbox, hint))
        if len(out) >= max(1, per_page):
            break
    return out


# ---------------------------------------------------------------------------
# Build paper structured data
# ---------------------------------------------------------------------------


def build_paper_structured(
    rec: SessionRecord,
    vision_index: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the *paper_structured* dict for *rec*.

    Extracts text, renders pages, splits chunks, proposes visual regions,
    and mutates *vision_index* in-place by appending paper-related entries.
    """
    pdf = rec.path / "paper.pdf"
    page_texts, extractor = (
        extract_pdf_pages_text(pdf) if pdf.exists() else ([], "pdf_missing")
    )
    if not page_texts:
        n = estimate_pdf_page_count(pdf) if pdf.exists() else 0
        page_texts = [""] * n

    page_images, page_renderer = render_pdf_pages(pdf, rec.path / "paper_pages")
    page_image_by_num: Dict[int, str] = {}
    for p in page_images:
        m = re.search(r"-(\d+)\.png$", p.name)
        if not m:
            continue
        page_image_by_num[int(m.group(1))] = p.name

    chunks: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    paper_pages: List[Dict[str, Any]] = []
    paper_visual_regions: List[Dict[str, Any]] = []
    per_page_regions = max(
        1,
        min(
            env_int(
                "PAPER_VISUAL_REGIONS_PER_PAGE",
                default=3,
                min_value=1,
                max_value=8,
            ),
            8,
        ),
    )
    chunk_counter = 0
    for page_idx, page_text in enumerate(page_texts, start=1):
        image_file = page_image_by_num.get(page_idx)
        if image_file:
            w, h = image_dimensions(rec.path / "paper_pages" / image_file)
        else:
            w, h = 1000, 1400
        page_id = f"{rec.sample_id}/paper_page/{page_idx:04d}"
        paper_pages.append(
            {
                "page_id": page_id,
                "page": page_idx,
                "image_file": image_file,
                "image_width": w,
                "image_height": h,
            }
        )

        if image_file:
            proposals = make_paper_region_proposals(
                w, h, page_text, per_page=per_page_regions
            )
            for ridx, (rtype, bbox, hint) in enumerate(proposals):
                region_id = f"{page_id}/region/{ridx:02d}"
                row = {
                    "region_id": region_id,
                    "page": page_idx,
                    "bbox": bbox,
                    "region_type": rtype,
                    "source_image": image_file,
                    "source_subdir": "paper_pages",
                    "semantic_hint": hint,
                }
                paper_visual_regions.append(row)
                vision_index["regions"].append(
                    {
                        "region_id": region_id,
                        "source_type": "paper_region",
                        "source_ref": page_id,
                        "page": page_idx,
                        "bbox": bbox,
                        "region_type": rtype,
                        "source_image": image_file,
                        "source_subdir": "paper_pages",
                        "semantic_hint": hint,
                    }
                )
                if hint == "Figure":
                    figures.append(
                        {
                            "figure_id": f"{rec.sample_id}/figure/{page_idx:03d}_{ridx:02d}",
                            "page": page_idx,
                            "bbox": bbox,
                            "caption": "",
                            "region_id": region_id,
                        }
                    )
                elif hint == "Table":
                    tables.append(
                        {
                            "table_id": f"{rec.sample_id}/table/{page_idx:03d}_{ridx:02d}",
                            "page": page_idx,
                            "bbox": bbox,
                            "caption": "",
                            "region_id": region_id,
                        }
                    )

        local_chunks = split_text_chunks(page_text) or (
            [""] if page_text == "" else []
        )
        if not local_chunks:
            local_chunks = [""]
        for ci, chunk in enumerate(local_chunks):
            chunk_id = f"{rec.sample_id}/paper_chunk/{chunk_counter:04d}"
            chunk_counter += 1
            bbox = [0, 0, 1000, 1400]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_idx,
                    "bbox": bbox,
                    "text": chunk,
                }
            )

            vision_index["regions"].append(
                {
                    "region_id": chunk_id,
                    "source_type": "paper_chunk",
                    "source_ref": chunk_id,
                    "page": page_idx,
                    "bbox": bbox,
                    "region_type": "text_block",
                }
            )

            low = chunk.lower()
            if "figure" in low:
                fig_id = f"{rec.sample_id}/figure/{page_idx:03d}_{ci:02d}"
                figures.append(
                    {
                        "figure_id": fig_id,
                        "page": page_idx,
                        "bbox": bbox,
                        "caption": chunk[:200],
                    }
                )
            if "table" in low:
                tab_id = f"{rec.sample_id}/table/{page_idx:03d}_{ci:02d}"
                tables.append(
                    {
                        "table_id": tab_id,
                        "page": page_idx,
                        "bbox": bbox,
                        "caption": chunk[:200],
                    }
                )

    return {
        "sample_id": rec.sample_id,
        "title": rec.metadata.get("title", ""),
        "authors": rec.metadata.get("authors", []),
        "paper_file": "paper.pdf" if pdf.exists() else None,
        "paper_url": rec.metadata.get("paper_url"),
        "pdf_text_extractor": extractor,
        "pdf_page_renderer": page_renderer,
        "paper_pages": paper_pages,
        "paper_visual_regions": paper_visual_regions,
        "paper_chunks": chunks,
        "figures": figures,
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embedding_env() -> Tuple[str, str, str]:
    """Return ``(base_url, api_key, model)`` from environment variables."""
    base = os.getenv("EMBED_BASE_URL", "").strip()
    key = os.getenv("EMBED_API_KEY", "").strip()
    model = os.getenv("EMBED_MODEL", "").strip()
    return base.rstrip("/"), key, model


def normalize_text_for_embedding(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def is_embedding_text_too_noisy(text: str) -> bool:
    """Heuristic to skip highly symbolic / OCR-noisy chunks."""
    t = normalize_text_for_embedding(text)
    if not t:
        return True
    n = len(t)
    if n < 24:
        return False
    alnum = sum(1 for ch in t if ch.isalnum())
    punct = sum(1 for ch in t if (not ch.isalnum()) and (not ch.isspace()))
    alnum_ratio = alnum / max(1, n)
    punct_ratio = punct / max(1, n)
    # Skip highly symbolic/OCR-noisy chunks (e.g. diagram glyph soup).
    if n >= 120 and alnum_ratio < 0.30 and punct_ratio > 0.35:
        return True
    if n >= 120 and ("…" in t) and punct_ratio > 0.42:
        return True
    return False


def call_openai_embeddings(
    texts: List[str],
    base: str,
    key: str,
    model: str,
) -> Tuple[Optional[List[List[float]]], str]:
    """Call an OpenAI-compatible ``/embeddings`` endpoint.

    Primary: batch request.  Fallback: per-text single requests.
    """
    if not base:
        return None, "missing_env:EMBED_BASE_URL"
    if not model:
        return None, "missing_env:EMBED_MODEL"
    clean_texts = [str(t).strip() for t in texts]
    if not clean_texts:
        return None, "empty_input"
    clean_texts = [t if t else " " for t in clean_texts]

    def _post(
        payload: Dict[str, Any], with_auth: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if with_auth and key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            safe_payload = sanitize_json_obj(payload)
            body = json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url=f"{base}/embeddings",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw_bytes = resp.read()
                raw = raw_bytes.decode("utf-8", "ignore")
                code = int(getattr(resp, "status", 200) or 200)
        except urllib.error.HTTPError as e:
            err_raw = e.read().decode("utf-8", "ignore")
            body_preview = (err_raw or "").replace("\n", " ").strip()[:400]
            return None, f"http_{e.code}:{body_preview}"
        except Exception as e:
            return None, f"http_exception:{e}"
        if code >= 300:
            body_preview = (raw or "").replace("\n", " ").strip()[:400]
            return None, f"http_{code}:{body_preview}"
        try:
            data = json.loads(raw)
        except Exception:
            raw_preview = (raw or "").replace("\n", " ").strip()[:240]
            return None, f"non_json_response:{raw_preview}"
        if not isinstance(data, dict):
            return None, "bad_response:not_object"
        return data, "ok"

    def _post_try_auth_then_noauth(
        payload: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        data, st = _post(payload, with_auth=True)
        if data is not None:
            return data, st if key else "ok_noauth"
        if key:
            data2, st2 = _post(payload, with_auth=False)
            if data2 is not None:
                return data2, "ok_noauth_fallback"
            return None, f"{st};noauth={st2}"
        return None, st

    def _parse(
        data: Dict[str, Any], expected_n: int
    ) -> Tuple[Optional[List[List[float]]], str]:
        arr = data.get("data")
        if not isinstance(arr, list) or not arr:
            return None, "bad_response:data_missing"
        arr_sorted = sorted(
            [x for x in arr if isinstance(x, dict)],
            key=lambda x: int(x.get("index", 0)),
        )
        vecs: List[List[float]] = []
        for item in arr_sorted:
            emb = item.get("embedding")
            if not isinstance(emb, list) or not emb:
                return None, "bad_response:embedding_missing"
            try:
                vecs.append([float(v) for v in emb])
            except Exception:
                return None, "bad_response:embedding_non_numeric"
        if len(vecs) != expected_n:
            return None, (
                f"bad_response:embedding_count_mismatch"
                f"(expected={expected_n},got={len(vecs)})"
            )
        return vecs, "ok"

    # Primary path: OpenAI-compatible batch request.
    data, st = _post_try_auth_then_noauth({"model": model, "input": clean_texts})
    if data is not None:
        vecs, pst = _parse(data, len(clean_texts))
        if vecs is not None:
            return vecs, "ok"
        st = f"{st};{pst}"

    # Fallback path for incompatible gateways:
    # 1) per-text with input as string
    # 2) if still failing, per-text with single-item list
    fallback_vecs: List[List[float]] = []
    last_err = st
    for t in clean_texts:
        d1, s1 = _post_try_auth_then_noauth({"model": model, "input": t})
        if d1 is not None:
            v1, p1 = _parse(d1, 1)
            if v1 is not None:
                fallback_vecs.append(v1[0])
                continue
            s1 = f"{s1};{p1}"

        d2, s2 = _post_try_auth_then_noauth({"model": model, "input": [t]})
        if d2 is not None:
            v2, p2 = _parse(d2, 1)
            if v2 is not None:
                fallback_vecs.append(v2[0])
                continue
            s2 = f"{s2};{p2}"

        last_err = f"single_str={s1}(len={len(t)});single_list={s2}(len={len(t)})"
        return None, f"embed_fallback_failed:{last_err}"

    return fallback_vecs, f"ok_fallback_single:{st}"


# ---------------------------------------------------------------------------
# Dense index construction
# ---------------------------------------------------------------------------


def build_paper_dense_index(paper_structured: Dict[str, Any]) -> Dict[str, Any]:
    """Build a dense (embedding) index over paper chunks.

    Uses faiss when available, falls back to numpy dot-product search.

    Returns a dict with keys ``enabled``, ``status``, ``backend``, and
    (when enabled) ``index``/``matrix``, ``rows``, ``base``, ``key``,
    ``model``.
    """
    rows: List[Dict[str, Any]] = []
    filtered_noise = 0
    for c in paper_structured.get("paper_chunks", []) or []:
        if not isinstance(c, dict):
            continue
        txt = normalize_text_for_embedding(c.get("text", ""))
        if not txt:
            continue
        if is_embedding_text_too_noisy(txt):
            filtered_noise += 1
            continue
        rows.append(
            {
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "bbox": c.get("bbox", [0, 0, 1000, 1400]),
                "text": txt,
            }
        )
    if not rows:
        return {
            "enabled": False,
            "status": f"no_paper_chunks(filtered_noise={filtered_noise})",
            "backend": "none",
        }

    base, key, model = embedding_env()
    if not (base and model):
        missing: List[str] = []
        if not base:
            missing.append("EMBED_BASE_URL")
        if not model:
            missing.append("EMBED_MODEL")
        return {
            "enabled": False,
            "status": "missing_embed_env:" + ",".join(missing),
            "backend": "none",
        }

    texts = [r["text"][:4000] for r in rows]
    vectors: List[List[float]] = []
    bs = max(1, int(os.getenv("EMBED_BATCH_SIZE", "1").strip() or "1"))
    base_dbg = base
    if "@" in base_dbg:
        base_dbg = base_dbg.split("@", 1)[-1]
    if len(base_dbg) > 120:
        base_dbg = base_dbg[:120] + "..."
    debug_meta = f"base={base_dbg};model={model};key_set={1 if bool(key) else 0}"
    for i in range(0, len(texts), bs):
        vecs, st = call_openai_embeddings(
            texts[i : i + bs], base=base, key=key, model=model
        )
        if vecs is None:
            return {
                "enabled": False,
                "status": (
                    f"embed_failed:{st};rows={len(rows)};"
                    f"filtered_noise={filtered_noise};"
                    f"batch={bs};first_batch_n={len(texts[i:i + bs])};"
                    f"first_len={len(texts[i]) if i < len(texts) else 0};"
                    f"{debug_meta}"
                ),
                "backend": "none",
            }
        vectors.extend(vecs)
    if np is None:
        return {
            "enabled": False,
            "status": "numpy_unavailable",
            "backend": "none",
        }
    mat = np.asarray(vectors, dtype="float32")
    if mat.ndim != 2 or mat.shape[0] != len(rows):
        return {"enabled": False, "status": "bad_embedding_shape", "backend": "none"}
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    if faiss is not None:
        try:
            idx = faiss.IndexFlatIP(mat.shape[1])
            idx.add(mat)
            return {
                "enabled": True,
                "status": (
                    f"ok(rows={len(rows)},filtered_noise={filtered_noise},"
                    f"batch={bs});{debug_meta}"
                ),
                "backend": "faiss",
                "index": idx,
                "matrix": mat,
                "rows": rows,
                "base": base,
                "key": key,
                "model": model,
            }
        except Exception as e:
            return {
                "enabled": True,
                "status": (
                    f"faiss_failed_fallback_numpy:{e};rows={len(rows)};"
                    f"filtered_noise={filtered_noise};"
                    f"batch={bs};{debug_meta}"
                ),
                "backend": "numpy",
                "matrix": mat,
                "rows": rows,
                "base": base,
                "key": key,
                "model": model,
            }
    return {
        "enabled": True,
        "status": (
            f"ok_fallback_numpy(rows={len(rows)},"
            f"filtered_noise={filtered_noise},batch={bs});{debug_meta}"
        ),
        "backend": "numpy",
        "matrix": mat,
        "rows": rows,
        "base": base,
        "key": key,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Dense search
# ---------------------------------------------------------------------------


def dense_search_top1(
    index_state: Dict[str, Any],
    query: str,
    query_cache: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """Return the top-1 closest paper chunk for *query*.

    Parameters
    ----------
    index_state : dict
        The dict returned by :func:`build_paper_dense_index`.
    query : str
        Query text to embed and search.
    query_cache : dict
        Simple ``{query_text: normalized_vector}`` cache (mutated in-place).

    Returns
    -------
    row : dict or None
        The best-matching chunk row, or ``None`` on failure.
    similarity : float
        Cosine similarity score.
    status : str
        Status tag (``"ok"`` on success).
    """
    if not index_state.get("enabled"):
        return None, 0.0, str(index_state.get("status", "index_disabled"))
    if np is None:
        return None, 0.0, "numpy_unavailable"
    q = query.strip()
    if not q:
        return None, 0.0, "empty_query"

    qvec = query_cache.get(q)
    if qvec is None:
        vecs, st = call_openai_embeddings(
            [q[:4000]],
            base=index_state["base"],
            key=index_state["key"],
            model=index_state["model"],
        )
        if vecs is None or not vecs:
            return None, 0.0, f"embed_query_failed:{st}"
        qarr = np.asarray(vecs[0], dtype="float32")
        if qarr.ndim != 1:
            return None, 0.0, "bad_query_embedding_shape"
        n = float(np.linalg.norm(qarr))
        if n <= 0:
            return None, 0.0, "zero_query_norm"
        qvec = qarr / n
        query_cache[q] = qvec

    backend = str(index_state.get("backend", "none"))
    rows = index_state.get("rows", [])
    if not rows:
        return None, 0.0, "index_rows_empty"
    if backend == "faiss":
        idx = index_state.get("index")
        if idx is None:
            return None, 0.0, "faiss_index_missing"
        try:
            D, I = idx.search(qvec.reshape(1, -1), 1)
            sim = float(D[0][0])
            pos = int(I[0][0])
            if pos < 0 or pos >= len(rows):
                return None, 0.0, "faiss_index_oob"
            return rows[pos], sim, "ok"
        except Exception as e:
            return None, 0.0, f"faiss_search_failed:{e}"

    mat = index_state.get("matrix")
    if mat is None:
        return None, 0.0, "matrix_missing"
    try:
        sims = mat @ qvec
    except Exception as e:
        return None, 0.0, f"numpy_search_failed:{e}"
    if sims.shape[0] == 0:
        return None, 0.0, "empty_similarity"
    pos = int(np.argmax(sims))
    sim = float(sims[pos])
    return rows[pos], sim, "ok"
