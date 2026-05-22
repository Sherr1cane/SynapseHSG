"""Visual semantics enrichment for the SynapseHSG pipeline.

Calls a vision LLM to extract OCR text, chart type, visual summary, and
entities from slide and paper regions, then writes results back into the
structured data.
"""

from __future__ import annotations

import base64
import io
import concurrent.futures
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .config import (
    VISUAL_REGION_SYSTEM_PROMPT,
    Config,
    SessionRecord,
    env_int,
)
from .io_utils import write_json
from .llm_client import (
    call_openai_multimodal,
    get_visual_llm_executor,
    parse_llm_json_object,
    sanitize_json_obj,
)


# ---------------------------------------------------------------------------
# Region image helpers
# ---------------------------------------------------------------------------


def crop_region_to_data_url(image_path: Path, bbox: List[int], max_bytes: int) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pillow is required for in-memory crop: {e}")

    with Image.open(image_path) as img:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(x1 + 1, min(x2, img.width))
        y2 = max(y1 + 1, min(y2, img.height))
        # Guard against degenerate crops (e.g., 500x1) that can break some VLM processors.
        min_side = 16
        w = x2 - x1
        h = y2 - y1
        if w < min_side:
            pad = (min_side - w + 1) // 2
            x1 = max(0, x1 - pad)
            x2 = min(img.width, x2 + pad)
        if h < min_side:
            pad = (min_side - h + 1) // 2
            y1 = max(0, y1 - pad)
            y2 = min(img.height, y2 + pad)
        x2 = max(x1 + 1, min(x2, img.width))
        y2 = max(y1 + 1, min(y2, img.height))
        crop = img.crop((x1, y1, x2, y2)).convert("RGB")

        quality = 90
        while True:
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=quality)
            b = buf.getvalue()
            if len(b) <= max_bytes or quality <= 40:
                break
            quality -= 10

    encoded = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _crop_region_image(image_path: Path, bbox: List[int]) -> Any:
    from PIL import Image  # type: ignore

    with Image.open(image_path) as img:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(x1 + 1, min(x2, img.width))
        y2 = max(y1 + 1, min(y2, img.height))
        return img.crop((x1, y1, x2, y2)).convert("L")


def resolve_region_image_path(rec: SessionRecord, region: Dict[str, Any]) -> Optional[Path]:
    source_image = region.get("source_image")
    if not source_image:
        return None
    source_subdir = str(region.get("source_subdir", "") or "").strip()
    search_dirs = [source_subdir] if source_subdir else []
    search_dirs.extend(["slides", "paper_pages"])
    for d in search_dirs:
        p = rec.path / d / str(source_image)
        if p.exists():
            return p
    return None


def should_prune_region_before_llm(rec: SessionRecord, region: Dict[str, Any]) -> Tuple[bool, str]:
    bbox = region.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return True, "low_info"
    img_path = resolve_region_image_path(rec, region)
    if img_path is None:
        return True, "low_info"
    try:
        gray = _crop_region_image(img_path, bbox)
    except Exception:
        return False, "keep"
    if np is None:
        # Conservative fallback if numpy is unavailable.
        return False, "keep"
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.size == 0:
        return True, "blank"

    std = float(arr.std())
    p05 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    dynamic = p95 - p05

    # Near-uniform image => blank/background-like.
    hist = np.bincount(arr.flatten(), minlength=256)
    dominant_ratio = float(hist.max() / max(1, arr.size))
    if dominant_ratio >= 0.985 or (dynamic < 6.0 and std < 2.5):
        return True, "blank"

    # Low-structure region => low information.
    gx = np.abs(np.diff(arr.astype(np.int16), axis=1))
    gy = np.abs(np.diff(arr.astype(np.int16), axis=0))
    edge_count = int((gx > 24).sum() + (gy > 24).sum())
    edge_total = max(1, gx.size + gy.size)
    edge_density = edge_count / edge_total
    if edge_density < 0.004 and dynamic < 18.0 and std < 10.0:
        return True, "low_info"

    return False, "keep"


# ---------------------------------------------------------------------------
# LLM call for a single region
# ---------------------------------------------------------------------------


def llm_visual_semantics_for_region(
    rec: SessionRecord,
    region: Dict[str, Any],
    max_image_bytes: int,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    bbox = region.get("bbox")
    if not isinstance(bbox, list):
        return None, "invalid_region_payload", ""
    img_path = resolve_region_image_path(rec, region)
    if img_path is None:
        return None, "source_image_missing", ""
    try:
        data_url = crop_region_to_data_url(img_path, bbox, max_image_bytes)
    except Exception as e:
        return None, f"crop_error:{e}", ""
    user_content = [
        {"type": "text", "text": "Extract visual semantics for this academic region and return JSON only."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    raw, status, _meta = call_openai_multimodal(
        messages=[
            {"role": "system", "content": VISUAL_REGION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=int(os.getenv("VISION_MAX_TOKENS", "32768").strip() or "32768"),
        temperature=0.0,
        response_format={"type": "json_object"},
        env_target="VISION",
    )
    if not raw:
        return None, status, ""
    parsed = parse_llm_json_object(raw)
    if parsed is None:
        return None, f"json_object_parse_failed:{status}", raw[:3000]
    chart_type = str(parsed.get("chart_type", "none")).strip().lower()
    if chart_type not in {"none", "line", "bar", "scatter", "table", "diagram", "equation", "image", "mixed"}:
        chart_type = "none"
    entities = parsed.get("entities")
    if not isinstance(entities, list):
        entities = []
    entities_clean: List[str] = []
    seen_entities = set()
    for x in entities:
        t = str(x).strip()
        if not t:
            continue
        k = re.sub(r"\s+", " ", t).strip().lower()
        if k in seen_entities:
            continue
        seen_entities.add(k)
        entities_clean.append(t)
    status_out = status if status and status != "ok" else "ok"
    return {
        "ocr_text": str(parsed.get("ocr_text", "")).strip(),
        "chart_type": chart_type,
        "visual_summary": str(parsed.get("visual_summary", "")).strip(),
        "entities": entities_clean,
    }, status_out, raw[:3000]


# ---------------------------------------------------------------------------
# Full-slide enrichment orchestrator
# ---------------------------------------------------------------------------


def enrich_slides_visual_semantics(
    rec: SessionRecord,
    slides_structured: Dict[str, Any],
    paper_structured: Dict[str, Any],
    vision_index: Dict[str, Any],
    cfg: Config,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # Visual semantics are only extracted when LLM path is enabled.
    if not cfg.enable_llm:
        return {
            "sample_id": rec.sample_id,
            "regions_total": len(slides_structured.get("visual_regions", [])) + len(paper_structured.get("paper_visual_regions", [])),
            "regions_targeted": 0,
            "regions_sent_to_llm": 0,
            "regions_enriched": 0,
            "regions_pruned_blank": 0,
            "regions_pruned_low_info": 0,
            "duplicate_summary_filtered": 0,
            "response_format_fallback_count": 0,
            "json_parse_fail_count": 0,
            "coverage": 0.0,
            "enabled": False,
        }

    per_slide = int(os.getenv("VISION_SEMANTICS_REGIONS_PER_SLIDE", "3").strip() or "3")
    per_slide = max(1, min(per_slide, 5))
    per_paper_page = max(1, min(env_int("VISION_SEMANTICS_REGIONS_PER_PAPER_PAGE", default=2, min_value=1, max_value=6), 6))

    slide_regions = slides_structured.get("visual_regions", [])
    paper_regions = paper_structured.get("paper_visual_regions", [])
    regions = list(slide_regions) + list(paper_regions)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in slide_regions:
        sid = r.get("slide_id")
        if not sid:
            continue
        grouped.setdefault(sid, []).append(r)
    for sid in grouped:
        grouped[sid].sort(key=lambda x: (0 if x.get("region_type") == "full" else 1, x.get("region_id", "")))

    targeted = []
    for sid, arr in grouped.items():
        targeted.extend(arr[:per_slide])

    paper_grouped: Dict[int, List[Dict[str, Any]]] = {}
    for r in paper_regions:
        pg = int(r.get("page", 0) or 0)
        if pg <= 0:
            continue
        paper_grouped.setdefault(pg, []).append(r)
    for pg in paper_grouped:
        paper_grouped[pg].sort(key=lambda x: x.get("region_id", ""))
        targeted.extend(paper_grouped[pg][:per_paper_page])

    pruned_blank = 0
    pruned_low_info = 0
    targeted_kept: List[Dict[str, Any]] = []
    for r in targeted:
        should_prune, reason = should_prune_region_before_llm(rec, r)
        if should_prune:
            if reason == "blank":
                pruned_blank += 1
            else:
                pruned_low_info += 1
            continue
        targeted_kept.append(r)

    # Use global VISUAL LLM executor for concurrency control across all sessions
    llm_min_interval_ms = env_int(
        "VISUAL_LLM_MIN_INTERVAL_MS",
        default=300,
        min_value=0,
        max_value=600000,
    )

    sem_by_region: Dict[str, Dict[str, Any]] = {}
    llm_failure_counts: Dict[str, int] = {}
    llm_failure_examples: List[Dict[str, Any]] = []
    llm_calls = 0
    llm_success = 0
    response_format_fallback_count = 0
    json_parse_fail_count = 0
    t0 = time.monotonic()
    indexed_targeted = list(enumerate(targeted_kept))

    def _run_one(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, Optional[Dict[str, Any]], str, str]:
        idx, region = item
        rid = str(region.get("region_id", "")).strip()
        if not rid:
            return idx, rid, None, "missing_region_id", ""
        sem, status, raw_preview = llm_visual_semantics_for_region(rec, region, cfg.max_image_bytes)
        return idx, rid, sem, status, raw_preview

    results: List[Tuple[int, str, Optional[Dict[str, Any]], str, str]] = []
    if indexed_targeted:
        # Use global VISUAL LLM executor
        ex = get_visual_llm_executor()
        futures = [ex.submit(_run_one, item) for item in indexed_targeted]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
        results.sort(key=lambda x: x[0])

    for _idx, rid, sem, status, raw_preview in results:
        if not rid:
            continue
        llm_calls += 1
        if "fallback_no_response_format" in status or "ctx_retry_no_response_format" in status:
            response_format_fallback_count += 1
        if sem:
            sem_by_region[rid] = sem
            llm_success += 1
            continue
        if status.startswith("json_object_parse_failed"):
            json_parse_fail_count += 1
        llm_failure_counts[status] = llm_failure_counts.get(status, 0) + 1
        if len(llm_failure_examples) < 5:
            llm_failure_examples.append({"region_id": rid, "status": status, "raw_preview": raw_preview})
        if debug_dir is not None:
            write_json(
                debug_dir / f"visual_{rid.split('/')[-3]}_{rid.split('/')[-1]}.json",
                {
                    "region_id": rid,
                    "status": status,
                    "raw_preview": raw_preview,
                },
            )

    # Remove templated/non-informative summaries and dedupe repeated summaries per source image.
    duplicate_summary_filtered = 0
    template_filtered = 0
    non_info_re = re.compile(
        r"\b(blank|empty|no visible text|no textual content|no visual elements|cannot determine)\b",
        flags=re.IGNORECASE,
    )
    source_seen: Dict[str, List[str]] = {}
    region_lookup: Dict[str, Dict[str, Any]] = {}
    for rr in slide_regions + paper_regions:
        rid = str(rr.get("region_id", "")).strip()
        if rid:
            region_lookup[rid] = rr

    def _token_set(s: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", s.lower()))

    def _high_overlap(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        ta = _token_set(a)
        tb = _token_set(b)
        if len(ta) < 4 or len(tb) < 4:
            return False
        inter = len(ta & tb)
        union = len(ta | tb)
        if union <= 0:
            return False
        return (inter / union) >= 0.9

    for rid in list(sem_by_region.keys()):
        sem = sem_by_region[rid]
        summary = str(sem.get("visual_summary", "") or "").strip()
        if summary and non_info_re.search(summary):
            del sem_by_region[rid]
            template_filtered += 1
            continue
        region = region_lookup.get(rid, {})
        source_key = f"{region.get('source_subdir','')}/{region.get('source_image','')}"
        seen_list = source_seen.setdefault(source_key, [])
        is_dup = any(_high_overlap(summary, s0) for s0 in seen_list if summary)
        if is_dup:
            sem["visual_summary"] = ""
            duplicate_summary_filtered += 1
            continue
        if summary:
            seen_list.append(summary)

    for r in slides_structured.get("visual_regions", []):
        rid = r.get("region_id")
        sem = sem_by_region.get(rid, {})
        r["ocr_text"] = sem.get("ocr_text", "")
        r["chart_type"] = sem.get("chart_type", "none")
        r["visual_summary"] = sem.get("visual_summary", "")
        r["entities"] = sem.get("entities", [])
    for r in paper_structured.get("paper_visual_regions", []):
        rid = r.get("region_id")
        sem = sem_by_region.get(rid, {})
        r["ocr_text"] = sem.get("ocr_text", "")
        r["chart_type"] = sem.get("chart_type", "none")
        r["visual_summary"] = sem.get("visual_summary", "")
        r["entities"] = sem.get("entities", [])
    for r in vision_index.get("regions", []):
        rid = r.get("region_id")
        sem = sem_by_region.get(rid, {})
        r["ocr_text"] = sem.get("ocr_text", "")
        r["chart_type"] = sem.get("chart_type", "none")
        r["visual_summary"] = sem.get("visual_summary", "")
        r["entities"] = sem.get("entities", [])

    targeted_n = len(targeted)
    sent_n = len(targeted_kept)
    enriched_n = len(sem_by_region)
    return {
        "sample_id": rec.sample_id,
        "regions_total": len(regions),
        "regions_targeted": targeted_n,
        "regions_sent_to_llm": sent_n,
        "regions_enriched": enriched_n,
        "regions_pruned_blank": pruned_blank,
        "regions_pruned_low_info": pruned_low_info,
        "duplicate_summary_filtered": duplicate_summary_filtered,
        "template_summary_filtered": template_filtered,
        "response_format_fallback_count": response_format_fallback_count,
        "json_parse_fail_count": json_parse_fail_count,
        "coverage": round(enriched_n / max(1, sent_n), 4),
        "enabled": True,
        "slides_regions_total": len(slide_regions),
        "paper_regions_total": len(paper_regions),
        "regions_per_slide": per_slide,
        "regions_per_paper_page": per_paper_page,
        "llm_concurrency": get_visual_llm_executor()._max_workers,
        "llm_min_interval_ms": llm_min_interval_ms,
        "wall_time_sec": round(time.monotonic() - t0, 3),
        "llm_calls": llm_calls,
        "llm_success": llm_success,
        "llm_failure_counts": llm_failure_counts,
        "llm_failure_examples": llm_failure_examples,
    }
