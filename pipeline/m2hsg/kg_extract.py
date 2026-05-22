"""Knowledge-graph extraction, scoring, repair, and quality reporting.

Centralises all functions related to building claim-centric KG triples from
multimodal evidence, validating them, scoring/tiering, repairing missing
provenance, and producing quality reports.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore[assignment]

from .alignment import (
    build_evidence_links,
    collect_metric_terms,
    normalize_content_key,
    retrieve_paper_chunks,
    split_sentences,
    stable_semantic_node_id,
    token_overlap_score,
)
from .config import (
    ALLOWED_ENTITY_TYPES,
    LOGICAL_RELATIONS,
    RELATION_EVIDENCE_WEIGHTS,
    RELATION_PHRASES,
    TIER_MEDIUM_MIN,
    TIER_STRONG_MIN,
    Config,
    SessionRecord,
    _safe_float,
    env_bool,
    env_int,
    to_float_or_none,
)
from .io_utils import write_json
from .llm_client import (
    call_openai_multimodal,
    get_triples_build_llm_executor,
    parse_llm_json_array,
    sanitize_json_obj,
)
from .paper import build_paper_dense_index, dense_search_top1
from .visual_semantics import crop_region_to_data_url


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def clean_weight(v: Any, default: float = 0.6) -> float:
    try:
        x = float(v)
    except Exception:
        x = default
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Triple validation / dedup
# ---------------------------------------------------------------------------


def validate_and_normalize_triples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .config import relation_allowed

    out = []
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        h = r.get("head") or {}
        t = r.get("tail") or {}
        rel = str(r.get("relation", "")).strip()
        prov = r.get("provenance")
        if not isinstance(h, dict) or not isinstance(t, dict):
            continue
        h_type = str(h.get("type", "")).strip()
        t_type = str(t.get("type", "")).strip()
        if h_type not in ALLOWED_ENTITY_TYPES or t_type not in ALLOWED_ENTITY_TYPES:
            continue
        if not relation_allowed(h_type, rel, t_type):
            continue
        if not isinstance(prov, list) or len(prov) == 0:
            continue

        row = {
            "head": {
                "id": str(h.get("id", "")),
                "type": h_type,
                "content": str(h.get("content", "")),
            },
            "relation": rel,
            "tail": {
                "id": str(t.get("id", "")),
                "type": t_type,
                "content": str(t.get("content", "")),
            },
            "provenance": prov,
            "weight": clean_weight(r.get("weight", 0.6)),
            "signal_source": str(r.get("signal_source", "semantic")),
        }
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def relation_count_map(triples: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for t in triples:
        rel = str(t.get("relation", ""))
        cnt[rel] = cnt.get(rel, 0) + 1
    return cnt


def dedupe_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _prov_key(p: Dict[str, Any]) -> str:
        try:
            return json.dumps(p, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(p)

    def _merge_provenance(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for p in (a or []) + (b or []):
            if not isinstance(p, dict):
                continue
            k = _prov_key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out

    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for t in triples:
        h = (t.get("head") or {}).get("id", "")
        r = t.get("relation", "")
        tl = (t.get("tail") or {}).get("id", "")
        if not (h and r and tl):
            continue
        key = (h, r, tl)
        prev = best.get(key)
        if prev is None:
            best[key] = t
            continue
        prev_w = _safe_float(prev.get("weight", 0.0), 0.0)
        cur_w = _safe_float(t.get("weight", 0.0), 0.0)
        keeper = prev if prev_w >= cur_w else t
        other = t if keeper is prev else prev
        merged = dict(keeper)
        merged["weight"] = round(max(prev_w, cur_w), 4)
        merged["provenance"] = _merge_provenance(
            keeper.get("provenance", []) if isinstance(keeper.get("provenance", []), list) else [],
            other.get("provenance", []) if isinstance(other.get("provenance", []), list) else [],
        )
        s1 = str(prev.get("signal_source", "")).strip()
        s2 = str(t.get("signal_source", "")).strip()
        merged["signal_source"] = s1 if s1 == s2 else "hybrid"
        rt_a = keeper.get("repair_trace", []) if isinstance(keeper.get("repair_trace", []), list) else []
        rt_b = other.get("repair_trace", []) if isinstance(other.get("repair_trace", []), list) else []
        if rt_a or rt_b:
            merged["repair_trace"] = _merge_provenance(rt_a, rt_b)
        if bool(keeper.get("is_repaired")) or bool(other.get("is_repaired")):
            merged["is_repaired"] = True
        best[key] = merged
    return list(best.values())


def dedupe_provenance_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for p in items:
        if not isinstance(p, dict):
            continue
        try:
            k = json.dumps(p, sort_keys=True, ensure_ascii=False)
        except Exception:
            k = str(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def provenance_modality(p: Dict[str, Any]) -> Optional[str]:
    if not isinstance(p, dict):
        return None
    st = str(p.get("source_type", "")).strip().lower()
    if st == "audio":
        return "audio"
    if st in {"slide", "slide_region", "visual_region"}:
        return "slide"
    if st in {"paper", "paper_chunk", "figure", "table"}:
        return "paper"
    if p.get("slide_id") is not None or p.get("region_id") is not None:
        return "slide"
    if p.get("chunk_id") is not None or p.get("page") is not None:
        return "paper"
    return None


def provenance_quality(modality: str, p: Dict[str, Any]) -> float:
    if modality == "audio":
        st = to_float_or_none(p.get("start_time"))
        et = to_float_or_none(p.get("end_time"))
        return 1.0 if (st is not None and et is not None and et > st) else 0.6

    if modality == "slide":
        slide_id = str(p.get("slide_id", "")).strip()
        region_id = str(p.get("region_id", "")).strip()
        bbox = p.get("bbox")
        has_bbox = isinstance(bbox, list) and len(bbox) == 4
        if region_id and has_bbox:
            return 1.0
        if slide_id:
            return 0.6
        return 0.0

    if modality == "paper":
        if bool(p.get("is_repaired")) and str(p.get("repair_method", "")).strip() == "dense_retrieval":
            return 0.9
        chunk_id = str(p.get("chunk_id", "")).strip()
        page = p.get("page")
        bbox = p.get("bbox")
        has_anchor = (page is not None) or (isinstance(bbox, list) and len(bbox) == 4)
        if chunk_id and has_anchor:
            return 1.0
        if str(p.get("source_type", "")).strip() in {"figure", "table"} and has_anchor:
            return 1.0
        if chunk_id or has_anchor:
            return 0.6
        return 0.0
    return 0.0


def triple_has_modality(triple: Dict[str, Any], modality: str) -> bool:
    prov = triple.get("provenance", [])
    if not isinstance(prov, list):
        return False
    for p in prov:
        if isinstance(p, dict) and provenance_modality(p) == modality:
            return True
    return False


def slide_id_for_time(slides: List[Dict[str, Any]], t: float) -> Optional[str]:
    for i, s in enumerate(slides):
        st = to_float_or_none(s.get("start_time"))
        et = to_float_or_none(s.get("end_time"))
        sid = str(s.get("slide_id", "")).strip()
        if not sid or st is None:
            continue
        if et is None:
            if t >= st:
                return sid
            continue
        if st <= t < et:
            return sid
        if i == len(slides) - 1 and t >= st:
            return sid
    return None


# ---------------------------------------------------------------------------
# Semantic node extraction from paper
# ---------------------------------------------------------------------------


def extract_semantic_nodes_from_paper(rec: SessionRecord, paper_structured: Dict[str, Any]) -> Dict[str, Any]:
    chunks = paper_structured.get("paper_chunks", [])
    cand_rows = []

    claim_pat = re.compile(r"\b(we propose|we present|our method|our approach|we introduce|we show|we demonstrate)\b", re.IGNORECASE)
    method_pat = re.compile(r"\b(method|framework|network|architecture|pipeline|module|optimizer)\b", re.IGNORECASE)
    result_pat = re.compile(r"\b(outperform|state[- ]of[- ]the[- ]art|improve|achieve|better than|superior)\b", re.IGNORECASE)
    limit_pat = re.compile(r"\b(limitation|future work|however|fails|challenge|difficult|still hard)\b", re.IGNORECASE)

    for c in chunks:
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]
        for sent in sentences:
            s = sent.strip()
            low = s.lower()
            if len(s) < 20:
                continue

            if claim_pat.search(low):
                cand_rows.append(("Claim", s, 0.78, c))
            if method_pat.search(low) and ("we " in low or "our " in low or "propose" in low):
                cand_rows.append(("Method", s, 0.72, c))
            if result_pat.search(low) and re.search(r"\d", s):
                cand_rows.append(("Result", s, 0.74, c))
            if limit_pat.search(low):
                cand_rows.append(("Limitation", s, 0.62, c))
            for m in collect_metric_terms(s):
                cand_rows.append(("Metric", m, 0.7, c))

    if chunks:
        first_non_empty = next((x for x in chunks if str(x.get("text", "")).strip()), None)
        if first_non_empty:
            title = str(rec.metadata.get("title", "")).strip()
            if title:
                cand_rows.append(("Claim", title, 0.6, first_non_empty))
                cand_rows.append(("Method", title, 0.58, first_non_empty))

    dedup: Dict[Tuple[str, str], Tuple[str, float, Dict[str, Any]]] = {}
    for typ, content, conf, c in cand_rows:
        key = (typ, normalize_content_key(content))
        prev = dedup.get(key)
        if prev is None or conf > prev[1]:
            dedup[key] = (content, conf, c)

    limits = {
        "Claim": max(1, int(os.getenv("SEM_LIMIT_CLAIM", "14").strip() or "14")),
        "Method": max(1, int(os.getenv("SEM_LIMIT_METHOD", "16").strip() or "16")),
        "Result": max(1, int(os.getenv("SEM_LIMIT_RESULT", "20").strip() or "20")),
        "Metric": max(1, int(os.getenv("SEM_LIMIT_METRIC", "20").strip() or "20")),
        "Limitation": max(1, int(os.getenv("SEM_LIMIT_LIMITATION", "10").strip() or "10")),
    }
    grouped: Dict[str, List[Tuple[str, float, Dict[str, Any]]]] = {}
    for (typ, _), row in dedup.items():
        grouped.setdefault(typ, []).append(row)

    nodes = []
    for typ, rows in grouped.items():
        rows.sort(key=lambda x: x[1], reverse=True)
        for content, conf, c in rows[: limits.get(typ, 8)]:
            chunk_id = c.get("chunk_id")
            if not chunk_id:
                continue
            nodes.append({
                "id": stable_semantic_node_id(rec.sample_id, typ, content),
                "type": typ,
                "content": content,
                "confidence": round(float(conf), 3),
                "source_chunk_id": chunk_id,
                "source_page": c.get("page"),
                "source_bbox": c.get("bbox", [0, 0, 1000, 1400]),
                "chunk_text_preview": str(c.get("text", ""))[:180],
            })

    counts: Dict[str, int] = {}
    for n in nodes:
        counts[n["type"]] = counts.get(n["type"], 0) + 1
    return {
        "sample_id": rec.sample_id,
        "nodes": nodes,
        "counts_by_type": counts,
    }


# ---------------------------------------------------------------------------
# Claim window building
# ---------------------------------------------------------------------------


def build_claim_windows(
    rec: SessionRecord,
    semantic_nodes: Dict[str, Any],
    evidence_links: Dict[str, Any],
    max_claim_chunks: int,
    max_claim_utts: int,
    max_claim_slides: int,
    max_claim_regions: int,
    max_claim_events: int,
    max_claim_signals: int,
) -> Dict[str, Any]:
    nodes = semantic_nodes.get("nodes", [])
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault(n.get("type", ""), []).append(n)
    for typ in by_type:
        by_type[typ].sort(key=lambda x: _safe_float(x.get("confidence", 0.0), 0.0), reverse=True)

    evidence_by_node = {x.get("node_id"): x for x in evidence_links.get("node_evidence", [])}
    claims = by_type.get("Claim", [])
    methods = by_type.get("Method", [])
    results = by_type.get("Result", [])
    metrics = by_type.get("Metric", [])
    limits_list = by_type.get("Limitation", [])

    per_claim = []
    for claim in claims:
        c_ev = evidence_by_node.get(claim.get("id"), {})

        def pick_related(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
            scored = []
            for r in rows:
                s = token_overlap_score(claim.get("content", ""), r.get("content", ""))
                scored.append((s + 0.1 * _safe_float(r.get("confidence", 0.0), 0.0), r))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored[:k]]

        rel_nodes = []
        rel_nodes.extend(pick_related(methods, 2))
        rel_nodes.extend(pick_related(results, 3))
        rel_nodes.extend(pick_related(metrics, 3))
        rel_nodes.extend(pick_related(limits_list, 1))

        merged_chunks = []
        merged_utts = []
        merged_slides = []
        merged_regions = []
        merged_events = []
        merged_signals = []

        node_bundle = [claim] + rel_nodes
        for n in node_bundle:
            ev = evidence_by_node.get(n.get("id"), {})
            merged_chunks.extend(ev.get("paper_chunks", []))
            merged_utts.extend(ev.get("utterances", []))
            merged_slides.extend(ev.get("slides", []))
            merged_regions.extend(ev.get("visual_regions", []))
            merged_events.extend(ev.get("prosody_events", []))
            merged_signals.extend(ev.get("pragmatic_signals", []))

        def dedupe(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
            out = []
            seen = set()
            for r in rows:
                k = r.get(key)
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(r)
            return out

        per_claim.append({
            "claim": claim,
            "related_nodes": rel_nodes,
            "evidence": {
                "paper_chunks": dedupe(merged_chunks, "chunk_id")[:max_claim_chunks],
                "utterances": dedupe(merged_utts, "utterance_id")[:max_claim_utts],
                "slides": dedupe(merged_slides, "slide_id")[:max_claim_slides],
                "visual_regions": dedupe(merged_regions, "region_id")[:max_claim_regions],
                "prosody_events": dedupe(merged_events, "event_id")[:max_claim_events],
                "pragmatic_signals": dedupe(merged_signals, "signal_id")[:max_claim_signals],
            },
        })

    return {
        "sample_id": rec.sample_id,
        "mode": "per_claim",
        "per_claim": per_claim,
    }


# ---------------------------------------------------------------------------
# Token budget helpers
# ---------------------------------------------------------------------------


def estimate_json_tokens_approx(obj: Any) -> int:
    try:
        s = json.dumps(sanitize_json_obj(obj), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        try:
            s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return 0
    return max(1, len(s) // 4)


def _split_text_semantic_windows(text: str, target_chars: int = 2200, overlap_chars: int = 250) -> List[str]:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return []
    if target_chars <= 0 or len(t) <= target_chars:
        return [t]

    sents = re.split(r"(?<=[.!?。！？；;])\s+", t)
    sents = [s.strip() for s in sents if s and s.strip()]
    if not sents:
        sents = [t]

    windows: List[str] = []
    cur = ""
    for s in sents:
        if len(s) > target_chars:
            if cur:
                windows.append(cur.strip())
                cur = ""
            step = max(1, target_chars - max(0, overlap_chars))
            for i in range(0, len(s), step):
                part = s[i:i + target_chars].strip()
                if part:
                    windows.append(part)
            continue
        if not cur:
            cur = s
            continue
        if len(cur) + 1 + len(s) <= target_chars:
            cur = f"{cur} {s}"
        else:
            windows.append(cur.strip())
            cur = s
    if cur:
        windows.append(cur.strip())

    if overlap_chars > 0 and len(windows) > 1:
        with_overlap: List[str] = []
        prev_tail = ""
        for w in windows:
            if prev_tail:
                merged = f"{prev_tail} {w}".strip()
                with_overlap.append(merged)
            else:
                with_overlap.append(w)
            prev_tail = w[-overlap_chars:].strip()
        windows = with_overlap
    return windows


def _expand_claim_payload_paper_chunks_for_budget(
    claim_payload: Dict[str, Any],
    target_chars: int,
    overlap_chars: int,
) -> Dict[str, Any]:
    cp = sanitize_json_obj(claim_payload)
    ev = cp.get("evidence", {}) if isinstance(cp.get("evidence", {}), dict) else {}
    chunks = ev.get("paper_chunks", [])
    if not isinstance(chunks, list) or not chunks:
        return cp

    expanded: List[Dict[str, Any]] = []
    for idx, c in enumerate(chunks):
        if not isinstance(c, dict):
            continue
        txt = str(c.get("text", "") or "").strip()
        if not txt:
            expanded.append(c)
            continue
        parts = _split_text_semantic_windows(txt, target_chars=target_chars, overlap_chars=overlap_chars)
        if len(parts) <= 1:
            expanded.append(c)
            continue
        base_id = str(c.get("chunk_id", f"chunk_{idx}")).strip() or f"chunk_{idx}"
        total = len(parts)
        for p_i, part in enumerate(parts, start=1):
            cc = dict(c)
            cc["source_chunk_id"] = base_id
            cc["chunk_id"] = f"{base_id}#part_{p_i:03d}"
            cc["text"] = part
            cc["text_part_index"] = p_i
            cc["text_part_total"] = total
            expanded.append(cc)
    ev["paper_chunks"] = expanded
    cp["evidence"] = ev
    return cp


def split_claim_payload_for_budget(
    claim_payload: Dict[str, Any],
    token_budget: int,
    max_shards: int,
) -> List[Dict[str, Any]]:
    if token_budget <= 0:
        return [claim_payload]
    max_shards = max(1, max_shards)
    split_target_chars = env_int("TRIPLES_BUILD_PAPER_CHUNK_SPLIT_CHARS", default=2200, min_value=256, max_value=20000)
    split_overlap_chars = env_int("TRIPLES_BUILD_PAPER_CHUNK_SPLIT_OVERLAP_CHARS", default=250, min_value=0, max_value=4000)
    working_payload = _expand_claim_payload_paper_chunks_for_budget(
        claim_payload=claim_payload,
        target_chars=split_target_chars,
        overlap_chars=split_overlap_chars,
    )
    est = estimate_json_tokens_approx(working_payload)
    if est <= token_budget or max_shards == 1:
        return [working_payload]

    ev = working_payload.get("evidence", {}) if isinstance(working_payload.get("evidence", {}), dict) else {}
    keys = ["paper_chunks", "utterances", "slides", "visual_regions", "prosody_events", "pragmatic_signals"]
    lengths = [len(ev.get(k, [])) if isinstance(ev.get(k, []), list) else 0 for k in keys]
    max_len = max(lengths) if lengths else 0
    if max_len <= 1:
        return [working_payload]

    n_shards = 1
    while n_shards < max_shards and (est / n_shards) > token_budget:
        n_shards *= 2
    n_shards = max(1, min(n_shards, max_shards, max_len))
    if n_shards <= 1:
        return [working_payload]

    def _slice_even(rows: List[Any], idx: int, total: int) -> List[Any]:
        if not rows:
            return []
        n = len(rows)
        start = (n * idx) // total
        end = (n * (idx + 1)) // total
        return rows[start:end]

    shards: List[Dict[str, Any]] = []
    for i in range(n_shards):
        evi: Dict[str, Any] = {}
        for k in keys:
            rows = ev.get(k, [])
            if not isinstance(rows, list):
                rows = []
            evi[k] = _slice_even(rows, i, n_shards)
        if not any(evi.get(k) for k in keys):
            continue
        shard = {
            "sample_id": working_payload.get("sample_id"),
            "mode": working_payload.get("mode", "per_claim"),
            "claim": working_payload.get("claim", {}),
            "related_nodes": working_payload.get("related_nodes", []),
            "evidence": evi,
            "shard_index": i,
            "shard_total": n_shards,
        }
        shards.append(shard)
    return shards or [working_payload]


# ---------------------------------------------------------------------------
# LLM KG extraction for a single claim
# ---------------------------------------------------------------------------


def llm_kg_extract_for_claim(
    rec: SessionRecord,
    claim_payload: Dict[str, Any],
    max_images_per_call: int,
    max_image_bytes: int,
    debug_dir: Optional[Path] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    from pathlib import Path

    base = os.getenv("TRIPLES_BUILD_BASE_URL", "").strip()
    key = os.getenv("TRIPLES_BUILD_API_KEY", "").strip()
    model = os.getenv("TRIPLES_BUILD_MODEL", "").strip()
    if not (base and key and model):
        return None, "missing_llm_env"

    safe_claim_payload = sanitize_json_obj(claim_payload)
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(safe_claim_payload, ensure_ascii=False)}]
    imgs = 0
    for reg in claim_payload.get("evidence", {}).get("visual_regions", []):
        if imgs >= max_images_per_call:
            break
        source_image = reg.get("source_image")
        bbox = reg.get("bbox")
        if not source_image or not isinstance(bbox, list):
            continue
        source_subdir = str(reg.get("source_subdir", "") or "").strip()
        search_dirs = [source_subdir] if source_subdir else []
        search_dirs.extend(["slides", "paper_pages"])
        img_path: Optional[Path] = None
        for d in search_dirs:
            p = rec.path / d / str(source_image)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        try:
            data_url = crop_region_to_data_url(img_path, bbox, max_image_bytes)
        except Exception:
            continue
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        imgs += 1

    triples_max_tokens = int(os.getenv("TRIPLES_BUILD_MAX_TOKENS", "32768").strip() or "32768")
    triples_retry_max_tokens = int(os.getenv("TRIPLES_BUILD_RETRY_MAX_TOKENS", "32768").strip() or "32768")
    raw, status, meta = call_openai_multimodal([
        {"role": "system", "content": os.getenv("TRIPLES_BUILD_SYSTEM_PROMPT", "")},
        {"role": "user", "content": user_content},
    ], max_tokens=triples_max_tokens, response_format={"type": "json_object"})
    if not raw:
        return None, status
    parsed = parse_llm_json_array(raw)
    if parsed is None:
        if meta.get("finish_reason") == "length" and triples_retry_max_tokens > triples_max_tokens:
            raw2, status2, meta2 = call_openai_multimodal(
                [
                    {"role": "system", "content": os.getenv("TRIPLES_BUILD_SYSTEM_PROMPT", "")},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=triples_retry_max_tokens,
                response_format={"type": "json_object"},
            )
            if raw2:
                parsed2 = parse_llm_json_array(raw2)
                if parsed2 is not None:
                    triples2 = validate_and_normalize_triples(parsed2)
                    if triples2:
                        return triples2, f"ok_retry_from_truncation:{status2}"
                raw = raw2
                status = f"json_array_parse_failed_after_retry:{status2}"
                meta = meta2
        if debug_dir is not None:
            cid = str((claim_payload.get("claim") or {}).get("id", "unknown")).replace("/", "_")
            write_json(
                debug_dir / f"kg_claim_{cid}.json",
                {
                    "status": "json_array_parse_failed",
                    "claim_id": (claim_payload.get("claim") or {}).get("id"),
                    "raw_preview": raw[:5000],
                    "llm_status": status,
                    "finish_reason": meta.get("finish_reason"),
                    "usage": meta.get("usage", {}),
                },
            )
        return None, "json_array_parse_failed"
    triples = validate_and_normalize_triples(parsed)
    if not triples:
        if debug_dir is not None:
            cid = str((claim_payload.get("claim") or {}).get("id", "unknown")).replace("/", "_")
            write_json(
                debug_dir / f"kg_claim_empty_{cid}.json",
                {
                    "status": "filtered_to_empty",
                    "claim_id": (claim_payload.get("claim") or {}).get("id"),
                    "raw_parsed": parsed,
                },
            )
        return [], "ok_empty"
    return triples, "ok"


# ---------------------------------------------------------------------------
# Repair helpers
# ---------------------------------------------------------------------------


def candidate_utterance_ids_for_edge(
    row: Dict[str, Any],
    evidence_by_node: Dict[str, Dict[str, Any]],
    prosody_to_utterance: Dict[str, str],
) -> List[str]:
    out: List[str] = []
    seen = set()

    def _add(uid: Optional[str]) -> None:
        if not uid:
            return
        u = str(uid).strip()
        if not u or u in seen:
            return
        seen.add(u)
        out.append(u)

    head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
    tail = row.get("tail", {}) if isinstance(row.get("tail", {}), dict) else {}
    h_type = str(head.get("type", "")).strip()
    t_type = str(tail.get("type", "")).strip()
    h_id = str(head.get("id", "")).strip()
    t_id = str(tail.get("id", "")).strip()

    if h_type == "Utterance":
        _add(h_id)
    if t_type == "Utterance":
        _add(t_id)
    if h_type == "ProsodyEvent":
        _add(prosody_to_utterance.get(h_id))
    if t_type == "ProsodyEvent":
        _add(prosody_to_utterance.get(t_id))

    for nid in [h_id, t_id]:
        ev = evidence_by_node.get(nid)
        if not ev:
            continue
        for u in ev.get("utterances", []) or []:
            if isinstance(u, dict):
                _add(u.get("utterance_id"))
            if len(out) >= 12:
                break
        if len(out) >= 12:
            break
    return out


def add_repair_entry(
    row: Dict[str, Any],
    provenance_obj: Dict[str, Any],
    trace_obj: Dict[str, Any],
) -> None:
    prov = row.get("provenance")
    if not isinstance(prov, list):
        prov = []
        row["provenance"] = prov
    prov.append(provenance_obj)
    trace = row.get("repair_trace")
    if not isinstance(trace, list):
        trace = []
        row["repair_trace"] = trace
    trace.append(trace_obj)
    row["is_repaired"] = True


def repair_slide_provenance_for_edge(
    row: Dict[str, Any],
    align_by_uid: Dict[str, Dict[str, Any]],
    utter_by_id: Dict[str, Dict[str, Any]],
    evidence_by_node: Dict[str, Dict[str, Any]],
    prosody_to_utterance: Dict[str, str],
    region_by_id: Dict[str, Dict[str, Any]],
    regions_by_slide: Dict[str, List[Dict[str, Any]]],
    slides: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    rel = str(row.get("relation", "")).strip()
    head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
    target = rel == "aligned_to_slide" or (rel == "referenced_by" and str(head.get("type", "")).strip() == "VisualRegion")
    if not target:
        return False, "not_target_relation"
    if triple_has_modality(row, "slide"):
        return False, "already_has_slide"

    h_id = str(head.get("id", "")).strip()
    if rel == "referenced_by" and str(head.get("type", "")).strip() == "VisualRegion" and h_id:
        reg = region_by_id.get(h_id)
        if reg:
            sid = str(reg.get("slide_id", "")).strip()
            bbox = reg.get("bbox", [0, 0, 1000, 1000])
            add_repair_entry(
                row,
                {
                    "source_type": "slide_region",
                    "slide_id": sid,
                    "region_id": h_id,
                    "bbox": bbox,
                    "is_repaired": True,
                    "repair_method": "time_alignment",
                    "repaired_granularity": "region_level",
                },
                {
                    "modality": "slide",
                    "repair_method": "time_alignment",
                    "repaired_granularity": "region_level",
                    "reason": "head_visual_region_anchor",
                    "region_id": h_id,
                    "slide_id": sid,
                },
            )
            return True, "region_level"

    utt_ids = candidate_utterance_ids_for_edge(row, evidence_by_node, prosody_to_utterance)
    candidate_times: List[float] = []

    for uid in utt_ids:
        a = align_by_uid.get(uid, {})
        sid = str(a.get("slide_id", "")).strip()
        vis = a.get("visual_regions", []) or []
        if sid:
            for r in vis:
                if not isinstance(r, dict):
                    continue
                rid = str(r.get("region_id", "")).strip()
                bbox = r.get("bbox")
                if rid and isinstance(bbox, list) and len(bbox) == 4:
                    add_repair_entry(
                        row,
                        {
                            "source_type": "slide_region",
                            "slide_id": sid,
                            "region_id": rid,
                            "bbox": bbox,
                            "is_repaired": True,
                            "repair_method": "time_alignment",
                            "repaired_granularity": "region_level",
                        },
                        {
                            "modality": "slide",
                            "repair_method": "time_alignment",
                            "repaired_granularity": "region_level",
                            "reason": "utterance_alignment",
                            "utterance_id": uid,
                            "slide_id": sid,
                            "region_id": rid,
                        },
                    )
                    return True, "region_level"
            s_regs = regions_by_slide.get(sid, [])
            if s_regs:
                fr = s_regs[0]
                rid = str(fr.get("region_id", "")).strip()
                bbox = fr.get("bbox", [0, 0, 1000, 1000])
                if rid and isinstance(bbox, list) and len(bbox) == 4:
                    add_repair_entry(
                        row,
                        {
                            "source_type": "slide_region",
                            "slide_id": sid,
                            "region_id": rid,
                            "bbox": bbox,
                            "is_repaired": True,
                            "repair_method": "time_alignment",
                            "repaired_granularity": "region_level",
                        },
                        {
                            "modality": "slide",
                            "repair_method": "time_alignment",
                            "repaired_granularity": "region_level",
                            "reason": "utterance_alignment_default_region",
                            "utterance_id": uid,
                            "slide_id": sid,
                            "region_id": rid,
                        },
                    )
                    return True, "region_level"

        u = utter_by_id.get(uid)
        if u:
            tm = to_float_or_none(u.get("start_time"))
            if tm is not None:
                candidate_times.append(tm)

    for tm in candidate_times:
        sid = slide_id_for_time(slides, tm)
        if sid:
            s_regs = regions_by_slide.get(sid, [])
            if s_regs:
                fr = s_regs[0]
                rid = str(fr.get("region_id", "")).strip()
                bbox = fr.get("bbox", [0, 0, 1000, 1000])
                if rid and isinstance(bbox, list) and len(bbox) == 4:
                    add_repair_entry(
                        row,
                        {
                            "source_type": "slide_region",
                            "slide_id": sid,
                            "region_id": rid,
                            "bbox": bbox,
                            "is_repaired": True,
                            "repair_method": "time_alignment",
                            "repaired_granularity": "region_level",
                        },
                        {
                            "modality": "slide",
                            "repair_method": "time_alignment",
                            "repaired_granularity": "region_level",
                            "reason": "audio_time_alignment",
                            "time": round(float(tm), 3),
                            "slide_id": sid,
                            "region_id": rid,
                        },
                    )
                    return True, "region_level"
        add_repair_entry(
            row,
            {
                "source_type": "slide",
                "slide_id": sid,
                "is_repaired": True,
                "repair_method": "time_alignment",
                "repaired_granularity": "page_level",
            },
            {
                "modality": "slide",
                "repair_method": "time_alignment",
                "repaired_granularity": "page_level",
                "reason": "audio_time_alignment_page",
                "time": round(float(tm), 3),
                "slide_id": sid,
            },
        )
        return True, "page_level"

    return False, "no_slide_candidate"


def repair_paper_provenance_for_edge(
    row: Dict[str, Any],
    index_state: Dict[str, Any],
    threshold: float,
    query_cache: Dict[str, Any],
) -> Tuple[bool, str, float]:
    rel = str(row.get("relation", "")).strip()
    w_paper = float(RELATION_EVIDENCE_WEIGHTS.get(rel, {}).get("paper", 0.0))
    if w_paper <= 0.0:
        return False, "not_paper_relation", 0.0
    if triple_has_modality(row, "paper"):
        return False, "already_has_paper", 0.0
    if not index_state.get("enabled"):
        return False, str(index_state.get("status", "index_disabled")), 0.0

    head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
    tail = row.get("tail", {}) if isinstance(row.get("tail", {}), dict) else {}
    phrase = RELATION_PHRASES.get(rel, rel.replace("_", " "))
    query = f'{str(head.get("content", "")).strip()} {phrase} {str(tail.get("content", "")).strip()}'.strip()
    match, sim, st = dense_search_top1(index_state=index_state, query=query, query_cache=query_cache)
    if match is None:
        return False, st, 0.0
    if sim < threshold:
        return False, "fused_below_threshold", sim

    add_repair_entry(
        row,
        {
            "source_type": "paper_chunk",
            "chunk_id": match.get("chunk_id"),
            "page": match.get("page"),
            "bbox": match.get("bbox", [0, 0, 1000, 1400]),
            "chunk_text_preview": str(match.get("text", ""))[:220],
            "is_repaired": True,
            "repair_method": "dense_retrieval",
            "similarity_score": round(float(sim), 4),
        },
        {
            "modality": "paper",
            "repair_method": "dense_retrieval",
            "repaired_granularity": "chunk_level",
            "similarity_score": round(float(sim), 4),
            "chunk_id": match.get("chunk_id"),
        },
    )
    return True, "repaired", sim


# ---------------------------------------------------------------------------
# Scoring / tiering
# ---------------------------------------------------------------------------


def score_and_tier_edge(row: Dict[str, Any]) -> Dict[str, Any]:
    prov = row.get("provenance", [])
    if not isinstance(prov, list):
        prov = []
        row["provenance"] = prov
    prov = dedupe_provenance_items([p for p in prov if isinstance(p, dict)])
    row["provenance"] = prov

    rel = str(row.get("relation", "")).strip()
    weights = RELATION_EVIDENCE_WEIGHTS.get(rel, {"audio": 0.0, "slide": 0.0, "paper": 0.0})
    qmax = {"audio": 0.0, "slide": 0.0, "paper": 0.0}
    repaired_grans: List[str] = []
    for p in prov:
        m = provenance_modality(p)
        if m not in qmax:
            continue
        q = provenance_quality(m, p)
        if q > qmax[m]:
            qmax[m] = q
        if bool(p.get("is_repaired")):
            g = str(p.get("repaired_granularity", "")).strip()
            if g:
                repaired_grans.append(g)

    score = 0.0
    for m in ["audio", "slide", "paper"]:
        w = float(weights.get(m, 0.0))
        present = 1.0 if qmax[m] > 0 else 0.0
        score += w * present * qmax[m]
    score = max(0.0, min(1.0, score))

    tier = "weak"
    if score >= TIER_STRONG_MIN:
        tier = "strong"
    elif score >= TIER_MEDIUM_MIN:
        tier = "medium"

    repaired = bool(row.get("is_repaired")) or any(bool(p.get("is_repaired")) for p in prov)
    if repaired and "page_level" in set(repaired_grans) and tier == "strong":
        tier = "medium"

    row["provenance_score"] = round(float(score), 4)
    row["tier"] = tier
    row["is_repaired"] = repaired
    rt = row.get("repair_trace", [])
    if isinstance(rt, list) and rt:
        row["repair_trace"] = dedupe_provenance_items([x for x in rt if isinstance(x, dict)])
    else:
        row.pop("repair_trace", None)
    return row


# ---------------------------------------------------------------------------
# Soft scoring + repairs (batch)
# ---------------------------------------------------------------------------


def apply_soft_scoring_and_repairs(
    rec: SessionRecord,
    triples: List[Dict[str, Any]],
    slides_structured: Dict[str, Any],
    paper_structured: Dict[str, Any],
    alignment: Dict[str, Any],
    evidence_links: Dict[str, Any],
    transcript_prosody: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    repair_enabled = env_bool("REPAIR_ENABLE", True)
    threshold = float(os.getenv("REPAIR_PAPER_SIM_THRESHOLD", "0.82").strip() or "0.82")

    align_by_uid: Dict[str, Dict[str, Any]] = {}
    for r in alignment.get("utterance_alignment", []) or []:
        if not isinstance(r, dict):
            continue
        uid = str(r.get("utterance_id", "")).strip()
        if uid:
            align_by_uid[uid] = r
    utter_by_id = {
        str(u.get("utterance_id", "")).strip(): u
        for u in transcript_prosody.get("utterances", []) or []
        if isinstance(u, dict) and str(u.get("utterance_id", "")).strip()
    }
    prosody_to_utterance = {
        str(e.get("event_id", "")).strip(): str(e.get("utterance_id", "")).strip()
        for e in transcript_prosody.get("prosody_events", []) or []
        if isinstance(e, dict) and str(e.get("event_id", "")).strip()
    }
    evidence_by_node = {
        str(x.get("node_id", "")).strip(): x
        for x in evidence_links.get("node_evidence", []) or []
        if isinstance(x, dict) and str(x.get("node_id", "")).strip()
    }
    slides = slides_structured.get("slides", []) or []
    region_by_id: Dict[str, Dict[str, Any]] = {}
    regions_by_slide: Dict[str, List[Dict[str, Any]]] = {}
    for r in slides_structured.get("visual_regions", []) or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("region_id", "")).strip()
        sid = str(r.get("slide_id", "")).strip()
        if rid:
            region_by_id[rid] = r
        if sid:
            regions_by_slide.setdefault(sid, []).append(r)
    for sid in regions_by_slide:
        regions_by_slide[sid].sort(key=lambda x: (0 if x.get("region_type") == "full" else 1, str(x.get("region_id", ""))))

    paper_candidates = 0
    for t in triples:
        rel = str(t.get("relation", "")).strip()
        if float(RELATION_EVIDENCE_WEIGHTS.get(rel, {}).get("paper", 0.0)) > 0.0 and not triple_has_modality(t, "paper"):
            paper_candidates += 1
    dense_index = {"enabled": False, "status": "repair_disabled", "backend": "none"}
    if repair_enabled and paper_candidates > 0:
        dense_index = build_paper_dense_index(paper_structured)
    query_cache: Dict[str, Any] = {}

    stats = {
        "enabled": repair_enabled,
        "slide_attempted": 0,
        "slide_repaired_region": 0,
        "slide_repaired_page": 0,
        "slide_repair_failed": 0,
        "paper_attempted": 0,
        "paper_repaired": 0,
        "paper_fused": 0,
        "paper_skipped": 0,
        "paper_backend": str(dense_index.get("backend", "none")),
        "paper_index_status": str(dense_index.get("status", "")),
        "paper_repair_threshold": threshold,
    }

    out: List[Dict[str, Any]] = []
    for t in triples:
        row = {
            "head": dict(t.get("head", {}) if isinstance(t.get("head", {}), dict) else {}),
            "relation": str(t.get("relation", "")).strip(),
            "tail": dict(t.get("tail", {}) if isinstance(t.get("tail", {}), dict) else {}),
            "provenance": [dict(p) for p in (t.get("provenance", []) or []) if isinstance(p, dict)],
            "weight": clean_weight(t.get("weight", 0.6)),
            "signal_source": str(t.get("signal_source", "semantic")),
        }
        if isinstance(t.get("repair_trace"), list):
            row["repair_trace"] = [dict(x) for x in t.get("repair_trace") if isinstance(x, dict)]
        if bool(t.get("is_repaired")):
            row["is_repaired"] = True

        if repair_enabled:
            rel = row["relation"]
            head_type = str((row.get("head") or {}).get("type", "")).strip()
            target_slide = rel == "aligned_to_slide" or (rel == "referenced_by" and head_type == "VisualRegion")
            if target_slide and not triple_has_modality(row, "slide"):
                stats["slide_attempted"] += 1
                repaired, level = repair_slide_provenance_for_edge(
                    row=row,
                    align_by_uid=align_by_uid,
                    utter_by_id=utter_by_id,
                    evidence_by_node=evidence_by_node,
                    prosody_to_utterance=prosody_to_utterance,
                    region_by_id=region_by_id,
                    regions_by_slide=regions_by_slide,
                    slides=slides,
                )
                if repaired:
                    if level == "region_level":
                        stats["slide_repaired_region"] += 1
                    elif level == "page_level":
                        stats["slide_repaired_page"] += 1
                else:
                    stats["slide_repair_failed"] += 1

            if float(RELATION_EVIDENCE_WEIGHTS.get(rel, {}).get("paper", 0.0)) > 0.0 and not triple_has_modality(row, "paper"):
                stats["paper_attempted"] += 1
                repaired, status, sim = repair_paper_provenance_for_edge(
                    row=row,
                    index_state=dense_index,
                    threshold=threshold,
                    query_cache=query_cache,
                )
                if repaired:
                    stats["paper_repaired"] += 1
                else:
                    if status == "fused_below_threshold":
                        stats["paper_fused"] += 1
                    else:
                        stats["paper_skipped"] += 1

        row["provenance"] = dedupe_provenance_items(row.get("provenance", []))
        row = score_and_tier_edge(row)
        out.append(row)

    tier_counts = {"strong": 0, "medium": 0, "weak": 0}
    for r in out:
        t = str(r.get("tier", "weak"))
        if t not in tier_counts:
            t = "weak"
        tier_counts[t] += 1

    meta = {
        "repair_stats": stats,
        "tier_counts": tier_counts,
        "paper_repair_fuse_count": stats["paper_fused"],
        "paper_repair_threshold": threshold,
        "repair_backend": stats["paper_backend"],
        "repair_backend_status": stats["paper_index_status"],
    }
    return out, meta


# ---------------------------------------------------------------------------
# Coverage / quality
# ---------------------------------------------------------------------------


def coverage_pass(triples: List[Dict[str, Any]]) -> bool:
    cnt = relation_count_map(triples)
    logic_total = sum(cnt.get(x, 0) for x in LOGICAL_RELATIONS)
    has_core = (cnt.get("supported_by", 0) + cnt.get("measured_by", 0)) >= 1
    return logic_total >= 3 and has_core


def relation_provenance_completeness(triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in triples:
        rel = str(t.get("relation", "")).strip()
        if not rel:
            continue
        grouped.setdefault(rel, []).append(t)
    out: Dict[str, Any] = {}
    for rel, rows in grouped.items():
        weights = RELATION_EVIDENCE_WEIGHTS.get(rel, {})
        expected = [m for m in ["audio", "slide", "paper"] if float(weights.get(m, 0.0)) > 0]
        cov_vals = []
        with_audio = 0
        with_slide = 0
        with_paper = 0
        for r in rows:
            present = {
                "audio": triple_has_modality(r, "audio"),
                "slide": triple_has_modality(r, "slide"),
                "paper": triple_has_modality(r, "paper"),
            }
            if present["audio"]:
                with_audio += 1
            if present["slide"]:
                with_slide += 1
            if present["paper"]:
                with_paper += 1
            if expected:
                cov_vals.append(sum(1 for m in expected if present[m]) / float(len(expected)))
            else:
                cov_vals.append(1.0)
        avg_cov = sum(cov_vals) / max(1, len(cov_vals))
        out[rel] = {
            "edges": len(rows),
            "expected_modalities": expected,
            "avg_expected_coverage": round(float(avg_cov), 4),
            "with_audio": with_audio,
            "with_slide": with_slide,
            "with_paper": with_paper,
        }
    return out


def build_kg_quality_report(
    rec: SessionRecord,
    triples: List[Dict[str, Any]],
    semantic_nodes: Dict[str, Any],
    evidence_links: Dict[str, Any],
    attempt_trace: List[Dict[str, Any]],
    repair_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rel = relation_count_map(triples)
    logic_total = sum(rel.get(x, 0) for x in LOGICAL_RELATIONS)
    support_or_measure = rel.get("supported_by", 0) + rel.get("measured_by", 0)

    emph_tails: Dict[str, int] = {}
    for t in triples:
        if t.get("relation") != "emphasizes":
            continue
        tid = str((t.get("tail") or {}).get("id", ""))
        emph_tails[tid] = emph_tails.get(tid, 0) + 1
    emph_total = sum(emph_tails.values())
    dominant_ratio = 0.0
    if emph_total > 0:
        dominant_ratio = max(emph_tails.values()) / emph_total

    visual_regions_total = 0
    visual_regions_with_semantics = 0
    for row in evidence_links.get("node_evidence", []) or []:
        for r in row.get("visual_regions", []) or []:
            visual_regions_total += 1
            if str(r.get("ocr_text", "")).strip() or str(r.get("visual_summary", "")).strip():
                visual_regions_with_semantics += 1
    visual_cov = visual_regions_with_semantics / max(1, visual_regions_total)

    tier_counts = {"strong": 0, "medium": 0, "weak": 0}
    for t in triples:
        tier = str(t.get("tier", "weak")).strip().lower()
        if tier not in tier_counts:
            tier = "weak"
        tier_counts[tier] += 1

    missing_prov = 0
    for t in triples:
        prov = t.get("provenance", [])
        if not isinstance(prov, list) or len(prov) == 0:
            missing_prov += 1

    relation_comp = relation_provenance_completeness(triples)

    repair_stats = {}
    paper_repair_threshold = float(os.getenv("REPAIR_PAPER_SIM_THRESHOLD", "0.82"))
    paper_repair_fuse_count = 0
    if isinstance(repair_meta, dict):
        repair_stats = repair_meta.get("repair_stats", {}) if isinstance(repair_meta.get("repair_stats", {}), dict) else {}
        paper_repair_threshold = float(repair_meta.get("paper_repair_threshold", paper_repair_threshold))
        paper_repair_fuse_count = int(repair_meta.get("paper_repair_fuse_count", 0))

    status = "ok" if coverage_pass(triples) else "failed_coverage"
    if visual_regions_total > 0 and visual_regions_with_semantics == 0:
        status = "failed_coverage"
    return {
        "sample_id": rec.sample_id,
        "quality_status": status,
        "coverage": {
            "logical_total": logic_total,
            "supported_or_measured": support_or_measure,
            "required_logical_total": 3,
            "required_supported_or_measured": 1,
        },
        "relation_counts": rel,
        "semantic_node_counts": semantic_nodes.get("counts_by_type", {}),
        "tier_counts": tier_counts,
        "emphasizes": {
            "total": emph_total,
            "unique_tail_nodes": len(emph_tails),
            "dominant_tail_ratio": round(dominant_ratio, 4),
        },
        "visual_semantics": {
            "regions_total": visual_regions_total,
            "regions_with_semantics": visual_regions_with_semantics,
            "coverage": round(visual_cov, 4),
        },
        "relation_provenance_completeness": relation_comp,
        "repair_stats": repair_stats,
        "paper_repair_threshold": round(paper_repair_threshold, 4),
        "paper_repair_fuse_count": paper_repair_fuse_count,
        "missing_provenance": missing_prov,
        "attempt_trace": attempt_trace,
    }


# ---------------------------------------------------------------------------
# Main claim-centric KG builder
# ---------------------------------------------------------------------------


def build_claim_centric_kg(
    rec: SessionRecord,
    cfg: Config,
    slides_structured: Dict[str, Any],
    paper_structured: Dict[str, Any],
    transcript_prosody: Dict[str, Any],
    transcript_enriched: Dict[str, Any],
    pragmatic: Dict[str, Any],
    alignment: Dict[str, Any],
    debug_dir: Optional[os.PathLike] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    from pathlib import Path

    debug_dir = Path(debug_dir) if debug_dir else None
    semantic_nodes = extract_semantic_nodes_from_paper(rec, paper_structured)
    llm_min_interval_ms = env_int(
        "TRIPLES_BUILD_MIN_INTERVAL_MS",
        default=300,
        min_value=0,
        max_value=600000,
    )
    attempts = [
        {
            "attempt_id": 1,
            "context_level": "K1_narrow",
            "topk_paper": cfg.topk_paper,
            "max_regions_per_slide": max(1, min(2, cfg.max_images_per_call)),
            "max_utterances_per_node": 6,
            "max_images_per_call": max(1, cfg.max_images_per_call),
            "window": {"chunks": 6, "utterances": 4, "slides": 2, "regions": 3, "events": 4, "signals": 4},
        },
        {
            "attempt_id": 2,
            "context_level": "K2_paragraph",
            "topk_paper": cfg.topk_paper + 3,
            "max_regions_per_slide": max(2, cfg.max_images_per_call + 1),
            "max_utterances_per_node": 10,
            "max_images_per_call": max(2, cfg.max_images_per_call + 1),
            "window": {"chunks": 10, "utterances": 8, "slides": 4, "regions": 8, "events": 8, "signals": 8},
        },
        {
            "attempt_id": 3,
            "context_level": "K3_section",
            "topk_paper": cfg.topk_paper + 6,
            "max_regions_per_slide": max(3, cfg.max_images_per_call + 2),
            "max_utterances_per_node": 14,
            "max_images_per_call": max(3, cfg.max_images_per_call + 2),
            "window": {"chunks": 14, "utterances": 12, "slides": 8, "regions": 14, "events": 14, "signals": 14},
        },
    ]

    final_evidence = {"sample_id": rec.sample_id, "node_evidence": []}
    final_prompt = {"sample_id": rec.sample_id, "mode": "per_claim", "per_claim": []}
    merged_triples: List[Dict[str, Any]] = []
    attempt_trace = []
    stop_on_coverage = os.getenv("STOP_ON_COVERAGE", "1").strip().lower() in {"1", "true", "yes"}

    for opt in attempts:
        evidence_links = build_evidence_links(
            rec=rec,
            semantic_nodes=semantic_nodes,
            alignment=alignment,
            slides_structured=slides_structured,
            paper_structured=paper_structured,
            transcript_prosody=transcript_prosody,
            transcript_enriched=transcript_enriched,
            pragmatic=pragmatic,
            topk_paper=opt["topk_paper"],
            max_regions_per_slide=opt["max_regions_per_slide"],
            max_utterances_per_node=opt["max_utterances_per_node"],
        )
        w = opt.get("window", {})
        claim_prompt = build_claim_windows(
            rec,
            semantic_nodes,
            evidence_links,
            max_claim_chunks=int(w.get("chunks", 8)),
            max_claim_utts=int(w.get("utterances", 8)),
            max_claim_slides=int(w.get("slides", 4)),
            max_claim_regions=int(w.get("regions", 8)),
            max_claim_events=int(w.get("events", 8)),
            max_claim_signals=int(w.get("signals", 8)),
        )

        triples: List[Dict[str, Any]] = []
        llm_calls = 0
        llm_ok = 0
        llm_ok_empty = 0
        llm_failures: Dict[str, int] = {}
        attempt_t0 = time.monotonic()
        claim_items = list(enumerate(claim_prompt.get("per_claim", [])))
        if cfg.enable_llm and claim_items:
            claim_budget_tokens = env_int("TRIPLES_BUILD_CLAIM_INPUT_TOKEN_BUDGET", default=7000, min_value=1000, max_value=120000)
            claim_max_shards = env_int("TRIPLES_BUILD_CLAIM_MAX_SHARDS", default=4, min_value=1, max_value=16)

            def _run_claim(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Optional[List[Dict[str, Any]]], str, int]:
                idx, claim_payload = item
                shards = split_claim_payload_for_budget(
                    claim_payload=claim_payload,
                    token_budget=claim_budget_tokens,
                    max_shards=claim_max_shards,
                )
                combined: List[Dict[str, Any]] = []
                statuses: List[str] = []
                calls = 0
                for shard in shards:
                    extracted, st = llm_kg_extract_for_claim(
                        rec=rec,
                        claim_payload=shard,
                        max_images_per_call=opt["max_images_per_call"],
                        max_image_bytes=cfg.max_image_bytes,
                        debug_dir=debug_dir,
                    )
                    calls += 1
                    statuses.append(st)
                    if extracted:
                        combined.extend(extracted)
                if combined:
                    return idx, combined, "ok", calls
                uniq_status = sorted(set(statuses))
                if uniq_status == ["ok_empty"]:
                    return idx, [], "ok_empty", calls
                st_out = "|".join(uniq_status[:3]) if uniq_status else "unknown"
                return idx, None, f"sharded_fail:{st_out}", calls

            claim_results: List[Tuple[int, Optional[List[Dict[str, Any]]], str, int]] = []
            ex = get_triples_build_llm_executor()
            futures = [ex.submit(_run_claim, item) for item in claim_items]
            for fut in concurrent.futures.as_completed(futures):
                claim_results.append(fut.result())
            claim_results.sort(key=lambda x: x[0])

            for _idx, extracted, st, calls in claim_results:
                llm_calls += max(1, calls)
                if st == "ok":
                    llm_ok += 1
                elif st == "ok_empty":
                    llm_ok_empty += 1
                else:
                    llm_failures[st] = llm_failures.get(st, 0) + 1
                if extracted:
                    triples.extend(extracted)

        triples = validate_and_normalize_triples(triples)
        merged_triples.extend(triples)
        merged_triples = dedupe_triples(validate_and_normalize_triples(merged_triples))
        rel = relation_count_map(merged_triples)
        logic_total = sum(rel.get(x, 0) for x in LOGICAL_RELATIONS)
        core = rel.get("supported_by", 0) + rel.get("measured_by", 0)
        passed = coverage_pass(merged_triples)
        attempt_trace.append({
            "attempt_id": opt["attempt_id"],
            "context_level": opt.get("context_level"),
            "topk_paper": opt["topk_paper"],
            "max_regions_per_slide": opt["max_regions_per_slide"],
            "max_utterances_per_node": opt["max_utterances_per_node"],
            "max_images_per_call": opt["max_images_per_call"],
            "triples": len(triples),
            "merged_triples": len(merged_triples),
            "relation_counts": rel,
            "logical_total": logic_total,
            "supported_or_measured": core,
            "coverage_pass": passed,
            "llm_calls": llm_calls,
            "llm_ok": llm_ok,
            "llm_ok_empty": llm_ok_empty,
            "llm_failures": llm_failures,
            "llm_concurrency": get_triples_build_llm_executor()._max_workers,
            "llm_min_interval_ms": llm_min_interval_ms,
            "wall_time_sec": round(time.monotonic() - attempt_t0, 3),
        })

        final_evidence = evidence_links
        final_prompt = claim_prompt
        if passed and stop_on_coverage:
            break

    final_triples = merged_triples
    final_triples, repair_meta = apply_soft_scoring_and_repairs(
        rec=rec,
        triples=final_triples,
        slides_structured=slides_structured,
        paper_structured=paper_structured,
        alignment=alignment,
        evidence_links=final_evidence,
        transcript_prosody=transcript_prosody,
    )
    quality = build_kg_quality_report(
        rec,
        final_triples,
        semantic_nodes,
        final_evidence,
        attempt_trace,
        repair_meta=repair_meta,
    )
    return semantic_nodes, final_evidence, final_prompt, final_triples, quality
