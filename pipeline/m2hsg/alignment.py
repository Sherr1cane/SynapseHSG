"""Alignment context and evidence-link construction for the SynapseHSG pipeline.

Provides hard utterance-to-slide alignment, paper chunk retrieval, and
comprehensive evidence-link assembly linking semantic nodes to their
multimodal evidence (slides, paper chunks, utterances, prosody, pragmatic
signals).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    LOGICAL_RELATIONS,
    RELATION_EVIDENCE_WEIGHTS,
    SessionRecord,
    _safe_float,
    env_int,
)


# ---------------------------------------------------------------------------
# Text / tokenisation helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def retrieve_paper_chunks(query: str, chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    q = set(tokenize(query))
    if not q:
        return chunks[:k]
    scored = []
    for c in chunks:
        cset = set(tokenize(c.get("text", "")))
        if not cset:
            score = 0.0
        else:
            score = len(q & cset) / max(1.0, len(q | cset))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for p in parts:
        s = re.sub(r"\s+", " ", p).strip()
        if len(s) < 20:
            continue
        out.append(s)
    return out


def normalize_content_key(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t[:120] or "empty"


def stable_semantic_node_id(sample_id: str, node_type: str, content: str) -> str:
    key = f"{sample_id}|{node_type}|{normalize_content_key(content)}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"{sample_id}/semantic/{node_type.lower()}/{h}"


def token_overlap_score(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / max(1.0, min(len(sa), len(sb)))


def collect_metric_terms(text: str) -> List[str]:
    low = text.lower()
    terms = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "ap",
        "map",
        "miou",
        "iou",
        "psnr",
        "ssim",
        "rmse",
        "mae",
        "bleu",
        "rouge",
        "wer",
        "cer",
    ]
    out = []
    for t in terms:
        if re.search(rf"\b{re.escape(t)}\b", low):
            out.append(t.upper() if len(t) <= 4 else t)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Hard alignment (utterance -> slide)
# ---------------------------------------------------------------------------


def hard_align_utterance_to_slide(utterances: List[Dict[str, Any]], slides: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for u in utterances:
        t = u["start_time"]
        match = None
        for i, s in enumerate(slides):
            st = s.get("start_time")
            et = s.get("end_time")
            if st is None:
                continue
            if et is None:
                if t >= st:
                    match = s
                    break
            elif st <= t < et:
                match = s
                break
            elif i == len(slides) - 1 and t >= st:
                match = s
                break
        out[u["utterance_id"]] = {
            "slide_id": match["slide_id"] if match else None,
            "slide_index": match["index"] if match else None,
        }
    return out


# ---------------------------------------------------------------------------
# Build alignment context
# ---------------------------------------------------------------------------


def build_alignment_context(
    rec: SessionRecord,
    slides_structured: Dict[str, Any],
    paper_structured: Dict[str, Any],
    transcript_prosody: Dict[str, Any],
    transcript_enriched: Dict[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    utterances = transcript_prosody.get("utterances", [])
    slides = slides_structured.get("slides", [])
    regions = slides_structured.get("visual_regions", [])
    chunks = paper_structured.get("paper_chunks", [])

    hard_map = hard_align_utterance_to_slide(utterances, slides)
    regions_by_slide: Dict[str, List[Dict[str, Any]]] = {}
    for r in regions:
        regions_by_slide.setdefault(r["slide_id"], []).append(r)

    enrich_by_uid = {u["utterance_id"]: u for u in transcript_enriched.get("utterances", [])}

    align_rows = []

    for u in utterances:
        uid = u["utterance_id"]
        slide_info = hard_map.get(uid, {})
        slide_id = slide_info.get("slide_id")
        vis = regions_by_slide.get(slide_id, []) if slide_id else []
        # Semantic-first region selection: prefer regions with informative OCR/summary/chart signals.
        # Avoid fixed first-N truncation that can drop critical visual evidence.
        align_vis_cap = env_int("ALIGN_VISUAL_REGIONS_PER_UTTERANCE", default=3, min_value=1, max_value=12)
        scored_vis: List[Tuple[int, Dict[str, Any]]] = []
        for r in vis:
            score = 0
            if str(r.get("ocr_text", "")).strip():
                score += 2
            if str(r.get("visual_summary", "")).strip():
                score += 2
            if str(r.get("chart_type", "")).strip().lower() not in {"", "none"}:
                score += 2
            ents = r.get("entities", [])
            if isinstance(ents, list) and len(ents) > 0:
                score += 1
            scored_vis.append((score, r))
        scored_vis.sort(key=lambda x: x[0], reverse=True)
        vis_for_model = [r for _, r in scored_vis[:align_vis_cap]]

        e = enrich_by_uid.get(uid, {})
        query = (u.get("text", "") + " " + e.get("enriched_text", "")).strip()
        top_chunks = retrieve_paper_chunks(query, chunks, cfg.topk_paper)

        align_rows.append({
            "utterance_id": uid,
            "slide_id": slide_id,
            "visual_region_id": vis_for_model[0]["region_id"] if vis_for_model else None,
            "bbox": vis_for_model[0]["bbox"] if vis_for_model else None,
            "visual_regions": [
                {
                    "region_id": r["region_id"],
                    "bbox": r["bbox"],
                    "source_image": r.get("source_image"),
                }
                for r in vis_for_model
            ],
            "top_paper_chunks": top_chunks,
        })

    alignment = {
        "sample_id": rec.sample_id,
        "step1_hard_alignment": align_rows,
        "utterance_alignment": align_rows,
    }
    return alignment


# ---------------------------------------------------------------------------
# Build evidence links
# ---------------------------------------------------------------------------


def build_evidence_links(
    rec: SessionRecord,
    semantic_nodes: Dict[str, Any],
    alignment: Dict[str, Any],
    slides_structured: Dict[str, Any],
    paper_structured: Dict[str, Any],
    transcript_prosody: Dict[str, Any],
    transcript_enriched: Dict[str, Any],
    pragmatic: Dict[str, Any],
    topk_paper: int,
    max_regions_per_slide: int,
    max_utterances_per_node: int,
) -> Dict[str, Any]:
    utterances = transcript_prosody.get("utterances", [])
    enriched_by_id = {u.get("utterance_id"): u for u in transcript_enriched.get("utterances", [])}
    align_rows = alignment.get("utterance_alignment", [])
    chunks = paper_structured.get("paper_chunks", [])
    paper_regions = paper_structured.get("paper_visual_regions", [])
    chunk_by_id = {c.get("chunk_id"): c for c in chunks}
    utter_by_id = {u.get("utterance_id"): u for u in utterances}
    align_by_uid = {x.get("utterance_id"): x for x in align_rows}
    regions = slides_structured.get("visual_regions", [])
    region_by_id: Dict[str, Dict[str, Any]] = {}
    regions_by_slide: Dict[str, List[Dict[str, Any]]] = {}
    for r in regions:
        sid = r.get("slide_id")
        if not sid:
            continue
        rid = r.get("region_id")
        if rid:
            region_by_id[rid] = r
        regions_by_slide.setdefault(sid, []).append(r)
    for sid in regions_by_slide:
        regions_by_slide[sid].sort(key=lambda x: x.get("region_id", ""))
    paper_regions_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for pr in paper_regions:
        pg = int(pr.get("page", 0) or 0)
        if pg <= 0:
            continue
        paper_regions_by_page.setdefault(pg, []).append(pr)
    for pg in paper_regions_by_page:
        paper_regions_by_page[pg].sort(key=lambda x: x.get("region_id", ""))
    max_paper_regions_per_page = max(1, min(env_int("PAPER_REGIONS_PER_NODE_PAGE", default=1, min_value=1, max_value=4), 4))

    prosody_by_uid: Dict[str, List[Dict[str, Any]]] = {}
    for ev in transcript_prosody.get("prosody_events", []):
        uid = ev.get("utterance_id")
        if uid:
            prosody_by_uid.setdefault(uid, []).append(ev)

    pragmatic_by_uid: Dict[str, List[Dict[str, Any]]] = {}
    for sg in pragmatic.get("signals", []):
        uid = sg.get("utterance_id")
        if uid:
            pragmatic_by_uid.setdefault(uid, []).append(sg)

    out_rows = []
    for n in semantic_nodes.get("nodes", []):
        n_text = n.get("content", "")
        scored = []
        for u in utterances:
            uid = u.get("utterance_id")
            if not uid:
                continue
            s = token_overlap_score(n_text, u.get("text", ""))
            if s <= 0:
                continue
            scored.append((s, uid))
        scored.sort(key=lambda x: x[0], reverse=True)
        utt_ids = [uid for _, uid in scored[:max_utterances_per_node]]

        slide_ids = []
        region_rows = []
        for uid in utt_ids:
            row = align_by_uid.get(uid, {})
            sid = row.get("slide_id")
            if sid:
                slide_ids.append(sid)
            # Prefer alignment-provided regions first.
            for r in row.get("visual_regions", []) or []:
                if not isinstance(r, dict):
                    continue
                rid = r.get("region_id")
                enrich_src = region_by_id.get(rid, {}) if rid else {}
                ocr_text = str(r.get("ocr_text", "") or "").strip() or str(enrich_src.get("ocr_text", "") or "").strip()
                visual_summary = str(r.get("visual_summary", "") or "").strip() or str(enrich_src.get("visual_summary", "") or "").strip()
                chart_type = str(r.get("chart_type", "") or "").strip() or str(enrich_src.get("chart_type", "") or "none")
                entities = r.get("entities", []) or enrich_src.get("entities", [])
                region_rows.append({
                    "region_id": rid,
                    "slide_id": sid,
                    "bbox": r.get("bbox", [0, 0, 1000, 1000]),
                    "source_image": r.get("source_image"),
                    "source_subdir": r.get("source_subdir", "slides"),
                    "ocr_text": ocr_text,
                    "chart_type": chart_type,
                    "visual_summary": visual_summary,
                    "entities": entities,
                })
            if sid and len([x for x in region_rows if x.get("slide_id") == sid]) < max_regions_per_slide:
                fallback_regions = regions_by_slide.get(sid, [])[:max_regions_per_slide]
                for fr in fallback_regions:
                    region_rows.append({
                        "region_id": fr.get("region_id"),
                        "slide_id": sid,
                        "bbox": fr.get("bbox", [0, 0, 1000, 1000]),
                        "source_image": fr.get("source_image"),
                        "source_subdir": fr.get("source_subdir", "slides"),
                        "ocr_text": fr.get("ocr_text", ""),
                        "chart_type": fr.get("chart_type", "none"),
                        "visual_summary": fr.get("visual_summary", ""),
                        "entities": fr.get("entities", []),
                    })

        uniq_slide = []
        seen_slide = set()
        for sid in slide_ids:
            if sid in seen_slide:
                continue
            seen_slide.add(sid)
            uniq_slide.append(sid)

        p_chunks = []
        src_chunk_id = n.get("source_chunk_id")
        if src_chunk_id and src_chunk_id in chunk_by_id:
            c = chunk_by_id[src_chunk_id]
            p_chunks.append({
                "chunk_id": src_chunk_id,
                "page": c.get("page"),
                "bbox": c.get("bbox", [0, 0, 1000, 1400]),
                "text": str(c.get("text", "")),
                "source": "node_anchor",
            })
        for c in retrieve_paper_chunks(n_text, chunks, topk_paper):
            cid = c.get("chunk_id")
            if not cid:
                continue
            p_chunks.append({
                "chunk_id": cid,
                "page": c.get("page"),
                "bbox": c.get("bbox", [0, 0, 1000, 1400]),
                "text": str(c.get("text", "")),
                "source": "retrieved",
            })

        uniq_chunks = []
        seen_chunks = set()
        for c in p_chunks:
            cid = c["chunk_id"]
            if cid in seen_chunks:
                continue
            seen_chunks.add(cid)
            uniq_chunks.append(c)
        uniq_chunks = uniq_chunks[: max(2, topk_paper)]
        for c in uniq_chunks:
            pg = int(c.get("page", 0) or 0)
            if pg <= 0:
                continue
            for pr in paper_regions_by_page.get(pg, [])[:max_paper_regions_per_page]:
                region_rows.append({
                    "region_id": pr.get("region_id"),
                    "slide_id": None,
                    "page": pg,
                    "bbox": pr.get("bbox", [0, 0, 1000, 1400]),
                    "source_image": pr.get("source_image"),
                    "source_subdir": pr.get("source_subdir", "paper_pages"),
                    "ocr_text": pr.get("ocr_text", ""),
                    "chart_type": pr.get("chart_type", "none"),
                    "visual_summary": pr.get("visual_summary", ""),
                    "entities": pr.get("entities", []),
                    "semantic_hint": pr.get("semantic_hint", ""),
                })

        by_region: Dict[str, Dict[str, Any]] = {}
        for r in region_rows:
            rid = r.get("region_id")
            if not rid:
                continue
            prev = by_region.get(rid)
            if prev is None:
                by_region[rid] = r
                continue
            prev_score = int(bool(str(prev.get("ocr_text", "")).strip())) + int(bool(str(prev.get("visual_summary", "")).strip()))
            cur_score = int(bool(str(r.get("ocr_text", "")).strip())) + int(bool(str(r.get("visual_summary", "")).strip()))
            if cur_score > prev_score:
                by_region[rid] = r
        uniq_region = list(by_region.values())
        region_cap = max(1, max_regions_per_slide * max(1, len(uniq_slide) or 1))
        region_cap = max(region_cap, max_paper_regions_per_page * max(1, len({int(c.get("page", 0) or 0) for c in uniq_chunks if int(c.get("page", 0) or 0) > 0})))
        uniq_region = uniq_region[:region_cap]

        utt_rows = []
        for uid in utt_ids:
            u = utter_by_id.get(uid)
            if not u:
                continue
            enr_u = enriched_by_id.get(uid, {})
            text_val = enr_u.get("enriched_text") or u.get("text", "")
            utt_rows.append({
                "utterance_id": uid,
                "start_time": u.get("start_time"),
                "end_time": u.get("end_time"),
                "text": text_val,
            })

        prosody_rows = []
        signal_rows = []
        for uid in utt_ids:
            prosody_rows.extend(prosody_by_uid.get(uid, []))
            signal_rows.extend(pragmatic_by_uid.get(uid, []))

        out_rows.append({
            "node_id": n.get("id"),
            "node_type": n.get("type"),
            "content": n.get("content", ""),
            "confidence": n.get("confidence", 0.5),
            "paper_chunks": uniq_chunks,
            "utterances": utt_rows,
            "slides": [{"slide_id": sid} for sid in uniq_slide],
            "visual_regions": uniq_region,
            "prosody_events": prosody_rows,
            "pragmatic_signals": signal_rows,
        })

    return {"sample_id": rec.sample_id, "node_evidence": out_rows}
