"""Supervision assembly for M2HSG stage-1 training."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .features import lexical_jaccard
from .frozen.rules import (
    RELATION_CONSTRAINTS,
    compute_heuristic_score_tier,
    infer_source_type,
    provenance_modality,
    provenance_quality,
    relation_allowed,
)
from .io_utils import read_json, read_jsonl, write_json, write_jsonl

TIER_TO_LABEL = {"weak": 0, "medium": 1, "strong": 2}


@dataclass
class SupervisionConfig:
    run_dir: Path
    output_dir: Path
    seed: int = 42
    negatives_per_positive: int = 3
    medium_weight: float = 0.7
    include_debug_prompt: bool = False
    include_weak_analysis: bool = True


def _safe_text(x: Any) -> str:
    return str(x or "").strip()


def _session_dir(run_dir: Path, sample_id: str) -> Path:
    conf, sess = sample_id.split("/", 1)
    return run_dir / "sessions" / conf / sess


def _modality_summary(provenance: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    counts = {"audio": 0, "slide": 0, "paper": 0}
    stats = {
        "audio": {"quality_max": 0.0, "duration_sum": 0.0},
        "slide": {"quality_max": 0.0, "has_chart_hint": 0.0},
        "paper": {"quality_max": 0.0, "repaired_count": 0.0, "similarity_max": 0.0},
    }
    for p in provenance:
        if not isinstance(p, dict):
            continue
        m = provenance_modality(p)
        if m not in counts:
            continue
        counts[m] += 1
        q = provenance_quality(m, p)
        stats[m]["quality_max"] = max(float(stats[m]["quality_max"]), float(q))
        if m == "audio":
            st = p.get("start_time")
            et = p.get("end_time")
            try:
                stf = float(st)
                etf = float(et)
                if etf > stf:
                    stats[m]["duration_sum"] += max(0.0, etf - stf)
            except Exception:
                pass
        elif m == "slide":
            if _safe_text(p.get("chart_type")) and _safe_text(p.get("chart_type")) != "none":
                stats[m]["has_chart_hint"] = 1.0
        elif m == "paper":
            if bool(p.get("is_repaired")):
                stats[m]["repaired_count"] += 1.0
            try:
                sim = float(p.get("similarity_score", 0.0))
                stats[m]["similarity_max"] = max(float(stats[m]["similarity_max"]), sim)
            except Exception:
                pass
    return counts, stats


def _make_sample_id_key(
    split: str,
    src_sample_id: str,
    head_id: str,
    relation: str,
    tail_id: str,
    source_type: str,
    negative_type: Optional[str],
) -> str:
    raw = "|".join([split, src_sample_id, head_id, relation, tail_id, source_type, negative_type or "none"])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{src_sample_id}/{digest}"


def _row_to_sample(
    row: Dict[str, Any],
    split: str,
    src_sample_id: str,
    source_type: str,
    tier_for_label: str,
    sample_weight: float,
    edge_target: float,
    include_in_loss: bool,
    negative_type: Optional[str] = None,
    base_tier: Optional[str] = None,
    origin: str = "v1_positive",
) -> Dict[str, Any]:
    head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
    tail = row.get("tail", {}) if isinstance(row.get("tail", {}), dict) else {}
    provenance = row.get("provenance", []) if isinstance(row.get("provenance", []), list) else []
    mcounts, mstats = _modality_summary([p for p in provenance if isinstance(p, dict)])

    heuristic_score, heuristic_tier, repaired, _ = compute_heuristic_score_tier(row)
    head_id = _safe_text(head.get("id"))
    relation = _safe_text(row.get("relation"))
    tail_id = _safe_text(tail.get("id"))

    return {
        "sample_key": _make_sample_id_key(split, src_sample_id, head_id, relation, tail_id, source_type, negative_type),
        "split": split,
        "sample_id": src_sample_id,
        "head_id": head_id,
        "head_type": _safe_text(head.get("type")),
        "head_content": _safe_text(head.get("content")),
        "relation": relation,
        "tail_id": tail_id,
        "tail_type": _safe_text(tail.get("type")),
        "tail_content": _safe_text(tail.get("content")),
        "provenance": provenance,
        "is_repaired": bool(row.get("is_repaired")) or repaired,
        "provenance_score": float(row.get("provenance_score", heuristic_score) or heuristic_score),
        "heuristic_score": heuristic_score,
        "heuristic_tier": heuristic_tier,
        "tier_label": int(TIER_TO_LABEL.get(tier_for_label, 0)),
        "edge_target": float(edge_target),
        "sample_weight": float(sample_weight),
        "source_type": source_type,
        "negative_type": negative_type,
        "base_tier": base_tier,
        "include_in_loss": bool(include_in_loss),
        "origin": origin,
        "modality_mask": {
            "audio": 1 if mcounts["audio"] > 0 else 0,
            "visual": 1 if mcounts["slide"] > 0 else 0,
            "paper": 1 if mcounts["paper"] > 0 else 0,
        },
        "audio_repr": {
            "count": mcounts["audio"],
            "quality_max": round(float(mstats["audio"]["quality_max"]), 4),
            "duration_sum": round(float(mstats["audio"]["duration_sum"]), 4),
        },
        "visual_repr": {
            "count": mcounts["slide"],
            "quality_max": round(float(mstats["slide"]["quality_max"]), 4),
            "has_chart_hint": float(mstats["slide"]["has_chart_hint"]),
        },
        "paper_repr": {
            "count": mcounts["paper"],
            "quality_max": round(float(mstats["paper"]["quality_max"]), 4),
            "repaired_count": int(mstats["paper"]["repaired_count"]),
            "similarity_max": round(float(mstats["paper"]["similarity_max"]), 4),
        },
    }


def _node_obj(node_id: str, node_type: str, content: str) -> Dict[str, str]:
    return {"id": _safe_text(node_id), "type": _safe_text(node_type), "content": _safe_text(content)}


def _collect_paper_node_pool(semantic_nodes: Dict[str, Any], evidence_links: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    by_type: Dict[str, Dict[str, Dict[str, str]]] = {}

    def add(entry: Dict[str, str]) -> None:
        ntype = entry["type"]
        nid = entry["id"]
        if not ntype or not nid:
            return
        by_type.setdefault(ntype, {})
        by_type[ntype][nid] = entry

    for n in semantic_nodes.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        add(_node_obj(n.get("id", ""), n.get("type", ""), n.get("content", "")))

    for row in evidence_links.get("node_evidence", []) or []:
        if not isinstance(row, dict):
            continue
        for c in row.get("paper_chunks", []) or []:
            if not isinstance(c, dict):
                continue
            add(_node_obj(c.get("chunk_id", ""), "PaperChunk", c.get("text", "")))
        for r in row.get("visual_regions", []) or []:
            if not isinstance(r, dict):
                continue
            rid = _safe_text(r.get("region_id"))
            if rid:
                add(_node_obj(rid, "VisualRegion", r.get("visual_summary", "") or r.get("ocr_text", "")))
        for u in row.get("utterances", []) or []:
            if not isinstance(u, dict):
                continue
            add(_node_obj(u.get("utterance_id", ""), "Utterance", u.get("text", "")))
        for s in row.get("slides", []) or []:
            if not isinstance(s, dict):
                continue
            sid = _safe_text(s.get("slide_id"))
            if sid:
                add(_node_obj(sid, "Slide", ""))

    return {k: list(v.values()) for k, v in by_type.items()}


def _collect_session_node_pool(triples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    by_type: Dict[str, Dict[str, Dict[str, str]]] = {}

    def add(entry: Dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        nid = _safe_text(entry.get("id"))
        ntype = _safe_text(entry.get("type"))
        if not nid or not ntype:
            return
        by_type.setdefault(ntype, {})
        by_type[ntype][nid] = _node_obj(nid, ntype, _safe_text(entry.get("content")))

    for row in triples:
        add(row.get("head"))
        add(row.get("tail"))

    return {k: list(v.values()) for k, v in by_type.items()}


def _edge_key(head_id: str, relation: str, tail_id: str) -> str:
    return f"{head_id}\t{relation}\t{tail_id}"


def _candidate_tail_types(relation: str, head_type: str) -> Set[str]:
    if relation not in RELATION_CONSTRAINTS:
        return set()
    hset, tset = RELATION_CONSTRAINTS[relation]
    if head_type in hset:
        return set(tset)
    return set()


def _choose_negative_tails(
    *,
    rng: random.Random,
    head: Dict[str, Any],
    relation: str,
    true_tail: Dict[str, Any],
    existing_edges: Set[str],
    pool: Dict[str, List[Dict[str, str]]],
    k: int,
) -> List[Dict[str, str]]:
    if k <= 0:
        return []
    head_id = _safe_text(head.get("id"))
    head_type = _safe_text(head.get("type"))
    true_tail_id = _safe_text(true_tail.get("id"))
    true_tail_type = _safe_text(true_tail.get("type"))
    true_tail_text = _safe_text(true_tail.get("content"))
    head_text = _safe_text(head.get("content"))

    tail_types = _candidate_tail_types(relation, head_type)
    if not tail_types and true_tail_type:
        tail_types = {true_tail_type}
    if true_tail_type and true_tail_type in pool:
        tail_types = {true_tail_type}

    candidates: List[Tuple[float, Dict[str, str]]] = []
    for ttype in tail_types:
        for cand in pool.get(ttype, []):
            cid = _safe_text(cand.get("id"))
            ctype = _safe_text(cand.get("type"))
            if not cid or cid == true_tail_id:
                continue
            if not relation_allowed(head_type, relation, ctype):
                continue
            ekey = _edge_key(head_id, relation, cid)
            if ekey in existing_edges:
                continue
            ctext = _safe_text(cand.get("content"))
            sim = lexical_jaccard(true_tail_text, ctext) + 0.15 * lexical_jaccard(head_text, ctext)
            candidates.append((sim, cand))

    rng.shuffle(candidates)
    candidates.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, str]] = []
    seen = set()
    for _, cand in candidates:
        cid = _safe_text(cand.get("id"))
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cand)
        if len(out) >= k:
            break
    return out


def _load_session_inputs(run_dir: Path, sample_id: str, include_debug_prompt: bool) -> Dict[str, Any]:
    sdir = _session_dir(run_dir, sample_id)
    required = {
        "kg_triples": sdir / "kg_triples.jsonl",
        "semantic_nodes": sdir / "semantic_nodes.json",
        "evidence_links": sdir / "evidence_links.json",
        "kg_quality_report": sdir / "kg_quality_report.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing required M2HSG inputs for {sample_id}: {missing}")

    out = {
        "kg_triples": read_jsonl(required["kg_triples"]),
        "semantic_nodes": read_json(required["semantic_nodes"]),
        "evidence_links": read_json(required["evidence_links"]),
        "kg_quality_report": read_json(required["kg_quality_report"]),
    }
    prompt_path = sdir / "kg_extract_prompt.json"
    if include_debug_prompt and prompt_path.exists():
        out["kg_extract_prompt"] = read_json(prompt_path)
    return out


def _iter_split_samples(splits: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for split in ["train", "dev", "test"]:
        for sample_id in splits.get(split, []) or []:
            yield split, str(sample_id)


def assemble_supervision(cfg: SupervisionConfig) -> Dict[str, Any]:
    rng = random.Random(cfg.seed)
    run_dir = cfg.run_dir.resolve()
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = read_json(run_dir / "splits.json")
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    stats: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "seed": cfg.seed,
        "negatives_per_positive": cfg.negatives_per_positive,
        "medium_weight": cfg.medium_weight,
        "counts": {
            "strong": 0,
            "medium": 0,
            "weak_analysis": 0,
            "hard_negative": 0,
            "paper_hard": 0,
            "session_hard": 0,
        },
        "missing_negatives": 0,
        "split_rows": {},
        "split_loss_rows": {},
    }

    for split, sample_id in _iter_split_samples(splits):
        data = _load_session_inputs(run_dir, sample_id, cfg.include_debug_prompt)
        triples = data["kg_triples"]
        semantic_nodes = data["semantic_nodes"]
        evidence_links = data["evidence_links"]

        paper_pool = _collect_paper_node_pool(semantic_nodes, evidence_links)
        session_pool = _collect_session_node_pool(triples)

        existing_edges: Set[str] = set()
        for row in triples:
            head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
            tail = row.get("tail", {}) if isinstance(row.get("tail", {}), dict) else {}
            existing_edges.add(_edge_key(_safe_text(head.get("id")), _safe_text(row.get("relation")), _safe_text(tail.get("id"))))

        for row in triples:
            head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
            tail = row.get("tail", {}) if isinstance(row.get("tail", {}), dict) else {}
            tier = _safe_text(row.get("tier")).lower() or "weak"
            if tier not in {"strong", "medium", "weak"}:
                tier = "weak"

            source_type, base_tier = infer_source_type(row, hard_negative=False)
            tier_for_label = base_tier or tier

            if tier in {"strong", "medium"}:
                weight = 1.0 if tier == "strong" else float(cfg.medium_weight)
                edge_target = 1.0 if tier == "strong" else 0.7
                sample = _row_to_sample(
                    row=row,
                    split=split,
                    src_sample_id=sample_id,
                    source_type=source_type,
                    tier_for_label=tier_for_label,
                    sample_weight=weight,
                    edge_target=edge_target,
                    include_in_loss=True,
                    negative_type=None,
                    base_tier=base_tier,
                    origin="v1_positive",
                )
                if cfg.include_debug_prompt:
                    sample["debug_has_kg_extract_prompt"] = "kg_extract_prompt" in data
                rows_by_split[split].append(sample)
                stats["counts"][tier] += 1

                paper_negs = _choose_negative_tails(
                    rng=rng,
                    head=head,
                    relation=_safe_text(row.get("relation")),
                    true_tail=tail,
                    existing_edges=existing_edges,
                    pool=paper_pool,
                    k=min(2, cfg.negatives_per_positive),
                )
                session_need = max(0, cfg.negatives_per_positive - len(paper_negs))
                session_negs = _choose_negative_tails(
                    rng=rng,
                    head=head,
                    relation=_safe_text(row.get("relation")),
                    true_tail=tail,
                    existing_edges=existing_edges,
                    pool=session_pool,
                    k=session_need,
                )
                if len(paper_negs) + len(session_negs) < cfg.negatives_per_positive:
                    stats["missing_negatives"] += cfg.negatives_per_positive - len(paper_negs) - len(session_negs)

                for neg_type, negs in [("paper_hard", paper_negs), ("session_hard", session_negs)]:
                    for neg_tail in negs:
                        neg_row = {
                            "head": head,
                            "relation": _safe_text(row.get("relation")),
                            "tail": neg_tail,
                            "provenance": [],
                            "is_repaired": False,
                            "provenance_score": 0.0,
                            "tier": "weak",
                        }
                        neg_sample = _row_to_sample(
                            row=neg_row,
                            split=split,
                            src_sample_id=sample_id,
                            source_type="hard_negative",
                            tier_for_label="weak",
                            sample_weight=1.0,
                            edge_target=0.0,
                            include_in_loss=True,
                            negative_type=neg_type,
                            base_tier=None,
                            origin="hard_negative_corrupt_tail",
                        )
                        rows_by_split[split].append(neg_sample)
                        stats["counts"]["hard_negative"] += 1
                        stats["counts"][neg_type] += 1

            elif cfg.include_weak_analysis:
                weak_sample = _row_to_sample(
                    row=row,
                    split=split,
                    src_sample_id=sample_id,
                    source_type="weak",
                    tier_for_label="weak",
                    sample_weight=0.0,
                    edge_target=0.0,
                    include_in_loss=False,
                    negative_type=None,
                    base_tier=None,
                    origin="v1_weak_analysis",
                )
                rows_by_split[split].append(weak_sample)
                stats["counts"]["weak_analysis"] += 1

    for split in ["train", "dev", "test"]:
        rows = rows_by_split[split]
        rows.sort(key=lambda x: (x["sample_id"], x["sample_key"]))
        out_path = output_dir / f"{split}.supervision.jsonl"
        write_jsonl(out_path, rows)
        stats["split_rows"][split] = len(rows)
        stats["split_loss_rows"][split] = sum(1 for x in rows if bool(x.get("include_in_loss")))

    write_json(output_dir / "supervision_stats.json", stats)
    return stats

