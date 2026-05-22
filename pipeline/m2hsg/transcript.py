"""Transcript construction for the SynapseHSG pipeline.

Builds utterance-level transcript from ASR payloads or silence-based
segmentation, and injects prosodic tags / events (STRESS, PAUSE, etc.)
into enriched transcripts.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from .audio import (
    median_filter,
    normalize_z,
    split_speech_segments,
)
from .config import DISFLUENCY_RE, _safe_float


# ---------------------------------------------------------------------------
# Utterance construction from ASR
# ---------------------------------------------------------------------------


def utterance_from_asr(asr: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = asr.get("segments") or []
    words = asr.get("words") or []
    utterances: List[Dict[str, Any]] = []

    if segs:
        for i, s in enumerate(segs):
            st = _safe_float(s.get("start", 0.0), 0.0)
            et = _safe_float(s.get("end", st + 0.1), st + 0.1)
            txt = str(s.get("text", "")).strip()
            if et <= st:
                et = st + 0.01
            utterances.append({
                "utterance_local_id": i,
                "start_time": round(st, 3),
                "end_time": round(max(et, st + 0.01), 3),
                "text": txt,
                "words": [],
            })

    if words and utterances:
        for w in words:
            st = _safe_float(w.get("start", 0.0), 0.0)
            et = _safe_float(w.get("end", st + 0.01), st + 0.01)
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            if et <= st:
                et = st + 0.01
            for u in utterances:
                if u["start_time"] <= st <= u["end_time"]:
                    u["words"].append({"text": text, "start": st, "end": et, "duration": max(0.01, et - st)})
                    break

    # words-only fallback: build utterances from word timestamps when segments are absent.
    if (not utterances) and isinstance(words, list) and words:
        norm_words = []
        for w in words:
            if not isinstance(w, dict):
                continue
            st = _safe_float(w.get("start", 0.0), 0.0)
            et = _safe_float(w.get("end", st + 0.01), st + 0.01)
            tok = str(w.get("word", w.get("text", ""))).strip()
            if not tok:
                continue
            if et <= st:
                et = st + 0.01
            norm_words.append({
                "text": tok,
                "start": st,
                "end": max(et, st + 0.01),
                "duration": max(0.01, et - st),
            })

        norm_words.sort(key=lambda x: (x["start"], x["end"]))
        if norm_words:
            groups: List[List[Dict[str, Any]]] = []
            cur_group: List[Dict[str, Any]] = [norm_words[0]]
            for ww in norm_words[1:]:
                prev = cur_group[-1]
                gap = ww["start"] - prev["end"]
                cur_dur = cur_group[-1]["end"] - cur_group[0]["start"]
                prev_text = prev["text"]
                end_punct = prev_text.endswith((".", "!", "?", ";", ":"))
                too_long = (cur_dur > 16.0 and len(cur_group) >= 12)
                if gap > 0.8 or too_long or (end_punct and gap > 0.2):
                    groups.append(cur_group)
                    cur_group = [ww]
                else:
                    cur_group.append(ww)
            if cur_group:
                groups.append(cur_group)

            for i, g in enumerate(groups):
                st = g[0]["start"]
                et = g[-1]["end"]
                txt = " ".join(x["text"] for x in g).strip()
                utterances.append({
                    "utterance_local_id": i,
                    "start_time": round(st, 3),
                    "end_time": round(max(et, st + 0.01), 3),
                    "text": txt,
                    "words": g,
                })
    return utterances


# ---------------------------------------------------------------------------
# Utterance construction from silence segmentation
# ---------------------------------------------------------------------------


def utterance_from_silence(duration: float, silences: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    utterances = []
    for i, (s, e) in enumerate(split_speech_segments(duration, silences)):
        utterances.append({
            "utterance_local_id": i,
            "start_time": round(s, 3),
            "end_time": round(e, 3),
            "text": f"utterance_{i}",
            "words": [],
        })
    return utterances


# ---------------------------------------------------------------------------
# Helpers for tag/event injection
# ---------------------------------------------------------------------------


def avg_rms_for_span(rms: List[Tuple[float, float]], s: float, e: float) -> float:
    vals = [v for t, v in rms if s <= t <= e]
    if not vals:
        return -60.0
    return statistics.mean(vals)


def nearest_pause(silences: List[Tuple[float, float]], utt_start: float) -> Optional[Tuple[float, float]]:
    best = None
    best_gap = 1e9
    for s, e in silences:
        gap = abs(e - utt_start)
        if gap < best_gap:
            best_gap = gap
            best = (s, e)
    return best


# ---------------------------------------------------------------------------
# Main tag / event injection
# ---------------------------------------------------------------------------


def inject_tags_and_events(sample_id: str, utterances: List[Dict[str, Any]], silences: List[Tuple[float, float]], rms_series: List[Tuple[float, float]], topk: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if topk < 1:
        topk = 1

    # Feature vectors
    energy = [avg_rms_for_span(rms_series, u["start_time"], u["end_time"]) for u in utterances]
    energy = median_filter(energy, window=5)
    z_energy = normalize_z(energy)

    durations = [u["end_time"] - u["start_time"] for u in utterances]
    z_duration = normalize_z(durations)

    z_pitch = [0.0 for _ in utterances]

    # PAUSE events raw (with short/long bucket)
    raw_events = []
    for i, (s, e) in enumerate(silences):
        d = e - s
        if d < 0.05:
            continue
        pause_kind = "PAUSE_LONG" if d >= 0.8 else "PAUSE"
        raw_events.append({
            "event_id": f"{sample_id}/prosody/pause/{i:04d}",
            "type": pause_kind,
            "start_time": round(s, 3),
            "end_time": round(e, 3),
            "duration": round(d, 3),
            "intensity": round(min(1.0, d / 2.0), 3),
            "confidence": 0.8,
            "utterance_id": None,
        })

    # Assign nearest pause to each utterance for candidate scoring.
    selected_events = []
    enriched_utterances = []

    for i, u in enumerate(utterances):
        text = (u.get("text") or "").strip()
        if not text:
            text = f"utterance_{i}"

        candidates: List[Tuple[str, float, Dict[str, Any]]] = []

        if i < len(z_energy) and z_energy[i] > 1.5:
            candidates.append(("STRESS", z_energy[i], {
                "start_time": u["start_time"], "end_time": u["end_time"], "confidence": 0.7
            }))

        if i < len(z_duration) and z_duration[i] > 1.5:
            candidates.append(("SLOW_DOWN", z_duration[i], {
                "start_time": u["start_time"], "end_time": u["end_time"], "confidence": 0.65
            }))

        if text.endswith("?"):
            # fallback rising-tone candidate without pitch library
            candidates.append(("RISING_TONE", 1.6, {
                "start_time": u["start_time"], "end_time": u["end_time"], "confidence": 0.55
            }))

        if DISFLUENCY_RE.search(text):
            candidates.append(("DISFLUENCY", 1.9, {
                "start_time": u["start_time"], "end_time": u["end_time"], "confidence": 0.75
            }))

        p = nearest_pause(silences, u["start_time"])
        if p is not None:
            ps, pe = p
            pdur = pe - ps
            if pdur >= 0.20:
                # PAUSE_LONG gets stronger score
                pause_score = 1.7 if pdur >= 0.8 else 1.3
                candidates.append(("PAUSE_LONG" if pdur >= 0.8 else "PAUSE", pause_score, {
                    "start_time": ps, "end_time": pe, "confidence": 0.8
                }))

        # Debounce: discard events shorter than 50ms.
        filtered = []
        for label, score, ev in candidates:
            if ev["end_time"] - ev["start_time"] < 0.05 and label in {"STRESS", "RISING_TONE"}:
                continue
            filtered.append((label, score, ev))

        # Top-K quota
        filtered.sort(key=lambda x: x[1], reverse=True)
        chosen = filtered[:topk]

        tags = []
        for idx, (label, score, ev) in enumerate(chosen):
            event_id = f"{sample_id}/prosody/{label.lower()}/{i:04d}_{idx:02d}"
            tags.append(label)
            selected_events.append({
                "event_id": event_id,
                "type": label,
                "start_time": round(ev["start_time"], 3),
                "end_time": round(ev["end_time"], 3),
                "intensity": round(min(1.0, score / 3.0), 3),
                "confidence": round(_safe_float(ev.get("confidence", 0.6), 0.6), 3),
                "utterance_id": None,
            })

        enriched = text
        # deterministic order for readability
        if "PAUSE_LONG" in tags:
            pause = next(x for x in chosen if x[0] == "PAUSE_LONG")
            dur = pause[2]["end_time"] - pause[2]["start_time"]
            enriched = f'<PAUSE dur="{dur:.2f}" kind="long"/> ' + enriched
        elif "PAUSE" in tags:
            pause = next(x for x in chosen if x[0] == "PAUSE")
            dur = pause[2]["end_time"] - pause[2]["start_time"]
            enriched = f'<PAUSE dur="{dur:.2f}"/> ' + enriched

        wrap_order = ["DISFLUENCY", "RISING_TONE", "SLOW_DOWN", "STRESS"]
        for t in wrap_order:
            if t in tags:
                enriched = f"<{t}>{enriched}</{t}>"

        enriched_utterances.append({
            "utterance_local_id": u["utterance_local_id"],
            "start_time": u["start_time"],
            "end_time": u["end_time"],
            "text": text,
            "enriched_text": enriched,
            "selected_tags": tags,
            "words": u.get("words", []),
        })

    # Link events to utterances by overlap
    for ev in selected_events + raw_events:
        best_uid = None
        best_ov = 0.0
        best_u = None
        for u in enriched_utterances:
            ov = max(0.0, min(ev["end_time"], u["end_time"]) - max(ev["start_time"], u["start_time"]))
            if ov > best_ov:
                best_ov = ov
                best_uid = f"{sample_id}/utterance/{u['utterance_local_id']:04d}"
                best_u = u
        if best_uid is None and enriched_utterances:
            mid = (ev["start_time"] + ev["end_time"]) / 2.0
            nearest = min(
                enriched_utterances,
                key=lambda u: abs(((u["start_time"] + u["end_time"]) / 2.0) - mid),
            )
            best_uid = f"{sample_id}/utterance/{nearest['utterance_local_id']:04d}"
            best_u = nearest
        if best_u is not None:
            s = max(ev["start_time"], best_u["start_time"])
            e = min(ev["end_time"], best_u["end_time"])
            if e <= s:
                e = min(best_u["end_time"], s + 0.01)
                if e <= s:
                    s = max(best_u["start_time"], e - 0.01)
            ev["start_time"] = round(s, 3)
            ev["end_time"] = round(e, 3)
        ev["utterance_id"] = best_uid

    transcript_prosody = {
        "sample_id": sample_id,
        "utterances": [
            {
                "utterance_id": f"{sample_id}/utterance/{u['utterance_local_id']:04d}",
                "start_time": u["start_time"],
                "end_time": u["end_time"],
                "text": u["text"],
                "word_timestamps": u.get("words", []),
            }
            for u in enriched_utterances
        ],
        "prosody_events_raw": raw_events,
        "prosody_events": selected_events,
    }

    combined_text = "\n".join(x["enriched_text"] for x in enriched_utterances)
    transcript_enriched = {
        "sample_id": sample_id,
        "topk_quota": topk,
        "utterances": [
            {
                "utterance_id": f"{sample_id}/utterance/{u['utterance_local_id']:04d}",
                "start_time": u["start_time"],
                "end_time": u["end_time"],
                "enriched_text": u["enriched_text"],
                "selected_tags": u["selected_tags"],
            }
            for u in enriched_utterances
        ],
        "combined_enriched_text": combined_text,
    }
    return transcript_prosody, transcript_enriched
