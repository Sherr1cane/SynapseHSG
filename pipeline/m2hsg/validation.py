"""Sample-level validation for SynapseHSG pipeline outputs.

Validates enriched transcript tags, prosody bounds, and triple schema
compliance for a processed session.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .kg_extract import validate_and_normalize_triples


def validate_sample(
    transcript_enriched: Dict[str, Any],
    transcript_prosody: Dict[str, Any],
    alignment: Dict[str, Any],
    triples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    errors = []

    for u in transcript_enriched.get("utterances", []):
        tags = u.get("selected_tags", [])
        if len(tags) > 2:
            errors.append(f"tag quota exceeded: {u.get('utterance_id')}")

    combined = transcript_enriched.get("combined_enriched_text", "")
    if "<STRESS>" not in combined and "<PAUSE" not in combined:
        errors.append("gate violation: enriched transcript has no STRESS/PAUSE tags")

    utter = {u["utterance_id"]: u for u in transcript_prosody.get("utterances", [])}
    for ev in transcript_prosody.get("prosody_events", []):
        uid = ev.get("utterance_id")
        if not uid or uid not in utter:
            errors.append(f"prosody missing utterance link: {ev.get('event_id')}")
            continue
        u = utter[uid]
        if not (u["start_time"] <= ev["start_time"] <= ev["end_time"] <= u["end_time"]):
            errors.append(f"prosody bounds violation: {ev.get('event_id')}")

    checked = validate_and_normalize_triples(triples)
    if len(checked) != len(triples):
        errors.append("invalid triples detected by schema/constraint validator")

    return {
        "ok": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "stats": {
            "utterances": len(transcript_prosody.get("utterances", [])),
            "prosody_events": len(transcript_prosody.get("prosody_events", [])),
            "align_rows": len(alignment.get("utterance_alignment", [])),
            "triples": len(triples),
        },
    }
