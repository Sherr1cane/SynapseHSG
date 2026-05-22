"""Pragmatic signal detection for the SynapseHSG pipeline.

Identifies PIVOT_MARKER, FOCUS_MARKER, and DEICTIC_MARKER signals from
enriched transcript utterances.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .config import DEICTIC_RE, TRANSITION_RE


def build_pragmatic_signals(sample_id: str, transcript_enriched: Dict[str, Any]) -> Dict[str, Any]:
    signals = []
    for u in transcript_enriched.get("utterances", []):
        text = u.get("enriched_text", "")
        low = text.lower()
        uid = u["utterance_id"]

        has_pause_long = "<PAUSE" in text and 'kind="long"' in text
        has_stress = "<STRESS>" in text
        has_slow = "<SLOW_DOWN>" in text

        if has_pause_long and TRANSITION_RE.search(low) and (has_stress or has_slow):
            signals.append({
                "signal_id": f"{sample_id}/pragmatic/pivot/{uid.split('/')[-1]}",
                "type": "PIVOT_MARKER",
                "utterance_id": uid,
                "confidence": 0.85,
                "evidence": "PAUSE_LONG + transition + STRESS/SLOW_DOWN",
            })

        if has_stress and re.search(r"\b\w+-\w+\b", text):
            signals.append({
                "signal_id": f"{sample_id}/pragmatic/focus/{uid.split('/')[-1]}",
                "type": "FOCUS_MARKER",
                "utterance_id": uid,
                "confidence": 0.78,
                "evidence": "stressed compound token",
            })

        if DEICTIC_RE.search(low) and "<PAUSE" in text:
            signals.append({
                "signal_id": f"{sample_id}/pragmatic/deictic/{uid.split('/')[-1]}",
                "type": "DEICTIC_MARKER",
                "utterance_id": uid,
                "confidence": 0.88,
                "evidence": "deictic phrase + pause",
            })

    return {"sample_id": sample_id, "signals": signals}
