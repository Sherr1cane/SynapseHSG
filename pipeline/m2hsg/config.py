"""Configuration, constants, and helper utilities for the SynapseHSG V1 pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# File-extension tuples
# ---------------------------------------------------------------------------

AUDIO_EXTS = (".mp3", ".m4a", ".wav")
SLIDE_EXTS = (".png", ".jpg", ".jpeg")

# ---------------------------------------------------------------------------
# Entity / relation taxonomies
# ---------------------------------------------------------------------------

ALLOWED_ENTITY_TYPES = {
    "Claim",
    "Method",
    "Dataset",
    "Metric",
    "Result",
    "Limitation",
    "Slide",
    "VisualRegion",
    "PaperChunk",
    "Figure",
    "Table",
    "Utterance",
    "ProsodyEvent",
}

RELATION_CONSTRAINTS = {
    "supported_by": ({"Claim"}, {"Result", "Figure", "Table", "VisualRegion", "PaperChunk"}),
    "compares": ({"Method", "Result"}, {"Method", "Result"}),
    "measured_by": ({"Result"}, {"Metric"}),
    "aligned_to_slide": ({"Utterance", "Claim", "Method", "Result"}, {"Slide", "VisualRegion"}),
    "grounded_in_paper": (
        {"Claim", "Method", "Result", "Slide", "VisualRegion"},
        {"PaperChunk", "Figure", "Table"},
    ),
    "emphasizes": ({"ProsodyEvent"}, {"Claim", "Method", "Result", "Metric"}),
    "referenced_by": (
        {"Figure", "Table", "VisualRegion", "PaperChunk"},
        {"Utterance", "PaperChunk", "Claim", "Method"},
    ),
}

LOGICAL_RELATIONS = {"supported_by", "measured_by", "compares", "referenced_by"}

RELATION_EVIDENCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "aligned_to_slide": {"audio": 0.2, "slide": 0.8, "paper": 0.0},
    "emphasizes": {"audio": 0.8, "slide": 0.1, "paper": 0.1},
    "supported_by": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "compares": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "measured_by": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "referenced_by": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "grounded_in_paper": {"audio": 0.0, "slide": 0.2, "paper": 0.8},
}

RELATION_PHRASES = {
    "supported_by": "is supported by",
    "measured_by": "is measured by",
    "compares": "is compared with",
    "referenced_by": "is referenced by",
    "aligned_to_slide": "is aligned to slide content",
    "grounded_in_paper": "is grounded in paper",
    "emphasizes": "emphasizes",
}

# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------

TIER_STRONG_MIN = 0.75
TIER_MEDIUM_MIN = 0.45

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

DISFLUENCY_RE = re.compile(
    r"\b(uh|um|erm|hmm|i mean|actually|well|let me rephrase|sorry)\b",
    flags=re.IGNORECASE,
)
DEICTIC_RE = re.compile(
    r"\b(here|this curve|this line|the red line|this figure|this table)\b",
    re.IGNORECASE,
)
TRANSITION_RE = re.compile(
    r"\b(however|but|nevertheless|yet|on the other hand|in contrast)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# LLM system prompts
# ---------------------------------------------------------------------------

TRIPLES_BUILD_SYSTEM_PROMPT = """
You are an information extraction engine for multimodal academic reasoning.
Your task is to analyze the provided claim, its related semantic nodes, and multimodal evidence, and output knowledge graph triples.

Output must be a strict JSON object containing a "triples" array. No markdown, no prose.
The top-level output must be a JSON object with exactly one key "triples" mapped to an array.
If nothing can be extracted, output {"triples": []}.
Each triple object in the array must include exactly:
- "head": {"id": "string", "type": "string", "content": "string"}
- "relation": "string"
- "tail": {"id": "string", "type": "string", "content": "string"}
- "provenance": array of structured objects
- "weight": float in [0,1]
- "signal_source": one of [stress, pause, semantic, hybrid]

Node taxonomy:
Semantic nodes: Claim, Method, Dataset, Metric, Result, Limitation.
Modality nodes: Slide, VisualRegion, PaperChunk, Figure, Table, Utterance, ProsodyEvent.

Allowed relations and type constraints:
- supported_by: Claim -> Result|Figure|Table|VisualRegion|PaperChunk
- compares: Method|Result -> Method|Result
- measured_by: Result -> Metric
- aligned_to_slide: Utterance|Claim|Method|Result -> Slide|VisualRegion
- grounded_in_paper: Claim|Method|Result|Slide|VisualRegion -> PaperChunk|Figure|Table
- emphasizes: ProsodyEvent -> Claim|Method|Result|Metric
- referenced_by: Figure|Table|VisualRegion|PaperChunk -> Utterance|PaperChunk|Claim|Method

Critical prosody rules:
- Input transcript is enriched with tags <STRESS>, <PAUSE>, <SLOW_DOWN>, <RISING_TONE>, <DISFLUENCY>.
- Entities wrapped by <STRESS> are likely novelty/innovation focus and should strongly influence emphasizes.
- <PAUSE>, <RISING_TONE>, <SLOW_DOWN>, <DISFLUENCY> are pragmatic evidence, not noise.
- All emphasizes edges must include audio provenance time anchors.

Claim-centric extraction preference:
- Prefer logical edges first when evidence exists: supported_by, measured_by, compares, referenced_by.
- Avoid returning only alignment/grounding/emphasizes edges.
- Use visual semantics from slide regions (OCR text, chart type, visual summary) as first-class evidence.
- If chart/table cues exist in visual semantics, prefer referenced_by/supported_by with concrete provenance.

Provenance schema examples (use these inside the provenance array):
- audio: {"source_type":"audio", "start_time": 0.0, "end_time": 0.0}
- slide_region: {"source_type":"slide_region", "slide_id": "...", "bbox": [], "region_id": "..."}
- paper_chunk: {"source_type":"paper_chunk", "page": 1, "bbox": [], "chunk_text_preview": "..."}

Never emit relation/entity types outside the whitelist.

EXAMPLE OUTPUT:
{
  "triples": [
    {
      "head": {"id": "neurips-2024/.../claim/...", "type": "Claim", "content": "..."},
      "relation": "supported_by",
      "tail": {"id": "neurips-2024/.../slide/0006/region/00", "type": "VisualRegion", "content": "..."},
      "provenance": [
        {"source_type": "slide_region", "slide_id": "neurips-2024/.../slide/0006", "bbox": [0,0,1000,1000], "region_id": "neurips-2024/.../slide/0006/region/00"}
      ],
      "weight": 0.9,
      "signal_source": "semantic"
    }
  ]
}
""".strip()

VISUAL_REGION_SYSTEM_PROMPT = """
You are a visual-semantic extractor for academic slide and paper regions.
Output strict JSON object only:
{
  "ocr_text": "string",
  "chart_type": "none|line|bar|scatter|table|diagram|equation|image|mixed",
  "visual_summary": "string",
  "entities": ["string", "..."]
}

Field intent:
- ocr_text: only key visible text inside this region; keep original wording when readable.
- chart_type: choose the closest actual visual type in this region; use "none" for pure text/non-chart regions.
- visual_summary: describe the academic semantic content carried by this region (claim/result/method/metric/limitation/evidence), not decorative appearance.
- entities: only KG-useful academic entities (method names, datasets, metrics, symbols, variables, baselines, model names).

Rules:
- Do not output generic placeholder text (e.g., "blank region", "no visible text", "contains slide elements").
- Do not repeat the same statement multiple times in one summary.
- Do not copy full-page context not present in this region.
- No markdown, no prose wrapper, no extra keys.
""".strip()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    conf: str
    session_id: str
    sample_id: str
    path: Path
    metadata: Dict[str, Any]


@dataclass
class Config:
    root: Path
    output_root: Path
    seed: int
    limit: Optional[int]
    random_sample: Optional[int]
    split_train: float
    split_dev: float
    split_test: float
    topk_tags: int
    topk_paper: int
    max_images_per_call: int
    max_image_bytes: int
    prune_missing_audio: bool
    enable_llm: bool
    require_asr: bool
    resume_run_dir: Optional[Path]


# ---------------------------------------------------------------------------
# Helper: relation validation
# ---------------------------------------------------------------------------


def relation_allowed(h_type: str, rel: str, t_type: str) -> bool:
    """Return True if *rel* is a known relation and (*h_type*, *t_type*) satisfy its constraints."""
    if rel not in RELATION_CONSTRAINTS:
        return False
    hs, ts = RELATION_CONSTRAINTS[rel]
    return (h_type in hs) and (t_type in ts)


# ---------------------------------------------------------------------------
# Environment-variable helpers
# ---------------------------------------------------------------------------


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, min_value: int = 1, max_value: int = 1024) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        val = int(str(raw).strip())
    except Exception:
        return default
    if val < min_value:
        return min_value
    if val > max_value:
        return max_value
    return val


def env_int_with_fallback(
    primary: str,
    fallback: str,
    default: int,
    min_value: int = 1,
    max_value: int = 1024,
) -> int:
    raw_p = os.getenv(primary)
    if raw_p is not None and str(raw_p).strip() != "":
        return env_int(primary, default=default, min_value=min_value, max_value=max_value)
    return env_int(fallback, default=default, min_value=min_value, max_value=max_value)


def to_float_or_none(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if math.isfinite(x):
        return x
    return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_json_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, list):
        return [sanitize_json_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    return obj


JSONL_APPEND_LOCK = threading.Lock()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(sanitize_json_obj(row), ensure_ascii=False) + "\n"
    with JSONL_APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def run_cmd(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)
