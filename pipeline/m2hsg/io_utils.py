"""I/O utilities for the SynapseHSG pipeline: JSON/JSONL helpers, subprocess wrapper, timestamps."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Thread-safe lock for JSONL append operations.
JSONL_APPEND_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def sanitize_json_obj(obj: Any) -> Any:
    """Recursively normalize *obj* so it can be safely serialized to JSON.

    In particular, replaces unpaired UTF-16 surrogates (which pypdf can emit)
    with the Unicode replacement character.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, list):
        return [sanitize_json_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    return obj


def write_json(path: Path, obj: Any) -> None:
    """Write a JSON object to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = sanitize_json_obj(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    """Read and parse a JSON file from *path*.

    Returns ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Overwrite *path* with one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize_json_obj(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, silently skipping blank and malformed lines."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single JSON line to *path* (thread-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(sanitize_json_obj(row), ensure_ascii=False) + "\n"
    with JSONL_APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _count_jsonl_rows(path: Path) -> int:
    """Count non-blank lines in a JSONL file (fast row-count heuristic)."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n


# ---------------------------------------------------------------------------
# Manifest recovery
# ---------------------------------------------------------------------------


def recover_manifest_from_session_outputs(
    run_dir: Path,
    selected_ids: set[str],
) -> Dict[str, Dict[str, Any]]:
    """Re-build a session manifest by scanning existing session output directories.

    Parameters
    ----------
    run_dir:
        Root run directory (expected to contain a ``sessions/`` sub-directory).
    selected_ids:
        Only include sessions whose ``conference/session_id`` key is in this set.
    """
    out: Dict[str, Dict[str, Any]] = {}
    sessions_root = run_dir / "sessions"
    if not sessions_root.exists():
        return out
    for conf_dir in sessions_root.iterdir():
        if not conf_dir.is_dir():
            continue
        conf = conf_dir.name
        for sess_dir in conf_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            sid = sess_dir.name
            sample_id = f"{conf}/{sid}"
            if sample_id not in selected_ids:
                continue
            kg = sess_dir / "kg_triples.jsonl"
            if not kg.exists():
                continue
            kg_trainable = sess_dir / "kg_triples_trainable.jsonl"
            quality = sess_dir / "kg_quality_report.json"
            validation = sess_dir / "pipeline_validation.json"
            asr_mode = ""
            asr_status = ""
            validation_ok = True
            error_count = 0
            quality_status = ""
            if validation.exists():
                try:
                    v = json.loads(validation.read_text(encoding="utf-8"))
                    validation_ok = bool(v.get("ok", True))
                    error_count = int(v.get("error_count", 0))
                except Exception:
                    pass
            if quality.exists():
                try:
                    q = json.loads(quality.read_text(encoding="utf-8"))
                    quality_status = str(q.get("quality_status", "")).strip()
                except Exception:
                    pass
            out[sample_id] = {
                "sample_id": sample_id,
                "session_id": sid,
                "conference": conf,
                "output_dir": str(sess_dir),
                "validation_ok": validation_ok,
                "triples": _count_jsonl_rows(kg),
                "triples_trainable": _count_jsonl_rows(kg_trainable),
                "quality_status": quality_status,
                "error_count": error_count,
                "asr_mode": asr_mode,
                "asr_status": asr_status,
            }
    return out


# ---------------------------------------------------------------------------
# Subprocess / timestamps
# ---------------------------------------------------------------------------


def run_cmd(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """Execute *cmd* via :func:`subprocess.run` and return ``(rc, stdout, stderr)``."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (``Z`` suffix)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
