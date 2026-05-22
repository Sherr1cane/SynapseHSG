"""Build Hypergraph Semantic Graphs (HSG) from raw session data.

Discovers conference/session directories under ``--root``, runs the full
per-session pipeline (paper parsing, slide processing, ASR, alignment,
KG extraction), and writes structured HSG triples to ``--output``.

Usage:
    python -m pipeline.build_hsg --root /path/to/raw_data --output /path/to/output
    python -m pipeline.build_hsg --root /path/to/raw_data --resume-latest
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure pipeline/ is on sys.path for sibling module imports
_pipeline_dir = str(Path(__file__).resolve().parent)
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

from m2hsg.config import Config, SessionRecord, env_int
from m2hsg.io_utils import (
    append_jsonl,
    now_iso,
    read_jsonl,
    recover_manifest_from_session_outputs,
    write_json,
    write_jsonl,
)
from m2hsg.process import process_session
from m2hsg.llm_client import shutdown_all_llm_executors


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

AUDIO_EXTS = (".mp3", ".m4a", ".wav")


def discover_sessions(root: Path) -> List[SessionRecord]:
    rows: List[SessionRecord] = []
    for conf_dir in sorted(root.iterdir()):
        if not conf_dir.is_dir():
            continue
        conf = conf_dir.name
        for session_dir in sorted(conf_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            meta_file = session_dir / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            session_id = str(metadata.get("session_id") or session_dir.name.split("_")[0])
            sample_id = f"{conf}/{session_id}"
            rows.append(SessionRecord(
                conf=conf, session_id=session_id,
                sample_id=sample_id, path=session_dir, metadata=metadata,
            ))
    return rows


def has_audio(session_path: Path) -> bool:
    return any((session_path / f"audio{ext}").exists() for ext in AUDIO_EXTS)


def prune_sessions_without_audio(
    records: List[SessionRecord], prune: bool, log_file: Path,
) -> List[SessionRecord]:
    kept: List[SessionRecord] = []
    for rec in records:
        if has_audio(rec.path):
            kept.append(rec)
            continue
        append_jsonl(log_file, {
            "timestamp": now_iso(),
            "sample_id": rec.sample_id,
            "stage": "prune_missing_audio",
            "status": "pruned" if prune else "skipped",
            "message": "missing audio",
            "path": str(rec.path),
        })
    return kept


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class GlobalRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire(self, min_interval_ms: int) -> None:
        if min_interval_ms <= 0:
            return
        min_interval_sec = min_interval_ms / 1000.0
        with self._lock:
            now = time.monotonic()
            self._next_allowed_time = max(self._next_allowed_time, now)
            wait_time = self._next_allowed_time - now
            self._next_allowed_time += min_interval_sec
        if wait_time > 0:
            time.sleep(wait_time)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config) -> Path:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    run_t0 = time.monotonic()

    if cfg.resume_run_dir is not None:
        run_dir = cfg.resume_run_dir.resolve()
        if not run_dir.exists():
            raise SystemExit(f"resume run dir not found: {run_dir}")
        run_id = run_dir.name
    else:
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = cfg.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run_log.jsonl"

    records = discover_sessions(cfg.root)
    append_jsonl(run_log, {
        "timestamp": now_iso(),
        "stage": "discover",
        "status": "ok",
        "sessions_found": len(records),
    })

    records = prune_sessions_without_audio(records, cfg.prune_missing_audio, run_log)

    if cfg.random_sample is not None and cfg.random_sample < len(records):
        rnd = random.Random(cfg.seed)
        records = sorted(rnd.sample(records, cfg.random_sample), key=lambda r: (r.conf, r.session_id))

    if cfg.limit is not None:
        records = records[: cfg.limit]
    selected_ids = {r.sample_id for r in records}

    manifest_by_id: Dict[str, Dict[str, Any]] = {}
    failures_by_id: Dict[str, Dict[str, Any]] = {}
    resume_skipped_completed = 0
    resume_rerun_failed_coverage = 0
    resume_loaded_manifest = 0
    resume_loaded_failures = 0

    if cfg.resume_run_dir is not None:
        for row in read_jsonl(run_dir / "manifest.jsonl"):
            sid = str(row.get("sample_id", "")).strip()
            if not sid or sid not in selected_ids:
                continue
            manifest_by_id[sid] = row
        for row in read_jsonl(run_dir / "failures.jsonl"):
            sid = str(row.get("sample_id", "")).strip()
            if not sid or sid not in selected_ids:
                continue
            failures_by_id[sid] = row
        recovered = recover_manifest_from_session_outputs(run_dir, selected_ids)
        for sid, row in recovered.items():
            if sid not in manifest_by_id:
                manifest_by_id[sid] = row
        resume_loaded_manifest = len(manifest_by_id)
        resume_loaded_failures = len(failures_by_id)
        append_jsonl(run_log, {
            "timestamp": now_iso(),
            "stage": "resume",
            "status": "ok",
            "run_dir": str(run_dir),
            "loaded_manifest": resume_loaded_manifest,
            "loaded_failures": resume_loaded_failures,
        })

    def _flush_progress() -> None:
        rows = sorted(manifest_by_id.values(), key=lambda x: str(x.get("sample_id", "")))
        errs = sorted(failures_by_id.values(), key=lambda x: str(x.get("sample_id", "")))
        write_jsonl(run_dir / "manifest.jsonl", rows)
        if errs:
            write_jsonl(run_dir / "failures.jsonl", errs)
        else:
            try:
                (run_dir / "failures.jsonl").unlink()
            except Exception:
                pass

    session_concurrency = env_int("SESSION_CONCURRENCY", default=1, min_value=1, max_value=64)
    session_min_interval_ms = env_int("SESSION_MIN_INTERVAL_MS", default=0, min_value=0, max_value=600000)
    append_jsonl(run_log, {
        "timestamp": now_iso(),
        "stage": "session_scheduler",
        "status": "ok",
        "session_concurrency": session_concurrency,
        "session_min_interval_ms": session_min_interval_ms,
    })

    processed_n = 0
    executed_n = 0
    pending_records: List[SessionRecord] = []
    for rec in records:
        if cfg.resume_run_dir is not None and rec.sample_id in manifest_by_id:
            prior = manifest_by_id.get(rec.sample_id, {})
            prior_quality = str(prior.get("quality_status", "")).strip()
            if prior_quality == "failed_coverage":
                resume_rerun_failed_coverage += 1
                append_jsonl(run_log, {
                    "timestamp": now_iso(),
                    "sample_id": rec.sample_id,
                    "stage": "resume_rerun",
                    "status": "ok",
                    "reason": "failed_coverage",
                })
                pending_records.append(rec)
                continue
            resume_skipped_completed += 1
            processed_n += 1
            append_jsonl(run_log, {
                "timestamp": now_iso(),
                "sample_id": rec.sample_id,
                "stage": "resume_skip",
                "status": "ok",
                "message": "already_completed",
            })
            elapsed = time.monotonic() - run_t0
            avg_per_item = elapsed / max(1, executed_n) if executed_n > 0 else 0
            remaining = max(0, len(records) - processed_n)
            eta_sec = remaining * avg_per_item if avg_per_item > 0 else 0
            completed_n = len(manifest_by_id) + len(failures_by_id)
            print(
                f"[{processed_n}/{len(records)}] resume-skip {rec.sample_id} | "
                f"elapsed={elapsed:.1f}s avg={avg_per_item:.1f}s eta={eta_sec:.1f}s "
                f"done={completed_n} ok={len(manifest_by_id)} fail={len(failures_by_id)}"
            )
        else:
            pending_records.append(rec)

    def _print_progress(prefix: str, sample_id: str, session_elapsed: float) -> None:
        elapsed = time.monotonic() - run_t0
        avg_per_item = elapsed / max(1, executed_n)
        remaining = max(0, len(records) - processed_n)
        eta_sec = remaining * avg_per_item
        print(
            f"  {prefix} {sample_id} | session={session_elapsed:.1f}s "
            f"elapsed={elapsed:.1f}s avg={avg_per_item:.1f}s eta={eta_sec:.1f}s "
            f"ok={len(manifest_by_id)} fail={len(failures_by_id)}"
        )

    session_start_limiter = GlobalRateLimiter()

    if session_concurrency <= 1:
        for rec in pending_records:
            print(f"[{processed_n + 1}/{len(records)}] {rec.sample_id}")
            session_t0 = time.monotonic()
            try:
                row = process_session(rec, cfg, run_dir, run_log)
                manifest_by_id[rec.sample_id] = row
                failures_by_id.pop(rec.sample_id, None)
                _flush_progress()
                processed_n += 1
                executed_n += 1
                _print_progress("done", rec.sample_id, time.monotonic() - session_t0)
            except Exception as e:
                err = {
                    "timestamp": now_iso(),
                    "sample_id": rec.sample_id,
                    "session_dir": str(rec.path),
                    "error": str(e),
                }
                failures_by_id[rec.sample_id] = err
                append_jsonl(run_log, {
                    "timestamp": now_iso(),
                    "sample_id": rec.sample_id,
                    "stage": "pipeline",
                    "status": "error",
                    "message": str(e),
                })
                _flush_progress()
                processed_n += 1
                executed_n += 1
                _print_progress("fail", rec.sample_id, time.monotonic() - session_t0)
    else:
        submitted = 0
        pending_iter = iter(pending_records)
        future_map: Dict[concurrent.futures.Future, Tuple[SessionRecord, float]] = {}
        initial_batch_size = session_concurrency

        def _submit_one(executor: concurrent.futures.ThreadPoolExecutor, wait_for_delay: bool = True) -> bool:
            nonlocal submitted
            try:
                rec_next = next(pending_iter)
            except StopIteration:
                return False

            if wait_for_delay and session_min_interval_ms > 0 and submitted >= 1:
                session_start_limiter.acquire(session_min_interval_ms)

            submitted += 1
            in_flight = len(future_map) + 1
            queued_pos = resume_skipped_completed + submitted
            print(f"[{queued_pos}/{len(records)}] queued {rec_next.sample_id} (in_flight={in_flight})")
            fut_next = executor.submit(process_session, rec_next, cfg, run_dir, run_log)
            future_map[fut_next] = (rec_next, time.monotonic())
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=session_concurrency) as ex:
            for i in range(initial_batch_size):
                if not _submit_one(ex, wait_for_delay=True):
                    break

            while future_map:
                done_set, _ = concurrent.futures.wait(
                    list(future_map.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done_set:
                    rec, session_t0 = future_map.pop(fut)
                    try:
                        row = fut.result()
                        manifest_by_id[rec.sample_id] = row
                        failures_by_id.pop(rec.sample_id, None)
                        _flush_progress()
                        processed_n += 1
                        executed_n += 1
                        _print_progress("done", rec.sample_id, time.monotonic() - session_t0)
                    except Exception as e:
                        err = {
                            "timestamp": now_iso(),
                            "sample_id": rec.sample_id,
                            "session_dir": str(rec.path),
                            "error": str(e),
                        }
                        failures_by_id[rec.sample_id] = err
                        append_jsonl(run_log, {
                            "timestamp": now_iso(),
                            "sample_id": rec.sample_id,
                            "stage": "pipeline",
                            "status": "error",
                            "message": str(e),
                        })
                        _flush_progress()
                        processed_n += 1
                        executed_n += 1
                        _print_progress("fail", rec.sample_id, time.monotonic() - session_t0)

                    _submit_one(ex, wait_for_delay=False)

    manifest_rows = sorted(manifest_by_id.values(), key=lambda x: str(x.get("sample_id", "")))
    failures = sorted(failures_by_id.values(), key=lambda x: str(x.get("sample_id", "")))
    _flush_progress()

    success = len(manifest_rows)
    total = len(records)
    summary = {
        "run_id": run_id,
        "started_root": str(cfg.root),
        "total_selected": total,
        "success": success,
        "failed": len(failures),
        "success_rate": round(success / max(1, total), 4),
        "validation_ok": sum(1 for x in manifest_rows if x.get("validation_ok")),
        "validation_warn": sum(1 for x in manifest_rows if not x.get("validation_ok")),
        "triples_total": sum(int(x.get("triples", 0)) for x in manifest_rows),
        "triples_trainable_total": sum(int(x.get("triples_trainable", 0)) for x in manifest_rows),
        "quality_ok_count": sum(1 for x in manifest_rows if x.get("quality_status") == "ok"),
        "quality_failed_count": sum(1 for x in manifest_rows if x.get("quality_status") == "failed_coverage"),
        "session_concurrency": session_concurrency,
        "session_min_interval_ms": session_min_interval_ms,
        "resume_mode": cfg.resume_run_dir is not None,
        "resume_loaded_manifest": resume_loaded_manifest,
        "resume_loaded_failures": resume_loaded_failures,
        "resume_skipped_completed": resume_skipped_completed,
        "resume_rerun_failed_coverage": resume_rerun_failed_coverage,
        "wall_time_sec": round(time.monotonic() - run_t0, 3),
        "generated_at": now_iso(),
    }
    write_json(run_dir / "run_summary.json", summary)

    print("\nRun summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynapseHSG — Build Hypergraph Semantic Graphs")
    p.add_argument("--root", default="raw_data", help="Input raw data root")
    p.add_argument("--output", default=None, help="Output root (default: <root>/_hsg_output)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--random-sample", type=int, default=None)
    p.add_argument("--topk-tags", type=int, default=2)
    p.add_argument("--topk-paper", type=int, default=5)
    p.add_argument("--max-images-per-call", type=int, default=1)
    p.add_argument("--max-image-bytes", type=int, default=400000)
    p.add_argument("--prune-missing-audio", action="store_true", default=True)
    p.add_argument("--no-prune-missing-audio", dest="prune_missing_audio", action="store_false")
    p.add_argument("--enable-llm", action="store_true", default=True)
    p.add_argument("--disable-llm", dest="enable_llm", action="store_false")
    p.add_argument("--require-asr", action="store_true", default=False,
                   help="Fail session when remote ASR is unavailable.")
    p.add_argument("--resume-run-dir", type=str, default=None,
                   help="Resume an existing run dir and continue unfinished sessions.")
    p.add_argument("--resume-latest", action="store_true", default=False,
                   help="Resume latest run under output root.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Input root not found: {root}")

    output_root = Path(args.output).resolve() if args.output else (root / "_hsg_output").resolve()
    resume_run_dir: Optional[Path] = None
    if args.resume_run_dir:
        resume_run_dir = Path(args.resume_run_dir).resolve()
    elif args.resume_latest:
        cands = sorted(
            [p for p in output_root.glob("run_*") if p.is_dir()],
            key=lambda x: x.name,
        )
        if not cands:
            raise SystemExit(f"no run_* found under output root for --resume-latest: {output_root}")
        resume_run_dir = cands[-1].resolve()

    cfg = Config(
        root=root,
        output_root=output_root,
        seed=args.seed,
        limit=args.limit,
        random_sample=args.random_sample,
        split_train=0.7,
        split_dev=0.15,
        split_test=0.15,
        topk_tags=args.topk_tags,
        topk_paper=args.topk_paper,
        max_images_per_call=args.max_images_per_call,
        max_image_bytes=args.max_image_bytes,
        prune_missing_audio=args.prune_missing_audio,
        enable_llm=args.enable_llm,
        require_asr=args.require_asr,
        resume_run_dir=resume_run_dir,
    )

    try:
        run_pipeline(cfg)
    finally:
        shutdown_all_llm_executors()


if __name__ == "__main__":
    main()
