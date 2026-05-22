from __future__ import annotations

import concurrent.futures
import json
import math
import os
import random
import shutil
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, SessionRecord
from .io_utils import write_json, write_jsonl, append_jsonl, read_jsonl, recover_manifest_from_session_outputs
from .llm_client import get_triples_build_llm_executor, shutdown_all_llm_executors, llm_preflight_status
from .slides import build_slides_and_vision_index
from .paper import build_paper_structured
from .visual_semantics import enrich_slides_visual_semantics
from .audio import ffprobe_duration, detect_silences, detect_rms_series, maybe_transcribe_openai, detect_pitch_slope_librosa
from .transcript import utterance_from_asr, utterance_from_silence, inject_tags_and_events
from .pragmatic import build_pragmatic_signals
from .alignment import build_alignment_context
from .kg_extract import build_claim_centric_kg
from .validation import validate_sample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTS = (".mp3", ".m4a", ".wav")

# ---------------------------------------------------------------------------
# Helpers (local to this module)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def median_filter(values: List[float], window: int = 5) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    half = window // 2
    out = []
    for i in range(len(values)):
        s = max(0, i - half)
        e = min(len(values), i + half + 1)
        out.append(statistics.median(values[s:e]))
    return out


def normalize_z(values: List[float]) -> List[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    mu = statistics.mean(values)
    sigma = statistics.pstdev(values)
    if sigma <= 1e-8:
        return [0.0 for _ in values]
    return [(v - mu) / sigma for v in values]


class GlobalRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0  # monotonic timestamp when next request is allowed

    def acquire(self, min_interval_ms: int) -> None:
        if min_interval_ms <= 0:
            return
        min_interval_sec = min_interval_ms / 1000.0
        with self._lock:
            now = time.monotonic()
            # Calculate when this request should be allowed
            self._next_allowed_time = max(self._next_allowed_time, now)
            wait_time = self._next_allowed_time - now
            # Advance the next allowed time for the next request
            self._next_allowed_time += min_interval_sec
        if wait_time > 0:
            time.sleep(wait_time)


# ---------------------------------------------------------------------------
# Session discovery & audio helpers
# ---------------------------------------------------------------------------


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
            rows.append(SessionRecord(conf=conf, session_id=session_id, sample_id=sample_id, path=session_dir, metadata=metadata))
    return rows


def has_audio(session_path: Path) -> bool:
    return any((session_path / f"audio{ext}").exists() for ext in AUDIO_EXTS)


def find_audio_file(session_path: Path) -> Optional[Path]:
    for ext in AUDIO_EXTS:
        p = session_path / f"audio{ext}"
        if p.exists():
            return p
    return None


def prune_sessions_without_audio(records: List[SessionRecord], prune: bool, log_file: Path) -> List[SessionRecord]:
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
        if prune:
            shutil.rmtree(rec.path, ignore_errors=True)
    return kept


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def split_ids(ids: List[str], seed: int, tr: float, dv: float, te: float) -> Dict[str, List[str]]:
    if not math.isclose(tr + dv + te, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1.0")
    arr = ids[:]
    rnd = random.Random(seed)
    rnd.shuffle(arr)
    n = len(arr)
    n_tr = int(n * tr)
    n_dv = int(n * dv)
    return {
        "train": arr[:n_tr],
        "dev": arr[n_tr:n_tr + n_dv],
        "test": arr[n_tr + n_dv:],
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def process_session(rec: SessionRecord, cfg: Config, run_dir: Path, run_log: Path) -> Dict[str, Any]:
    session_out = run_dir / "sessions" / rec.conf / rec.session_id
    session_out.mkdir(parents=True, exist_ok=True)
    llm_debug_dir = session_out / "llm_debug"

    def log(stage: str, status: str, message: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        row = {
            "timestamp": now_iso(),
            "sample_id": rec.sample_id,
            "stage": stage,
            "status": status,
            "message": message,
        }
        if extra:
            row.update(extra)
        append_jsonl(run_log, row)

    log("start", "ok", str(rec.path))

    if cfg.enable_llm:
        ready, detail = llm_preflight_status()
        if not ready:
            raise RuntimeError(f"LLM enabled but unavailable ({detail})")
        log("llm_preflight", "ok", extra={"detail": detail})
        log(
            "llm_runtime_config",
            "ok",
            extra={
                "triples_build_concurrency": env_int(
                    "TRIPLES_BUILD_CONCURRENCY",
                    default=10,
                    min_value=1,
                    max_value=128,
                ),
                "triples_build_min_interval_ms": env_int(
                    "TRIPLES_BUILD_MIN_INTERVAL_MS",
                    default=300,
                    min_value=0,
                    max_value=600000,
                ),
                "visual_llm_concurrency": env_int(
                    "VISION_LLM_CONCURRENCY",
                    default=50,
                    min_value=1,
                    max_value=128,
                ),
                "visual_llm_min_interval_ms": env_int(
                    "VISION_LLM_MIN_INTERVAL_MS",
                    default=300,
                    min_value=0,
                    max_value=600000,
                ),
                "asr_concurrency": env_int(
                    "ASR_CONCURRENCY",
                    default=5,
                    min_value=1,
                    max_value=32,
                ),
                "asr_min_interval_ms": env_int(
                    "ASR_MIN_INTERVAL_MS",
                    default=100,
                    min_value=0,
                    max_value=600000,
                ),
            },
        )

    audio_path = find_audio_file(rec.path)
    if audio_path is None:
        raise RuntimeError("audio missing after pruning")

    slides_structured, vision_index = build_slides_and_vision_index(rec)
    paper_structured = build_paper_structured(rec, vision_index)
    log("slides", "ok", extra={"slides": len(slides_structured["slides"]), "regions": len(slides_structured["visual_regions"])})
    non_empty_chunks = sum(1 for c in paper_structured.get("paper_chunks", []) if str(c.get("text", "")).strip())
    log(
        "paper",
        "ok",
        extra={
            "chunks": len(paper_structured["paper_chunks"]),
            "non_empty_chunks": non_empty_chunks,
            "pdf_text_extractor": paper_structured.get("pdf_text_extractor", ""),
            "pdf_page_renderer": paper_structured.get("pdf_page_renderer", ""),
            "paper_pages": len(paper_structured.get("paper_pages", [])),
            "paper_visual_regions": len(paper_structured.get("paper_visual_regions", [])),
        },
    )
    if non_empty_chunks == 0 and os.getenv("ALLOW_EMPTY_PAPER_TEXT", "0").strip() not in {"1", "true", "yes"}:
        raise RuntimeError(
            f"paper text extraction is empty (extractor={paper_structured.get('pdf_text_extractor')}); "
            "install pypdf/PyPDF2 or set ALLOW_EMPTY_PAPER_TEXT=1 to bypass"
        )

    def _run_visual_branch() -> Dict[str, Any]:
        return enrich_slides_visual_semantics(
            rec,
            slides_structured,
            paper_structured,
            vision_index,
            cfg,
            debug_dir=llm_debug_dir,
        )

    def _run_audio_branch() -> Dict[str, Any]:
        duration = ffprobe_duration(audio_path) or 0.0
        silences = detect_silences(audio_path)
        rms_series = detect_rms_series(audio_path)

        asr_payload, asr_status = maybe_transcribe_openai(audio_path, duration=duration, silences=silences)
        asr_mode = "silence_fallback"
        if asr_payload:
            utterances = utterance_from_asr(asr_payload)
            if utterances:
                asr_mode = "remote_asr"
            else:
                asr_status = f"{asr_status};no_segments_from_payload"
                utterances = utterance_from_silence(duration, silences)
        else:
            utterances = utterance_from_silence(duration, silences)

        if cfg.require_asr and asr_mode != "remote_asr":
            raise RuntimeError(f"ASR required but unavailable ({asr_status})")

        pitch_slopes = detect_pitch_slope_librosa(audio_path, utterances)
        if pitch_slopes and len(pitch_slopes) == len(utterances):
            z_pitch = normalize_z(median_filter(pitch_slopes, window=5))
            for i, z in enumerate(z_pitch):
                if z > 1.5 and not utterances[i]["text"].strip().endswith("?"):
                    utterances[i]["text"] = utterances[i]["text"].rstrip(".") + "?"

        transcript_prosody, transcript_enriched = inject_tags_and_events(
            sample_id=rec.sample_id,
            utterances=utterances,
            silences=silences,
            rms_series=rms_series,
            topk=cfg.topk_tags,
        )
        pragmatic = build_pragmatic_signals(rec.sample_id, transcript_enriched)
        return {
            "transcript_prosody": transcript_prosody,
            "transcript_enriched": transcript_enriched,
            "pragmatic": pragmatic,
            "asr_mode": asr_mode,
            "asr_status": asr_status,
        }

    log("parallel_stage", "ok", extra={"branches": ["visual_semantics", "audio_prosody_pragmatic"]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_audio = ex.submit(_run_audio_branch)
        fut_visual = ex.submit(_run_visual_branch)
        audio_out = fut_audio.result()
        visual_semantics_report = fut_visual.result()

    transcript_prosody = audio_out["transcript_prosody"]
    transcript_enriched = audio_out["transcript_enriched"]
    pragmatic = audio_out["pragmatic"]
    asr_mode = str(audio_out["asr_mode"])
    asr_status = str(audio_out["asr_status"])

    write_json(session_out / "slides_structured.json", slides_structured)
    write_json(session_out / "paper_structured.json", paper_structured)
    write_json(session_out / "vision_index.json", vision_index)
    write_json(session_out / "visual_semantics_report.json", visual_semantics_report)
    write_json(session_out / "transcript_prosody.json", transcript_prosody)
    write_json(session_out / "transcript_enriched.json", transcript_enriched)
    write_json(session_out / "pragmatic_signals.json", pragmatic)

    log(
        "visual_semantics",
        "ok",
        extra={
            "enabled": visual_semantics_report.get("enabled"),
            "regions_targeted": visual_semantics_report.get("regions_targeted"),
            "regions_sent_to_llm": visual_semantics_report.get("regions_sent_to_llm"),
            "regions_enriched": visual_semantics_report.get("regions_enriched"),
            "regions_pruned_blank": visual_semantics_report.get("regions_pruned_blank"),
            "regions_pruned_low_info": visual_semantics_report.get("regions_pruned_low_info"),
            "duplicate_summary_filtered": visual_semantics_report.get("duplicate_summary_filtered"),
            "response_format_fallback_count": visual_semantics_report.get("response_format_fallback_count"),
            "json_parse_fail_count": visual_semantics_report.get("json_parse_fail_count"),
            "coverage": visual_semantics_report.get("coverage"),
        },
    )
    log(
        "audio_prosody",
        "ok",
        extra={
            "utterances": len(transcript_prosody["utterances"]),
            "events": len(transcript_prosody["prosody_events"]),
            "asr_mode": asr_mode,
            "asr_status": asr_status,
        },
    )
    log("pragmatic", "ok", extra={"signals": len(pragmatic["signals"])})

    alignment = build_alignment_context(
        rec=rec,
        slides_structured=slides_structured,
        paper_structured=paper_structured,
        transcript_prosody=transcript_prosody,
        transcript_enriched=transcript_enriched,
        cfg=cfg,
    )
    semantic_nodes, evidence_links, kg_prompt_payload, triples, quality = build_claim_centric_kg(
        rec=rec,
        cfg=cfg,
        slides_structured=slides_structured,
        paper_structured=paper_structured,
        transcript_prosody=transcript_prosody,
        transcript_enriched=transcript_enriched,
        pragmatic=pragmatic,
        alignment=alignment,
        debug_dir=llm_debug_dir,
    )

    write_json(session_out / "alignment_index.json", alignment)
    write_json(session_out / "semantic_nodes.json", semantic_nodes)
    write_json(session_out / "evidence_links.json", evidence_links)
    write_json(session_out / "kg_extract_prompt.json", kg_prompt_payload)
    trainable_triples = [t for t in triples if str(t.get("tier", "")).strip() in {"strong", "medium"}]
    write_jsonl(session_out / "kg_triples.jsonl", triples)
    write_jsonl(session_out / "kg_triples_trainable.jsonl", trainable_triples)
    write_json(session_out / "kg_quality_report.json", quality)
    log(
        "kg_extract",
        "ok",
        extra={
            "triples": len(triples),
            "triples_trainable": len(trainable_triples),
            "quality_status": quality.get("quality_status"),
        },
    )

    validation = validate_sample(transcript_enriched, transcript_prosody, alignment, triples)
    validation["stats"]["quality_status"] = quality.get("quality_status")
    write_json(session_out / "pipeline_validation.json", validation)
    log(
        "validate",
        "ok" if validation["ok"] else "warn",
        extra={"errors": validation["error_count"], "quality_status": quality.get("quality_status")},
    )

    return {
        "sample_id": rec.sample_id,
        "session_id": rec.session_id,
        "conference": rec.conf,
        "output_dir": str(session_out),
        "validation_ok": validation["ok"],
        "triples": len(triples),
        "triples_trainable": len(trainable_triples),
        "quality_status": quality.get("quality_status"),
        "error_count": validation["error_count"],
        "asr_mode": asr_mode,
        "asr_status": asr_status,
    }


# ---------------------------------------------------------------------------
# Multi-session pipeline scheduler with resume support
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
            "recovered_from_sessions": max(0, len(recovered) - len(read_jsonl(run_dir / "manifest.jsonl"))),
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

    processed_n = 0  # Total records processed (includes skips)
    executed_n = 0   # Actually executed sessions (excludes skips)
    pending_records: List[SessionRecord] = []
    for rec in records:
        if cfg.resume_run_dir is not None and rec.sample_id in manifest_by_id:
            prior = manifest_by_id.get(rec.sample_id, {})
            prior_quality = str(prior.get("quality_status", "")).strip()
            # Resume policy: always re-run sessions that previously failed coverage.
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
            # For resume-skip, estimate based on executed sessions only (not including skips)
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
            """Submit one session. If wait_for_delay is True, apply session_min_interval_ms."""
            nonlocal submitted
            try:
                rec_next = next(pending_iter)
            except StopIteration:
                return False

            # Apply delay for initial batch (except the very first one)
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
            # Initial batch: submit with delays to stagger startup
            for i in range(initial_batch_size):
                if not _submit_one(ex, wait_for_delay=True):
                    break  # No more records

            # Main loop: FIFO - submit immediately upon completion
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

                    # FIFO: submit next session immediately (no delay)
                    _submit_one(ex, wait_for_delay=False)

    manifest_rows = sorted(manifest_by_id.values(), key=lambda x: str(x.get("sample_id", "")))
    failures = sorted(failures_by_id.values(), key=lambda x: str(x.get("sample_id", "")))
    _flush_progress()

    ids = [x["sample_id"] for x in manifest_rows]
    splits = split_ids(ids, cfg.seed, cfg.split_train, cfg.split_dev, cfg.split_test)
    write_json(run_dir / "splits.json", splits)

    for split_name, split_ids_list in splits.items():
        rows = []
        rows_trainable = []
        for sid in split_ids_list:
            conf, sess = sid.split("/", 1)
            kg = run_dir / "sessions" / conf / sess / "kg_triples.jsonl"
            kg_trainable = run_dir / "sessions" / conf / sess / "kg_triples_trainable.jsonl"
            if kg.exists():
                for line in kg.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            if kg_trainable.exists():
                for line in kg_trainable.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows_trainable.append(json.loads(line))
        write_jsonl(run_dir / f"{split_name}.kg_triples.jsonl", rows)
        write_jsonl(run_dir / f"{split_name}.kg_triples_trainable.jsonl", rows_trainable)
        with open(run_dir / f"{split_name}_ids.txt", "w", encoding="utf-8") as f:
            for sid in split_ids_list:
                f.write(sid + "\n")

    success = len(manifest_rows)
    total = len(records)
    summary = {
        "run_id": run_id,
        "started_root": str(cfg.root),
        "total_selected": total,
        "success": success,
        "failed": len(failures),
        "success_rate": round(success / max(1, total), 4),
        "acceptance_target_90pct": (success / max(1, total)) >= 0.9,
        "validation_ok": sum(1 for x in manifest_rows if x.get("validation_ok")),
        "validation_warn": sum(1 for x in manifest_rows if not x.get("validation_ok")),
        "asr_remote_count": sum(1 for x in manifest_rows if x.get("asr_mode") == "remote_asr"),
        "asr_fallback_count": sum(1 for x in manifest_rows if x.get("asr_mode") != "remote_asr"),
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
