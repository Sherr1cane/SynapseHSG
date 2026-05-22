"""Audio processing for the SynapseHSG pipeline.

Handles duration probing, silence detection, RMS analysis, speech segmentation,
ASR chunking, speech-onset detection, pitch-slope extraction, and OpenAI-compatible
remote transcription.
"""

from __future__ import annotations

import io
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from .config import _safe_float
from .io_utils import run_cmd


# ---------------------------------------------------------------------------
# Duration probing
# ---------------------------------------------------------------------------


def ffprobe_duration(audio_path: Path) -> Optional[float]:
    code, out, _ = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ])
    if code != 0:
        return None
    try:
        return float(out.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Silence / RMS detection
# ---------------------------------------------------------------------------


def detect_silences(audio_path: Path) -> List[Tuple[float, float]]:
    code, _, err = run_cmd([
        "ffmpeg", "-i", str(audio_path), "-af", "silencedetect=noise=-30dB:d=0.20", "-f", "null", "-"
    ], timeout=300)
    text = err
    if code != 0 and "silence_start" not in text:
        return []

    starts: List[float] = []
    rows: List[Tuple[float, float]] = []
    for line in text.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m1:
            starts.append(float(m1.group(1)))
            continue
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m2 and starts:
            s = starts.pop(0)
            e = float(m2.group(1))
            if e > s:
                rows.append((s, e))
    return rows


def detect_rms_series(audio_path: Path) -> List[Tuple[float, float]]:
    code, out, err = run_cmd([
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f",
        "null",
        "-",
    ], timeout=300)
    text = out + "\n" + err
    if code != 0 and "lavfi.astats.Overall.RMS_level" not in text:
        return []

    rows: List[Tuple[float, float]] = []
    cur_t: Optional[float] = None
    for line in text.splitlines():
        m_t = re.search(r"pts_time:([0-9.]+)", line)
        if m_t:
            cur_t = float(m_t.group(1))
            continue
        m_r = re.search(r"lavfi\.astats\.Overall\.RMS_level=([-0-9.]+)", line)
        if m_r and cur_t is not None:
            try:
                v = float(m_r.group(1))
            except ValueError:
                cur_t = None
                continue
            rows.append((cur_t, v))
            cur_t = None
    return rows


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Speech segmentation
# ---------------------------------------------------------------------------


def split_speech_segments(duration: float, silences: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []
    silences = sorted(silences)
    segs: List[Tuple[float, float]] = []
    cur = 0.0
    for s, e in silences:
        if s > cur:
            segs.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        segs.append((cur, duration))
    return [(s, e) for s, e in segs if e - s >= 0.15]


# ---------------------------------------------------------------------------
# ASR chunk-range helpers
# ---------------------------------------------------------------------------


def asr_chunk_ranges_by_speech(
    duration: float,
    silences: List[Tuple[float, float]],
    target_sec: float = 30.0,
    min_sec: float = 20.0,
    max_sec: float = 35.0,
    overlap_sec: float = 1.5,
) -> List[Tuple[float, float]]:
    if duration <= 0:
        return [(0.0, 0.0)]
    # Prefer cut points at silence starts (speech paragraph boundaries).
    pause_points = sorted(
        [s for s, e in silences if (e - s) >= 0.2 and 0.0 < s < duration]
    )
    out: List[Tuple[float, float]] = []
    cur = 0.0
    guard = 0
    while cur < duration and guard < 10000:
        guard += 1
        target = min(duration, cur + target_sec)
        lo = min(duration, cur + min_sec)
        hi = min(duration, cur + max_sec)
        if hi <= cur + 0.05:
            break

        cands = [p for p in pause_points if lo <= p <= hi]
        if cands:
            cut = min(cands, key=lambda p: abs(p - target))
        else:
            cut = hi

        if cut <= cur + 0.05:
            cut = min(duration, cur + target_sec)
        out.append((round(cur, 3), round(cut, 3)))
        if cut >= duration:
            break
        nxt = max(0.0, cut - overlap_sec)
        if nxt <= cur + 0.01:
            nxt = cut
        cur = nxt
    if not out:
        out = [(0.0, round(duration, 3))]
    return out


def split_range_max(start_sec: float, end_sec: float, max_sec: float) -> List[Tuple[float, float]]:
    if end_sec <= start_sec:
        return []
    if max_sec <= 0:
        return [(start_sec, end_sec)]
    out: List[Tuple[float, float]] = []
    cur = start_sec
    while cur < end_sec:
        nxt = min(end_sec, cur + max_sec)
        out.append((round(cur, 3), round(nxt, 3)))
        if nxt >= end_sec:
            break
        cur = nxt
    return out


# ---------------------------------------------------------------------------
# Word-token sanity check
# ---------------------------------------------------------------------------


def _is_reasonable_word_token(w: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        st = float(w.get("start", 0.0))
        et = float(w.get("end", st))
    except Exception:
        return None
    tok = str(w.get("word", w.get("text", ""))).strip()
    if not tok:
        return None
    dur = et - st
    if dur <= 0.0:
        return None
    return st, dur


# ---------------------------------------------------------------------------
# Speech-onset inference
# ---------------------------------------------------------------------------


def infer_speech_onset_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    words = payload.get("words")
    if isinstance(words, list) and words:
        norm: List[Tuple[float, float]] = []
        for w in words:
            if not isinstance(w, dict):
                continue
            x = _is_reasonable_word_token(w)
            if x is None:
                continue
            norm.append(x)
        norm.sort(key=lambda x: x[0])
        if len(norm) >= 4:
            # Find first stable run of short/normal-duration words.
            for i in range(0, len(norm) - 3):
                run = norm[i : i + 4]
                durs = [d for _, d in run]
                starts = [s for s, _ in run]
                if any(d > 1.2 for d in durs):
                    continue
                if any((starts[j + 1] - starts[j]) > 1.5 for j in range(3)):
                    continue
                return float(starts[0])
        if norm:
            for st, dur in norm:
                if dur <= 1.2:
                    return float(st)
    segs = payload.get("segments")
    if isinstance(segs, list) and segs:
        cand: List[float] = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            try:
                st = float(s.get("start", 0.0))
                et = float(s.get("end", st))
            except Exception:
                continue
            if et - st < 0.2:
                continue
            cand.append(st)
        if cand:
            cand.sort()
            return float(cand[0])
    return None


# ---------------------------------------------------------------------------
# Automatic speech-onset detection
# ---------------------------------------------------------------------------


def detect_speech_onset_auto(
    audio_path: Path,
    duration: float,
    silences: List[Tuple[float, float]],
    url: str,
    key: str,
    model: str,
    language: str,
) -> Tuple[float, str]:
    if duration <= 0:
        return 0.0, "onset=0.000;reason=no_duration"

    # Heuristic-A: leading silence from ffmpeg silencedetect.
    silence_onset = 0.0
    if silences:
        for s, e in sorted(silences):
            if s <= 0.08 and e > s and (e - s) >= 0.5:
                silence_onset = float(e)
                break

    # Heuristic-B: short ASR probe on the first ~35s.
    probe_len = min(duration, float(os.getenv("ASR_SPEECH_ONSET_PROBE_SEC", "35").strip() or "35"))
    asr_onset: Optional[float] = None
    probe_diag = "probe=skip"
    if probe_len >= 8.0:
        raw, st = extract_audio_chunk_bytes(audio_path, 0.0, probe_len)
        if raw is not None:
            payload, ast = asr_request_verbose(
                url=url,
                key=key,
                model=model,
                file_name=f"{audio_path.stem}_onset_probe.mp3",
                file_bytes=raw,
                language=language,
            )
            asr_onset = infer_speech_onset_from_payload(payload)
            probe_diag = f"probe={st};{ast};asr_onset={'none' if asr_onset is None else f'{asr_onset:.3f}'}"
        else:
            probe_diag = f"probe={st}"

    onset = silence_onset
    reason = "silence"
    if asr_onset is not None and asr_onset >= silence_onset + 1.0:
        onset = asr_onset
        reason = "asr_probe"

    max_onset = float(os.getenv("ASR_SPEECH_ONSET_MAX_SEC", "20").strip() or "20")
    if max_onset < 0:
        max_onset = 0.0
    onset = max(0.0, min(onset, max_onset, max(0.0, duration - 1.0)))
    return round(onset, 3), f"onset={onset:.3f};reason={reason};silence_onset={silence_onset:.3f};{probe_diag}"


# ---------------------------------------------------------------------------
# Audio chunk extraction (ffmpeg)
# ---------------------------------------------------------------------------


def extract_audio_chunk_bytes(audio_path: Path, start_sec: float, duration_sec: float) -> Tuple[Optional[bytes], str]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(max(0.0, start_sec)),
        "-t",
        str(max(0.01, duration_sec)),
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "mp3",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception as e:
        return None, f"chunk_ffmpeg_exception:{e}"
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="ignore").replace("\n", " ").strip()[:240]
        return None, f"chunk_ffmpeg_error:{err}"
    if not proc.stdout:
        return None, "chunk_ffmpeg_empty"
    return proc.stdout, "ok"


# ---------------------------------------------------------------------------
# ASR payload time-shifting
# ---------------------------------------------------------------------------


def shift_asr_payload_time(payload: Dict[str, Any], offset: float) -> Dict[str, Any]:
    out = dict(payload)
    segs = []
    for s in payload.get("segments") or []:
        if not isinstance(s, dict):
            continue
        x = dict(s)
        try:
            x["start"] = round(float(x.get("start", 0.0)) + offset, 3)
            x["end"] = round(float(x.get("end", 0.0)) + offset, 3)
        except Exception:
            # Keep raw segment as-is if time cast fails, for downstream diagnostics.
            pass
        segs.append(x)
    words = []
    for w in payload.get("words") or []:
        if not isinstance(w, dict):
            continue
        x = dict(w)
        try:
            x["start"] = round(float(x.get("start", 0.0)) + offset, 3)
            x["end"] = round(float(x.get("end", 0.0)) + offset, 3)
        except Exception:
            pass
        words.append(x)
    out["segments"] = segs
    out["words"] = words
    return out


# ---------------------------------------------------------------------------
# Text normalisation (shared with transcript)
# ---------------------------------------------------------------------------


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ---------------------------------------------------------------------------
# ASR payload merging (de-overlap)
# ---------------------------------------------------------------------------


def merge_asr_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged_text = " ".join(str(p.get("text", "")).strip() for p in payloads if str(p.get("text", "")).strip()).strip()
    raw_segments: List[Dict[str, Any]] = []
    raw_words: List[Dict[str, Any]] = []
    seg_id = 0
    for p in payloads:
        for s in p.get("segments") or []:
            if not isinstance(s, dict):
                continue
            x = dict(s)
            x["id"] = seg_id
            seg_id += 1
            raw_segments.append(x)
        for w in p.get("words") or []:
            if isinstance(w, dict):
                raw_words.append(dict(w))

    # Dedupe overlapped segment duplicates caused by chunk overlap.
    raw_segments.sort(key=lambda x: (_safe_float(x.get("start", 0.0), 0.0), _safe_float(x.get("end", 0.0), 0.0)))
    merged_segments: List[Dict[str, Any]] = []
    for s in raw_segments:
        if not merged_segments:
            merged_segments.append(s)
            continue
        prev = merged_segments[-1]
        s0, s1 = _safe_float(s.get("start", 0.0), 0.0), _safe_float(s.get("end", 0.0), 0.0)
        p0, p1 = _safe_float(prev.get("start", 0.0), 0.0), _safe_float(prev.get("end", 0.0), 0.0)
        t1 = _norm_text(str(s.get("text", "")))
        t0 = _norm_text(str(prev.get("text", "")))
        overlap = min(s1, p1) - max(s0, p0)
        if overlap > 0 and (t1 == t0 or (t1 and t1 in t0) or (t0 and t0 in t1)):
            # Keep the longer textual segment span.
            if len(t1) > len(t0):
                merged_segments[-1] = s
            continue
        merged_segments.append(s)

    # Dedupe words by rounded time + token.
    seen_words = set()
    merged_words: List[Dict[str, Any]] = []
    for w in sorted(raw_words, key=lambda x: (_safe_float(x.get("start", 0.0), 0.0), _safe_float(x.get("end", 0.0), 0.0))):
        key = (
            round(_safe_float(w.get("start", 0.0), 0.0), 2),
            round(_safe_float(w.get("end", 0.0), 0.0), 2),
            _norm_text(str(w.get("word", w.get("text", "")))),
        )
        if key in seen_words:
            continue
        seen_words.add(key)
        merged_words.append(w)
    return {
        "text": merged_text,
        "segments": merged_segments,
        "words": merged_words,
    }


# ---------------------------------------------------------------------------
# ASR HTTP request helpers
# ---------------------------------------------------------------------------


def _asr_request_verbose_single(
    url: str,
    key: str,
    model: str,
    file_name: str,
    file_bytes: bytes,
    language: str,
    granularity: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    files = {"file": (file_name, io.BytesIO(file_bytes), "audio/mpeg")}
    # Use tuple-list form fields so multipart repeats `timestamp_granularities[]`
    # exactly like curl -F.
    data: List[Tuple[str, str]] = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", granularity),
    ]
    if language:
        data.append(("language", language))
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {key}"}, files=files, data=data, timeout=300)
    except Exception as e:
        return None, f"http_exception:{e}"

    if resp.status_code >= 300:
        body = (resp.text or "").replace("\n", " ").strip()[:240]
        return None, f"http_{resp.status_code}:{body}"
    try:
        payload = resp.json()
    except Exception:
        raw = (resp.text or "").replace("\n", " ").strip()[:240]
        return None, f"non_json_response:{raw}"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    if not (payload.get("segments") or payload.get("words") or str(payload.get("text", "")).strip()):
        return None, "empty_asr_payload"
    return payload, "ok"


def asr_request_verbose(
    url: str,
    key: str,
    model: str,
    file_name: str,
    file_bytes: bytes,
    language: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    # Memory-constrained servers can fail when asking segment+word together.
    # Request twice and merge: once for segment timestamps, once for word timestamps.
    seg_payload, seg_status = _asr_request_verbose_single(
        url=url,
        key=key,
        model=model,
        file_name=file_name,
        file_bytes=file_bytes,
        language=language,
        granularity="segment",
    )
    word_payload, word_status = _asr_request_verbose_single(
        url=url,
        key=key,
        model=model,
        file_name=file_name,
        file_bytes=file_bytes,
        language=language,
        granularity="word",
    )

    if seg_payload is None and word_payload is None:
        return None, f"seg={seg_status};word={word_status}"

    merged: Dict[str, Any] = {}
    base = seg_payload if seg_payload is not None else word_payload
    if base is None:
        return None, f"seg={seg_status};word={word_status}"
    merged.update(base)
    merged["segments"] = (seg_payload or {}).get("segments", []) if isinstance((seg_payload or {}).get("segments", []), list) else []
    merged["words"] = (word_payload or {}).get("words", []) if isinstance((word_payload or {}).get("words", []), list) else []
    if not str(merged.get("text", "")).strip():
        merged["text"] = str((seg_payload or {}).get("text", "") or (word_payload or {}).get("text", "")).strip()
    return merged, f"seg={seg_status};word={word_status}"


# ---------------------------------------------------------------------------
# Normalise z-score helper (local to audio)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pitch-slope extraction via librosa
# ---------------------------------------------------------------------------


def detect_pitch_slope_librosa(audio_path: Path, utterances: List[Dict[str, Any]]) -> List[float]:
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
    except Exception:
        return [0.0 for _ in utterances]

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        f0 = librosa.yin(y, fmin=60, fmax=450, sr=sr, frame_length=2048, hop_length=256)
        hop_sec = 256 / sr
        out = []
        for u in utterances:
            s = u["start_time"]
            e = u["end_time"]
            a = int(max(0, s / hop_sec))
            b = int(min(len(f0), e / hop_sec))
            seq = f0[a:b]
            seq = seq[np.isfinite(seq)]
            if len(seq) < 6:
                out.append(0.0)
                continue
            start_mean = float(np.mean(seq[: max(1, len(seq) // 3)]))
            end_mean = float(np.mean(seq[-max(1, len(seq) // 3):]))
            out.append(end_mean - start_mean)
        return out
    except Exception:
        return [0.0 for _ in utterances]


# ---------------------------------------------------------------------------
# Main ASR entry point
# ---------------------------------------------------------------------------


def maybe_transcribe_openai(
    audio_path: Path,
    duration: Optional[float] = None,
    silences: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    # ASR endpoint/credentials are independent from other LLM services.
    base = os.getenv("ASR_BASE_URL", "").strip().rstrip("/")
    key = os.getenv("ASR_API_KEY", "").strip()
    model = os.getenv("ASR_MODEL", "").strip()
    if requests is None:
        return None, "requests_unavailable"
    if not base:
        return None, "missing_env:ASR_BASE_URL"
    if not key:
        return None, "missing_env:ASR_API_KEY"
    if not model:
        return None, "missing_env:ASR_MODEL"

    transcribe_path = os.getenv("ASR_TRANSCRIBE_PATH", "/audio/transcriptions").strip() or "/audio/transcriptions"
    language = os.getenv("ASR_LANGUAGE", "").strip()
    word_max_chunk_sec = float(os.getenv("ASR_WORD_MAX_CHUNK_SEC", "30").strip() or "30")
    if word_max_chunk_sec < 5:
        word_max_chunk_sec = 5.0
    if not transcribe_path.startswith("/"):
        transcribe_path = "/" + transcribe_path
    url = f"{base}{transcribe_path}"
    if duration is None:
        duration = ffprobe_duration(audio_path) or 0.0
    if silences is None:
        silences = detect_silences(audio_path)

    onset_mode = os.getenv("ASR_SPEECH_ONSET_MODE", "AUTO").strip().upper()
    onset_sec = 0.0
    onset_diag = "onset=0.000;reason=off"
    if duration > 0:
        if onset_mode == "AUTO":
            onset_sec, onset_diag = detect_speech_onset_auto(
                audio_path=audio_path,
                duration=duration,
                silences=silences,
                url=url,
                key=key,
                model=model,
                language=language,
            )
        elif onset_mode == "OFF":
            onset_sec = 0.0
            onset_diag = "onset=0.000;reason=off"
        else:
            forced_str = onset_mode
            if onset_mode.startswith("FORCE_SEC:"):
                forced_str = onset_mode.split(":", 1)[1].strip()
            try:
                onset_sec = max(0.0, float(forced_str))
            except Exception:
                onset_sec = 0.0
            onset_sec = round(min(onset_sec, max(0.0, duration - 1.0)), 3)
            onset_diag = f"onset={onset_sec:.3f};reason=forced"

    effective_duration = max(0.0, duration - onset_sec)
    if onset_sec > 0:
        shifted_silences = [(max(0.0, s - onset_sec), max(0.0, e - onset_sec)) for s, e in silences if e > onset_sec]
    else:
        shifted_silences = silences
    ranges_rel = asr_chunk_ranges_by_speech(effective_duration, shifted_silences, target_sec=30.0, min_sec=20.0, max_sec=35.0, overlap_sec=1.5)
    ranges = [(round(s + onset_sec, 3), round(e + onset_sec, 3)) for s, e in ranges_rel]
    payloads: List[Dict[str, Any]] = []
    errors: List[str] = []
    diag: List[str] = []

    # unknown duration: single full-file attempt in verbose mode.
    if duration <= 0:
        try:
            with open(audio_path, "rb") as f:
                raw = f.read()
        except Exception as e:
            return None, f"read_audio_exception:{e}"
        payload, st = asr_request_verbose(url, key, model, audio_path.name, raw, language)
        if payload is None:
            return None, st
        return payload, f"ok_verbose_single;{onset_diag}"

    for i, (s, e) in enumerate(ranges):
        sub_ranges = split_range_max(s, e, max_sec=word_max_chunk_sec)
        if not sub_ranges:
            continue
        for j, (ss, ee) in enumerate(sub_ranges):
            chunk_bytes, cst = extract_audio_chunk_bytes(audio_path, ss, ee - ss)
            if chunk_bytes is None:
                return None, f"chunk_{i}.{j}:{cst}"
            payload, st = asr_request_verbose(
                url,
                key,
                model,
                f"{audio_path.stem}_chunk_{i:04d}_{j:02d}.mp3",
                chunk_bytes,
                language,
            )
            if payload is None:
                errors.append(f"chunk_{i}.{j}@{ss:.2f}-{ee:.2f}:{st}")
                return None, "|".join(errors)
            segs = payload.get("segments")
            words = payload.get("words")
            seg_len = len(segs) if isinstance(segs, list) else -1
            word_len = len(words) if isinstance(words, list) else -1
            text_ok = bool(str(payload.get("text", "")).strip())
            diag.append(f"{i}.{j}:seg={seg_len},word={word_len},text={int(text_ok)}")
            payloads.append(shift_asr_payload_time(payload, ss))

    merged = merge_asr_payloads(payloads)
    mseg = merged.get("segments")
    mword = merged.get("words")
    mseg_len = len(mseg) if isinstance(mseg, list) else -1
    mword_len = len(mword) if isinstance(mword, list) else -1
    return merged, (
        f"ok_verbose_chunked:{len(ranges)} segments_from_{len(payloads)} requests;"
        f"merged_seg={mseg_len};merged_word={mword_len};{onset_diag};diag={','.join(diag[:20])}"
    )
