"""LLM client module for the SynapseHSG pipeline.

Centralises all rate-limiting, executor management, API-call helpers,
and JSON-parsing utilities used by the pipeline's LLM interactions.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover
    repair_json = None  # type: ignore[assignment]

from .config import env_int


# ---------------------------------------------------------------------------
# Sanitisation helper (shared by LLM and I/O paths)
# ---------------------------------------------------------------------------


def sanitize_json_obj(obj: Any) -> Any:
    """Recursively normalise strings in *obj* to valid UTF-8."""
    if isinstance(obj, str):
        # pypdf can emit unpaired surrogates; normalise to valid UTF-8.
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, list):
        return [sanitize_json_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# GlobalRateLimiter
# ---------------------------------------------------------------------------


class GlobalRateLimiter:
    """Token-bucket-style rate limiter that serialises ``acquire`` calls."""

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
# Global instances
# ---------------------------------------------------------------------------

GLOBAL_LLM_RATE_LIMITER = GlobalRateLimiter()
GLOBAL_TRIPLES_BUILD_RATE_LIMITER = GlobalRateLimiter()
GLOBAL_VISUAL_LLM_RATE_LIMITER = GlobalRateLimiter()
JSONL_APPEND_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Global executors (lazily initialised, per-service concurrency control)
# ---------------------------------------------------------------------------

GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
GLOBAL_VISUAL_LLM_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
GLOBAL_ASR_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None

# Locks for executor initialisation
TRIPLES_BUILD_LLM_LOCK = threading.Lock()
VISUAL_LLM_LOCK = threading.Lock()
ASR_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Executor management
# ---------------------------------------------------------------------------


def get_triples_build_llm_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the global TRIPLES_BUILD LLM executor with configured concurrency."""
    global GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR
    if GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR is not None:
        return GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR
    with TRIPLES_BUILD_LLM_LOCK:
        if GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR is None:
            concurrency = env_int(
                "TRIPLES_BUILD_CONCURRENCY",
                default=10,
                min_value=1,
                max_value=128,
            )
            GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="triples_build_llm",
            )
    return GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR


def get_visual_llm_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the global VISUAL LLM executor with configured concurrency."""
    global GLOBAL_VISUAL_LLM_EXECUTOR
    if GLOBAL_VISUAL_LLM_EXECUTOR is not None:
        return GLOBAL_VISUAL_LLM_EXECUTOR
    with VISUAL_LLM_LOCK:
        if GLOBAL_VISUAL_LLM_EXECUTOR is None:
            concurrency = env_int(
                "VISION_LLM_CONCURRENCY",
                default=50,
                min_value=1,
                max_value=128,
            )
            GLOBAL_VISUAL_LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="visual_llm",
            )
    return GLOBAL_VISUAL_LLM_EXECUTOR


def get_asr_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the global ASR executor with configured concurrency."""
    global GLOBAL_ASR_EXECUTOR
    if GLOBAL_ASR_EXECUTOR is not None:
        return GLOBAL_ASR_EXECUTOR
    with ASR_LOCK:
        if GLOBAL_ASR_EXECUTOR is None:
            concurrency = env_int(
                "ASR_CONCURRENCY",
                default=5,
                min_value=1,
                max_value=32,
            )
            GLOBAL_ASR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="asr",
            )
    return GLOBAL_ASR_EXECUTOR


def shutdown_all_llm_executors() -> None:
    """Shutdown all global LLM executors (call on exit)."""
    global GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR, GLOBAL_VISUAL_LLM_EXECUTOR, GLOBAL_ASR_EXECUTOR
    if GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR is not None:
        GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR.shutdown(wait=True)
        GLOBAL_TRIPLES_BUILD_LLM_EXECUTOR = None
    if GLOBAL_VISUAL_LLM_EXECUTOR is not None:
        GLOBAL_VISUAL_LLM_EXECUTOR.shutdown(wait=True)
        GLOBAL_VISUAL_LLM_EXECUTOR = None
    if GLOBAL_ASR_EXECUTOR is not None:
        GLOBAL_ASR_EXECUTOR.shutdown(wait=True)
        GLOBAL_ASR_EXECUTOR = None


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def retry_sleep_ms(attempt: int, base_ms: int, max_ms: int, jitter_ms: int) -> int:
    """Exponential back-off with jitter, returning milliseconds to sleep."""
    exp = base_ms * (2 ** max(0, attempt))
    capped = min(max_ms, exp)
    jitter = random.randint(0, max(0, jitter_ms))
    return max(0, capped + jitter)


def is_retryable_chat_failure(status: str, body: str = "") -> bool:
    """Return ``True`` if the LLM error *status* is transient and worth retrying."""
    s = (status or "").lower()
    b = (body or "").lower()
    if s.startswith("http_429"):
        return True
    m = re.match(r"http_(\d+)", s)
    if m:
        try:
            code = int(m.group(1))
        except Exception:
            code = 0
        if code >= 500:
            return True
        if code in {400, 401, 403, 404}:
            return False
    if "http_exception" in s:
        return True
    if "empty_content" in s:
        return True
    if "do_request_failed" in s or "do_request_failed" in b:
        return True
    return False


# ---------------------------------------------------------------------------
# Preflight & endpoint resolution
# ---------------------------------------------------------------------------


def llm_preflight_status() -> Tuple[bool, str]:
    """Check that all required TRIPLES_BUILD env vars and ``requests`` are available."""
    missing: list[str] = []
    if requests is None:
        missing.append("requests_unavailable")
    if not os.getenv("TRIPLES_BUILD_BASE_URL", "").strip():
        missing.append("TRIPLES_BUILD_BASE_URL")
    if not os.getenv("TRIPLES_BUILD_API_KEY", "").strip():
        missing.append("TRIPLES_BUILD_API_KEY")
    if not os.getenv("TRIPLES_BUILD_MODEL", "").strip():
        missing.append("TRIPLES_BUILD_MODEL")
    if missing:
        return False, "missing_or_unavailable:" + ",".join(missing)
    return True, "ok"


def resolve_chat_endpoint_env(target: str = "TRIPLES_BUILD") -> Tuple[str, str, str, str]:
    """Resolve ``(base_url, api_key, model, resolved_target)`` from env vars for *target*."""
    t = (target or "TRIPLES_BUILD").strip().upper()
    if t == "TRIPLES_BUILD":
        base = os.getenv("TRIPLES_BUILD_BASE_URL", "").strip().rstrip("/")
        key = os.getenv("TRIPLES_BUILD_API_KEY", "").strip()
        model = os.getenv("TRIPLES_BUILD_MODEL", "").strip()
        return base, key, model, "TRIPLES_BUILD"
    base = os.getenv(f"{t}_BASE_URL", "").strip().rstrip("/")
    key = os.getenv(f"{t}_API_KEY", "").strip()
    model = os.getenv(f"{t}_MODEL", "").strip()
    return base, key, model, t


# ---------------------------------------------------------------------------
# Core chat completions poster
# ---------------------------------------------------------------------------


def _post_chat_completions(
    base: str,
    key: str,
    payload: Dict[str, Any],
    queue_scope: str = "TRIPLES_BUILD",
) -> Tuple[Optional[Dict[str, Any]], str, int]:
    """POST to ``/chat/completions`` with rate-limiting and retries.

    Returns ``(data_dict | None, status_string, retries_done)``.
    """
    if queue_scope == "VISION":
        llm_min_interval_ms = env_int(
            "VISION_LLM_MIN_INTERVAL_MS",
            default=300,
            min_value=0,
            max_value=600000,
        )
        rate_limiter = GLOBAL_VISUAL_LLM_RATE_LIMITER
    elif queue_scope == "TRIPLES_BUILD":
        llm_min_interval_ms = env_int(
            "TRIPLES_BUILD_MIN_INTERVAL_MS",
            default=300,
            min_value=0,
            max_value=600000,
        )
        rate_limiter = GLOBAL_TRIPLES_BUILD_RATE_LIMITER
    else:
        llm_min_interval_ms = env_int("LLM_MIN_INTERVAL_MS", default=300, min_value=0, max_value=600000)
        rate_limiter = GLOBAL_LLM_RATE_LIMITER
    max_retries = env_int("LLM_MAX_RETRIES", default=3, min_value=0, max_value=20)
    retry_base_ms = env_int("LLM_RETRY_BASE_MS", default=800, min_value=1, max_value=600000)
    retry_max_ms = env_int("LLM_RETRY_MAX_MS", default=8000, min_value=1, max_value=600000)
    retry_jitter_ms = env_int("LLM_RETRY_JITTER_MS", default=300, min_value=0, max_value=600000)
    retries_done = 0

    for attempt in range(max_retries + 1):
        status = ""
        body = ""
        try:
            rate_limiter.acquire(llm_min_interval_ms)
            resp = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=300,
            )
            if resp.status_code >= 300:
                body = (resp.text or "").replace("\n", " ").strip()[:240]
                status = f"http_{resp.status_code}:{body}"
                if attempt < max_retries and is_retryable_chat_failure(status, body):
                    retries_done += 1
                    time.sleep(retry_sleep_ms(attempt, retry_base_ms, retry_max_ms, retry_jitter_ms) / 1000.0)
                    continue
                return None, status, retries_done
            data = resp.json()
            if not isinstance(data, dict):
                status = "bad_response:not_object"
                return None, status, retries_done
            return data, "ok", retries_done
        except Exception as e:
            status = f"http_exception:{e}"
            if attempt < max_retries and is_retryable_chat_failure(status):
                retries_done += 1
                time.sleep(retry_sleep_ms(attempt, retry_base_ms, retry_max_ms, retry_jitter_ms) / 1000.0)
                continue
            return None, status, retries_done
    return None, "unknown_error", retries_done


# ---------------------------------------------------------------------------
# Context-window guard
# ---------------------------------------------------------------------------


def _derive_context_retry_max_tokens(status: str, requested_max_tokens: int) -> Optional[int]:
    """Try to derive a smaller ``max_tokens`` from a context-length error."""
    if requested_max_tokens <= 0:
        return None
    s = str(status or "")
    m_in = re.search(r"input length\s*(?:is|=)?\s*(\d+)", s, flags=re.IGNORECASE)
    if not m_in:
        m_in = re.search(r"messages?\s+(?:resulted in|have)\s*(\d+)\s*tokens?", s, flags=re.IGNORECASE)
    if not m_in:
        return None
    try:
        input_tokens = int(m_in.group(1))
    except Exception:
        return None
    if input_tokens <= 0:
        return None
    ctx_limit = env_int("LLM_CONTEXT_MAX_TOKENS", default=32768, min_value=2048, max_value=1000000)
    m_ctx = re.search(r"maximum context length\s*(?:is|=)?\s*(\d+)", s, flags=re.IGNORECASE)
    if m_ctx:
        try:
            ctx_limit = max(2048, int(m_ctx.group(1)))
        except Exception:
            pass
    reserve = env_int("LLM_CONTEXT_RESERVE_TOKENS", default=512, min_value=0, max_value=65536)
    budget = ctx_limit - input_tokens - reserve
    retry_max_tokens = max(128, min(requested_max_tokens, budget))
    if retry_max_tokens >= requested_max_tokens:
        return None
    return retry_max_tokens


# ---------------------------------------------------------------------------
# High-level chat completions caller
# ---------------------------------------------------------------------------


def call_openai_multimodal(
    messages: List[Dict[str, Any]],
    max_tokens: int = 32768,
    temperature: float = 0.0,
    response_format: Optional[Dict[str, Any]] = None,
    env_target: str = "TRIPLES_BUILD",
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Call an OpenAI-compatible chat/completions endpoint.

    Returns ``(content_text | None, status_string, meta_dict)``.
    Includes automatic fallback for gateways that reject ``response_format``,
    context-window auto-shrink, and an empty-content retry.
    """
    base, key, model, resolved_target = resolve_chat_endpoint_env(env_target)
    if requests is None:
        return None, "requests_unavailable", {}
    if not base:
        return None, f"missing_env:{resolved_target}_BASE_URL", {}
    if not key:
        return None, f"missing_env:{resolved_target}_API_KEY", {}
    if not model:
        return None, f"missing_env:{resolved_target}_MODEL", {}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "metadata": {"service": "synapsehsg"},
    }
    if response_format:
        payload["response_format"] = response_format
    data, status, retries_1 = _post_chat_completions(base, key, payload, queue_scope=resolved_target)
    total_retries = retries_1
    if data is None and response_format:
        # Some OpenAI-compatible gateways do not support response_format.
        fallback_payload = dict(payload)
        fallback_payload.pop("response_format", None)
        data, st2, retries_2 = _post_chat_completions(base, key, fallback_payload, queue_scope=resolved_target)
        total_retries += retries_2
        if data is None:
            # Context-window guard: auto-shrink max_tokens when gateway reports input too long.
            retry_max_tokens = _derive_context_retry_max_tokens(status, max_tokens)
            if retry_max_tokens is None:
                return None, status, {"retry_count": total_retries}
            retry_payload = dict(payload)
            retry_payload["max_tokens"] = retry_max_tokens
            data, st3, retries_3 = _post_chat_completions(base, key, retry_payload, queue_scope=resolved_target)
            total_retries += retries_3
            if data is None and response_format:
                retry_fallback = dict(retry_payload)
                retry_fallback.pop("response_format", None)
                data, st4, retries_4 = _post_chat_completions(base, key, retry_fallback, queue_scope=resolved_target)
                total_retries += retries_4
                if data is None:
                    return None, f"{status};ctx_retry_failed:{st4}", {"retry_count": total_retries}
                status = f"{status};ctx_retry_no_response_format:{st4};max_tokens={retry_max_tokens}"
            elif data is None:
                return None, f"{status};ctx_retry_failed:{st3}", {"retry_count": total_retries}
            else:
                status = f"{status};ctx_retry:{st3};max_tokens={retry_max_tokens}"
        status = f"fallback_no_response_format:{st2}"
    elif data is None:
        retry_max_tokens = _derive_context_retry_max_tokens(status, max_tokens)
        if retry_max_tokens is None:
            return None, status, {"retry_count": total_retries}
        retry_payload = dict(payload)
        retry_payload["max_tokens"] = retry_max_tokens
        data, st3, retries_3 = _post_chat_completions(base, key, retry_payload, queue_scope=resolved_target)
        total_retries += retries_3
        if data is None:
            return None, f"{status};ctx_retry_failed:{st3}", {"retry_count": total_retries}
        status = f"{status};ctx_retry:{st3};max_tokens={retry_max_tokens}"

    try:
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return None, "bad_response:choices_missing", {"retry_count": total_retries}
        finish_reason = str(choices[0].get("finish_reason", "") or "")
        usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            # Some OpenAI-compatible endpoints return structured content parts.
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            content = "".join(text_parts)
        if not isinstance(content, str):
            content = str(content or "")
        content = content.strip()
        if not content:
            # Retry once more round-trip for empty content on retryable gateways.
            empty_retry = env_int("LLM_MAX_RETRIES", default=3, min_value=0, max_value=20)
            if empty_retry > 0:
                data2, status2, retries_3 = _post_chat_completions(base, key, payload, queue_scope=resolved_target)
                total_retries += retries_3
                if data2 is not None:
                    choices2 = data2.get("choices", [])
                    if isinstance(choices2, list) and choices2:
                        finish_reason = str(choices2[0].get("finish_reason", "") or "")
                        usage = data2.get("usage", {}) if isinstance(data2.get("usage", {}), dict) else {}
                        message = choices2[0].get("message", {})
                        content = message.get("content")
                        if isinstance(content, list):
                            text_parts2: list[str] = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts2.append(str(part.get("text", "")))
                            content = "".join(text_parts2)
                        if not isinstance(content, str):
                            content = str(content or "")
                        content = content.strip()
                else:
                    status = f"{status};empty_content_retry_failed:{status2}"
            if not content:
                return None, "empty_content", {"finish_reason": finish_reason, "usage": usage, "retry_count": total_retries}
        if finish_reason == "length":
            status = f"{status};truncated"
        return content, status, {"finish_reason": finish_reason, "usage": usage, "retry_count": total_retries}
    except Exception as e:
        return None, f"parse_response_exception:{e}", {"retry_count": total_retries}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embedding_env() -> Tuple[str, str, str]:
    """Return ``(base_url, api_key, model)`` for the embedding endpoint."""
    base = os.getenv("EMBED_BASE_URL", "").strip()
    key = os.getenv("EMBED_API_KEY", "").strip()
    model = os.getenv("EMBED_MODEL", "").strip()
    return base.rstrip("/"), key, model


def call_openai_embeddings(
    texts: List[str],
    base: str,
    key: str,
    model: str,
) -> Tuple[Optional[List[List[float]]], str]:
    """Call an OpenAI-compatible ``/embeddings`` endpoint.

    Tries batch first, then falls back to per-text requests with ``input``
    as either a bare string or a single-element list.
    """
    if requests is None:
        # Keep going: embeddings path can use urllib only.
        pass
    if not base:
        return None, "missing_env:EMBED_BASE_URL"
    if not model:
        return None, "missing_env:EMBED_MODEL"
    clean_texts = [str(t).strip() for t in texts]
    if not clean_texts:
        return None, "empty_input"
    clean_texts = [t if t else " " for t in clean_texts]

    # -- inner helpers -------------------------------------------------------

    def _post(payload: Dict[str, Any], with_auth: bool = True) -> Tuple[Optional[Dict[str, Any]], str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if with_auth and key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            safe_payload = sanitize_json_obj(payload)
            body = json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url=f"{base}/embeddings",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw_bytes = resp.read()
                raw = raw_bytes.decode("utf-8", "ignore")
                code = int(getattr(resp, "status", 200) or 200)
        except urllib.error.HTTPError as e:
            err_raw = e.read().decode("utf-8", "ignore")
            body_preview = (err_raw or "").replace("\n", " ").strip()[:400]
            return None, f"http_{e.code}:{body_preview}"
        except Exception as e:
            return None, f"http_exception:{e}"
        if code >= 300:
            body_preview = (raw or "").replace("\n", " ").strip()[:400]
            return None, f"http_{code}:{body_preview}"
        try:
            data = json.loads(raw)
        except Exception:
            raw_preview = (raw or "").replace("\n", " ").strip()[:240]
            return None, f"non_json_response:{raw_preview}"
        if not isinstance(data, dict):
            return None, "bad_response:not_object"
        return data, "ok"

    def _post_try_auth_then_noauth(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        data, st = _post(payload, with_auth=True)
        if data is not None:
            return data, st if key else "ok_noauth"
        if key:
            data2, st2 = _post(payload, with_auth=False)
            if data2 is not None:
                return data2, "ok_noauth_fallback"
            return None, f"{st};noauth={st2}"
        return None, st

    def _parse(data: Dict[str, Any], expected_n: int) -> Tuple[Optional[List[List[float]]], str]:
        arr = data.get("data")
        if not isinstance(arr, list) or not arr:
            return None, "bad_response:data_missing"
        arr_sorted = sorted([x for x in arr if isinstance(x, dict)], key=lambda x: int(x.get("index", 0)))
        vecs: List[List[float]] = []
        for item in arr_sorted:
            emb = item.get("embedding")
            if not isinstance(emb, list) or not emb:
                return None, "bad_response:embedding_missing"
            try:
                vecs.append([float(v) for v in emb])
            except Exception:
                return None, "bad_response:embedding_non_numeric"
        if len(vecs) != expected_n:
            return None, f"bad_response:embedding_count_mismatch(expected={expected_n},got={len(vecs)})"
        return vecs, "ok"

    # -- primary path: OpenAI-compatible batch request ------------------------

    data, st = _post_try_auth_then_noauth({"model": model, "input": clean_texts})
    if data is not None:
        vecs, pst = _parse(data, len(clean_texts))
        if vecs is not None:
            return vecs, "ok"
        st = f"{st};{pst}"

    # -- fallback: per-text requests -----------------------------------------

    fallback_vecs: List[List[float]] = []
    last_err = st
    for t in clean_texts:
        d1, s1 = _post_try_auth_then_noauth({"model": model, "input": t})
        if d1 is not None:
            v1, p1 = _parse(d1, 1)
            if v1 is not None:
                fallback_vecs.append(v1[0])
                continue
            s1 = f"{s1};{p1}"

        d2, s2 = _post_try_auth_then_noauth({"model": model, "input": [t]})
        if d2 is not None:
            v2, p2 = _parse(d2, 1)
            if v2 is not None:
                fallback_vecs.append(v2[0])
                continue
            s2 = f"{s2};{p2}"

        last_err = f"single_str={s1}(len={len(t)});single_list={s2}(len={len(t)})"
        return None, f"embed_fallback_failed:{last_err}"

    return fallback_vecs, f"ok_fallback_single:{st}"


# ---------------------------------------------------------------------------
# LLM JSON parsing helpers
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences wrapping LLM output."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _try_raw_decode(text: str) -> List[Any]:
    """Attempt ``json.JSONDecoder.raw_decode`` at every ``[`` or ``{`` offset."""
    dec = json.JSONDecoder()
    out: list[Any] = []
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            out.append(obj)
        except Exception:
            continue
    return out


def parse_llm_json_array(text: str) -> Optional[List[Any]]:
    """Parse an LLM response that should contain a JSON array.

    Handles code fences, ``{"triples": [...]}`` wrappers, and ``json_repair``
    fallbacks.
    """
    if not text:
        return None
    text = _strip_code_fence(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("triples"), list):
            return obj.get("triples")
    except Exception:
        pass
    if repair_json is not None:
        try:
            repaired = repair_json(text)
            obj2 = json.loads(repaired)
            if isinstance(obj2, list):
                return obj2
            if isinstance(obj2, dict) and isinstance(obj2.get("triples"), list):
                return obj2.get("triples")
        except Exception:
            pass
    for obj in _try_raw_decode(text):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("triples"), list):
            return obj.get("triples")
    return None


def parse_llm_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse an LLM response that should contain a JSON object.

    Handles code fences and ``json_repair`` fallbacks.
    """
    if not text:
        return None
    text = _strip_code_fence(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    if repair_json is not None:
        try:
            repaired = repair_json(text)
            obj2 = json.loads(repaired)
            if isinstance(obj2, dict):
                return obj2
        except Exception:
            pass
    for obj in _try_raw_decode(text):
        if isinstance(obj, dict):
            return obj
    return None
