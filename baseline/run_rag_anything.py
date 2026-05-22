#!/usr/bin/env python3
"""Baseline: RAG-Anything (LightRAG-based multimodal RAG) evaluation.

Pipeline per session:
  1. Parse paper text chunks + slide images → RAG-Anything content list
  2. Insert into per-session LightRAG knowledge base
  3. For each question: aquery() → answer
  4. LLM-as-judge evaluation

Uses RAG-Anything's own retrieval (LightRAG graph search), NOT our HSG.
Generator: Qwen3.6-27B via vLLM API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
import uuid
from multiprocessing import Pool, cpu_count
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# LLM helpers (shared with other baselines)
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    if not text:
        return text
    idx = text.find("</think")
    if idx >= 0:
        rest = text[idx:]
        nl = rest.find("\n")
        if nl >= 0:
            return rest[nl + 1:].strip()
        return text[idx + len("</think"):].strip()
    return text


def call_llm(api_url, api_key, model, messages, max_tokens=8192,
             temperature=0.3, max_retries=5, base_timeout=300):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (1 + attempt)
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if not content:
                    continue
                return strip_thinking(content)
            print(f"  LLM error {resp.status_code}: {resp.text[:200]}")
            if resp.status_code < 500:
                break
        except requests.exceptions.Timeout:
            print(f"  LLM timeout (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"  LLM error: {e}")
    return None


def extract_binary(response):
    if not response:
        return None
    matches = re.findall(r"[10]", response)
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Session data loader (simplified, no HSG dependency)
# ---------------------------------------------------------------------------

class SessionLoader:
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.sessions_dir = os.path.join(data_root, "sessions")
        self._download_dirs: dict[str, str] = {}
        self._index_download_dirs()

    def _index_download_dirs(self):
        parent = os.path.dirname(self.data_root)
        for conf in os.listdir(parent):
            conf_path = os.path.join(parent, conf)
            if not os.path.isdir(conf_path) or conf.startswith("_"):
                continue
            for entry in os.listdir(conf_path):
                entry_path = os.path.join(conf_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                parts = entry.split("_", 1)
                if parts[0].isdigit():
                    sid = f"{conf}/{parts[0]}"
                    self._download_dirs[sid] = entry_path

    def get_session_path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id)

    def get_download_dir(self, session_id: str) -> str | None:
        return self._download_dirs.get(session_id)

    def load_paper_text(self, session_id: str) -> str:
        path = os.path.join(self.get_session_path(session_id), "paper_structured.json")
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("paper_chunks", [])
        return "\n\n".join(c.get("text", "") for c in chunks)

    def load_transcript(self, session_id: str) -> str:
        path = os.path.join(self.get_session_path(session_id), "transcript_enriched.json")
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        combined = data.get("combined_enriched_text", "")
        if combined:
            return combined
        utterances = data.get("utterances", [])
        return "\n".join(u.get("enriched_text", "") for u in utterances)

    def get_slide_image_paths(self, session_id: str) -> list[str]:
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return []
        slides_dir = os.path.join(dl_dir, "slides")
        if not os.path.isdir(slides_dir):
            return []
        paths = []
        for fname in sorted(os.listdir(slides_dir)):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(slides_dir, fname))
        return paths

    def get_paper_page_paths(self, session_id: str) -> list[str]:
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return []
        pages_dir = os.path.join(dl_dir, "paper_pages")
        if not os.path.isdir(pages_dir):
            return []
        paths = []
        for fname in sorted(os.listdir(pages_dir)):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(pages_dir, fname))
        return paths


# ---------------------------------------------------------------------------
# RAG-Anything session index builder (multiprocess parallel)
# ---------------------------------------------------------------------------

# Global config passed to worker processes
_WORKER_CONFIG = None


def _init_worker(config):
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _build_one_session(args):
    """Worker function: build index for a single session in its own process."""
    session_id, working_dir_base = args

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    cfg = _WORKER_CONFIG
    working_dir = os.path.join(working_dir_base, session_id.replace("/", "_"))

    # Skip if already built (resume support)
    vdb_file = os.path.join(working_dir, "vdb_chunks.json")
    if os.path.exists(vdb_file) and os.path.getsize(vdb_file) > 100:
        return (session_id, "cached", 0)

    # Build content list
    loader = SessionLoader(cfg["data_root"])
    content_list = []
    paper_text = loader.load_paper_text(session_id)
    if paper_text:
        chunk_size = 2000
        for i in range(0, len(paper_text), chunk_size):
            content_list.append(paper_text[i:i + chunk_size])
    transcript = loader.load_transcript(session_id)
    if transcript:
        content_list.append(f"=== Presentation Transcript ===\n{transcript}")
    if not content_list:
        return (session_id, "no_content", 0)

    full_text = "\n\n".join(content_list)
    os.makedirs(working_dir, exist_ok=True)

    # Each process runs its own async loop
    async def _async_build():
        api_url = cfg["api"].replace("/chat/completions", "")
        api_key = cfg["api_key"]
        model = cfg["model"]

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            from lightrag.llm.openai import openai_complete_if_cache
            return await openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=api_url,
                api_key=api_key,
                **kwargs,
            )

        async def embed_func(texts):
            embed_api = "http://localhost:8001/v1/embeddings"
            headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-000"}
            resp = requests.post(embed_api, headers=headers,
                                json={"model": "bge-m3", "input": texts}, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding error: {resp.text[:200]}")
            import numpy as np
            return np.array([d["embedding"] for d in resp.json()["data"]])

        embedding_func = EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed_func)

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            workspace=session_id.replace("/", "_"),
        )
        await rag.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status(workspace=session_id.replace("/", "_"))
        await rag.ainsert(full_text)
        return rag

    t0 = time.time()
    try:
        asyncio.run(_async_build())
        elapsed = time.time() - t0
        return (session_id, "ok", elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        return (session_id, f"error:{e}", elapsed)


def build_all_indexes_parallel(session_ids: list[str], working_dir_base: str,
                                cfg: dict, num_workers: int = 1):
    """Build indexes with resume support. Serial if num_workers=1, parallel if >1."""
    total = len(session_ids)
    built = cached = failed = 0

    # Always use Pool so each worker has its own event loop (avoids nested asyncio.run)
    actual_workers = max(1, min(num_workers, len(session_ids)))
    print(f"Building indexes for {total} sessions ({actual_workers} worker(s))...")
    with Pool(processes=actual_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, result in enumerate(pool.imap_unordered(
                _build_one_session, [(sid, working_dir_base) for sid in session_ids]), 1):
            sid, status, elapsed = result
            if status == "cached":
                cached += 1
                if cached <= 5 or cached % 20 == 0:
                    print(f"[{i}/{total}] {sid}: cached")
            elif status == "ok":
                built += 1
                print(f"[{i}/{total}] {sid}: done in {elapsed:.0f}s")
            elif status == "no_content":
                failed += 1
                print(f"[{i}/{total}] {sid}: no content")
            else:
                failed += 1
                print(f"[{i}/{total}] {sid}: FAILED {status}")

    print(f"\nIndex building complete: {built} built, {cached} cached, {failed} failed")


# ---------------------------------------------------------------------------
# RAG-Anything Baseline Evaluator
# ---------------------------------------------------------------------------

class RAGAnythingBaseline:
    def __init__(self, api_url, api_key, model, loader, working_dir_base):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.loader = loader
        self.working_dir_base = working_dir_base
        self._session_rags: dict[str, object] = {}

    def _make_rag_factory(self):
        """Create LLM and embedding functions for RAG-Anything."""
        from lightrag.utils import EmbeddingFunc

        api_url = self.api_url.replace("/chat/completions", "")
        api_key = self.api_key
        model = self.model

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            from lightrag.llm.openai import openai_complete_if_cache
            return await openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=api_url,
                api_key=api_key,
                **kwargs,
            )

        async def embed_func(texts):
            import numpy as np
            embed_api = "http://localhost:8001/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-000",
            }
            resp = requests.post(
                embed_api,
                headers=headers,
                json={"model": "bge-m3", "input": texts},
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding error: {resp.text[:200]}")
            data = resp.json()
            return np.array([d["embedding"] for d in data["data"]])

        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embed_func,
        )

        return {"llm_func": llm_func, "embedding_func": embedding_func}

    async def _get_or_build_index(self, session_id: str):
        if session_id in self._session_rags:
            return self._session_rags[session_id]

        factory = self._make_rag_factory()
        working_dir = os.path.join(self.working_dir_base, session_id.replace("/", "_"))
        workspace = session_id.replace("/", "_")

        from lightrag import LightRAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=factory["llm_func"],
            embedding_func=factory["embedding_func"],
            workspace=workspace,
        )
        await rag.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status(workspace=workspace)

        # If index doesn't exist yet, build it
        vdb_file = os.path.join(working_dir, "vdb_chunks.json")
        if not os.path.exists(vdb_file) or os.path.getsize(vdb_file) < 100:
            content_list = []
            paper_text = self.loader.load_paper_text(session_id)
            if paper_text:
                for i in range(0, len(paper_text), 2000):
                    content_list.append(paper_text[i:i + 2000])
            transcript = self.loader.load_transcript(session_id)
            if transcript:
                content_list.append(f"=== Presentation Transcript ===\n{transcript}")
            if content_list:
                await rag.ainsert("\n\n".join(content_list))

        self._session_rags[session_id] = rag
        return rag

    async def evaluate_sample(self, sample: dict) -> dict:
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        session_id = sample.get("session_id", "")
        is_unanswerable = sample.get("answerable", True) is False

        t0 = time.time()
        try:
            rag = await self._get_or_build_index(session_id)
            if rag is None:
                generated = "No content available for this session."
            else:
                from lightrag import QueryParam
                param = QueryParam(mode="hybrid")
                generated = await rag.aquery(question, param=param)
                if not generated:
                    generated = "Failed to generate answer"
                else:
                    generated = strip_thinking(generated)

            llm_eval = self._evaluate(question, generated, ground_truth, is_unanswerable)
            return {
                "sample_id": sample.get("sample_id", ""),
                "session_id": session_id,
                "question": question,
                "ground_truth_answer": ground_truth,
                "generated_answer": generated,
                "llm_evaluation": llm_eval,
                "timing": time.time() - t0,
                "error": None,
            }
        except Exception as e:
            return {
                "sample_id": sample.get("sample_id", ""),
                "session_id": session_id,
                "question": question,
                "error": str(e),
                "traceback": traceback.format_exc()[:500],
                "timing": time.time() - t0,
            }

    def _evaluate(self, question, generated, ground_truth, is_unanswerable):
        if is_unanswerable:
            prompt = (
                f"Does the answer indicate the question cannot be answered?\n\n"
                f"Question: {question}\nGenerated Answer: {generated}\n\n"
                f"Respond 1 if the answer says info is unavailable. "
                f"0 if it attempts a specific answer.\n\nEvaluation:"
            )
            resp = call_llm(self.api_url, self.api_key, self.model,
                            [{"role": "user", "content": prompt}],
                            max_tokens=8192, temperature=0.1)
            properly_refused = extract_binary(resp) == "1"
            return {
                "is_correct": properly_refused,
                "properly_refused": properly_refused,
                "is_unanswerable": True,
            }
        else:
            corr_prompt = (
                f"Is the generated answer correct compared to the ground truth?\n\n"
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Generated Answer: {generated}\n\n"
                f"Respond 1 if correct (matches in meaning), 0 if not.\n\nEvaluation:"
            )
            corr_resp = call_llm(self.api_url, self.api_key, self.model,
                                 [{"role": "user", "content": corr_prompt}],
                                 max_tokens=8192, temperature=0.1)
            is_correct = extract_binary(corr_resp) == "1"
            return {
                "is_correct": is_correct,
                "is_unanswerable": False,
            }

    async def evaluate_dataset(self, data_path, max_samples=0, output_path=None):
        print(f"Loading {data_path}...")
        samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if max_samples > 0:
            samples = samples[:max_samples]

        # Resume support
        results = []
        done_ids = set()
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path, encoding="utf-8") as f:
                    old = json.load(f)
                results = [r for r in old.get("results", [])
                           if r.get("generated_answer") != "Failed to generate answer"
                           and not r.get("error")]
                done_ids = {r["sample_id"] for r in results}
                print(f"  Resumed: {len(results)} completed, "
                      f"{len(samples) - len(done_ids)} remaining")
            except Exception:
                results = []

        print(f"Evaluating {len(samples)} samples with RAG-Anything...\n")

        for i, s in enumerate(samples, 1):
            if s.get("sample_id") in done_ids:
                continue
            print(f"\n[{i}/{len(samples)}] {s.get('question', '')[:80]}...")
            r = await self.evaluate_sample(s)
            results.append(r)
            if r.get("error"):
                print(f"  Error: {r['error'][:100]}")
            else:
                ev = r.get("llm_evaluation", {})
                print(f"  Correct: {ev.get('is_correct')} | Time: {r['timing']:.1f}s")

            if output_path and i % 5 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"summary": {}, "results": results}, f, indent=2, ensure_ascii=False)

        summary = self._summary(results)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

        self._print_summary(summary, output_path)
        return summary

    def _summary(self, results):
        valid = [r for r in results if not r.get("error")]
        ans = [r for r in valid if not r.get("llm_evaluation", {}).get("is_unanswerable")]
        unans = [r for r in valid if r.get("llm_evaluation", {}).get("is_unanswerable")]

        s = {
            "mode": "rag_anything",
            "model": self.model,
            "total_samples": len(results),
            "error_count": len(results) - len(valid),
            "answerable_count": len(ans),
            "unanswerable_count": len(unans),
        }

        ans_eval = [r for r in ans if r.get("llm_evaluation", {}).get("is_correct") is not None]
        if ans_eval:
            correct = sum(1 for r in ans_eval if r["llm_evaluation"]["is_correct"])
            s["answerable_accuracy"] = correct / len(ans_eval)
            s["answerable_correct_count"] = correct
            s["answerable_total"] = len(ans_eval)

        all_eval = [r for r in valid if r.get("llm_evaluation", {}).get("is_correct") is not None]
        if all_eval:
            s["overall_accuracy"] = sum(
                1 for r in all_eval if r["llm_evaluation"]["is_correct"]
            ) / len(all_eval)

        times = [r.get("timing", 0) for r in valid]
        s["avg_time"] = sum(times) / len(times) if times else 0
        return s

    def _print_summary(self, s, output_path):
        print(f"\n{'='*60}")
        print(f"RAG-ANYTHING BASELINE SUMMARY")
        print(f"Model: {s['model']}")
        print(f"{'='*60}")
        print(f"Samples: {s['total_samples']}  "
              f"Answerable: {s.get('answerable_count', 0)}  "
              f"Unanswerable: {s.get('unanswerable_count', 0)}  "
              f"Errors: {s['error_count']}")
        if "answerable_accuracy" in s:
            print(f"Answerable Accuracy: {s['answerable_accuracy']:.1%}")
        if "overall_accuracy" in s:
            print(f"Overall Accuracy: {s['overall_accuracy']:.1%}")
        print(f"Avg Time: {s['avg_time']:.1f}s/sample")
        print(f"Output: {output_path}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    p = argparse.ArgumentParser(description="RAG-Anything baseline evaluation")
    p.add_argument("--data-root", default="hsg_output")
    p.add_argument("--golden-data", default="dataset/output/gold_test.jsonl")
    p.add_argument("--output", default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--api", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--api-key",
                   default="your_api_key")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--working-dir", default="baseline/rag_anything_indexes")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--build-index", action="store_true",
                   help="Only build indexes for all sessions, skip evaluation")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel workers for index building")
    args = p.parse_args()

    loader = SessionLoader(args.data_root)

    if args.build_index:
        # Collect unique session IDs from test set
        session_ids = set()
        with open(args.golden_data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    session_ids.add(json.loads(line)["session_id"])
        session_ids = sorted(session_ids)

        cfg = {
            "data_root": args.data_root,
            "api": args.api,
            "api_key": args.api_key,
            "model": args.model,
        }
        build_all_indexes_parallel(
            session_ids, args.working_dir, cfg,
            num_workers=min(args.workers, len(session_ids)),
        )
        return

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline/results_rag_anything_{ts}.json"

    print("=" * 60)
    print(f"BASELINE: RAG-Anything (LightRAG + Qwen3.6-27B)")
    print("=" * 60)

    evaluator = RAGAnythingBaseline(
        api_url=args.api,
        api_key=args.api_key,
        model=args.model,
        loader=loader,
        working_dir_base=args.working_dir,
    )
    await evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
