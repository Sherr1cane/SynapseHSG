#!/usr/bin/env python3
"""Baseline: EventRAG (Event Knowledge Graph RAG) evaluation.

Implements EventRAG's event-centric KG + iterative retrieval approach:
  1. Build event knowledge graph from paper text chunks + transcript (per session)
  2. Extract Event/Actor/Action/Result/Tool entities with temporal/causal relations
  3. Retrieve relevant context via EventRAG's KG query (local/hybrid mode)
  4. Also retrieve relevant images via BGE-M3 embedding
  5. Feed retrieved text + images to Qwen3.6-27B VLM
  6. LLM-as-judge evaluation

Uses EventRAG's own KG construction + retrieval pipeline (NOT our HSG).
Generator: Qwen3.6-27B via vLLM API.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from functools import partial

import concurrent.futures

import numpy as np
import requests


# Thread pool for running sync LLM/embedding calls from async context
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)


# ---------------------------------------------------------------------------
# LLM helpers
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
             temperature=0.3, max_retries=5, base_timeout=300,
             enable_thinking=True):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (1 + attempt)
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content is None:
                    continue
                content = content.strip()
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
# Session data loader
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

    def load_paper_chunks(self, session_id: str) -> list[dict]:
        path = os.path.join(self.get_session_path(session_id), "paper_structured.json")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("paper_chunks", [])

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
        return sorted([
            os.path.join(slides_dir, f)
            for f in os.listdir(slides_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])

    def get_paper_page_paths(self, session_id: str) -> list[str]:
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return []
        pages_dir = os.path.join(dl_dir, "paper_pages")
        if not os.path.isdir(pages_dir):
            return []
        return sorted([
            os.path.join(pages_dir, f)
            for f in os.listdir(pages_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])


# ---------------------------------------------------------------------------
# BGE-M3 Embedding service client
# ---------------------------------------------------------------------------

class BGEM3Embedder:
    def __init__(self, api_url: str, api_key: str, model: str = "bge-m3"):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def encode(self, texts: list[str]) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "input": texts,
        }
        resp = requests.post(
            f"{self.api_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        embeddings = [d["embedding"] for d in data["data"]]
        return np.array(embeddings)


# ---------------------------------------------------------------------------
# EventRAG wrapper (uses EventRAG library from baseline/EventRAG/)
# ---------------------------------------------------------------------------

class EventRAGSessionManager:
    """Manages EventRAG instances per session with custom LLM/embedding."""

    def __init__(self, embedder: BGEM3Embedder,
                 api_url: str, api_key: str, model: str,
                 cache_dir: str = "baseline/eventrag_indexes"):
        self.embedder = embedder
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self._sessions: dict[str, object] = {}
        self._image_data: dict[str, dict] = {}

    def _is_indexed(self, session_id: str) -> bool:
        """Check if session has been indexed."""
        work_dir = os.path.join(self.cache_dir, session_id.replace("/", "_"))
        graph_file = os.path.join(work_dir, "chunk_entity_relation_graph.graphml")
        return os.path.exists(graph_file) and os.path.getsize(graph_file) > 100

    def _create_llm_func(self):
        """Create async LLM function compatible with EventRAG."""
        api_url = self.api_url
        api_key = self.api_key
        model = self.model

        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.pop("hashing_kv", None)
            kwargs.pop("keyword_extraction", None)
            kwargs.pop("mode", None)
            kwargs.pop("response_format", None)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            if prompt is not None:
                messages.append({"role": "user", "content": prompt})

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                _THREAD_POOL,
                lambda: call_llm(
                    api_url, api_key, model, messages,
                    max_tokens=kwargs.get("max_tokens", 32768),
                    temperature=kwargs.get("temperature", 0.3),
                    enable_thinking=False,
                )
            )
            return resp or ""

        return llm_func

    def _create_embed_func(self):
        """Create async embedding function compatible with EventRAG's EmbeddingFunc."""
        embedder = self.embedder

        async def embed_func(texts: list[str], **kwargs) -> np.ndarray:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                _THREAD_POOL, lambda: embedder.encode(texts)
            )

        # Wrap with EmbeddingFunc class from EventRAG
        from eventrag.utils import EmbeddingFunc as EF
        return EF(embedding_dim=1024, max_token_size=8192, func=embed_func)

    def build_session_index(self, session_id: str, loader: SessionLoader) -> bool:
        """Build EventRAG index for a session."""
        if self._is_indexed(session_id):
            print(f"  {session_id}: already indexed")
            return True

        chunks = loader.load_paper_chunks(session_id)
        chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]
        transcript = loader.load_transcript(session_id)

        if not chunk_texts and not transcript:
            return False

        # Concatenate text
        all_text = "\n\n".join(chunk_texts)
        if transcript:
            all_text += f"\n\n--- Presentation Transcript ---\n{transcript}"

        work_dir = os.path.join(self.cache_dir, session_id.replace("/", "_"))
        os.makedirs(work_dir, exist_ok=True)

        # Also prepare image data for later retrieval
        image_paths = loader.get_paper_page_paths(session_id) + loader.get_slide_image_paths(session_id)
        if image_paths:
            img_texts = []
            for p in image_paths:
                fname = os.path.basename(p)
                parts = fname.replace(".png", "").replace(".jpg", "").replace("_", " ")
                img_texts.append(f"document page image {parts}")
            img_embs = self.embedder.encode(img_texts)
            self._image_data[session_id] = {
                "image_paths": image_paths,
                "image_embeddings": img_embs,
            }

        try:
            # Import EventRAG
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EventRAG"))
            from eventrag import EventRAG as EventRAGLib

            erag = EventRAGLib(
                working_dir=work_dir,
                llm_model_func=self._create_llm_func(),
                embedding_func=self._create_embed_func(),
                llm_model_name=self.model,
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
                entity_extract_max_gleaning=1,
                entity_summary_to_max_tokens=65536,
                enable_llm_cache=True,
            )

            print(f"  Inserting {len(all_text)} chars into EventRAG...")
            erag.insert([all_text])
            self._sessions[session_id] = erag
            print(f"  EventRAG index built for {session_id}")
            return True
        except Exception as e:
            print(f"  EventRAG indexing failed for {session_id}: {e}")
            traceback.print_exc()
            return False

    def load_session(self, session_id: str) -> object | None:
        """Load an existing EventRAG session from disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        work_dir = os.path.join(self.cache_dir, session_id.replace("/", "_"))
        if not os.path.exists(work_dir):
            return None

        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EventRAG"))
            from eventrag import EventRAG as EventRAGLib

            erag = EventRAGLib(
                working_dir=work_dir,
                llm_model_func=self._create_llm_func(),
                embedding_func=self._create_embed_func(),
                llm_model_name=self.model,
                enable_llm_cache=False,  # No cache for query
            )
            self._sessions[session_id] = erag
            return erag
        except Exception as e:
            print(f"  Failed to load EventRAG session {session_id}: {e}")
            return None

    def query_session(self, session_id: str, query: str,
                      loader: SessionLoader = None,
                      top_k_images: int = 5) -> tuple[str, list[str]]:
        """Query a session's EventRAG index and retrieve images."""
        # Build index if not yet indexed
        if not self._is_indexed(session_id) and loader is not None:
            self.build_session_index(session_id, loader)

        erag = self.load_session(session_id)
        if erag is None:
            return "", []

        try:
            from eventrag.base import QueryParam
            # Use hybrid mode for best retrieval
            param = QueryParam(mode="hybrid", only_need_context=False)
            result = erag.query(query, param=param)
            kg_context = result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"  EventRAG query failed: {e}")
            kg_context = ""

        # Retrieve images
        retrieved_images = []
        img_data = self._image_data.get(session_id)
        if img_data is None:
            # Try loading from cache
            img_cache = os.path.join(self.cache_dir, session_id.replace("/", "_") + "_images.npz")
            img_meta = os.path.join(self.cache_dir, session_id.replace("/", "_") + "_images_meta.json")
            if os.path.exists(img_cache) and os.path.exists(img_meta):
                d = np.load(img_cache)
                with open(img_meta) as f:
                    meta = json.load(f)
                img_data = {
                    "image_paths": meta["image_paths"],
                    "image_embeddings": d["image_embeddings"],
                }
                self._image_data[session_id] = img_data

        if img_data and img_data["image_embeddings"] is not None and len(img_data["image_embeddings"]) > 0:
            q_emb = self.embedder.encode([query])
            scores = (q_emb @ img_data["image_embeddings"].T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k_images]
            retrieved_images = [img_data["image_paths"][i] for i in top_indices]

        return kg_context, retrieved_images


# ---------------------------------------------------------------------------
# EventRAG Baseline Evaluator
# ---------------------------------------------------------------------------

EVENTRAG_PROMPT = """# Task Description
You are a multi-modal Q&A assistant. Answer the user's question using information from the event knowledge graph context and relevant document images.

# Input Data
1. **Question**: The user's query.
2. **Event KG Context**: Knowledge extracted from documents, showing events, actors, actions, and their relationships.
3. **Images**: Document page images with relevant figures, tables, or text.

# Guidelines
1. Use both the event knowledge graph context and images to answer the question.
2. Be specific and include concrete details from the documents.
3. Answer in 2-4 sentences.
4. If the information is not available, say so.

# Question
{question}

# Event Knowledge Graph Context
{kg_context}

Answer based on the provided event knowledge graph context and images above.
"""


class EventRAGBaseline:
    def __init__(self, api_url, api_key, model, loader, session_mgr,
                 top_k_images=5):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.loader = loader
        self.session_mgr = session_mgr
        self.top_k_images = top_k_images

    def evaluate_sample(self, sample: dict) -> dict:
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        session_id = sample.get("session_id", "")
        is_unanswerable = sample.get("answerable", True) is False

        t0 = time.time()
        try:
            kg_context, images = self.session_mgr.query_session(
                session_id, question,
                loader=self.loader,
                top_k_images=self.top_k_images,
            )

            if not kg_context and not images:
                generated = "No documents available for this session."
            else:
                prompt = EVENTRAG_PROMPT.format(question=question, kg_context=kg_context[:8000])
                content = [{"type": "text", "text": prompt}]
                for img_path in images:
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })

                generated = call_llm(
                    self.api_url, self.api_key, self.model,
                    [{"role": "user", "content": content}],
                )
                if not generated:
                    generated = "Failed to generate answer"

            llm_eval = self._evaluate(question, generated, ground_truth, is_unanswerable)
            return {
                "sample_id": sample.get("sample_id", ""),
                "session_id": session_id,
                "question": question,
                "ground_truth_answer": ground_truth,
                "generated_answer": generated,
                "kg_context_length": len(kg_context),
                "retrieved_images": len(images),
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
            return {"is_correct": properly_refused, "properly_refused": properly_refused,
                    "is_unanswerable": True}
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
            return {"is_correct": is_correct, "is_unanswerable": False}

    def evaluate_dataset(self, data_path, max_samples=0, output_path=None,
                         shard=0, num_shards=1):
        print(f"Loading {data_path}...")
        samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if max_samples > 0:
            samples = samples[:max_samples]

        # Shard: each worker processes a subset
        if num_shards > 1:
            my_samples = [s for i, s in enumerate(samples) if i % num_shards == shard]
            print(f"  Shard {shard}/{num_shards}: {len(my_samples)} samples "
                  f"(from {len(samples)} total)")
            samples = my_samples

        # Per-shard output path
        if output_path and num_shards > 1:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_shard{shard}{ext}"

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

        print(f"Evaluating {len(samples)} samples with EventRAG...\n")

        for i, s in enumerate(samples, 1):
            if s.get("sample_id") in done_ids:
                continue
            print(f"\n[{shard}][{i}/{len(samples)}] {s.get('question', '')[:80]}...")
            r = self.evaluate_sample(s)
            results.append(r)
            if r.get("error"):
                print(f"  Error: {r['error'][:100]}")
            else:
                ev = r.get("llm_evaluation", {})
                print(f"  Correct: {ev.get('is_correct')} | "
                      f"KG ctx: {r.get('kg_context_length', 0)} chars | "
                      f"Images: {r.get('retrieved_images', 0)} | "
                      f"Time: {r['timing']:.1f}s")

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
            "mode": "eventrag",
            "model": self.model,
            "top_k_images": self.top_k_images,
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
        print(f"EVENTRAG BASELINE SUMMARY")
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

def main():
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    p = argparse.ArgumentParser(description="EventRAG baseline evaluation")
    p.add_argument("--data-root", default="hsg_output")
    p.add_argument("--golden-data", default="dataset/gold_test.jsonl")
    p.add_argument("--output", default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--api", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--api-key",
                   default="your_api_key")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--embed-api", default="http://localhost:8001/v1")
    p.add_argument("--embed-key", default="sk-000")
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--top-k-images", type=int, default=5)
    p.add_argument("--cache-dir", default="baseline/eventrag_indexes")
    p.add_argument("--build-index", action="store_true",
                   help="Only pre-build indexes, skip evaluation")
    p.add_argument("--shard", type=int, default=0,
                   help="Shard index (0-based) for parallel runs")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards for parallel runs")
    args = p.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline/results_eventrag_{ts}.json"

    print("=" * 60)
    print(f"BASELINE: EventRAG (event-centric KG + iterative retrieval)")
    print(f"Generator: {args.model}")
    print("=" * 60)

    loader = SessionLoader(args.data_root)
    embedder = BGEM3Embedder(args.embed_api, args.embed_key, args.embed_model)
    session_mgr = EventRAGSessionManager(
        embedder, args.api, args.api_key, args.model,
        cache_dir=args.cache_dir,
    )

    if args.build_index:
        session_ids = set()
        with open(args.golden_data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    session_ids.add(json.loads(line)["session_id"])
        print(f"Building EventRAG indexes for {len(session_ids)} sessions...")
        for i, sid in enumerate(sorted(session_ids), 1):
            if session_mgr._is_indexed(sid):
                print(f"[{i}/{len(session_ids)}] {sid}: already indexed")
                continue
            print(f"[{i}/{len(session_ids)}] {sid}: building...")
            t0 = time.time()
            try:
                session_mgr.build_session_index(sid, loader)
                print(f"  Done in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
        print("Index building complete.")
        return

    evaluator = EventRAGBaseline(
        api_url=args.api,
        api_key=args.api_key,
        model=args.model,
        loader=loader,
        session_mgr=session_mgr,
        top_k_images=args.top_k_images,
    )
    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
        shard=args.shard, num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
