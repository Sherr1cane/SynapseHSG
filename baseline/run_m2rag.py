#!/usr/bin/env python3
"""Baseline: M²RAG (Multi-modal Retrieval Augmented Multi-modal Generation) evaluation.

Implements M²RAG's single-stage VLM summarization approach adapted for local documents:
  1. Retrieve relevant text chunks via BGE-M3 embedding similarity (session-scoped)
  2. Retrieve relevant document/slide page images via text-to-image matching
  3. Single-stage generation: feed retrieved text + images to Qwen3.6-27B VLM
  4. LLM-as-judge evaluation

Uses M²RAG's own retrieval+generation approach (embedding-based retrieval + VLM generation),
NOT our HSG pipeline. Generator: Qwen3.6-27B via vLLM API.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests


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
# M²RAG Retriever (session-scoped)
# ---------------------------------------------------------------------------

class M2RAGRetriever:
    """Embedding-based retriever for text chunks and document images."""

    def __init__(self, embedder: BGEM3Embedder, cache_dir: str = "baseline/m2rag_indexes"):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self._session_data: dict[str, dict] = {}

    def _get_or_build(self, session_id: str, loader: SessionLoader) -> dict | None:
        if session_id in self._session_data:
            return self._session_data[session_id]

        # Try cache
        cache_path = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}.npz")
        meta_path = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}_meta.json")

        if os.path.exists(cache_path) and os.path.exists(meta_path):
            data = np.load(cache_path)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._session_data[session_id] = {
                "chunk_embeddings": data["chunk_embeddings"],
                "chunk_texts": meta["chunk_texts"],
                "image_paths": meta["image_paths"],
                "image_embeddings": data.get("image_embeddings", None),
            }
            return self._session_data[session_id]

        # Build from scratch
        chunks = loader.load_paper_chunks(session_id)
        chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]

        image_paths = loader.get_paper_page_paths(session_id) + loader.get_slide_image_paths(session_id)

        if not chunk_texts and not image_paths:
            return None

        # Embed text chunks
        chunk_embeddings = None
        if chunk_texts:
            chunk_embeddings = self.embedder.encode(chunk_texts)

        # For images, we use their filenames as pseudo-text for retrieval
        # (M²RAG's original approach uses image captions from webpages)
        image_embeddings = None
        if image_paths:
            # Use filenames as retrieval keys (limited but consistent with M²RAG's
            # text-based image retrieval)
            img_texts = []
            for p in image_paths:
                fname = os.path.basename(p)
                # Extract meaningful parts from filename
                parts = fname.replace(".png", "").replace(".jpg", "").replace("_", " ")
                img_texts.append(f"document page image {parts}")
            image_embeddings = self.embedder.encode(img_texts)

        # Cache
        os.makedirs(self.cache_dir, exist_ok=True)
        save_data = {}
        if chunk_embeddings is not None:
            save_data["chunk_embeddings"] = chunk_embeddings
        else:
            save_data["chunk_embeddings"] = np.array([])
        if image_embeddings is not None:
            save_data["image_embeddings"] = image_embeddings
        else:
            save_data["image_embeddings"] = np.array([])
        np.savez(cache_path, **save_data)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "chunk_texts": chunk_texts,
                "image_paths": image_paths,
            }, f)

        self._session_data[session_id] = {
            "chunk_embeddings": chunk_embeddings,
            "chunk_texts": chunk_texts,
            "image_paths": image_paths,
            "image_embeddings": image_embeddings,
        }
        return self._session_data[session_id]

    def retrieve(self, session_id: str, query: str, loader: SessionLoader,
                 top_k_chunks: int = 10, top_k_images: int = 5) -> tuple[list[str], list[str]]:
        """Retrieve top-k text chunks and top-k images for a query."""
        data = self._get_or_build(session_id, loader)
        if data is None:
            return [], []

        q_emb = self.embedder.encode([query])

        # Retrieve text chunks
        retrieved_chunks = []
        if data["chunk_embeddings"] is not None and len(data["chunk_embeddings"]) > 0:
            scores = (q_emb @ data["chunk_embeddings"].T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k_chunks]
            retrieved_chunks = [data["chunk_texts"][i] for i in top_indices]

        # Retrieve images
        retrieved_images = []
        if data["image_embeddings"] is not None and len(data["image_embeddings"]) > 0:
            scores = (q_emb @ data["image_embeddings"].T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k_images]
            retrieved_images = [data["image_paths"][i] for i in top_indices]

        return retrieved_chunks, retrieved_images


# ---------------------------------------------------------------------------
# M²RAG Baseline Evaluator
# ---------------------------------------------------------------------------

M2RAG_PROMPT = """# Task Description
You are a multi-modal Q&A assistant. Your role is to answer a user's question using information from relevant documents that include both text and images.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your answer.
2. **Documents**: Text passages from the document.
3. **Images**: Document page images that may contain relevant figures, tables, or text.

# Guidelines
1. **Understand the Question**: Determine which content from the documents best answers the user's question.
2. **Text Answer**: Create a coherent answer to fully address the user's question.
3. **Image Integration**: Use information from both text and images to provide a comprehensive answer.
4. **Image Relevance**: Reference relevant visual information when it supports your answer.

# Precautions
1. Be specific and include concrete details from the documents.
2. Answer in 2-4 sentences.
3. If the information is not available in the provided documents, say so.

# Question
{question}

# Documents
{documents}

Answer based on the provided documents and images above.
"""


class M2RAGBaseline:
    def __init__(self, api_url, api_key, model, loader, retriever,
                 top_k_chunks=10, top_k_images=5):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.loader = loader
        self.retriever = retriever
        self.top_k_chunks = top_k_chunks
        self.top_k_images = top_k_images

    def evaluate_sample(self, sample: dict) -> dict:
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        session_id = sample.get("session_id", "")
        is_unanswerable = sample.get("answerable", True) is False

        t0 = time.time()
        try:
            # M²RAG retrieval: get relevant chunks and images
            chunks, images = self.retriever.retrieve(
                session_id, question, self.loader,
                top_k_chunks=self.top_k_chunks,
                top_k_images=self.top_k_images,
            )

            if not chunks and not images:
                generated = "No documents available for this session."
            else:
                # Build M²RAG-style single-stage multimodal message
                docs_text = ""
                for i, chunk in enumerate(chunks):
                    docs_text += f"## Document {i}\n```\n{chunk}\n```\n\n"

                prompt = M2RAG_PROMPT.format(question=question, documents=docs_text)

                content = [{"type": "text", "text": prompt}]

                # Append retrieved images
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
                "retrieved_chunks": len(chunks),
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

    def evaluate_dataset(self, data_path, max_samples=0, output_path=None):
        print(f"Loading {data_path}...")
        samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if max_samples > 0:
            samples = samples[:max_samples]

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

        print(f"Evaluating {len(samples)} samples with M2RAG "
              f"(top-{self.top_k_chunks} chunks, top-{self.top_k_images} images)...\n")

        for i, s in enumerate(samples, 1):
            if s.get("sample_id") in done_ids:
                continue
            print(f"\n[{i}/{len(samples)}] {s.get('question', '')[:80]}...")
            r = self.evaluate_sample(s)
            results.append(r)
            if r.get("error"):
                print(f"  Error: {r['error'][:100]}")
            else:
                ev = r.get("llm_evaluation", {})
                print(f"  Correct: {ev.get('is_correct')} | "
                      f"Chunks: {r.get('retrieved_chunks', 0)} | "
                      f"Images: {r.get('retrieved_images', 0)} | "
                      f"Time: {r['timing']:.1f}s")

            if output_path and i % 10 == 0:
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
            "mode": "m2rag",
            "model": self.model,
            "top_k_chunks": self.top_k_chunks,
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
        print(f"M2RAG BASELINE SUMMARY")
        print(f"Model: {s['model']}")
        print(f"Top-K chunks: {s.get('top_k_chunks', '?')} | Top-K images: {s.get('top_k_images', '?')}")
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

    p = argparse.ArgumentParser(description="M2RAG baseline evaluation")
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
    p.add_argument("--top-k-chunks", type=int, default=10)
    p.add_argument("--top-k-images", type=int, default=5)
    p.add_argument("--cache-dir", default="baseline/m2rag_indexes")
    p.add_argument("--build-index", action="store_true",
                   help="Only pre-build indexes, skip evaluation")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline/results_m2rag_top{args.top_k_chunks}c{args.top_k_images}i_{ts}.json"

    print("=" * 60)
    print(f"BASELINE: M2RAG (top-{args.top_k_chunks} chunks + top-{args.top_k_images} images)")
    print(f"Generator: {args.model}")
    print("=" * 60)

    loader = SessionLoader(args.data_root)
    embedder = BGEM3Embedder(args.embed_api, args.embed_key, args.embed_model)
    retriever = M2RAGRetriever(embedder, cache_dir=args.cache_dir)

    if args.build_index:
        # Pre-build indexes for all sessions
        session_ids = set()
        with open(args.golden_data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    session_ids.add(json.loads(line)["session_id"])
        print(f"Building M2RAG indexes for {len(session_ids)} sessions...")
        for i, sid in enumerate(sorted(session_ids), 1):
            cache_path = os.path.join(args.cache_dir, f"{sid.replace('/', '_')}.npz")
            if os.path.exists(cache_path):
                print(f"[{i}/{len(session_ids)}] {sid}: cached")
                continue
            print(f"[{i}/{len(session_ids)}] {sid}: encoding...")
            t0 = time.time()
            try:
                data = retriever._get_or_build(sid, loader)
                if data:
                    print(f"  Done in {time.time()-t0:.1f}s "
                          f"({len(data['chunk_texts'])} chunks, {len(data['image_paths'])} images)")
                else:
                    print(f"  No data")
            except Exception as e:
                print(f"  FAILED: {e}")
        print("Index building complete.")
        return

    evaluator = M2RAGBaseline(
        api_url=args.api,
        api_key=args.api_key,
        model=args.model,
        loader=loader,
        retriever=retriever,
        top_k_chunks=args.top_k_chunks,
        top_k_images=args.top_k_images,
    )
    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
    )


if __name__ == "__main__":
    main()
