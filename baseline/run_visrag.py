#!/usr/bin/env python3
"""Baseline: VisRAG (Vision-based RAG) evaluation.

Pipeline per session:
  1. Encode paper pages + slides with VisRAG-Ret (image → embedding)
  2. For each question: encode question text → cosine sim → top-k page images
  3. Feed top-k images + question to Qwen3.6-27B for answer generation
  4. LLM-as-judge evaluation

Uses VisRAG's own retrieval (visual document embedding), NOT our HSG.
Generator: Qwen3.6-27B via vLLM API.
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

    def get_download_dir(self, session_id: str) -> str | None:
        return self._download_dirs.get(session_id)

    def get_document_image_paths(self, session_id: str) -> list[tuple[str, str]]:
        """Get all document image paths with type labels: [(path, label), ...]"""
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return []
        paths = []
        # Paper pages
        pages_dir = os.path.join(dl_dir, "paper_pages")
        if os.path.isdir(pages_dir):
            for fname in sorted(os.listdir(pages_dir)):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    paths.append((os.path.join(pages_dir, fname), "paper"))
        # Slides
        slides_dir = os.path.join(dl_dir, "slides")
        if os.path.isdir(slides_dir):
            for fname in sorted(os.listdir(slides_dir)):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    paths.append((os.path.join(slides_dir, fname), "slide"))
        return paths


# ---------------------------------------------------------------------------
# VisRAG-Ret encoder
# ---------------------------------------------------------------------------

class VisRAGEncoder:
    """Encode images and text using VisRAG-Ret model."""

    def __init__(self, model_path: str = "baseline/visrag_model"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.cache_utils import DynamicCache
        import torch.nn.functional as F

        # Monkey-patch: newer transformers removed get_usable_length
        if not hasattr(DynamicCache, 'get_usable_length'):
            DynamicCache.get_usable_length = lambda self, seq_len=0, layer_idx=0: self.get_seq_length(layer_idx)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading VisRAG-Ret from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.F = F
        self.torch = torch
        self.instruction = "Represent this query for retrieving relevant documents: "
        print("VisRAG-Ret loaded.")

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        from PIL import Image
        with self.torch.no_grad():
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = {
                "text": [""] * len(images),
                "image": images,
                "tokenizer": self.tokenizer,
            }
            outputs = self.model(**inputs)
            reps = self._weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
            return self.F.normalize(reps, p=2, dim=1).cpu().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        texts = [self.instruction + t for t in texts]
        with self.torch.no_grad():
            inputs = {
                "text": texts,
                "image": [None] * len(texts),
                "tokenizer": self.tokenizer,
            }
            outputs = self.model(**inputs)
            reps = self._weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
            return self.F.normalize(reps, p=2, dim=1).cpu().numpy()

    def _weighted_mean_pooling(self, hidden, attention_mask):
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = (hidden * attention_mask_.unsqueeze(-1).float()).sum(dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        return s / d


# ---------------------------------------------------------------------------
# VisRAG Baseline Evaluator
# ---------------------------------------------------------------------------

class VisRAGBaseline:
    def __init__(self, api_url, api_key, model, loader, encoder,
                 top_k=5, cache_dir="baseline/visrag_indexes"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.loader = loader
        self.encoder = encoder
        self.top_k = top_k
        self.cache_dir = cache_dir
        self._session_indexes: dict[str, tuple[np.ndarray, list[str]]] = {}

    def _get_or_build_index(self, session_id: str):
        if session_id in self._session_indexes:
            return self._session_indexes[session_id]

        # Try loading from cache
        cache_path = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}.npz")
        img_paths_file = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}_paths.json")

        if os.path.exists(cache_path) and os.path.exists(img_paths_file):
            data = np.load(cache_path)
            with open(img_paths_file) as f:
                paths = json.load(f)
            self._session_indexes[session_id] = (data["embeddings"], paths)
            return self._session_indexes[session_id]

        # Build index
        doc_images = self.loader.get_document_image_paths(session_id)
        if not doc_images:
            return None

        paths = [p for p, _ in doc_images]
        embeddings = self.encoder.encode_images(paths)

        # Cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(cache_path, embeddings=embeddings)
        with open(img_paths_file, "w") as f:
            json.dump(paths, f)

        self._session_indexes[session_id] = (embeddings, paths)
        return self._session_indexes[session_id]

    def evaluate_sample(self, sample: dict) -> dict:
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        session_id = sample.get("session_id", "")
        is_unanswerable = sample.get("answerable", True) is False

        t0 = time.time()
        try:
            index = self._get_or_build_index(session_id)
            if index is None:
                generated = "No document images available for this session."
            else:
                embeddings, img_paths = index
                # Encode question and retrieve top-k
                q_emb = self.encoder.encode_text([question])
                scores = (q_emb @ embeddings.T).flatten()
                top_indices = np.argsort(scores)[::-1][:self.top_k]

                # Build multimodal message with top-k images
                content = []
                content.append({
                    "type": "text",
                    "text": (
                        f"You are an expert research assistant. Answer the question "
                        f"based on the document page images below.\n\n"
                        f"Question: {question}\n\n"
                        f"Be specific and include concrete details. "
                        f"Answer in 2-4 sentences. "
                        f"If the information is not available, say so.\n"
                    ),
                })
                for idx in top_indices:
                    with open(img_paths[idx], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })

                generated = call_llm(self.api_url, self.api_key, self.model,
                                     [{"role": "user", "content": content}])
                if not generated:
                    generated = "Failed to generate answer"

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

        print(f"Evaluating {len(samples)} samples with VisRAG (top-{self.top_k})...\n")

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
                print(f"  Correct: {ev.get('is_correct')} | Time: {r['timing']:.1f}s")

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
            "mode": "visrag",
            "model": self.model,
            "top_k": self.top_k,
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
        print(f"VISRAG BASELINE SUMMARY (top-{s.get('top_k', 5)})")
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

    p = argparse.ArgumentParser(description="VisRAG baseline evaluation")
    p.add_argument("--data-root", default="hsg_output")
    p.add_argument("--golden-data", default="dataset/gold_test.jsonl")
    p.add_argument("--output", default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--api", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--api-key",
                   default="your_api_key")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--visrag-model", default="baseline/visrag_model",
                   help="Path to VisRAG-Ret model")
    p.add_argument("--top-k", type=int, default=5, help="Number of images to retrieve")
    p.add_argument("--cache-dir", default="baseline/visrag_indexes")
    p.add_argument("--build-index", action="store_true",
                   help="Only pre-build image embeddings, skip evaluation")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline/results_visrag_top{args.top_k}_{ts}.json"

    print("=" * 60)
    print(f"BASELINE: VisRAG (top-{args.top_k}) + Qwen3.6-27B")
    print("=" * 60)

    loader = SessionLoader(args.data_root)
    encoder = VisRAGEncoder(model_path=args.visrag_model)

    if args.build_index:
        # Pre-build indexes
        session_ids = set()
        with open(args.golden_data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    session_ids.add(json.loads(line)["session_id"])
        print(f"Building VisRAG image indexes for {len(session_ids)} sessions...")
        for i, sid in enumerate(sorted(session_ids), 1):
            index = Baseline = None
            cache_path = os.path.join(args.cache_dir, f"{sid.replace('/', '_')}.npz")
            if os.path.exists(cache_path):
                print(f"[{i}/{len(session_ids)}] {sid}: cached")
                continue
            print(f"[{i}/{len(session_ids)}] {sid}: encoding...")
            t0 = time.time()
            try:
                doc_images = loader.get_document_image_paths(sid)
                if doc_images:
                    paths = [p for p, _ in doc_images]
                    embeddings = encoder.encode_images(paths)
                    os.makedirs(args.cache_dir, exist_ok=True)
                    np.savez(cache_path, embeddings=embeddings)
                    with open(cache_path.replace(".npz", "_paths.json"), "w") as f:
                        json.dump(paths, f)
                    print(f"  Done in {time.time()-t0:.1f}s ({len(paths)} images)")
                else:
                    print(f"  No images")
            except Exception as e:
                print(f"  FAILED: {e}")
        print("Index building complete.")
        return

    evaluator = VisRAGBaseline(
        api_url=args.api,
        api_key=args.api_key,
        model=args.model,
        loader=loader,
        encoder=encoder,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
    )
    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
    )


if __name__ == "__main__":
    main()
