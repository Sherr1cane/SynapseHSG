#!/usr/bin/env python3
"""Baseline: Pure Qwen3.6-27B (multimodal) QA evaluation — no RAG pipeline.

This baseline feeds the model raw multimodal content from the session:
  - Full paper text (from paper_chunks)
  - Slide images (as base64)
  - Presentation transcript (enriched utterances)
and asks it to answer questions directly.

No decomposition, no retrieval, no reranking — just the model's raw capability.
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

import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Strip Qwen3.x <think ...>...</think\\> blocks."""
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
    """Call OpenAI-compatible chat API (supports multimodal messages)."""
    import time as _time
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
            if resp.status_code == 500 and attempt < max_retries - 1:
                _time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code < 500:
                break
        except requests.exceptions.Timeout:
            print(f"  LLM timeout (attempt {attempt+1}/{max_retries}, {timeout}s)")
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
    """Load raw multimodal content for a session from the build directory."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.sessions_dir = os.path.join(data_root, "sessions")
        self._download_dirs: dict[str, str] = {}
        self._index_download_dirs()

    def _index_download_dirs(self):
        """Build a mapping from session_id to its download directory (with images)."""
        # data_root is like hsg_output
        # raw downloads are at downloads/iclr-2025/, downloads/neurips-2024/ etc.
        parent = os.path.dirname(self.data_root)  # downloads/
        for conf in os.listdir(parent):
            conf_path = os.path.join(parent, conf)
            if not os.path.isdir(conf_path) or conf.startswith("_"):
                continue
            for entry in os.listdir(conf_path):
                entry_path = os.path.join(conf_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                # entry like "39035345_Optimized_Multi_Token..."
                parts = entry.split("_", 1)
                if parts[0].isdigit():
                    sid = f"{conf}/{parts[0]}"
                    self._download_dirs[sid] = entry_path

    def get_session_path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id)

    def get_download_dir(self, session_id: str) -> str | None:
        return self._download_dirs.get(session_id)

    def load_paper_text(self, session_id: str) -> str:
        """Load paper text directly from raw PDF."""
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return ""
        pdf_path = os.path.join(dl_dir, "paper.pdf")
        if not os.path.exists(pdf_path):
            return ""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            return "\n\n".join(page.get_text() for page in doc)
        except Exception:
            return ""

    def load_paper_page_images(self, session_id: str) -> list[str]:
        """Load paper page images as base64 strings."""
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return []
        pages_dir = os.path.join(dl_dir, "paper_pages")
        if not os.path.isdir(pages_dir):
            return []
        images = []
        for fname in sorted(os.listdir(pages_dir)):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(pages_dir, fname)
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                images.append(b64)
        return images

    def load_raw_transcript(self, session_id: str) -> str:
        """Load raw transcript text (not M2HSG-processed)."""
        path = os.path.join(self.get_session_path(session_id), "transcript_enriched.json")
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("combined_enriched_text", "")

    @staticmethod
    def _resize_image_b64(b64: str, size: int = 336) -> str:
        """Resize image to reduce token usage and avoid vLLM cache issues."""
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            img = img.convert("RGB").resize((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return b64

    def load_slides_images(self, session_id: str) -> list[str]:
        """Load slide images as base64 strings, resized to 336x336."""
        dl_dir = self.get_download_dir(session_id)
        if not dl_dir:
            return []
        slides_dir = os.path.join(dl_dir, "slides")
        if not os.path.isdir(slides_dir):
            return []
        images = []
        for fname in sorted(os.listdir(slides_dir)):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(slides_dir, fname)
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                images.append(self._resize_image_b64(b64))
        return images

    def load_transcript(self, session_id: str) -> str:
        """Load enriched transcript text."""
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

    def load_slides_text(self, session_id: str) -> str:
        """Load slide text content from slides_structured.json."""
        path = os.path.join(self.get_session_path(session_id), "slides_structured.json")
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        regions = data.get("visual_regions", [])
        return "\n".join(r.get("text", r.get("content", "")) for r in regions if r.get("text") or r.get("content"))


# ---------------------------------------------------------------------------
# Baseline evaluator
# ---------------------------------------------------------------------------

MODES = ["text_only", "multimodal", "slides_only"]


class BaselineEvaluator:
    """Pure Qwen3.6-27B baseline — no RAG, no decomposition."""

    def __init__(self, api_url: str, api_key: str, model: str,
                 session_loader: SessionLoader, mode: str = "text_only",
                 max_context_chars: int = 60000,
                 eval_api: str = None, eval_key: str = None, eval_model: str = None):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.loader = session_loader
        self.mode = mode
        self.max_context_chars = max_context_chars
        self.eval_api = eval_api or api_url
        self.eval_key = eval_key or api_key
        self.eval_model = eval_model or model

    def _build_messages(self, question: str, session_id: str):
        """Build multimodal messages based on the selected mode."""
        if self.mode == "text_only":
            return self._build_text_only(question, session_id)
        elif self.mode == "multimodal":
            return self._build_multimodal(question, session_id)
        elif self.mode == "slides_only":
            return self._build_slides_only(question, session_id)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_text_only(self, question: str, session_id: str):
        """Text-only: paper + transcript, no images."""
        paper = self.loader.load_paper_text(session_id)
        transcript = self.loader.load_transcript(session_id)
        slides_text = self.loader.load_slides_text(session_id)

        parts = []
        if paper:
            parts.append(f"=== Paper Content ===\n{paper}")
        if transcript:
            parts.append(f"=== Presentation Transcript ===\n{transcript}")
        if slides_text:
            parts.append(f"=== Slide Text Content ===\n{slides_text}")

        context = "\n\n".join(parts)
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "\n\n[...truncated...]"

        prompt = (
            f"You are an expert research assistant answering questions about academic papers.\n\n"
            f"Below is the full content from an academic paper presentation session.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions: Answer the question based on the content above. "
            f"Be specific and include concrete details (numbers, method names, dataset names). "
            f"Answer in 2-4 sentences. "
            f"If the information is not available in the provided content, say so.\n\n"
            f"Answer:"
        )
        return [{"role": "user", "content": prompt}]

    def _build_multimodal(self, question: str, session_id: str):
        """Multimodal: raw PDF text + slide images + transcript text.

        Strictly fits within 32K tokens:
        - 5 slides × ~576 tokens = ~2880 tokens for images
        - Prompt + output = ~4300 tokens
        - Text budget: ~25800 tokens ≈ ~100K chars
        """
        paper = self.loader.load_paper_text(session_id)
        slide_images = self.loader.load_slides_images(session_id)
        transcript = self.loader.load_raw_transcript(session_id)

        # Text budget: paper + transcript share ~60K chars
        # 3 slides × ~1955 tokens = ~5865, prompt ~100, output 2048
        # Total input: 5865 + ~15000(text) + 100 = ~21K, well within 30K
        text_budget = 20000 # 60000
        paper_text = paper[:text_budget] if paper else ""
        remaining = text_budget - len(paper_text)
        transcript_text = transcript[:max(remaining, 2000)] if transcript else "" # 5000

        content = []
        content.append({
            "type": "text",
            "text": (
                "You are an expert research assistant answering questions about academic papers.\n\n"
                "Below is content from an academic paper presentation session.\n\n"
            ),
        })

        # Paper text (from raw PDF)
        if paper_text:
            content.append({
                "type": "text",
                "text": f"=== Paper Content ===\n{paper_text}\n\n",
            })

        # Slide images (limit to 3 to stay within 32K tokens, ~1955 tokens each)
        max_slides = min(len(slide_images), 3)
        if max_slides > 0:
            content.append({
                "type": "text",
                "text": f"=== Presentation Slides ({max_slides}) ===\n",
            })
            for i in range(max_slides):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{slide_images[i]}"},
                })

        # Transcript text
        if transcript_text:
            content.append({
                "type": "text",
                "text": f"\n=== Transcript ===\n{transcript_text}\n",
            })

        content.append({
            "type": "text",
            "text": f"\nQuestion: {question}\n\n"
                    f"Answer the question using the content above. "
                    f"Be specific. 2-4 sentences. "
                    f"If not available, say so.\n\nAnswer:",
        })

        return [{"role": "user", "content": content}]

    def _build_slides_only(self, question: str, session_id: str):
        """Slides-only: only slide images + slide text, no paper/transcript."""
        slide_images = self.loader.load_slides_images(session_id)
        slides_text = self.loader.load_slides_text(session_id)

        content = []
        content.append({
            "type": "text",
            "text": "You are an expert research assistant. Answer the question based on the presentation slides below.\n\n",
        })

        max_slides = min(len(slide_images), 15)
        if max_slides > 0:
            for i in range(max_slides):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{slide_images[i]}"},
                })

        text_section = ""
        if slides_text:
            text_section = f"\n\n=== Slide Text Content ===\n{slides_text[:30000]}\n"

        content.append({
            "type": "text",
            "text": f"{text_section}\nQuestion: {question}\n\n"
                    f"Instructions: Answer the question based on the slides above. "
                    f"Be specific and include concrete details. Answer in 2-4 sentences. "
                    f"If the information is not available, say so.\n\nAnswer:",
        })

        return [{"role": "user", "content": content}]

    def evaluate_sample(self, sample: dict) -> dict:
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        session_id = sample.get("session_id", "")
        is_unanswerable = sample.get("answerable", True) is False

        t0 = time.time()
        try:
            messages = self._build_messages(question, session_id)
            max_out = 2048 if "llava" in self.model.lower() or "7b" in self.model.lower() else 8192
            generated = call_llm(self.api_url, self.api_key, self.model,
                                 messages, max_tokens=max_out, temperature=0.3)
            # Throttle llava requests to avoid vLLM mm_cache race condition
            if "llava" in self.model.lower():
                time.sleep(3)
            if not generated:
                generated = "Failed to generate answer"

            # Evaluate using same LLM-as-judge
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
                f"Respond 1 if the answer says info is unavailable/cannot be answered. "
                f"0 if it attempts a specific answer.\n\nEvaluation:"
            )
            resp = call_llm(self.eval_api, self.eval_key, self.eval_model,
                            [{"role": "user", "content": prompt}],
                            max_tokens=8192, temperature=0.1)
            properly_refused = extract_binary(resp) == "1"
            return {
                "is_correct": properly_refused,
                "properly_refused": properly_refused,
                "is_unanswerable": True,
                "context_supports_answer": None,
            }
        else:
            corr_prompt = (
                f"Is the generated answer correct compared to the ground truth?\n\n"
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Generated Answer: {generated}\n\n"
                f"Respond 1 if correct (matches in meaning), 0 if not.\n\nEvaluation:"
            )
            corr_resp = call_llm(self.eval_api, self.eval_key, self.eval_model,
                                 [{"role": "user", "content": corr_prompt}],
                                 max_tokens=8192, temperature=0.1)
            is_correct = extract_binary(corr_resp) == "1"
            return {
                "is_correct": is_correct,
                "context_supports_answer": None,
                "is_unanswerable": False,
            }

    def evaluate_dataset(self, data_path: str, max_samples: int = 0,
                         output_path: str | None = None, resume: bool = True):
        print(f"Loading {data_path}...")
        samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if max_samples > 0:
            samples = samples[:max_samples]
        print(f"Evaluating {len(samples)} samples (mode={self.mode})...\n")

        # Resume from existing results
        results = []
        done_ids = set()
        if resume and output_path and os.path.exists(output_path):
            try:
                with open(output_path, encoding="utf-8") as f:
                    old = json.load(f)
                results = old.get("results", [])
                done_ids = {r["sample_id"] for r in results if r.get("sample_id")}
                # Filter out failed results for retry
                results = [r for r in results
                           if r.get("generated_answer") != "Failed to generate answer"
                           and not r.get("error")]
                done_ids = {r["sample_id"] for r in results}
                print(f"  Resumed: {len(results)} completed, "
                      f"{len(samples) - len(done_ids)} remaining")
            except Exception as e:
                print(f"  Resume failed ({e}), starting fresh")
                results = []
                done_ids = set()

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

            # Incremental save every 10 samples
            if output_path and i % 10 == 0:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"summary": {}, "results": results},
                              f, indent=2, ensure_ascii=False)

        summary = self._summary(results)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results},
                          f, indent=2, ensure_ascii=False)

        self._print_summary(summary, output_path)
        return summary

    def _summary(self, results):
        valid = [r for r in results if not r.get("error")]
        ans = [r for r in valid if not r.get("llm_evaluation", {}).get("is_unanswerable")]
        unans = [r for r in valid if r.get("llm_evaluation", {}).get("is_unanswerable")]

        s = {
            "mode": self.mode,
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

        unans_eval = [r for r in unans if r.get("llm_evaluation", {}).get("properly_refused") is not None]
        if unans_eval:
            refused = sum(1 for r in unans_eval if r["llm_evaluation"]["properly_refused"])
            s["unanswerable_refusal_rate"] = refused / len(unans_eval)
            s["unanswerable_refused_count"] = refused
            s["unanswerable_total"] = len(unans_eval)

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
        print(f"BASELINE EVALUATION SUMMARY ({s['mode']})")
        print(f"Model: {s['model']}")
        print(f"{'='*60}")
        print(f"Samples: {s['total_samples']}  "
              f"Answerable: {s.get('answerable_count', 0)}  "
              f"Unanswerable: {s.get('unanswerable_count', 0)}  "
              f"Errors: {s['error_count']}")
        if "answerable_accuracy" in s:
            print(f"Answerable Accuracy: {s['answerable_accuracy']:.1%} "
                  f"({s['answerable_correct_count']}/{s['answerable_total']})")
        if "unanswerable_refusal_rate" in s:
            print(f"Unanswerable Refusal: {s['unanswerable_refusal_rate']:.1%}")
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

    p = argparse.ArgumentParser(description="Pure Qwen3.6-27B baseline evaluation")
    p.add_argument("--data-root", default="hsg_output",
                   help="Path to build directory with sessions/")
    p.add_argument("--golden-data", default="dataset/output/gold_test.jsonl",
                   help="Path to gold test JSONL")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: baseline/results_{mode}_{timestamp}.json)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples to evaluate (0=all)")
    p.add_argument("--api", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--api-key", default="your_api_key")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--eval-api", default=None,
                   help="Eval model API (default: same as --api)")
    p.add_argument("--eval-key", default=None,
                   help="Eval model API key (default: same as --api-key)")
    p.add_argument("--eval-model", default=None,
                   help="Eval model name (default: same as --model)")
    p.add_argument("--mode", choices=MODES, default="text_only",
                   help="text_only=paper+transcript text, "
                        "multimodal=text+slide images, "
                        "slides_only=slide images+text only")
    p.add_argument("--max-context-chars", type=int, default=60000,
                   help="Max text context characters before truncation")
    p.add_argument("--no-resume", action="store_true",
                   help="Start fresh, ignore existing results file")
    args = p.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline/results_{args.mode}_{ts}.json"

    print("=" * 60)
    print(f"BASELINE: Pure {args.model} ({args.mode})")
    print("=" * 60)

    loader = SessionLoader(args.data_root)
    evaluator = BaselineEvaluator(
        api_url=args.api,
        api_key=args.api_key,
        model=args.model,
        session_loader=loader,
        mode=args.mode,
        max_context_chars=args.max_context_chars,
        eval_api=args.eval_api,
        eval_key=args.eval_key,
        eval_model=args.eval_model,
    )
    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
