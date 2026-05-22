#!/usr/bin/env python3
"""Baseline: KGGen (Knowledge Graph Generation from Any Text) evaluation.

Implements KGGen's approach adapted for multimodal document QA:
  1. Extract knowledge graph from paper text chunks + transcript (per session)
  2. Generate entity embeddings via BGE-M3
  3. Retrieve relevant subgraph context via embedding similarity + graph traversal
  4. Also retrieve relevant images via BGE-M3 text-to-image matching
  5. Feed retrieved text + images to Qwen3.6-27B VLM
  6. LLM-as-judge evaluation

Uses KGGen's own KG extraction pipeline (NOT our HSG).
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
# LLM helpers (same as run_m2rag.py)
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
                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content") or ""
                # Handle thinking mode: check reasoning_content or reasoning keys
                if not content.strip():
                    for rc_key in ("reasoning_content", "reasoning"):
                        rc = msg.get(rc_key)
                        if rc and isinstance(rc, str) and rc.strip():
                            # Try to extract the answer from the reasoning
                            # Usually reasoning ends with the actual answer
                            content = rc
                            break
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
# Session data loader (same as run_m2rag.py)
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
# KGGen Knowledge Graph builder + retriever
# ---------------------------------------------------------------------------

class KGGenRetriever:
    """Build KG with KGGen library, retrieve via BGE-M3 embedding over entities."""

    def __init__(self, embedder: BGEM3Embedder,
                 api_url: str, api_key: str, model: str,
                 cache_dir: str = "baseline/kkgen_indexes"):
        self.embedder = embedder
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self._session_data: dict[str, dict] = {}

    def _get_or_build(self, session_id: str, loader: SessionLoader) -> dict | None:
        if session_id in self._session_data:
            return self._session_data[session_id]

        cache_path = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}.json")
        emb_path = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}_emb.npz")
        meta_path = os.path.join(self.cache_dir, f"{session_id.replace('/', '_')}_meta.json")

        # Try loading from cache
        if os.path.exists(cache_path) and os.path.exists(emb_path) and os.path.exists(meta_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    graph_data = json.load(f)
                emb_data = np.load(emb_path)
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)

                entities = graph_data.get("entities", [])
                relations = graph_data.get("relations", [])
                edges = graph_data.get("edges", [])

                self._session_data[session_id] = {
                    "entities": entities,
                    "relations": relations,
                    "edges": edges,
                    "entity_embeddings": emb_data["entity_embeddings"],
                    "entity_names": meta["entity_names"],
                    "image_paths": meta["image_paths"],
                    "image_embeddings": emb_data.get("image_embeddings", None),
                }
                return self._session_data[session_id]
            except Exception as e:
                print(f"  Cache load failed for {session_id}: {e}")

        # Build from scratch
        chunks = loader.load_paper_chunks(session_id)
        chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]
        transcript = loader.load_transcript(session_id)

        if not chunk_texts and not transcript:
            return None

        # Concatenate all text for KG extraction
        all_text = "\n\n".join(chunk_texts)
        if transcript:
            all_text += f"\n\n--- Transcript ---\n{transcript}"

        print(f"  Extracting KG from {len(all_text)} chars...")

        # Extract KG using LLM-based prompts (KGGen paper approach, without buggy library)
        try:
            entities, relations = self._extract_kg(all_text)
            edges = [(r[0], r[2]) for r in relations]
            print(f"  Extracted: {len(entities)} entities, {len(relations)} relations")
        except Exception as e:
            print(f"  KG extraction failed: {e}")
            traceback.print_exc()
            return None

        if not entities:
            return None

        # Generate embeddings for entities using BGE-M3
        entity_texts = [str(e) for e in entities]
        batch_size = 64
        all_embs = []
        for i in range(0, len(entity_texts), batch_size):
            batch = entity_texts[i:i + batch_size]
            embs = self.embedder.encode(batch)
            all_embs.append(embs)
        entity_embeddings = np.vstack(all_embs) if all_embs else np.array([])

        # Embed images for retrieval
        image_paths = loader.get_paper_page_paths(session_id) + loader.get_slide_image_paths(session_id)
        image_embeddings = None
        if image_paths:
            img_texts = []
            for p in image_paths:
                fname = os.path.basename(p)
                parts = fname.replace(".png", "").replace(".jpg", "").replace("_", " ")
                img_texts.append(f"document page image {parts}")
            image_embeddings = self.embedder.encode(img_texts)

        # Cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"entities": entities, "relations": relations, "edges": edges}, f)

        save_data = {"entity_embeddings": entity_embeddings}
        if image_embeddings is not None:
            save_data["image_embeddings"] = image_embeddings
        else:
            save_data["image_embeddings"] = np.array([])
        np.savez(emb_path, **save_data)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "entity_names": entity_texts,
                "image_paths": image_paths,
            }, f)

        self._session_data[session_id] = {
            "entities": entities,
            "relations": relations,
            "edges": edges,
            "entity_embeddings": entity_embeddings,
            "entity_names": entity_texts,
            "image_paths": image_paths,
            "image_embeddings": image_embeddings,
        }
        return self._session_data[session_id]

    def _extract_kg(self, text: str) -> tuple[list[str], list[tuple]]:
        """Extract knowledge graph from text using direct LLM calls.

        Follows KGGen paper approach: extract entities, then extract relations between them.
        Processes text in chunks to handle long documents.
        """
        chunk_size = 6000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_entities = set()
        all_relations = set()

        def _get_json_response(prompt: str) -> str | None:
            """Call LLM and return the content. Falls back to extracting JSON from reasoning."""
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            for attempt in range(3):
                try:
                    resp = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                    if resp.status_code == 200:
                        msg = resp.json()["choices"][0]["message"]
                        content = msg.get("content")
                        if content and content.strip():
                            return strip_thinking(content.strip())
                        # content is None — model used thinking mode
                        # Extract JSON from reasoning field (answer is at the end)
                        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                        if reasoning:
                            # Find the last JSON array/object in the reasoning
                            matches = list(re.finditer(r'\[.*?\]', reasoning, re.DOTALL))
                            if matches:
                                return matches[-1].group()
                            matches = list(re.finditer(r'\{.*?\}', reasoning, re.DOTALL))
                            if matches:
                                return matches[-1].group()
                except Exception as e:
                    print(f"  _get_json_response error: {e}")
            return None

        entity_prompt = """Extract all key entities from the following text. Return a JSON list of entity strings.
Focus on: methods, models, concepts, datasets, metrics, components, techniques, results.
Text:
```
{text}
```
Return ONLY a JSON list, e.g. ["entity1", "entity2", ...]:"""

        relation_prompt = """Given the following text and entities, extract subject-predicate-object triples.
Return a JSON list of [subject, predicate, object] triples.

Text:
```
{text}
```

Entities: {entities}

Return ONLY a JSON list, e.g. [["subject1", "predicate1", "object1"], ...]:"""

        for chunk in chunks[:5]:
            ent_resp = _get_json_response(entity_prompt.format(text=chunk[:3000]))
            if ent_resp:
                try:
                    match = re.search(r'\[.*\]', ent_resp, re.DOTALL)
                    if match:
                        ents = json.loads(match.group())
                        all_entities.update(str(e) for e in ents if isinstance(e, str))
                except (json.JSONDecodeError, ValueError):
                    pass

        if all_entities:
            ent_list = list(all_entities)
            for chunk in chunks[:3]:
                ent_str = ", ".join(ent_list[:50])
                rel_resp = _get_json_response(relation_prompt.format(
                    text=chunk[:3000], entities=ent_str))
                if rel_resp:
                    try:
                        match = re.search(r'\[.*\]', rel_resp, re.DOTALL)
                        if match:
                            rels = json.loads(match.group())
                            for r in rels:
                                if isinstance(r, list) and len(r) >= 3:
                                    all_relations.add((str(r[0]), str(r[1]), str(r[2])))
                    except (json.JSONDecodeError, ValueError):
                        pass

        return list(all_entities), list(all_relations)

    def retrieve(self, session_id: str, query: str, loader: SessionLoader,
                 top_k_entities: int = 10, top_k_images: int = 5,
                 graph_depth: int = 2) -> tuple[str, list[str]]:
        """Retrieve KG context text and relevant images for a query."""
        data = self._get_or_build(session_id, loader)
        if data is None:
            return "", []

        q_emb = self.embedder.encode([query])

        # Retrieve relevant entities via embedding similarity
        context_triples = set()
        if data["entity_embeddings"] is not None and len(data["entity_embeddings"]) > 0:
            scores = (q_emb @ data["entity_embeddings"].T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k_entities]

            top_entity_names = set(data["entity_names"][i] for i in top_indices)

            # Graph traversal: find relations involving top entities
            relations = data.get("relations", [])
            for rel in relations:
                rel_list = list(rel)
                if len(rel_list) >= 3:
                    src, pred, tgt = str(rel_list[0]), str(rel_list[1]), str(rel_list[2])
                    if src in top_entity_names or tgt in top_entity_names:
                        context_triples.add(f"{src} --[{pred}]--> {tgt}")

        # Also add direct entity descriptions for top entities
        context_lines = list(context_triples)
        if not context_lines:
            # Fallback: just list the top entity names
            context_lines = [data["entity_names"][i] for i in
                             np.argsort((q_emb @ data["entity_embeddings"].T).flatten())[::-1][:top_k_entities]]

        context_text = "\n".join(context_lines[:50])  # Limit context size

        # Retrieve images
        retrieved_images = []
        if data["image_embeddings"] is not None and len(data["image_embeddings"]) > 0:
            scores = (q_emb @ data["image_embeddings"].T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k_images]
            retrieved_images = [data["image_paths"][i] for i in top_indices]

        return context_text, retrieved_images


# ---------------------------------------------------------------------------
# KGGen Baseline Evaluator
# ---------------------------------------------------------------------------

KKGEN_PROMPT = """# Task Description
You are a multi-modal Q&A assistant. Your role is to answer a user's question using information from a knowledge graph and relevant document images.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your answer.
2. **Knowledge Graph Context**: Structured knowledge extracted from the documents, showing entities and their relationships.
3. **Images**: Document page images that may contain relevant figures, tables, or text.

# Guidelines
1. **Understand the Question**: Determine which information from the knowledge graph and images best answers the question.
2. **Use KG Context**: Leverage the entity relationships and structured knowledge to provide accurate answers.
3. **Image Integration**: Use information from images to supplement and enrich your answer.
4. **Answer Quality**: Be specific, include concrete details, and answer in 2-4 sentences.
5. If the information is not available, say so.

# Question
{question}

# Knowledge Graph Context
{kg_context}

Answer based on the provided knowledge graph context and images above.
"""


class KGGenBaseline:
    def __init__(self, api_url, api_key, model, loader, retriever,
                 top_k_entities=10, top_k_images=5):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.loader = loader
        self.retriever = retriever
        self.top_k_entities = top_k_entities
        self.top_k_images = top_k_images

    def evaluate_sample(self, sample: dict) -> dict:
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        session_id = sample.get("session_id", "")
        is_unanswerable = sample.get("answerable", True) is False

        t0 = time.time()
        try:
            kg_context, images = self.retriever.retrieve(
                session_id, question, self.loader,
                top_k_entities=self.top_k_entities,
                top_k_images=self.top_k_images,
            )

            if not kg_context and not images:
                generated = "No documents available for this session."
            else:
                prompt = KKGEN_PROMPT.format(question=question, kg_context=kg_context)

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

        print(f"Evaluating {len(samples)} samples with KGGen "
              f"(top-{self.top_k_entities} entities, top-{self.top_k_images} images)...\n")

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
                      f"KG ctx: {r.get('kg_context_length', 0)} chars | "
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
            "mode": "kkgen",
            "model": self.model,
            "top_k_entities": self.top_k_entities,
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
        print(f"KKGEN BASELINE SUMMARY")
        print(f"Model: {s['model']}")
        print(f"Top-K entities: {s.get('top_k_entities', '?')} | Top-K images: {s.get('top_k_images', '?')}")
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
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    p = argparse.ArgumentParser(description="KGGen baseline evaluation")
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
    p.add_argument("--top-k-entities", type=int, default=10)
    p.add_argument("--top-k-images", type=int, default=5)
    p.add_argument("--cache-dir", default="baseline/kkgen_indexes")
    p.add_argument("--build-index", action="store_true",
                   help="Only pre-build indexes, skip evaluation")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"baseline/results_kkgen_top{args.top_k_entities}e{args.top_k_images}i_{ts}.json"

    print("=" * 60)
    print(f"BASELINE: KGGen (top-{args.top_k_entities} entities + top-{args.top_k_images} images)")
    print(f"Generator: {args.model}")
    print("=" * 60)

    loader = SessionLoader(args.data_root)
    embedder = BGEM3Embedder(args.embed_api, args.embed_key, args.embed_model)
    retriever = KGGenRetriever(
        embedder, args.api, args.api_key, args.model,
        cache_dir=args.cache_dir,
    )

    if args.build_index:
        session_ids = set()
        with open(args.golden_data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    session_ids.add(json.loads(line)["session_id"])
        print(f"Building KGGen indexes for {len(session_ids)} sessions...")
        for i, sid in enumerate(sorted(session_ids), 1):
            cache_path = os.path.join(args.cache_dir, f"{sid.replace('/', '_')}.json")
            if os.path.exists(cache_path):
                print(f"[{i}/{len(session_ids)}] {sid}: cached")
                continue
            print(f"[{i}/{len(session_ids)}] {sid}: extracting KG...")
            t0 = time.time()
            try:
                data = retriever._get_or_build(sid, loader)
                if data:
                    print(f"  Done in {time.time()-t0:.1f}s "
                          f"({len(data['entities'])} entities, {len(data['relations'])} relations)")
                else:
                    print(f"  No data")
            except Exception as e:
                print(f"  FAILED: {e}")
        print("Index building complete.")
        return

    evaluator = KGGenBaseline(
        api_url=args.api,
        api_key=args.api_key,
        model=args.model,
        loader=loader,
        retriever=retriever,
        top_k_entities=args.top_k_entities,
        top_k_images=args.top_k_images,
    )
    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
    )


if __name__ == "__main__":
    main()
