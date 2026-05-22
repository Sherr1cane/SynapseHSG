#!/usr/bin/env python3
"""End-to-end QA evaluation with the new pipeline.

Full pipeline:
1. Load HSG (with inverted index + session mapping)
2. Decompose question (using trained decomposer)
3. Session-scoped multi-channel retrieval
4. Semantic reranking
5. Generate answer (Qwen3.6-27B)
6. Evaluate answer (Qwen3.6-27B, binary)

No imports from stage2/scripts/ — fully self-contained.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import requests

from constructor import HSG, Triple, BGEEmbeddingClient
from retriever import SessionScopedRetriever, get_retrieval_channels
from reranker import SemanticReranker


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(api_url, api_key, model, prompt, max_tokens=8192, temperature=0.3,
              max_retries=3, base_timeout=120):
    """Call an OpenAI-compatible chat API with retry on timeout."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature,
    }
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (1 + attempt)  # 120, 240, 360s
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                msg = resp.json()["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                # Strip Qwen3.5 thinking block
                think_end = content.find("</think")
                if think_end >= 0:
                    rest = content[think_end:]
                    nl = rest.find("\n")
                    if nl >= 0:
                        content = rest[nl + 1:].strip()
                    else:
                        content = content[think_end + len("</think"):].strip()
                return content
            print(f"  LLM error {resp.status_code}: {resp.text[:200]}")
            if resp.status_code < 500:
                break  # client error, retry won't help
        except requests.exceptions.Timeout:
            print(f"  LLM timeout (attempt {attempt+1}/{max_retries}, {timeout}s)")
        except Exception as e:
            print(f"  LLM error: {e}")
    return None


def extract_binary(response):
    """Extract last 1 or 0 from LLM response."""
    if not response:
        return None
    matches = re.findall(r"[10]", response)
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Raw text retriever (for wo HSG ablation)
# ---------------------------------------------------------------------------

class RawTextRetriever:
    """Vector retrieval over raw PDF/transcript/slides text (no HSG graph)."""

    def __init__(self, embed_client, data_root):
        self.embed_client = embed_client
        self.data_root = data_root
        self.sessions_dir = os.path.join(data_root, "sessions")
        self._download_dirs = {}
        self._index_download_dirs()
        self._text_cache = {}

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

    def _get_session_path(self, session_id):
        return os.path.join(self.sessions_dir, session_id)

    def _load_paper_text(self, session_id):
        dl_dir = self._download_dirs.get(session_id)
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

    def _load_transcript(self, session_id):
        path = os.path.join(self._get_session_path(session_id), "transcript_enriched.json")
        if not os.path.exists(path):
            return ""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            segments = data.get("segments", [])
            return "\n".join(s.get("text", "") for s in segments if s.get("text"))
        except Exception:
            return ""

    def _load_slides_text(self, session_id):
        path = os.path.join(self._get_session_path(session_id), "slides_structured.json")
        if not os.path.exists(path):
            return ""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("visual_regions", [])
            return "\n".join(
                r.get("text", r.get("content", ""))
                for r in regions if r.get("text") or r.get("content")
            )
        except Exception:
            return ""

    def _load_all_text(self, session_id):
        if session_id in self._text_cache:
            return self._text_cache[session_id]
        parts = []
        paper = self._load_paper_text(session_id)
        if paper:
            parts.append(f"[Paper]\n{paper}")
        transcript = self._load_transcript(session_id)
        if transcript:
            parts.append(f"[Transcript]\n{transcript}")
        slides = self._load_slides_text(session_id)
        if slides:
            parts.append(f"[Slides]\n{slides}")
        text = "\n\n".join(parts)
        self._text_cache[session_id] = text
        return text

    @staticmethod
    def _chunk(text, chunk_size=500, overlap=100):
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def retrieve(self, session_id, query, top_k=10):
        import numpy as np
        text = self._load_all_text(session_id)
        if not text:
            return []
        chunks = self._chunk(text, chunk_size=500, overlap=100)
        if not chunks:
            return []

        query_emb = self.embed_client.encode(query)
        if query_emb is None:
            return chunks[:top_k]

        scored = []
        for c in chunks:
            c_emb = self.embed_client.encode(c[:4000])
            if c_emb is not None:
                sim = float(np.dot(query_emb, c_emb))
                scored.append((sim, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


# ---------------------------------------------------------------------------
# Decomposer (loads the fine-tuned Qwen3.5-9B LoRA)
# ---------------------------------------------------------------------------

class Decomposer:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, get_peft_model

        # Offline mode: model is cached locally
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        tokenizer_path = model_path
        for p in [model_path, model_path + "/../base_model"]:
            if os.path.exists(os.path.join(p, "tokenizer.json")):
                tokenizer_path = p
                break

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, padding_side="right",
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-9B", torch_dtype="auto",
            device_map="auto", trust_remote_code=True, local_files_only=True,
        )

        # Load LoRA adapter manually to avoid peft's remote file check
        adapter_dir = model_path
        adapter_weights_path = os.path.join(model_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_weights_path):
            adapter_dir = os.path.join(model_path, "final")
            adapter_weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")

        with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
            import json as _json
            lora_cfg = _json.load(f)
        lora_config = LoraConfig(**{k: v for k, v in lora_cfg.items()
                                     if k in LoraConfig.__dataclass_fields__})
        self.model = get_peft_model(base, lora_config)

        from safetensors.torch import load_file
        adapter_weights = load_file(adapter_weights_path)
        self.model.load_state_dict(adapter_weights, strict=False)
        self.model.eval()

        self.system_prompt = (
            "You are a semantic decomposition model for academic paper QA. "
            "Given a question, output JSON with: target_semantic_unit, target_relation, "
            "constraints (time/page/baseline/metric/section/experiment_setting), "
            "modality_requirement, reasoning_pattern, answer_type, answerable. "
            "Output ONLY valid JSON.\n\n"
            "target_relation MUST be one of:\n"
            "- supported_by: default factual support (e.g. 'What is X?')\n"
            "- emphasizes: asks about emphasis/highlights/focus/key points (e.g. 'What does X emphasize?')\n"
            "- compares: asks about comparison/contrast/difference/advantage (e.g. 'How does X compare to Y?')\n"
            "- grounded_in_paper: asks if/how something is backed by the paper (e.g. 'Is X grounded in the paper?')\n"
            "- measured_by: asks about metrics/evaluation/quantitative results (e.g. 'What metric is used?')\n"
            "- aligned_to_slide: asks about slide content/presentation material\n"
            "- referenced_by: asks about citations/references/related work"
        )

    def decompose(self, question):
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        import torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.3,
                do_sample=True, pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        resp = self.tokenizer.decode(gen, skip_special_tokens=True)
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            s, e = resp.find("{"), resp.rfind("}")
            if s >= 0 and e > s:
                try:
                    return json.loads(resp[s:e + 1])
                except Exception:
                    pass
        # Infer relation from question keywords rather than hardcoding supported_by
        q_low = question.lower()
        if any(w in q_low for w in ["compar", "contrast", "differ", "versus", "vs", "advantage", "better"]):
            relation = "compares"
        elif any(w in q_low for w in ["emphas", "highlight", "focus", "key point", "stress"]):
            relation = "emphasizes"
        elif any(w in q_low for w in ["metric", "evaluat", "measure", "quantit", "score", "accuracy", "f1", "bleu"]):
            relation = "measured_by"
        elif any(w in q_low for w in ["grounded", "backed by", "supported by the paper", "evidence in"]):
            relation = "grounded_in_paper"
        elif any(w in q_low for w in ["slide", "presentation", "figure", "chart"]):
            relation = "aligned_to_slide"
        else:
            relation = "supported_by"
        return {
            "target_semantic_unit": "Claim", "target_relation": relation,
            "constraints": {}, "modality_requirement": "paper",
            "reasoning_pattern": "single_hop", "answer_type": "explanation",
            "answerable": True,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, hsg, decomposer, retriever, reranker,
                 gen_api, gen_key, eval_api, eval_key,
                 use_hyperedge=False, use_linearize=False,
                 use_hyperedge_anchor=False, gen_model="Qwen3.6-27B",
                 no_hsg=False, raw_retriever=None, skip_eval=False,
                 rerank_top_k=10):
        self.hsg = hsg
        self.decomposer = decomposer
        self.retriever = retriever
        self.reranker = reranker
        self.gen_api = gen_api
        self.gen_key = gen_key
        self.eval_api = eval_api
        self.eval_key = eval_key
        self.use_hyperedge = use_hyperedge
        self.use_linearize = use_linearize
        self.use_hyperedge_anchor = use_hyperedge_anchor
        self.gen_model = gen_model
        self.no_hsg = no_hsg
        self.raw_retriever = raw_retriever
        self.skip_eval = skip_eval
        self.rerank_top_k = rerank_top_k

    def evaluate_sample(self, sample):
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        gold_decomp = sample.get("decomposition", {})
        gold_subgraph = sample.get("gold_subgraph", {})
        session_id = sample.get("session_id", "")
        is_unanswerable = gold_decomp.get("answerable", True) is False

        t0 = time.time()
        try:
            if self.no_hsg:
                return self._evaluate_no_hsg(sample, question, ground_truth,
                                             session_id, is_unanswerable, t0)

            # 1. Decompose
            print("  [1/5] Decomposing...")
            if self.decomposer is not None:
                decomposition = self.decomposer.decompose(question)
            else:
                decomposition = {}

            # 2. Retrieve (session-scoped)
            print("  [2/5] Retrieving...")
            channels = get_retrieval_channels(question, decomposition)
            raw_hits = self.retriever.retrieve_multi_channel(session_id, channels)

            # 3. Rerank
            print("  [3/5] Reranking...")
            scored = self.reranker.rerank(
                raw_hits, channels, decomposition,
                use_hyperedge=self.use_hyperedge,
            )
            top = scored[:self.rerank_top_k]

            # Hyperedge anchor expansion: pull in cross-modal triples
            if self.use_hyperedge_anchor:
                anchor_nodes = list(set(
                    nid for t, _ in top for nid in (t.head_id, t.tail_id)
                ))
                extra_triples = self.hsg.expand_hyperedge_anchors(
                    anchor_nodes, max_extra_triples=15
                )
                if extra_triples:
                    existing_keys = {(t.head_id, t.relation, t.tail_id) for t, _ in top}
                    for et in extra_triples:
                        key = (et.head_id, et.relation, et.tail_id)
                        if key not in existing_keys:
                            top.append((et, 0.0))
                            existing_keys.add(key)

            subgraph = self.hsg.build_subgraph(
                top, decomposition, use_hyperedge=self.use_hyperedge,
            )

            print(f"  FINAL: {len(top)} triples from {subgraph['metadata']['num_nodes']} nodes")

            # 4. Generate answer
            print("  [4/5] Generating answer...")
            if self.use_linearize:
                ctx_str = self._linearize_subgraph(subgraph)
                gen_prompt = self._graph_aware_prompt(question, ctx_str, decomposition, subgraph)
            elif self.use_hyperedge and subgraph.get("hyperedges"):
                ctx_str = self._build_hyperedge_context(subgraph)
                gen_prompt = self._default_prompt(question, ctx_str)
            else:
                context = [n["content"] for n in subgraph["nodes"]]
                ctx_str = "\n".join(f"Evidence {i+1}: {c[:500]}" for i, c in enumerate(context[:10]))
                gen_prompt = self._default_prompt(question, ctx_str)
            max_out = 2048 if "llava" in self.gen_model.lower() or "7b" in self.gen_model.lower() else 8192
            generated = call_llm(self.gen_api, self.gen_key, self.gen_model,
                                 gen_prompt, max_tokens=max_out, temperature=0.3)
            if not generated:
                generated = "Failed to generate answer"

            # 5. Evaluate
            if self.skip_eval:
                llm_eval = {}
            else:
                print("  [5/5] Evaluating...")
                llm_eval = self._evaluate(question, generated, ground_truth,
                                          ctx_str, is_unanswerable)

            retrieval_metrics = self._retrieval_metrics(gold_subgraph, subgraph, session_id)

            return {
                "sample_id": sample.get("sample_id", ""),
                "session_id": session_id,
                "question": question,
                "ground_truth_answer": ground_truth,
                "generated_answer": generated,
                "our_decomposition": decomposition,
                "gold_decomposition": gold_decomp,
                "retrieval_metrics": retrieval_metrics,
                "llm_evaluation": llm_eval,
                "subgraph_stats": {
                    "num_nodes": subgraph["metadata"]["num_nodes"],
                    "num_edges": subgraph["metadata"]["num_edges"],
                    "score": subgraph["score"],
                },
                "timing": time.time() - t0,
                "error": None,
            }
        except Exception as e:
            return {
                "sample_id": sample.get("sample_id", ""),
                "question": question,
                "error": str(e),
                "traceback": traceback.format_exc()[:500],
                "timing": time.time() - t0,
            }

    def _evaluate_no_hsg(self, sample, question, ground_truth,
                         session_id, is_unanswerable, t0):
        """Evaluate without HSG: decomposer + raw text vector retrieval."""
        # 1. Decompose
        print("  [1/3] Decomposing...")
        decomposition = self.decomposer.decompose(question) if self.decomposer else {}

        # 2. Retrieve from raw text via embedding similarity
        print("  [2/3] Retrieving raw text...")
        query = question
        if decomposition.get("target_semantic_unit"):
            query = f"{question} {decomposition['target_semantic_unit']}"
        chunks = self.raw_retriever.retrieve(session_id, query, top_k=10)
        ctx_str = "\n".join(f"Evidence {i+1}: {c}" for i, c in enumerate(chunks))

        # 3. Generate answer
        print("  [3/3] Generating answer...")
        gen_prompt = self._default_prompt(question, ctx_str)
        generated = call_llm(self.gen_api, self.gen_key, self.gen_model,
                             gen_prompt, max_tokens=8192, temperature=0.3)
        if not generated:
            generated = "Failed to generate answer"

        # 4. Evaluate
        llm_eval = self._evaluate(question, generated, ground_truth,
                                  ctx_str, is_unanswerable)

        return {
            "sample_id": sample.get("sample_id", ""),
            "session_id": session_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "generated_answer": generated,
            "our_decomposition": decomposition,
            "retrieval_metrics": {},
            "llm_evaluation": llm_eval,
            "subgraph_stats": {},
            "timing": time.time() - t0,
            "error": None,
        }

    def _evaluate(self, question, generated, ground_truth, context_str,
                  is_unanswerable):
        if is_unanswerable:
            prompt = (
                f"Does the answer indicate the question cannot be answered?\n\n"
                f"Question: {question}\nGenerated Answer: {generated}\n\n"
                f"Respond 1 if the answer says info is unavailable/cannot be answered. "
                f"0 if it attempts a specific answer.\n\nEvaluation:"
            )
            resp = call_llm(self.eval_api, self.eval_key, "Qwen3.6-27B",
                            prompt, max_tokens=8192, temperature=0.1)
            properly_refused = extract_binary(resp) == "1"
            return {
                "is_correct": properly_refused,
                "properly_refused": properly_refused,
                "is_unanswerable": True,
                "context_supports_answer": None,
            }
        else:
            # Context support: does context support the generated answer?
            cs_prompt = (
                f"Does the context contain information that supports the generated answer?\n\n"
                f"Context: {context_str[:2000]}\n\n"
                f"Generated Answer: {generated}\n\n"
                f"Respond 1 if yes, 0 if no.\n\nEvaluation:"
            )
            cs_resp = call_llm(self.eval_api, self.eval_key, "Qwen3.6-27B",
                               cs_prompt, max_tokens=8192, temperature=0.1)
            context_supports = extract_binary(cs_resp) == "1"

            # Correctness: does generated match ground truth?
            corr_prompt = (
                f"Is the generated answer correct compared to the ground truth?\n\n"
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Generated Answer: {generated}\n\n"
                f"Respond 1 if correct (matches in meaning), 0 if not.\n\nEvaluation:"
            )
            corr_resp = call_llm(self.eval_api, self.eval_key, "Qwen3.6-27B",
                                 corr_prompt, max_tokens=8192, temperature=0.1)
            is_correct = extract_binary(corr_resp) == "1"

            return {
                "is_correct": is_correct,
                "context_supports_answer": context_supports,
                "is_unanswerable": False,
            }

    def _build_hyperedge_context(self, subgraph):
        parts = []
        for he in subgraph.get("hyperedges", [])[:5]:
            parts.append(f"Claim: {he['center_content'][:300]}")
            evidence = he.get("evidence_summary", {})
            mod_info = []
            if evidence.get("paper_chunks", 0):
                mod_info.append(f"{evidence['paper_chunks']} paper passages")
            if evidence.get("slides", 0):
                mod_info.append(f"{evidence['slides']} slides")
            if evidence.get("visual_regions", 0):
                mod_info.append(f"{evidence['visual_regions']} visual regions")
            if evidence.get("utterances", 0):
                mod_info.append(f"{evidence['utterances']} spoken segments")
            if evidence.get("prosody_events", 0):
                mod_info.append(f"{evidence['prosody_events']} emphasis events")
            if mod_info:
                parts.append(f"  Cross-modal evidence: {', '.join(mod_info)}")
            parts.append(f"  Relations: {', '.join(he.get('relations', []))}")
        # Also include node content for completeness
        context = [n["content"] for n in subgraph["nodes"][:10]]
        if context:
            parts.append("\nAdditional evidence:")
            for i, c in enumerate(context[:8]):
                parts.append(f"Evidence {i+1}: {c[:300]}")
        return "\n".join(parts)

    # -- Prompt construction --------------------------------------------------

    def _default_prompt(self, question, context_str):
        return (
            f"You are an expert research assistant answering questions about academic papers.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context_str}\n\n"
            f"Instructions: Answer the question using the evidence above. "
            f"Synthesize information across multiple evidence items when possible. "
            f"Be specific and include concrete details (numbers, method names, dataset names). "
            f"Answer in 2-4 sentences. "
            f"If some details are missing from the evidence, provide the best answer you can "
            f"with what is available rather than refusing to answer.\n\nAnswer:"
        )

    def _graph_aware_prompt(self, question, linearized_graph, decomposition, subgraph):
        """Prompt that leverages graph structure to guide LLM."""
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        type_counts = {}
        for n in nodes:
            t = n.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        type_summary = ", ".join(f"{t} ({c})" for t, c in sorted(type_counts.items()))

        return (
            f"You are an expert research assistant. Below is a knowledge graph extracted from "
            f"an academic presentation. Each node has a type and content; arrows show relations.\n\n"
            f"Question: {question}\n\n"
            f"Knowledge Graph ({len(nodes)} nodes: {type_summary}, {len(edges)} relations):\n"
            f"{linearized_graph}\n\n"
            f"Instructions:\n"
            f"- Trace the relations to find the most relevant evidence chain.\n"
            f"- Focus on the specific claim or fact the question asks about.\n"
            f"- Be precise: include numbers, method names, and dataset names.\n"
            f"- Synthesize across connected nodes when the answer requires multiple pieces.\n"
            f"- Answer in 2-4 sentences.\n\n"
            f"Answer:"
        )

    def _linearize_subgraph(self, subgraph):
        """Linearize subgraph as nodes + relations (Eq.21).

        Node content listed once with full detail; relations reference by ID.
        """
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])

        short_ids = {}
        for i, n in enumerate(nodes):
            short_ids[n["id"]] = f"N{i+1}"

        parts = []
        parts.append("=== Evidence Nodes ===")
        for n in nodes:
            sid = short_ids[n["id"]]
            ntype = n.get("type", "Unknown")
            content = n.get("content", "")
            parts.append(f"[{sid}|{ntype}] {content}")

        if edges:
            parts.append("\n=== Relations ===")
            for e in edges:
                src = short_ids.get(e.get("source", ""), "?")
                tgt = short_ids.get(e.get("target", ""), "?")
                rel = e.get("relation", "related_to")
                parts.append(f"{src} └[{rel}]─> {tgt}")

        return "\n".join(parts)

    def _retrieval_metrics(self, gold_subgraph, our_subgraph, session_id):
        gold_nodes = set(gold_subgraph.get("node_ids", []))
        our_nodes = set()
        for node in our_subgraph.get("nodes", []):
            nid = node.get("id", "")
            if not nid.startswith(session_id.split("/")[0]):
                nid = f"{session_id}/{nid}"
            our_nodes.add(nid)
        overlap = len(gold_nodes & our_nodes)
        prec = overlap / len(our_nodes) if our_nodes else 0
        rec = overlap / len(gold_nodes) if gold_nodes else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return {
            "gold_node_count": len(gold_nodes),
            "retrieved_node_count": len(our_nodes),
            "node_overlap": overlap,
            "node_precision": prec,
            "node_recall": rec,
            "node_f1": f1,
        }

    def evaluate_dataset(self, data_path, max_samples=0, output_path=None):
        print(f"Loading {data_path}...")
        samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if max_samples > 0:
            samples = samples[:max_samples]
        print(f"Evaluating {len(samples)} samples...\n")

        # Resume from existing results
        results = []
        done_ids = set()
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path, encoding="utf-8") as f:
                    old = json.load(f)
                results = old.get("results", [])
                done_ids = {r["sample_id"] for r in results if r.get("sample_id")}
                results = [r for r in results
                           if r.get("generated_answer") != "Failed to generate answer"
                           and not r.get("error")]
                done_ids = {r["sample_id"] for r in results}
                print(f"  Resumed: {len(results)} completed, "
                      f"{len(samples) - len(done_ids)} remaining")
            except Exception:
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
                ret = r.get("retrieval_metrics", {})
                print(f"  Correct: {ev.get('is_correct')} | "
                      f"Ctx: {ev.get('context_supports_answer')} | "
                      f"Time: {r['timing']:.1f}s")
                print(f"  Node Recall: {ret.get('node_recall', 0):.1%} | "
                      f"F1: {ret.get('node_f1', 0):.1%}")

            # Incremental save every 10 samples
            if output_path and len(results) % 10 == 0:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"summary": {}, "results": results},
                              f, indent=2, ensure_ascii=False)

        summary = self._summary(results)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results}, f,
                          indent=2, ensure_ascii=False)

        self._print_summary(summary, output_path)
        return summary

    def _summary(self, results):
        valid = [r for r in results if not r.get("error")]
        ans = [r for r in valid if not r.get("llm_evaluation", {}).get("is_unanswerable")]
        unans = [r for r in valid if r.get("llm_evaluation", {}).get("is_unanswerable")]

        s = {"total_samples": len(results), "error_count": len(results) - len(valid),
             "answerable_count": len(ans), "unanswerable_count": len(unans)}

        ans_eval = [r for r in ans if r.get("llm_evaluation", {}).get("is_correct") is not None]
        if ans_eval:
            correct = sum(1 for r in ans_eval if r["llm_evaluation"]["is_correct"])
            s["answerable_accuracy"] = correct / len(ans_eval)
            s["answerable_correct_count"] = correct
            s["answerable_total"] = len(ans_eval)
            ctx = sum(1 for r in ans_eval if r["llm_evaluation"].get("context_supports_answer"))
            s["context_support_ratio"] = ctx / len(ans_eval)

        unans_eval = [r for r in unans if r.get("llm_evaluation", {}).get("properly_refused") is not None]
        if unans_eval:
            refused = sum(1 for r in unans_eval if r["llm_evaluation"]["properly_refused"])
            s["unanswerable_refusal_rate"] = refused / len(unans_eval)
            s["unanswerable_refused_count"] = refused
            s["unanswerable_total"] = len(unans_eval)

        all_eval = [r for r in valid if r.get("llm_evaluation", {}).get("is_correct") is not None]
        if all_eval:
            s["overall_accuracy"] = sum(1 for r in all_eval if r["llm_evaluation"]["is_correct"]) / len(all_eval)

        recalls = [r.get("retrieval_metrics", {}).get("node_recall", 0) for r in valid]
        f1s = [r.get("retrieval_metrics", {}).get("node_f1", 0) for r in valid]
        s["avg_node_recall"] = sum(recalls) / len(recalls) if recalls else 0
        s["avg_node_f1"] = sum(f1s) / len(f1s) if f1s else 0
        s["avg_time"] = sum(r.get("timing", 0) for r in valid) / len(valid) if valid else 0
        return s

    def _print_summary(self, s, output_path):
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Samples: {s['total_samples']}  "
              f"Answerable: {s.get('answerable_count', 0)}  "
              f"Unanswerable: {s.get('unanswerable_count', 0)}  "
              f"Errors: {s['error_count']}")
        if "answerable_accuracy" in s:
            print(f"Answerable Accuracy: {s['answerable_accuracy']:.1%} "
                  f"({s['answerable_correct_count']}/{s['answerable_total']})")
            print(f"Context Support: {s['context_support_ratio']:.1%}")
        if "unanswerable_refusal_rate" in s:
            print(f"Unanswerable Refusal: {s['unanswerable_refusal_rate']:.1%}")
        if "overall_accuracy" in s:
            print(f"Overall Accuracy: {s['overall_accuracy']:.1%}")
        print(f"Avg Node Recall: {s['avg_node_recall']:.1%}")
        print(f"Avg Node F1: {s['avg_node_f1']:.1%}")
        print(f"Avg Time: {s['avg_time']:.1f}s/sample")
        print(f"Output: {output_path}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Disable proxy for local APIs
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="hsg_output")
    p.add_argument("--embedding-dir", default="dataset/output/embeddings")
    p.add_argument("--decomposer-path", required=True,
                   help="Decomposer LoRA model path")
    p.add_argument("--golden-data", default="dataset/gold_test.jsonl")
    p.add_argument("--output", default="pipeline/eval_results.json")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--gen-api", default="http://localhost:8000/v1")
    p.add_argument("--gen-key", default="your_api_key")
    p.add_argument("--eval-api", default="http://localhost:8000/v1")
    p.add_argument("--eval-key", default="your_api_key")
    p.add_argument("--embed-api", default="http://localhost:8001")
    p.add_argument("--embed-key", default="sk-000")
    p.add_argument("--type-bonus", type=float, default=1.2)
    args = p.parse_args()

    print("=" * 60)
    print("PIPELINE E2E QA EVALUATION")
    print("=" * 60)

    # Load HSG
    hsg = HSG(args.data_root, embedding_dir=args.embedding_dir)

    # Embedding client
    embed_client = BGEEmbeddingClient(args.embed_api, args.embed_key)

    # Decomposer
    print(f"\nLoading decomposer from {args.decomposer_path}...")
    decomposer = Decomposer(args.decomposer_path)

    # Retriever + Reranker
    retriever = SessionScopedRetriever(hsg, embed_client)
    reranker = SemanticReranker(type_bonus=args.type_bonus)

    # Evaluator
    evaluator = Evaluator(hsg, decomposer, retriever, reranker,
                          f"{args.gen_api}/chat/completions", args.gen_key,
                          f"{args.eval_api}/chat/completions", args.eval_key)

    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples, output_path=args.output,
    )


if __name__ == "__main__":
    main()
