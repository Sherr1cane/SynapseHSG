# Baseline Reproduction

This directory contains evaluation scripts for baseline methods compared in the SynapseHSG paper.
Each script is self-contained except for the third-party baseline libraries that must be installed separately.

## Prerequisites

All baselines require:
- An OpenAI-compatible LLM API endpoint (e.g., vLLM serving Qwen3.6-27B)
- A BGE-M3 embedding API endpoint
- The processed session data (`--data-root` pointing to the build directory)

## Baselines

### 1. Native LLM (no RAG)

**Script**: `run_native_baseline.py`

Feeds raw multimodal content (paper text, slide images, transcript) directly to the LLM without retrieval.

```bash
python run_native_baseline.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions \
    --model Qwen3.6-27B
```

No third-party library required.

### 2. M2RAG

**Script**: `run_m2rag.py`

Multi-modal retrieval via BGE-M3 embedding + VLM generation.

**Source**: https://github.com/maziao/M2RAG

**Modifications**: Our script uses BGE-M3 for both text and image retrieval (session-scoped), then feeds results to a VLM generator. The original M2RAG retrieves image captions from webpages; we adapt it for local document images.

```bash
python run_m2rag.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions \
    --model Qwen3.6-27B
```

No third-party library required (uses direct embedding API calls).

### 3. GraphRAG

**Script**: `run_graphrag.py`

Microsoft GraphRAG: KG construction + community detection + local search.

**Source**: https://github.com/microsoft/graphrag (pip install graphrag)

**Additional dependency**: `graphrag-llm` — a wrapper that configures GraphRAG to use custom LLM endpoints. Clone and install:
```bash
pip install graphrag
# graphrag-llm must be available on PYTHONPATH
```

**Modifications**: We added multimodal image retrieval on top of GraphRAG's text-only local search results.

```bash
python run_graphrag.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions \
    --build-index
```

### 4. EventRAG

**Script**: `run_eventrag.py`

Event-centric KG + iterative retrieval.

**Source**: https://github.com/Ryaang/EventRAG

**Setup**: Clone the repo as a sibling directory or adjust `sys.path` in the script:
```bash
git clone https://github.com/Ryaang/EventRAG.git
# The script adds EventRAG/ to sys.path at runtime
```

**Modifications**: We adapted EventRAG for session-scoped evaluation on academic documents. Original is designed for generic text.

```bash
python run_eventrag.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions \
    --build-index
```

### 5. RAG-Anything

**Script**: `run_rag_anything.py`

LightRAG-based multimodal RAG.

**Source**: https://github.com/HKUDS/RAG-Anything

**Setup**: Install LightRAG:
```bash
pip install lightrag-hku
```

**Modifications**: We use per-session LightRAG instances for academic document QA instead of the original web-based setup.

```bash
python run_rag_anything.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions
```

### 6. VisRAG

**Script**: `run_visrag.py`

Vision-based document retrieval using VisRAG-Ret encoder.

**Source**: https://github.com/openbmb/VisRAG

**Setup**: Download VisRAG-Ret model weights:
```bash
git clone https://github.com/openbmb/VisRAG.git
# Download VisRAG-Ret weights to baseline/visrag_model/ or specify via --visrag-model
```

**Modifications**: We encode paper pages + slides with VisRAG-Ret for retrieval, then feed top-k images to Qwen3.6-27B for answer generation.

```bash
python run_visrag.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions \
    --visrag-model /path/to/VisRAG-Ret \
    --build-index
```

### 7. KGGen

**Script**: `run_kkgen.py`

Knowledge graph generation from text via LLM prompting + embedding retrieval.

No third-party library required. KG extraction is done via direct LLM calls following the KGGen paper approach.

```bash
python run_kkgen.py \
    --data-root /path/to/build \
    --golden-data /path/to/gold_test.jsonl \
    --api http://localhost:1088/v1/chat/completions \
    --model Qwen3.6-27B
```

## Common CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `hsg_output` | Path to processed session data |
| `--golden-data` | `dataset/gold_test.jsonl` | Path to QA test set |
| `--output` | auto-generated | Output JSON path |
| `--max-samples` | 0 (all) | Limit number of samples |
| `--api` | (varies) | LLM API endpoint URL |
| `--model` | `Qwen3.6-27B` | LLM model name |
| `--build-index` | flag | Build index from scratch (first run) |

## Data Format

Each script expects:
- `--data-root/{conference}/{session_id}_*/` directories containing `paper_chunks.jsonl`, `slides/`, `utterances.jsonl`, etc.
- `--golden-data` JSONL with fields: `session_id`, `question`, `answer`, `answerable`
