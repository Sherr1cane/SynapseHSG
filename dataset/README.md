# MM-ConfQA Benchmark

MM-ConfQA is a multimodal conference question-answering benchmark for evaluating structured reasoning over academic presentations.

## Overview

Each sample contains a question about an academic presentation (paper + slides + talk audio), paired with a gold answer and metadata. Questions span 7 reasoning patterns and include both answerable and unanswerable queries.

## Statistics

| Split | QA Pairs | Sessions | Answerable | Unanswerable |
|-------|----------|----------|------------|--------------|
| Train | 10,973 | 1,203 | 9,818 | 1,155 |
| Dev | 400 | 133 | 352 | 48 |
| Test | 500 | 139 | 440 | 60 |
| **Total** | **11,873** | **1,475** | **10,610** | **1,263** |

### Reasoning Patterns

| Pattern | Count | Description |
|---------|-------|-------------|
| evidence_linking | 4,254 | Link claims to supporting evidence across modalities |
| cross_slide | 2,344 | Synthesize information across multiple slides |
| multi_hop_aggregation | 2,326 | Aggregate information through multi-hop reasoning |
| unanswerable | 1,263 | Questions that cannot be answered from the presentation |
| comparison | 886 | Compare methods, results, or metrics |
| limitation_or_caveat | 415 | Identify limitations, caveats, or assumptions |
| single_hop | 383 | Direct single-step lookups |


## Data Format

Each line in the JSONL files is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Conference/session identifier (e.g., `iclr-2025/39035345`) |
| `question` | string | The question text |
| `answer` | string | Gold answer (empty string for unanswerable questions) |
| `answerable` | bool | Whether the question can be answered from the presentation |
| `split` | string | Data split (`train`, `dev`, or `test`) |
| `source_type` | string | Source modality that grounds the answer |
| `reasoning_pattern` | string | One of the 7 patterns above |

## Full Data

The release includes QA pairs (questions, answers, and metadata). The raw multimodal session data (paper PDFs, slide images, audio recordings) can be obtained from SlidesLive (https://slideslive.com).

## License

This dataset is released for research purposes only.
