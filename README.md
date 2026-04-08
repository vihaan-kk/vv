# vv: Retrieval Benchmark for Legal Clause Extraction

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#setup)
[![Run](https://img.shields.io/badge/run-python%20run.py-success)](#quick-start)
[![HPC](https://img.shields.io/badge/cluster-Longleaf%20l40--gpu-6f42c1)](#option-b-longleaf-gpu-workflow-for-full-hichunk-stack)

`vv` is a reproducible benchmark that compares retrieval strategies for LLM analysis of legal text.

It runs the same model prompt against multiple context-construction methods and measures:

- response quality versus a ground-truth pass
- latency
- token usage
- retrieval coverage and hallucination behavior

The benchmark target document is `nda_section.md` (NDA Section 8).

## TL;DR

- Goal: compare retrieval strategies (flat vs semantic vs hierarchical) on the same legal extraction task.
- Entry point: `run.py`.
- Output: `results.json` + terminal summary table.
- Local run: `BACKEND=ollama python run.py`.
- Longleaf run: `sbatch submit.sl`.

## Quick start

```bash
# Local
pip install requests transformers torch langchain-experimental langchain-huggingface sentence-transformers
BACKEND=ollama python run.py

# HPC
sbatch submit.sl
```

## Why this exists

Most "RAG quality" discussions mix retrieval and generation errors together. This project separates them:

- Retrieval quality: did the retriever surface enough source text?
- Extraction quality: given that text, did the LLM answer correctly?

`run.py` evaluates both with field-level scoring against a ground-truth run.

## Repository layout

- `run.py` — benchmark driver, chunking/retrieval logic, model invocation, scoring, and summary output
- `nda_section.md` — fixed legal source text under test
- `prompt.md` — fixed extraction prompt template (`{context}` placeholder)
- `results.json` — machine-readable benchmark output
- `submit.sl` — Slurm job script for Longleaf GPU execution
- `CLAUDE.md` — project notes/instructions for coding-agent workflows

## Benchmark runs

The script currently executes:

1. `GROUND TRUTH — Full Section 8`
   - passes full source text into context
   - acts as reference output for comparison

2. `RUN 2 — BASELINE (flat chunking, top 2 chunks)`
   - fixed-size word chunks (approx 500 tokens/chunk)
   - retrieves first 2 chunks only (naive baseline)

3. `RUN 3 — SEMANTIC CHUNKING (topic-shift, top 2 chunks)`
   - LangChain `SemanticChunker` with MiniLM embeddings
   - still retrieves top 2 chunks only

4. `RUN 4 — HICHUNK + AUTO-MERGE (hierarchical, top 5)`
   - uses hierarchical chunk representation and auto-merge logic
   - retrieval embeds leaf chunks (`BAAI/bge-m3`), retrieves top-k, then merges siblings toward parent context
   - if full HiChunk repo is unavailable, `run.py` falls back to a structure-aware chunker for local compatibility

### Run matrix (at a glance)

| Run | Strategy | Retrieval depth | Expected behavior |
| --- | --- | --- | --- |
| Ground Truth | Full-context pass | Entire document | Reference output |
| Run 2 | Flat chunking | Top 2 fixed chunks | Misses deep sub-clauses |
| Run 3 | Semantic chunking | Top 2 semantic chunks | Better topical grouping, still flat |
| Run 4 | Hierarchical + auto-merge | Top 5 leaf chunks + parent promotion | Better structural coverage |

## Backends

`run.py` supports two LLM backends:

- `BACKEND=ollama`
  - local inference via Ollama API
  - default model in code: `qwen2.5:7b`

- default (`BACKEND` unset): Hugging Face
  - `Qwen/Qwen2.5-7B-Instruct` via `transformers.pipeline`
  - prefers CUDA when available

## Setup

### Option A: Local Mac workflow (typical development path)

1. Install Python dependencies:

```bash
pip install requests transformers torch langchain-experimental langchain-huggingface sentence-transformers
```

2. Install and run Ollama (if using local backend):

```bash
ollama pull qwen2.5:7b
```

3. Run benchmark:

```bash
BACKEND=ollama python run.py
```

Notes:

- RUN 1-3 work in this mode.
- RUN 4 executes if required embedding dependencies are available; behavior depends on available HiChunk assets.

### Option B: Longleaf GPU workflow (for full HiChunk stack)

`submit.sl` is configured for the `l40-gpu` partition and expects:

- `module load anaconda`
- `conda activate hichunk`
- project cloned at `~/vv_bench`

Run:

```bash
sbatch submit.sl
```

## Output and metrics

Results are written to `results.json` as a list of run objects. Each run includes:

- `run`
- `latency_seconds`
- `tokens_prompt`
- `tokens_response`
- `context_used`
- `output`

For non-ground-truth runs, a `comparison` object is also included:

- `exact_score` — strict string match score
- `llm_score` — semantic score from LLM judge
- `retrieval_adjusted_score` — score on answerable fields only
- `retrieval_coverage` — fraction of fields answerable from retrieved context
- `hallucinations`
- `retrieval_gaps`
- detailed field-level diagnostics under `details`

The terminal also prints a compact summary table:

- latency
- semantic score
- retrieval-adjusted score
- coverage
- hallucination count

## How scoring works

`compare_to_ground_truth()` parses structured field outputs and compares each run to run 1.

The judge prompt in `run.py` enforces a two-step evaluation:

1. Determine whether the needed information existed in the provided context.
2. Score extraction quality only after checking retrieval sufficiency.

This avoids penalizing a model for missing data it never received, and flags hallucinations when it invents unsupported facts.

## Reproducibility notes

- `prompt.md` and `nda_section.md` are benchmark fixtures; keep them stable when comparing retrieval strategies.
- Summary trends are more informative than one-off run variance.
- For fair comparisons, keep backend/model, prompt, and source document constant.

## What success looks like

- RUN 4 should generally improve coverage vs RUN 3 on deeply nested clauses.
- `retrieval_coverage` should increase when hierarchical retrieval is working.
- `retrieval_adjusted_score` helps separate extraction mistakes from retrieval misses.
- Hallucinations should remain low even as context breadth increases.

## Customization points

Safe modifications for experiments:

- retrieval parameters (`top_k`, chunk settings)
- retrieval algorithms/chunkers
- backend model choice

Avoid modifying while benchmarking strategy quality:

- `run()` interface
- scoring logic in `compare_to_ground_truth()`
- fixture semantics in `prompt.md` and `nda_section.md`

## Troubleshooting

- `Skipping RUN 4 — sentence-transformers not installed`
  - install `sentence-transformers` and related deps

- slow or failed Hugging Face model load
  - use `BACKEND=ollama` locally, or run on GPU environment

- unexpected retrieval quality regressions
  - inspect `context_used` in `results.json`
  - verify `top_k`, merge behavior, and chunk boundaries

- no summary differences across runs
  - ensure chunking/retrieval settings actually differ per run
