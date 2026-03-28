# vv

A benchmarking tool that compares different document retrieval strategies for LLM question-answering over legal text. It tests how well each strategy extracts structured information from an NDA section, running the same prompt against different context-building approaches and recording latency and token usage.

## What it tests

- **Ground truth** — full document passed directly into context
- **Flat chunking** — naive fixed-size (~500 token) word splits, first 2 chunks retrieved
- **Semantic chunking** — embedding-based topic-shift splitting (via `sentence-transformers/all-MiniLM-L6-v2`), first 2 chunks retrieved

Results are saved to `results.json` with latency, token counts, and full model output for each run.

## Requirements

- [Ollama](https://ollama.com) running locally with `qwen3:8b` pulled
- Python 3.10+
- An Apple Silicon Mac (MPS acceleration used for embeddings)

```bash
pip install requests langchain-experimental langchain-huggingface sentence-transformers
ollama pull qwen3:8b
```

## Running

```bash
python run.py
```

Results are printed as a summary table in the terminal and written to `results.json`.

## Customizing

- Edit `prompt.md` to change the question asked of the model
- Edit `nda_section.md` to swap in a different document section
