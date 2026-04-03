# CLAUDE.md — Context Layer Benchmark (vv)

## What This Project Is

A demonstration that structured context retrieval produces measurably better output than naive chunking. Three runs on the same document, same prompt, same LLM. Only the context strategy changes.

**The headline claim**: flat chunking and semantic chunking produce `[NOT FOUND]` on deep sub-clauses that HiChunk + Auto-Merge correctly surfaces.

This is not a product. It is a benchmark script that produces a `results.json` comparing run quality.

---

## Current State

All files are in `/Users/vihaankerekatte/Developer/Optimization/vv/` (local) and `~/vv_bench/` (Longleaf HPC).

```
run.py          — main script, all logic lives here
nda_section.md  — NDA Section 8, the document under test
prompt.md       — the prompt template ({context} placeholder)
results.json    — output from the last run
submit.sl       — Slurm job script for Longleaf (l40-gpu partition)
```

Three runs are already implemented in `run.py`:

| Run label | What it does |
|---|---|
| `GROUND TRUTH — Full Section 8` | Full doc in context. The reference. |
| `RUN 2 — BASELINE (flat chunking, top 2 chunks)` | 500-token word-split chunks, take first 2. |
| `RUN 3 — SEMANTIC CHUNKING (topic-shift, top 2 chunks)` | LangChain SemanticChunker with MiniLM embeddings. |

After Run 3 there is a comment `# NOTE: RUN 4 — RAG (HiChunk + Auto-Merge) comes later`. **This is where your work goes.**

The `run(label, context)` function is the only interface you need to call. Pass it a label string and a context string (the assembled retrieved text), and it handles LLM dispatch, timing, and token counting.

---

## What to Add: RUN 4 — HiChunk + Auto-Merge Retrieval

### Goal

Replace flat/semantic chunking with **hierarchical chunking** using HiChunk, then use **Auto-Merge retrieval** to assemble the context. Pass the result into the existing `run()` function. Log the same metrics as other runs.

### HiChunk Overview

Repo: https://github.com/TencentCloudADP/hichunk

HiChunk uses a fine-tuned LLaMA-based model to identify hierarchical boundaries in text (not just topic shifts). It produces a tree of chunks: document → sections → subsections → clauses. 

Auto-Merge retrieval: embed leaf nodes → retrieve top-k leaves by cosine similarity → if enough sibling leaves are retrieved, merge up to the parent node → send the merged (more complete) context to the LLM.

The key advantage over semantic chunking: it respects the document's own hierarchy (8.3(a)(i), 8.3(a)(ii), etc.) rather than splitting on embedding distance. Sub-clauses stay with their parent section.

### Workflow

**Claude Code runs on Mac and edits project files only.** It does not execute anything on Longleaf.

At the end of the session, Claude Code must output a clearly labeled block of shell commands for the user to run manually on Longleaf. The user will then `git pull` the updated project files and run those commands.

Format the handoff block like this at the end of your response:

```
--- LONGLEAF SETUP COMMANDS (run these manually) ---
<commands here>
--- END ---
```

### Longleaf Setup (user runs these manually — HiChunk not yet installed)

HiChunk environment and model weights are not yet set up on Longleaf. The handoff block must include:

```bash
module load anaconda

# Clone HiChunk
git clone https://github.com/TencentCloudADP/hichunk.git ~/hichunk
cd ~/hichunk

# Create conda env (use their environment.yml if it exists, otherwise install manually)
conda env create -f environment.yml   # or: conda create -n hichunk python=3.10 -y
conda activate hichunk

# Install deps if no environment.yml
pip install torch transformers sentence-transformers chromadb

# Download bge-m3 embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Pull updated project files
cd ~/vv_bench
git pull
```

The HiChunk chunking model is a fine-tuned LLaMA checkpoint. Check the repo README for the HuggingFace model ID or download URL. Include the download command in the handoff block once you know it. If you cannot determine it without running code, note it as a `# TODO: download model weights per HiChunk README` placeholder in the handoff block.

### Integration Plan

Add the following to `run.py` in order:

**1. Import block** (top of file, guarded by a `HICHUNK_AVAILABLE` flag):

```python
try:
    import sys
    sys.path.insert(0, os.path.expanduser("~/hichunk"))
    from hichunk import HiChunker          # adjust import to match actual module name
    from retrieval_algo import auto_merge  # adjust to match actual function name
    from sentence_transformers import SentenceTransformer
    import chromadb
    HICHUNK_AVAILABLE = True
except ImportError:
    HICHUNK_AVAILABLE = False
    print("HiChunk not available — skipping RUN 4")
```

**2. `hichunk_chunk(text)` function**:

```python
def hichunk_chunk(text):
    """
    Use HiChunk to produce a hierarchical chunk tree from the document.
    Returns the raw chunk tree object (structure depends on HiChunk's API).
    """
    chunker = HiChunker(model_path="~/hichunk/models/...")  # adjust path
    return chunker.chunk(text)
```

**3. `hichunk_retrieve(chunk_tree, query, top_k=3)` function**:

```python
def hichunk_retrieve(chunk_tree, query, top_k=3):
    """
    Embed leaf nodes with bge-m3, retrieve top_k by cosine similarity,
    run Auto-Merge to promote to parent nodes where siblings are present,
    return assembled context string.
    """
    embed_model = SentenceTransformer("BAAI/bge-m3")
    
    # Extract leaf nodes from the chunk tree
    # (structure depends on HiChunk output — read retrieval_algo.py to confirm)
    leaves = chunk_tree.get_leaves()  # adjust to actual API
    
    # Embed leaves and store in Chroma
    client = chromadb.Client()
    collection = client.create_collection("nda_hichunk")
    for i, leaf in enumerate(leaves):
        embedding = embed_model.encode(leaf.text).tolist()
        collection.add(documents=[leaf.text], embeddings=[embedding], ids=[str(i)])
    
    # Query
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_ids = [int(id) for id in results["ids"][0]]
    retrieved_leaves = [leaves[i] for i in retrieved_ids]
    
    # Auto-Merge: promote to parent if siblings retrieved
    merged_nodes = auto_merge(chunk_tree, retrieved_leaves)  # adjust to actual API
    
    # Assemble context string
    return "\n\n---\n\n".join(node.text for node in merged_nodes)
```

**4. RUN 4 block** (slot in after Run 3, before the comparison block):

```python
# -------------------------------------------------------
# RUN 4 — HICHUNK + AUTO-MERGE RETRIEVAL
# hierarchical chunking respects the document's own structure
# auto-merge promotes sibling leaf nodes to parent context
# this is what should correctly surface deep sub-clauses
# -------------------------------------------------------
if HICHUNK_AVAILABLE:
    QUERY = "What are all the conditions under which this agreement can be terminated?"
    chunk_tree = hichunk_chunk(NDA_SECTION)
    hichunk_context = hichunk_retrieve(chunk_tree, QUERY, top_k=3)
    results.append(run(
        "RUN 4 — HICHUNK + AUTO-MERGE (hierarchical, top 3)",
        hichunk_context
    ))
else:
    print("\nSkipping RUN 4 — HiChunk not installed")
```

### What to Read in the HiChunk Repo

Before writing any code, read these files in the repo:

- `README.md` — setup instructions, model download, basic usage
- `hichunk.py` (or equivalent) — the `HiChunker` class interface: what it takes as input, what it returns
- `retrieval_algo.py` — the `auto_merge` function: what it expects (chunk tree + retrieved leaves), what it returns
- `requirements.txt` or `environment.yml` — exact dependencies

The integration code above uses placeholder function names. **Match the actual API from the repo.** Do not guess — read the source before writing the wrapper.

### Environment Notes

- Longleaf partition: `l40-gpu` (single A40 GPU, 16GB RAM allocated in submit.sl)
- The HiChunk LLaMA model needs GPU. Do not attempt CPU inference on Longleaf — it will time out.
- `submit.sl` needs to activate the hichunk conda env before `python run.py`. Update it to add `conda activate hichunk` after `module load anaconda`.
- Local Mac uses `BACKEND=ollama`. HiChunk is Longleaf-only for now. The `HICHUNK_AVAILABLE` flag handles this gracefully — runs 1–3 still execute on Mac.

### Metrics to Verify

After running RUN 4, confirm `results.json` contains:
- `latency_seconds`
- `tokens_prompt`
- `tokens_response`
- `output` — the structured termination analysis
- `comparison` — auto-computed score against ground truth (the comparison block at the bottom of `run.py` handles this automatically for any run after index 0)

### Success Criteria

RUN 4 should score higher than RUN 3 (semantic chunking, score: 0.190) and comparable to RUN 2 (flat chunking, score: 0.714). The specific fields that RUN 3 missed — all of 8.3 sub-clauses, all of 8.4, all of 8.5, all of 8.6 — should be present in RUN 4's output because Auto-Merge will pull the parent section into context when sibling leaves are retrieved.

If RUN 4 scores lower than RUN 2, something is wrong with the retrieval (probably not enough top_k, or Auto-Merge is not merging correctly). Increase `top_k` to 5 and re-run before investigating further.

---

## Do Not Touch

- `nda_section.md` — the document is fixed
- `prompt.md` — the prompt is locked
- The `run()` function interface
- The `compare_to_ground_truth()` scoring logic
- The summary table at the bottom of `run.py`

---

## Running

```bash
# Longleaf
sbatch submit.sl

# Mac (runs 1–3 only, skips HiChunk)
BACKEND=ollama python run.py
```

Results write to `results.json` in the same directory.
