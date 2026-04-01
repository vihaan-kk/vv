import re
import json
import time
import os
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results.json")
NDA_SECTION = Path(os.path.join(os.path.dirname(__file__), "nda_section.md")).read_text()
PROMPT = Path(os.path.join(os.path.dirname(__file__), "prompt.md")).read_text()

# --- BACKEND SELECTION ---
# set BACKEND=ollama on your Mac, leave unset on Longleaf
# on Mac: BACKEND=ollama python run.py
# on Longleaf: python run.py
BACKEND = os.environ.get("BACKEND", "huggingface")

if BACKEND == "ollama":
    import requests
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen2.5:7b"
    print(f"Using Ollama backend ({OLLAMA_MODEL})")
else:
    from transformers import pipeline
    import torch
    HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using HuggingFace backend ({HF_MODEL}) on {DEVICE}")
    print("Loading model...")
    pipe = pipeline(
        "text-generation",
        model=HF_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded.")

# --- DETERMINISTIC COMPARISON ---
# parses the structured output into {field: value} pairs and compares
# each run against the ground truth line-by-line

def parse_output(output):
    """Extract '- Field: Value' lines into a dict."""
    fields = {}
    for line in output.split("\n"):
        match = re.match(r'\s*-\s+(.+?):\s*(.*)', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            fields[key] = value
    return fields

def compare_to_ground_truth(ground_truth, run_result):
    """Compare a run's parsed output against the ground truth field by field."""
    gt_fields = parse_output(ground_truth["output"])
    run_fields = parse_output(run_result["output"])

    missing = []     # GT has a value, run has [NOT FOUND]
    mismatches = []  # Both have values but they differ
    exact_matches = 0

    for field, gt_value in gt_fields.items():
        run_value = run_fields.get(field, "[NOT FOUND]")
        gt_not_found = gt_value == "[NOT FOUND]"
        run_not_found = run_value == "[NOT FOUND]"

        if gt_not_found:
            continue  # GT itself didn't find this — not a fair comparison point
        elif run_not_found:
            missing.append({"field": field, "expected": gt_value})
        elif gt_value.lower().strip() == run_value.lower().strip():
            exact_matches += 1
        else:
            mismatches.append({"field": field, "expected": gt_value, "actual": run_value})

    gt_fields_with_value = sum(1 for v in gt_fields.values() if v != "[NOT FOUND]")
    score = round(exact_matches / gt_fields_with_value, 3) if gt_fields_with_value > 0 else 0

    return {
        "compared_to": ground_truth["run"],
        "gt_fields_with_value": gt_fields_with_value,
        "exact_matches": exact_matches,
        "missing_in_run": len(missing),
        "mismatches": len(mismatches),
        "score": score,
        "details": {
            "missing": missing,
            "mismatches": mismatches
        }
    }

# --- RUN FUNCTION ---
# this function takes a label (what to call the run) and a context (what to
# inject into the prompt), sends it to the model, and returns a results dict
def run(label, context):
    print(f"\nRunning: {label}...")

    # swap the {context} placeholder with the actual text for this run
    prompt = PROMPT.replace("{context}", context)

    # record the time before the request
    start = time.time()

    if BACKEND == "ollama":
        # send the request to ollama
        # json= means we're sending a python dict as json in the request body
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,       # wait for the full response before returning
            "options": {
                "temperature": 0.1,  # low temperature = more deterministic output
                "num_ctx": 8192      # context window size in tokens
            }
        })

        # parse the response body from json into a python dict
        data = response.json()

        # extract the fields we care about
        # .get() means "return this key if it exists, otherwise return the default"
        output = data.get("response", "")
        tokens_prompt = data.get("prompt_eval_count", "N/A")
        tokens_response = data.get("eval_count", "N/A")

    else:
        result = pipe(
            prompt,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        # strip the input prompt from the output — huggingface returns the full text
        output = result[0]["generated_text"][len(prompt):]
        tokens_prompt = len(pipe.tokenizer.encode(prompt))
        tokens_response = len(pipe.tokenizer.encode(output))

    # calculate how many seconds the run took
    elapsed = round(time.time() - start, 2)

    # print a quick summary to terminal so you can see progress
    print(f"  Done. Latency: {elapsed}s | Prompt tokens: {tokens_prompt} | Response tokens: {tokens_response}")

    # return everything as a dict — this gets saved to json later
    return {
        "run": label,
        "latency_seconds": elapsed,
        "tokens_prompt": tokens_prompt,
        "tokens_response": tokens_response,
        "context_used": context,
        "output": output
    }

# --- FLAT CHUNKER ---
# this simulates what a basic AI tool does today
# it splits the document into fixed-size blocks, ignoring section boundaries
# max_tokens is approximate — we split by words, ~0.75 words per token
def flat_chunk(text, max_tokens=500):
    words = text.split()                        # split full text into individual words
    words_per_chunk = int(max_tokens * 0.75)    # convert token limit to word limit
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])  # join each group of words back into a string
        chunks.append(chunk)
    return chunks

# --- FAKE RETRIEVER ---
# in a real system, a vector database would find the most relevant chunks
# here we simulate that by just grabbing the first 2 chunks
# this is intentionally naive — it represents the worst case of flat chunking
# in practice a real retriever might do slightly better, but will still miss
# sub-clauses that got split across chunk boundaries
def fake_retrieve(chunks, top_k=2):
    return chunks[:top_k]   # return only the first top_k chunks

def semantic_chunk(text):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "mps" if BACKEND == "ollama" else DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )

    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]

if __name__ == "__main__":

    # run the ground truth pass — full section 8 goes directly into context
    # this confirms the prompt works before we test retrieval
    results = []
    results.append(run("GROUND TRUTH — Full Section 8", NDA_SECTION))

    # -------------------------------------------------------
    # RUN 2 — BASELINE (flat chunking)
    # simulates what chatgpt / a naive tool does today
    # splits the document into fixed ~500 token blocks
    # retrieves only the first 2 chunks — ignores the rest
    # this is where [NOT FOUND] will appear on deep sub-clauses
    # -------------------------------------------------------
    chunks = flat_chunk(NDA_SECTION, max_tokens=500)
    retrieved = fake_retrieve(chunks, top_k=2)

    # join the retrieved chunks with a separator so the model
    # can see where one chunk ends and the next begins
    baseline_context = "\n\n---\n\n".join(retrieved)

    results.append(run(
        "RUN 2 — BASELINE (flat chunking, top 2 chunks)",
        baseline_context
    ))

    # -------------------------------------------------------
    # RUN 3 — SEMANTIC CHUNKING
    # uses embedding similarity to detect topic shifts and split there
    # smarter than flat chunking but still flat — no hierarchy awareness
    # this represents the current best practice in most AI tools today
    # -------------------------------------------------------
    semantic_chunks = semantic_chunk(NDA_SECTION)
    semantic_retrieved = fake_retrieve(semantic_chunks, top_k=2)
    semantic_context = "\n\n---\n\n".join(semantic_retrieved)

    results.append(run(
        "RUN 3 — SEMANTIC CHUNKING (topic-shift, top 2 chunks)",
        semantic_context
    ))

    # NOTE: RUN 4 — RAG (HiChunk + Auto-Merge) comes later
    # once your friend's tool is available (or you build it from the repo)
    # it will slot in here between runs 2 and 3

    # --- DETERMINISTIC COMPARISON AGAINST GROUND TRUTH ---
    ground_truth = results[0]
    for r in results[1:]:
        r["comparison"] = compare_to_ground_truth(ground_truth, r)

    # write all results to a json file in the same folder as this script
    # "w" means write mode — creates the file if it doesn't exist
    # indent=2 makes the json human-readable with 2-space indentation
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # -------------------------------------------------------
    # QUICK COMPARISON SUMMARY
    # prints a compact table to terminal so you don't have to
    # open the json file to see the headline numbers
    # -------------------------------------------------------
    print("\n--- SUMMARY ---")
    print(f"{'Run':<45} {'Latency':>10} {'Prompt Tokens':>15} {'Response Tokens':>17} {'Score vs GT':>13} {'Missing':>9} {'Mismatch':>10}")
    print("-" * 122)
    for r in results:
        cmp = r.get("comparison", {})
        score    = f"{cmp['score']:.1%}"       if cmp else "— (baseline)"
        missing  = str(cmp["missing_in_run"])  if cmp else "—"
        mismatch = str(cmp["mismatches"])      if cmp else "—"
        print(f"{r['run']:<45} {str(r['latency_seconds']) + 's':>10} {str(r['tokens_prompt']):>15} {str(r['tokens_response']):>17} {score:>13} {missing:>9} {mismatch:>10}")
