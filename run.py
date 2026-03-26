import requests
import json
import time
import os
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results.json")


NDA_SECTION = Path(os.path.join(os.path.dirname(__file__), "nda_section.md")).read_text()
PROMPT = Path(os.path.join(os.path.dirname(__file__), "prompt.md")).read_text()

# --- RUN FUNCTION ---
# this function takes a label (what to call the run) and a context (what to
# inject into the prompt), sends it to ollama, and returns a results dict
def run(label, context):
    print(f"\nRunning: {label}...")

    # swap the {context} placeholder with the actual text for this run
    prompt = PROMPT.replace("{context}", context)

    # record the time before the request
    start = time.time()

    # send the request to ollama
    # json= means we're sending a python dict as json in the request body
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,       # wait for the full response before returning
        "options": {
            "temperature": 0.1,  # low temperature = more deterministic output
            "num_ctx": 8192      # context window size in tokens
        }
    })

    # calculate how many seconds the run took
    elapsed = round(time.time() - start, 2)

    # parse the response body from json into a python dict
    data = response.json()

    # extract the fields we care about
    # .get() means "return this key if it exists, otherwise return the default"
    output = data.get("response", "")
    tokens_prompt = data.get("prompt_eval_count", "N/A")
    tokens_response = data.get("eval_count", "N/A")

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
        model_kwargs={"device": "mps"},          # M4 Metal acceleration
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

    # RUN 2.5 — SEMANTIC CHUNKING
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

    # -------------------------------------------------------
    # RUN 3 — CAG (cache augmented generation)
    # the full document goes into context — no retrieval step
    # in production this would be preloaded into the kv cache once
    # and reused across multiple queries in the same session
    # for the prototype we simulate it by just passing the full text
    # -------------------------------------------------------
    # results.append(run(
    #     "RUN 3 — CAG (full document in context)",
    #     NDA_SECTION
    # ))

    # NOTE: RUN 4 — RAG (HiChunk + Auto-Merge) comes later
    # once your friend's tool is available (or you build it from the repo)
    # it will slot in here between runs 2 and 3

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
    print(f"{'Run':<45} {'Latency':>10} {'Prompt Tokens':>15} {'Response Tokens':>17}")
    print("-" * 90)
    for r in results:
        print(f"{r['run']:<45} {str(r['latency_seconds']) + 's':>10} {str(r['tokens_prompt']):>15} {str(r['tokens_response']):>17}")