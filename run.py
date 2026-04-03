import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import json
import time
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# --- HICHUNK AVAILABILITY CHECK ---
# HiChunk requires: sentence-transformers, numpy, chromadb for retrieval
# The full HiChunk model (LLaMA-based) requires vLLM + GPU on Longleaf
# On Mac, we use a structure-aware fallback chunker with the same retrieval logic
HICHUNK_AVAILABLE = False
HICHUNK_FULL = False  # True if actual HiChunk repo is available
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    # chromadb is optional - we use numpy for similarity search
    HICHUNK_AVAILABLE = True
    # Check if full HiChunk repo is available (Longleaf only)
    HICHUNK_REPO = os.path.expanduser("~/HiChunk")
    if os.path.exists(HICHUNK_REPO):
        import sys
        sys.path.insert(0, HICHUNK_REPO)
        HICHUNK_FULL = True
        print(f"HiChunk repo found at {HICHUNK_REPO}")
except ImportError as e:
    print(f"HiChunk dependencies not available ({e}) — skipping RUN 4")

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


# --- LLM-AS-JUDGE ---
# Uses the same LLM to evaluate semantic equivalence between expected/actual answers
# More accurate than string matching — handles formatting differences, paraphrasing, etc.

LLM_JUDGE_PROMPT = """You are evaluating a legal clause extraction system. Your job is to assess whether an extracted answer captures the correct legal meaning, regardless of phrasing differences.

You will be given:
- The source context (the actual text the system had access to)
- A field name describing what was being extracted
- The expected answer (ground truth)
- The extracted answer (what the system produced)

---

Score the extraction on the following scale:

1.0 — Correct and complete
  The extracted answer conveys the same legal meaning as the expected answer.
  Minor wording differences, tense changes, added preamble, or trailing punctuation do NOT reduce this score.
  Example: "is reduced" vs "shall be reduced" → 1.0
  Example: "Notwithstanding § 8.5(a), the Receiving Party may retain..." vs "The Receiving Party may retain..." → 1.0

0.7 — Correct but over- or under-extracted
  The core legal meaning is present but the answer includes significant surrounding text that
  obscures the specific answer, OR omits minor detail that does not change the legal outcome.
  Example: returning a full verbatim paragraph when only the deadline ("10 business days") was asked for → 0.7
  Example: "Successive one-year periods" when expected "This Agreement shall automatically renew for successive one-year periods" → 0.7

0.4 — Partially correct
  The answer captures some but not all of the required legal meaning. Key terms, conditions,
  or qualifications are missing or wrong in a way that would matter legally.
  Example: "Section 3 (Confidentiality)" when the full list is § 3, § 8.5, § 8.6, § 9, § 10, § 11 → 0.4

0.0 — Wrong or missing
  The answer is absent ([NOT FOUND]), is a hallucination, or conveys incorrect legal meaning.
  Note: if the source context does not contain the information, a [NOT FOUND] answer is correct
  and should be scored based on whether that is accurate given the source context.

---

Watch for these specific failure patterns and handle them carefully:

SELF-CONTRADICTION: If the extracted answer contains the correct information but also tags itself
  [NOT FOUND] or [AMBIGUOUS], treat the correct information as the answer and score accordingly.
  Do not penalize for the tag if the substance is right.

CONTEXT-APPROPRIATE GAPS: If the source context does not contain the information needed to answer
  the field, and the system responded [NOT FOUND] or [AMBIGUOUS], this is a CORRECT response.
  Score it 1.0 and set "gap_is_retrieval_fault" to true.

HALLUCINATION: If the extracted answer asserts something that is not in the source context and
  is not in the expected answer, flag it. This is more serious than a missing answer.

---

Source context:
<context>
{chunk_text}
</context>

Field: {field_name}

Expected answer:
<expected>
{ground_truth}
</expected>

Extracted answer:
<extracted>
{run_output}
</extracted>

---

Respond with valid JSON only. No preamble, no explanation outside the JSON.

{
  "score": float,              // 0.0, 0.4, 0.7, or 1.0
  "reasoning": string,         // one sentence explaining the score
  "hallucinated": boolean,     // true if the extracted answer asserts something not in the source context
  "gap_is_retrieval_fault": boolean,  // true if the field was unanswerable given the source context
  "category": string           // one of: "correct", "cosmetic_difference", "over_extracted", "under_extracted", "partial", "missing", "hallucinated", "correct_gap"
}"""


def llm_judge(field, expected, actual, chunk_text=""):
    """
    Ask the LLM to evaluate semantic equivalence between expected and actual answers.
    Returns a dict with score, reasoning, and other metadata from the judge.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        chunk_text=chunk_text if chunk_text else "[Full document context]",
        field_name=field,
        ground_truth=expected,
        run_output=actual
    )
    
    if BACKEND == "ollama":
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # deterministic
                "num_ctx": 4096
            }
        })
        answer = response.json().get("response", "").strip()
    else:
        result = pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        answer = result[0]["generated_text"][len(prompt):].strip()
    
    # Parse JSON response from judge
    try:
        # Handle potential markdown code blocks
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0].strip()
        elif "```" in answer:
            answer = answer.split("```")[1].split("```")[0].strip()
        
        judge_result = json.loads(answer)
        return {
            "score": judge_result.get("score", 0.0),
            "reasoning": judge_result.get("reasoning", ""),
            "hallucinated": judge_result.get("hallucinated", False),
            "gap_is_retrieval_fault": judge_result.get("gap_is_retrieval_fault", False),
            "category": judge_result.get("category", "unknown")
        }
    except (json.JSONDecodeError, IndexError):
        # Fallback if JSON parsing fails
        return {
            "score": 0.0,
            "reasoning": f"Failed to parse judge response: {answer[:100]}",
            "hallucinated": False,
            "gap_is_retrieval_fault": False,
            "category": "parse_error"
        }


def compare_to_ground_truth(ground_truth, run_result, use_llm_judge=True):
    """Compare a run's parsed output against the ground truth field by field."""
    gt_fields = parse_output(ground_truth["output"])
    run_fields = parse_output(run_result["output"])
    chunk_text = run_result.get("context_used", "")

    missing = []     # GT has a value, run has [NOT FOUND]
    mismatches = []  # Both have values but they differ
    exact_matches = 0
    weighted_score = 0.0  # Sum of LLM judge scores
    total_fields = 0
    hallucinations = 0
    retrieval_gaps = 0
    judge_results = []  # Detailed per-field results

    for field, gt_value in gt_fields.items():
        run_value = run_fields.get(field, "[NOT FOUND]")
        gt_not_found = gt_value == "[NOT FOUND]"
        run_not_found = run_value == "[NOT FOUND]"

        if gt_not_found:
            continue  # GT itself didn't find this — not a fair comparison point
        
        total_fields += 1
        
        if run_not_found:
            # Use LLM judge to check if this is a retrieval gap
            if use_llm_judge:
                judge = llm_judge(field, gt_value, run_value, chunk_text)
                judge_results.append({"field": field, **judge})
                weighted_score += judge["score"]
                if judge["gap_is_retrieval_fault"]:
                    retrieval_gaps += 1
            missing.append({"field": field, "expected": gt_value})
        elif gt_value.lower().strip() == run_value.lower().strip():
            exact_matches += 1
            weighted_score += 1.0
            judge_results.append({
                "field": field,
                "score": 1.0,
                "reasoning": "Exact match",
                "category": "correct"
            })
        else:
            # Not an exact match — use LLM to judge semantic equivalence
            if use_llm_judge:
                judge = llm_judge(field, gt_value, run_value, chunk_text)
                judge_results.append({"field": field, **judge})
                weighted_score += judge["score"]
                if judge["hallucinated"]:
                    hallucinations += 1
                mismatches.append({
                    "field": field, 
                    "expected": gt_value, 
                    "actual": run_value,
                    "llm_judge": judge
                })
            else:
                mismatches.append({"field": field, "expected": gt_value, "actual": run_value})

    exact_score = round(exact_matches / total_fields, 3) if total_fields > 0 else 0
    llm_score = round(weighted_score / total_fields, 3) if total_fields > 0 else 0

    return {
        "compared_to": ground_truth["run"],
        "total_fields": total_fields,
        "exact_matches": exact_matches,
        "missing_in_run": len(missing),
        "mismatches": len(mismatches),
        "hallucinations": hallucinations,
        "retrieval_gaps": retrieval_gaps,
        "exact_score": exact_score,
        "llm_score": llm_score,
        "details": {
            "missing": missing,
            "mismatches": mismatches,
            "judge_results": judge_results
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


# --- HICHUNK FUNCTIONS ---
# These functions implement hierarchical chunking with Auto-Merge retrieval
# HiChunk produces a tree of chunks that respects document structure
# Auto-Merge promotes sibling leaf nodes to parent context when retrieved together

def hichunk_parse_splits(splits):
    """
    Parse HiChunk splits format into a hierarchical tree structure.
    
    HiChunk output format: [["chunk text", level], ...]
    where level indicates the hierarchical depth (1 = top level section).
    
    Returns a list of chunk dicts with parent/child relationships.
    """
    if not splits:
        return []
    
    chunks = []
    for i, (text, level) in enumerate(splits):
        chunks.append({
            'id': f'chunk_{i}',
            'text': text.strip(),
            'level': level,
            'left_index_idx': i,
            'right_index_idx': i + 1,
            'parent': None,
            'children': []
        })
    
    # Build parent-child relationships based on levels
    # Lower level number = higher in hierarchy (level 1 is parent of level 2)
    for i, chunk in enumerate(chunks):
        # Find parent: nearest preceding chunk with lower level number
        for j in range(i - 1, -1, -1):
            if chunks[j]['level'] < chunk['level']:
                chunk['parent'] = chunks[j]
                chunks[j]['children'].append(chunk)
                break
    
    return chunks


def hichunk_get_leaves(chunks):
    """
    Extract leaf nodes (chunks with no children) from the chunk tree.
    These are the finest-grained chunks for embedding and retrieval.
    """
    return [c for c in chunks if not c['children']]


def hichunk_auto_merge(chunks, retrieved_leaves, similarity_threshold=2):
    """
    Auto-Merge retrieval algorithm.
    
    If multiple sibling leaves are retrieved, merge them up to their parent node.
    This ensures that related sub-clauses (like 8.3(a)(i), 8.3(a)(ii)) are kept 
    together with their parent section.
    
    Args:
        chunks: Full list of chunk dicts with parent/child relationships
        retrieved_leaves: List of retrieved leaf chunk dicts
        similarity_threshold: Min siblings required to trigger merge (default 2)
    
    Returns:
        List of merged chunk dicts (may include parent nodes)
    """
    if not retrieved_leaves:
        return []
    
    # Group retrieved leaves by parent
    parent_groups = {}
    for leaf in retrieved_leaves:
        parent = leaf.get('parent')
        parent_id = parent['id'] if parent else 'root'
        if parent_id not in parent_groups:
            parent_groups[parent_id] = {'parent': parent, 'leaves': []}
        parent_groups[parent_id]['leaves'].append(leaf)
    
    # Decide which nodes to include in final context
    merged_nodes = []
    seen_ids = set()
    
    for parent_id, group in parent_groups.items():
        parent = group['parent']
        leaves = group['leaves']
        
        # If enough siblings retrieved and parent exists, use parent
        if parent and len(leaves) >= similarity_threshold:
            if parent['id'] not in seen_ids:
                merged_nodes.append(parent)
                seen_ids.add(parent['id'])
                # Mark all children as seen
                for child in parent['children']:
                    seen_ids.add(child['id'])
        else:
            # Otherwise use individual leaves
            for leaf in leaves:
                if leaf['id'] not in seen_ids:
                    merged_nodes.append(leaf)
                    seen_ids.add(leaf['id'])
    
    # Sort by document order
    merged_nodes.sort(key=lambda c: c['left_index_idx'])
    return merged_nodes


def hichunk_retrieve(chunks, query, top_k=3):
    """
    Embed leaf nodes with bge-m3, retrieve top_k by cosine similarity,
    run Auto-Merge to promote to parent nodes where siblings are present,
    return assembled context string.
    
    Args:
        chunks: List of chunk dicts from hichunk_parse_splits()
        query: Query string for retrieval
        top_k: Number of leaf nodes to retrieve
    
    Returns:
        Assembled context string with merged chunks
    """
    leaves = hichunk_get_leaves(chunks)
    
    if not leaves:
        return ""
    
    # Use BGE-M3 for embeddings (high quality multilingual model)
    embed_model = SentenceTransformer("BAAI/bge-m3")
    
    # Embed all leaves
    leaf_texts = [leaf['text'] for leaf in leaves]
    leaf_embeddings = embed_model.encode(leaf_texts, normalize_embeddings=True)
    
    # Embed query
    query_embedding = embed_model.encode([query], normalize_embeddings=True)[0]
    
    # Compute cosine similarity (embeddings are normalized, so dot product = cosine)
    similarities = np.dot(leaf_embeddings, query_embedding)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    retrieved_leaves = [leaves[i] for i in top_indices]
    
    # Apply Auto-Merge
    merged_nodes = hichunk_auto_merge(chunks, retrieved_leaves)
    
    # Build full text for each merged node
    def get_node_text(node):
        """Get full text for a node, including all children if it's a parent."""
        if not node['children']:
            return node['text']
        # For parent nodes, concatenate all descendant text
        texts = []
        start_idx = node['left_index_idx']
        end_idx = node['right_index_idx']
        for chunk in chunks:
            if start_idx <= chunk['left_index_idx'] < end_idx:
                if not chunk['children']:  # Only add leaf text to avoid duplication
                    texts.append(chunk['text'])
        return '\n'.join(texts) if texts else node['text']
    
    # Assemble context string
    context_parts = [get_node_text(node) for node in merged_nodes]
    return "\n\n---\n\n".join(context_parts)


def structure_aware_chunk(text):
    """
    Fallback hierarchical chunker that uses regex patterns to identify
    legal document structure (sections, subsections, clauses).
    
    This mimics HiChunk's output format: [["chunk text", level], ...]
    
    Patterns recognized (handles markdown formatting):
    - Level 1: Major sections like "**8.1", "## 8.1", "8.1 Term"
    - Level 2: Subsections like "*(a)*", "(a)", "**(a)**"
    - Level 3: Sub-clauses like "- (i)", "(i)", "(ii)", "(iii)"
    
    Returns list of [text, level] pairs.
    """
    lines = text.split('\n')
    splits = []
    current_chunk = []
    current_level = 1
    
    # Patterns for legal document structure (with optional markdown)
    # Level 1: **8.X** or ## 8.X or just 8.X section headers
    section_pattern = re.compile(r'^(\*\*|##?\s*)?(\d+\.\d+)\s*(\*\*)?')
    # Level 2: *(a)* or (a) or **(a)** style subsections
    subsection_pattern = re.compile(r'^\s*(\*+)?\(([a-z])\)(\*+)?\s')
    # Level 3: - (i), (i), (ii), (iii) style sub-clauses
    clause_pattern = re.compile(r'^\s*[-*]?\s*\((?:i{1,3}|iv|v|vi{0,3})\)\s', re.IGNORECASE)
    
    def get_level(line):
        """Determine the hierarchical level of a line."""
        stripped = line.strip()
        if not stripped:
            return None
        if section_pattern.match(stripped):
            return 1
        if subsection_pattern.match(stripped):
            return 2
        if clause_pattern.match(stripped):
            return 3
        return None
    
    def flush_chunk():
        """Save the current chunk if non-empty."""
        nonlocal current_chunk, current_level
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                splits.append([chunk_text, current_level])
            current_chunk = []
    
    for line in lines:
        level = get_level(line)
        
        if level is not None:
            # New section/subsection/clause detected
            flush_chunk()
            current_level = level
            current_chunk.append(line)
        else:
            # Continuation of current chunk
            current_chunk.append(line)
    
    # Don't forget the last chunk
    flush_chunk()
    
    # If no structure was detected, fall back to paragraph-based chunks
    if not splits:
        paragraphs = text.split('\n\n')
        splits = [[p.strip(), 1] for p in paragraphs if p.strip()]
    
    return splits

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

    # -------------------------------------------------------
    # RUN 4 — HICHUNK + AUTO-MERGE RETRIEVAL
    # hierarchical chunking respects the document's own structure
    # auto-merge promotes sibling leaf nodes to parent context
    # this is what should correctly surface deep sub-clauses
    # -------------------------------------------------------
    if HICHUNK_AVAILABLE:
        print("\n--- HiChunk + Auto-Merge Retrieval ---")
        
        # Check if pre-computed HiChunk splits exist (from full HiChunk model)
        HICHUNK_CACHE = os.path.join(os.path.dirname(__file__), ".hichunk_cache.json")
        
        if os.path.exists(HICHUNK_CACHE):
            print("Loading cached HiChunk splits...")
            with open(HICHUNK_CACHE, 'r') as f:
                cached = json.load(f)
                hichunk_splits = cached.get('splits', [])
        else:
            # Fallback: use a structure-aware regex chunker that mimics HiChunk
            # This identifies section headers (8.1, 8.2, 8.3(a), etc.) and assigns levels
            if HICHUNK_FULL:
                print("Full HiChunk available but no cache — using structure-aware fallback...")
            else:
                print("Using structure-aware fallback chunker (HiChunk repo not installed)...")
            hichunk_splits = structure_aware_chunk(NDA_SECTION)
            # Cache for future runs
            with open(HICHUNK_CACHE, 'w') as f:
                json.dump({'splits': hichunk_splits}, f)
        
        if hichunk_splits:
            QUERY = "What are all the conditions under which this agreement can be terminated?"
            chunks = hichunk_parse_splits(hichunk_splits)
            hichunk_context = hichunk_retrieve(chunks, QUERY, top_k=5)
            
            results.append(run(
                "RUN 4 — HICHUNK + AUTO-MERGE (hierarchical, top 5)",
                hichunk_context
            ))
        else:
            print("No HiChunk splits available — skipping RUN 4")
    else:
        print("\nSkipping RUN 4 — sentence-transformers not installed")

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
    print(f"{'Run':<50} {'Latency':>8} {'Exact':>8} {'LLM':>8} {'Missing':>9} {'Halluc':>8} {'Gaps':>6}")
    print("-" * 100)
    for r in results:
        cmp = r.get("comparison", {})
        if cmp:
            exact    = f"{cmp['exact_score']:.1%}"
            llm      = f"{cmp['llm_score']:.1%}"
            missing  = str(cmp["missing_in_run"])
            halluc   = str(cmp["hallucinations"])
            gaps     = str(cmp["retrieval_gaps"])
        else:
            exact = "—"
            llm = "(baseline)"
            missing = "—"
            halluc = "—"
            gaps = "—"
        print(f"{r['run']:<50} {str(r['latency_seconds']) + 's':>8} {exact:>8} {llm:>8} {missing:>9} {halluc:>8} {gaps:>6}")
