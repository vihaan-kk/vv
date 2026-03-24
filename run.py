import requests
import json
import time
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results.json")


NDA_SECTION = """
**SECTION 8 — TERM AND TERMINATION**

**8.1 Term**
This Agreement shall commence on the Effective Date and continue in force for a period of two (2) years, unless earlier terminated in accordance with this Section 8. Upon expiration of the initial term, this Agreement shall automatically renew for successive one-year periods unless either Party provides written notice of non-renewal no fewer than sixty (60) days prior to the end of the then-current term.

**8.2 Termination by Notice**

*(a)* Either Party may terminate this Agreement for any reason upon thirty (30) days prior written notice to the other Party. Such notice shall be delivered by certified mail or overnight courier to the address set forth in Section 12.4 of this Agreement.

*(b)* During the notice period, both Parties shall continue to perform all obligations under this Agreement, including obligations of confidentiality. Termination shall become effective upon the expiration of the notice period unless withdrawn by mutual written consent prior to that date.

*(c)* Notwithstanding the foregoing, if the terminating Party is in material breach of this Agreement at the time notice is delivered, the non-terminating Party may elect to treat such notice as void and pursue its rights under Section 8.3.

**8.3 Termination for Material Breach**

*(a)* Either Party (the "Non-Breaching Party") may terminate this Agreement upon written notice to the other Party (the "Breaching Party") if the Breaching Party has committed a material breach of any obligation under this Agreement, provided that:

- (i) the Non-Breaching Party has delivered a written Default Notice specifying the nature of the breach in reasonable detail;
- (ii) the Breaching Party has failed to cure such breach within sixty (60) days following receipt of the Default Notice; and
- (iii) the Non-Breaching Party delivers a subsequent written termination notice within thirty (30) days after expiration of the cure period.

*(b)* Notwithstanding Section 8.3(a), for any breach solely involving the failure to make a payment obligation, the cure period shall be reduced to thirty (30) days from the date of the Default Notice.

*(c)* If the Breaching Party disputes in good faith that a material breach has occurred and provides written notice of such dispute prior to expiration of the cure period, termination shall not take effect pending resolution of such dispute pursuant to Section 11.1 (Dispute Resolution). The cure period shall be tolled during the pendency of any dispute resolution proceeding initiated in good faith.

*(d)* If a breach is not capable of being remedied within the cure period specified above but the Breaching Party is making bona fide efforts to remedy it, the Breaching Party may request in writing a reasonable extension of the cure period, not to exceed an additional ninety (90) days, by delivering to the Non-Breaching Party a written remediation plan within fifteen (15) days of receipt of the Default Notice.

**8.4 Termination for Insolvency**

Either Party may terminate this Agreement immediately upon written notice if the other Party: (a) becomes the subject of a voluntary or involuntary petition in bankruptcy; (b) makes a general assignment for the benefit of creditors; (c) has a receiver, trustee, or liquidator appointed for all or substantially all of its assets; or (d) ceases to conduct business in the ordinary course. Such termination shall be effective upon delivery of written notice and shall not be subject to any cure period.

**8.5 Effect of Termination**

*(a) Return or Destruction of Confidential Information.* Upon termination or expiration of this Agreement for any reason, and upon written request by the Disclosing Party, the Receiving Party shall within ten (10) business days: (i) return all tangible materials containing Confidential Information, including all copies, extracts, and summaries thereof; or (ii) permanently destroy all such materials and certify such destruction in writing to the Disclosing Party. The Receiving Party shall, to the extent technically feasible, expunge all Confidential Information from any electronic storage systems.

*(b) Retention Exception.* Notwithstanding Section 8.5(a), the Receiving Party may retain one (1) archival copy of Confidential Information solely to the extent required by applicable law or regulation, subject to the confidentiality obligations of this Agreement, which shall continue to apply to such retained materials for a period of five (5) years following termination.

*(c) Accrued Rights.* Termination of this Agreement shall not affect any rights or obligations of either Party that accrued prior to the effective date of termination, including any claims for breach arising before such date.

**8.6 Survival**
The following provisions shall survive any expiration or termination of this Agreement: Section 3 (Confidentiality Obligations), Section 8.5 (Effect of Termination), Section 8.6 (Survival), Section 9 (Disclaimer of Warranties), Section 10 (Limitation of Liability), and Section 11 (Dispute Resolution). With respect to Confidential Information that constitutes a trade secret under applicable law, the confidentiality obligations of Section 3 shall survive for so long as such information remains a trade secret, notwithstanding any fixed survival period stated herein.
"""

PROMPT = """TASK:
You are a legal document analyst reviewing a Non-Disclosure Agreement.
Using only the text provided below, identify and summarize every
condition under which this agreement can be terminated.

OUTPUT FORMAT:
Return your answer using exactly this structure:

1. Termination by Notice
   - Who can terminate:
   - Notice period required:
   - Delivery method:
   - Obligations during notice period:
   - Exception or override condition:

2. Termination for Material Breach
   - Step 1 (Default Notice):
   - Step 2 (Cure period):
   - Step 3 (Final termination notice window):
   - Payment breach exception (cure period):
   - Good faith dispute exception:
   - Extension if breach cannot be cured:

3. Termination for Insolvency
   - Triggers (list all):
   - Cure period:
   - Effective date:

4. Effect of Termination
   - Return/destruction deadline:
   - Retention exception:
   - Accrued rights:

5. Survival After Termination
   - Sections that survive:
   - Trade secret carveout:

6. Term and Renewal
   - Initial duration:
   - Auto-renewal:
   - Notice required to prevent renewal:

RULES:
- Only include information explicitly stated in the provided text
- If a field has no information in the provided text, write [NOT FOUND]
- If a condition is ambiguous or contradictory, flag it with [AMBIGUOUS: explain why]
- Do not infer, assume, or import knowledge from outside the provided text
- Do not summarize — reproduce the specific terms (numbers, timeframes, conditions) exactly as stated

CONTEXT:
{context}

QUESTION:
What are all the conditions under which this agreement can be terminated,
and what are the obligations of each party upon termination? /no_think"""


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