#!/usr/bin/env python3
"""
RQ4 step 3: classify review comments into the Bacchelli & Bird taxonomy
(+ SETUP) using GPT-4.1 as an LLM-as-a-Judge.

Classifies:
  (a) the 3,000-comment proportional sample  -> produces the RQ4 findings
  (b) the 383 gold comments                  -> for human-AI agreement (kappa)

The prompt (below, PROMPT_TEMPLATE) is the artifact to place in the appendix /
replication package: it contains the explicit category definitions and a
decision rule for ambiguous cases, per the reviewer request.

Requires: OPENAI_API_KEY in the environment, and `pip install openai`.
Cost control: processes in batches, caches results, resumable.
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI

MODEL = "gpt-4.1"
SAMPLE_CSV = "rq4_outputs/rq4_classification_sample.csv"
GOLD_CSV = "rq4_outputs/gold_384.csv"
OUT_DIR = "rq4_outputs"

CATEGORIES = [
    "DEFECT", "CODE_IMPROVEMENT", "UNDERSTANDING", "KNOWLEDGE_TRANSFER",
    "TESTING", "EXTERNAL_IMPACT", "SOCIAL", "REVIEW_TOOL", "SETUP", "MISC",
]

# --------------------------------------------------------------------------- #
# Classification prompt. Definitions and decision tree reproduce the manual
# labeling guidelines used by our annotators (Bacchelli & Bird 2013 taxonomy).
# This block is the artifact to place in the appendix / replication package.
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are classifying a single code-review comment into exactly ONE category, using the definitions and decision tree below.

Categories:
- CODE_IMPROVEMENT: Readability, commenting, consistency, naming, dead-code removal, formatting. Improving working code, not correctness. NOT this if it causes incorrect behavior at runtime (that is DEFECT).
- DEFECT: Faults causing incorrect behavior at execution time: logic errors, security bugs, wrong exception handling, missing validation, resource-management issues. NOT this if the code works but could be improved (that is CODE_IMPROVEMENT or KNOWLEDGE_TRANSFER).
- UNDERSTANDING: Seeks clarification about rationale, design intent, or purpose. Asks WHY something was done and does NOT suggest a concrete change. NOT this if it asks AND suggests a change (classify by the suggestion instead).
- KNOWLEDGE_TRANSFER: Directs the author to external resources, explains team conventions, shares project-specific knowledge, or educates about APIs, system design, or best practices. NOT this if it asks a question instead of sharing knowledge (that is UNDERSTANDING).
- TESTING: Specifically about adding, updating, fixing, or improving tests: unit/integration tests, edge-case coverage, test naming. NOT this if it is general code style unrelated to tests (that is CODE_IMPROVEMENT).
- EXTERNAL_IMPACT: Consequences beyond the local diff: system-level effects, API contracts, backward compatibility, cross-team or cross-service implications. NOT this if the impact is local to the changed code only (that is DEFECT or CODE_IMPROVEMENT).
- SOCIAL: Interpersonal communication: appreciation, encouragement, affirmation, humor, general supportive comments not about the code. NOT this if it contains any technical content (use the technical category).
- REVIEW_TOOL: About the review tool or process itself, including bot invocation commands (e.g. @copilot, @coderabbit). Not about the code. NOT this if it contains actual code feedback beyond the bot command.
- SETUP: Instructions or guidance about setting up, installing, configuring, or running the project or its environment: installing dependencies, required language/runtime versions, virtual environments, build/run steps, or development-environment configuration (e.g. "install the dependencies from requirements.txt; the Python version is 3.10; use Conda for a virtual environment"). Educating the user about environment setup belongs here. NOT this if the educational content is about APIs, conventions, or design rather than setup (that is KNOWLEDGE_TRANSFER), or about the code's logic (use the technical category).
- MISC: Does not fit any other category. Use very sparingly. If any other category fits even loosely, use that instead.

Decision tree (apply in order):
1. Is it a bot invocation command (@copilot, @coderabbit, ...) or about the review tool/process? -> REVIEW_TOOL
2. Is it purely social with no technical content (LGTM, thanks, great work)? -> SOCIAL
3. Does it suggest or identify something concrete?
   - NO, it only asks a question -> UNDERSTANDING
   - YES -> what does it address?
       - Code is broken / will fail / security vulnerability -> DEFECT
       - Tests -> TESTING
       - Effects beyond the local diff (API/compat/cross-service) -> EXTERNAL_IMPACT
       - Environment setup / installation / configuration / build-run instructions (even if it educates the user) -> SETUP
       - Shares external resources / conventions / teaches about APIs or design -> KNOWLEDGE_TRANSFER
       - Readability / naming / style / refactor of working code -> CODE_IMPROVEMENT
4. If nothing above fits -> MISC

Return ONLY a JSON object: {{"category": "<ONE_CATEGORY>"}}

Comment:
\"\"\"{comment}\"\"\"
"""


def classify_one(client, text):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(comment=str(text)[:4000])}],
        temperature=0,
        max_tokens=20,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        cat = json.loads(raw)["category"].strip().upper()
    except Exception:
        cat = "MISC"
    return cat if cat in CATEGORIES else "MISC"


def classify_file(client, in_csv, text_col, out_csv):
    df = pd.read_csv(in_csv)
    key = "row_key" if "row_key" in df.columns else ("comment_id" if "comment_id" in df.columns else "id")
    # resume if partial output exists
    done = {}
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        if key in prev.columns:
            done = dict(zip(prev[key], prev["gpt_label"]))
        print(f"[resume] {len(done)} already classified in {out_csv}")

    results = []
    for i, row in df.iterrows():
        k = row[key]
        if k in done:
            results.append({key: k, "gpt_label": done[k]})
            continue
        label = classify_one(client, row[text_col])
        results.append({key: k, "gpt_label": label})
        if (i + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(out_csv, index=False)
            print(f"  {i+1}/{len(df)} classified")
            time.sleep(0.2)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[done] {out_csv}  ({len(results)} comments)")
    return pd.DataFrame(results)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = OpenAI()   # reads OPENAI_API_KEY

    # (b) gold first -- small, cheap, gives the human-AI agreement number
    print("=== classifying gold 383 (validation) ===")
    classify_file(client, GOLD_CSV, "comment_text", f"{OUT_DIR}/gpt_labels_gold.csv")

    # (a) the 3000-comment sample -> findings
    print("\n=== classifying 3000 sample (findings) ===")
    # the sample index has ids but not text; text_col must exist -- see note below
    classify_file(client, SAMPLE_CSV, "comment_text", f"{OUT_DIR}/gpt_labels_sample.csv")


if __name__ == "__main__":
    main()