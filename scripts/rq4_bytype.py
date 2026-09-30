#!/usr/bin/env python3
"""
RQ5 Finding 3: technical acceptance by suggestion type.

Steps:
  1. Classify each of the 1,500 RQ5 suggestions into the Bacchelli & Bird (+SETUP)
     taxonomy, reusing the exact RQ4 prompt (GPT-4.1).
  2. Join to the implemented/not judgments (rq5_judgments.csv).
  3. Compute conditional implementation rate per suggestion type (excl. UNCLEAR).

Requires OPENAI_API_KEY. Resumable; caches type labels to rq5_type_labels.csv.
"""
import os, json, time
import pandas as pd
from openai import OpenAI

MODEL = "gpt-4.1"
OUT_DIR = "rq5_outputs"
DIFFS = f"{OUT_DIR}/rq5_sample_api_diffs.csv"     # sugg_id, body, ...
JUDG = f"{OUT_DIR}/rq5_judgments.csv"             # sugg_id, label
TYPE_OUT = f"{OUT_DIR}/rq5_type_labels.csv"
RATE_OUT = f"{OUT_DIR}/rq5_rate_by_type.csv"

CATEGORIES = ["DEFECT","CODE_IMPROVEMENT","UNDERSTANDING","KNOWLEDGE_TRANSFER",
              "TESTING","EXTERNAL_IMPACT","SOCIAL","REVIEW_TOOL","SETUP","MISC"]

# reuse the RQ4 classification prompt (same taxonomy, same rules)
PROMPT = """You are classifying a single code-review comment into exactly ONE category.

Categories:
- CODE_IMPROVEMENT: Readability, naming, refactoring, style of working code. Not correctness.
- DEFECT: Bug, incorrect behavior, security issue, wrong handling.
- UNDERSTANDING: Asks why/clarification, no concrete change suggested.
- KNOWLEDGE_TRANSFER: Shares resources, conventions, teaches about APIs/design.
- TESTING: About tests, coverage, verification.
- EXTERNAL_IMPACT: Effects beyond the local diff (API/compat/cross-service).
- SOCIAL: Appreciation, encouragement, interpersonal, no technical request.
- REVIEW_TOOL: About the review tool/process or a bot invocation command.
- SETUP: Environment setup, installation, configuration, build/run instructions.
- MISC: None of the above (use sparingly).

Return ONLY JSON: {{"category": "<ONE_CATEGORY>"}}

Comment:
\"\"\"{c}\"\"\"
"""

def classify(client, text):
    r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=20,
        messages=[{"role":"user","content":PROMPT.format(c=str(text)[:4000])}])
    raw = r.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try: cat = json.loads(raw)["category"].strip().upper()
    except Exception: cat = "MISC"
    return cat if cat in CATEGORIES else "MISC"

def main():
    client = OpenAI()
    d = pd.read_csv(DIFFS)[["sugg_id","body"]]
    done = {}
    if os.path.exists(TYPE_OUT):
        prev = pd.read_csv(TYPE_OUT); done = dict(zip(prev["sugg_id"], prev["type"]))
        print(f"[resume] {len(done)} typed")
    rows = []
    for i,r in d.iterrows():
        sid = r["sugg_id"]
        t = done.get(sid) or classify(client, r["body"])
        rows.append({"sugg_id": sid, "type": t})
        if (i+1)%50==0:
            pd.DataFrame(rows).to_csv(TYPE_OUT, index=False); print(f"  {i+1}/{len(d)}"); time.sleep(0.2)
    types = pd.DataFrame(rows); types.to_csv(TYPE_OUT, index=False)

    # join to judgments, compute conditional rate per type
    j = pd.read_csv(JUDG)[["sugg_id","label"]]
    m = types.merge(j, on="sugg_id")
    m = m[m["label"]!="UNCLEAR"]
    g = m.groupby("type")["label"].agg(
        n="count", implemented=lambda s:(s=="IMPLEMENTED").sum())
    g["rate_pct"] = (g["implemented"]/g["n"]*100).round(2)
    g = g.sort_values("rate_pct", ascending=False)
    g.to_csv(RATE_OUT)
    print("\n=== conditional implementation rate by suggestion type ==="); print(g.to_string())

if __name__ == "__main__":
    main()