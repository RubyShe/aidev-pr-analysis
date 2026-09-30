#!/usr/bin/env python3
"""
RQ5 step 2: LLM-judge whether each AI review suggestion was implemented.

Input: rq5_sample_api_diffs.csv (suggestion body + file_diff + diff_status).
For each suggestion the judge returns IMPLEMENTED / NOT_IMPLEMENTED / UNCLEAR.

Handling by diff_status:
  ok                    -> judge from the file patch
  file_not_changed      -> the commented file was not modified in the PR; the
                           judge is told this and decides (usually NOT_IMPLEMENTED)
  pr_unavailable /
  changed_no_patch_text -> no diff evidence -> UNCLEAR without an LLM call

The prompt is the appendix artifact. Requires OPENAI_API_KEY, `pip install openai`.
Resumable, checkpoints every 50.
"""
import os, json, time
import pandas as pd
from openai import OpenAI

MODEL = "gpt-4.1"
OUT_DIR = "rq5_outputs"
IN = f"{OUT_DIR}/rq5_sample_api_diffs.csv"
OUT = f"{OUT_DIR}/rq5_judgments.csv"
LABELS = {"IMPLEMENTED", "NOT_IMPLEMENTED", "UNCLEAR"}

PROMPT = """You are judging whether a code-review suggestion was implemented in a pull request.

You are given: (1) the reviewer's suggestion, and (2) the diff of the file the
suggestion was about (the actual code change made in the pull request).

Decide whether the suggested change was carried out in the diff.

Return exactly one label as JSON {{"label": "<LABEL>"}}:
- IMPLEMENTED: the diff clearly reflects the suggested change (the code was changed as suggested).
- NOT_IMPLEMENTED: the diff does not contain the suggested change (the code was not changed, or changed differently).
- UNCLEAR: the diff is insufficient to tell.

Judge only from the evidence given. Do not assume changes not shown.

SUGGESTION:
\"\"\"{body}\"\"\"

FILE DIFF:
\"\"\"{diff}\"\"\"
"""

PROMPT_NOCHANGE = """You are judging whether a code-review suggestion was implemented in a pull request.

The reviewer's suggestion is below. The file the suggestion was about was NOT
modified anywhere in the pull request (no diff exists for it).

Return exactly one label as JSON {{"label": "<LABEL>"}}:
- NOT_IMPLEMENTED: the suggested change requires modifying that file, which was not modified, so it was not implemented.
- UNCLEAR: the suggestion may have been addressed without modifying this file, or does not require a code change.

SUGGESTION:
\"\"\"{body}\"\"\"
"""

def judge(client, body, diff, status):
    if status in ("pr_unavailable", "changed_no_patch_text"):
        return "UNCLEAR"        # no evidence, skip the call
    if status == "file_not_changed":
        prompt = PROMPT_NOCHANGE.format(body=str(body)[:4000])
    else:
        prompt = PROMPT.format(body=str(body)[:4000], diff=str(diff)[:12000])
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        lab = json.loads(raw)["label"].strip().upper()
    except Exception:
        lab = "UNCLEAR"
    return lab if lab in LABELS else "UNCLEAR"

def main():
    client = OpenAI()
    df = pd.read_csv(IN)
    done = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = dict(zip(prev["sugg_id"], prev["label"]))
        print(f"[resume] {len(done)} done")
    results = []
    for i, r in df.iterrows():
        sid = r["sugg_id"]
        if sid in done:
            results.append({"sugg_id": sid, "label": done[sid]}); continue
        lab = judge(client, r.get("body",""), r.get("file_diff",""), r.get("diff_status",""))
        results.append({"sugg_id": sid, "label": lab})
        if (i+1) % 50 == 0:
            pd.DataFrame(results).to_csv(OUT, index=False)
            print(f"  {i+1}/{len(df)}"); time.sleep(0.2)
    out = pd.DataFrame(results).merge(df[["sugg_id","agent","diff_status"]], on="sugg_id")
    out.to_csv(OUT, index=False)
    print(f"[done] {OUT}")
    print("\nOverall:"); print(out["label"].value_counts().to_string())
    print("\nConditional (excl. UNCLEAR):")
    j = out[out["label"]!="UNCLEAR"]
    print(f"  implemented rate: {(j['label']=='IMPLEMENTED').mean()*100:.2f}%  (n={len(j)})")
    print("\nBy agent (conditional implementation rate):")
    for a,g in out.groupby("agent"):
        gj = g[g["label"]!="UNCLEAR"]
        if len(gj):
            print(f"  {a:14s} {(gj['label']=='IMPLEMENTED').mean()*100:5.2f}%  (n={len(gj)})")

if __name__ == "__main__":
    main()