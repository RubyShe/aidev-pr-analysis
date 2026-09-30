#!/usr/bin/env python3
"""
RQ4 step 4: (a) validate the classifier against the 383 gold labels
            (b) compute comment-type distributions per group for the findings.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

OUT_DIR = "rq4_outputs"
GOLD = f"{OUT_DIR}/gold_384.csv"                    # comment_id, comment_text, final_label, group, ...
GPT_GOLD = f"{OUT_DIR}/gpt_labels_gold.csv"          # comment_id, gpt_label
SAMPLE = f"{OUT_DIR}/rq4_classification_sample.csv"  # comment_id, group, agent, ...
GPT_SAMPLE = f"{OUT_DIR}/gpt_labels_sample.csv"      # comment_id, gpt_label

CATS = ["DEFECT", "CODE_IMPROVEMENT", "UNDERSTANDING", "KNOWLEDGE_TRANSFER",
        "TESTING", "EXTERNAL_IMPACT", "SOCIAL", "REVIEW_TOOL", "SETUP", "MISC"]


def validation():
    gold = pd.read_csv(GOLD).drop_duplicates("comment_id")
    gpt = pd.read_csv(GPT_GOLD).drop_duplicates("comment_id", keep="last")
    label_col = "final_label" if "final_label" in gold.columns else "human_label"
    m = gold[["comment_id", label_col]].merge(gpt, on="comment_id", how="inner")
    m[label_col] = m[label_col].str.upper().str.strip()
    m["gpt_label"] = m["gpt_label"].str.upper().str.strip()
    print(f"[validation] matched {len(m)} gold comments")

    kappa = cohen_kappa_score(m[label_col], m["gpt_label"])
    raw = (m[label_col] == m["gpt_label"]).mean()
    print(f"  human-AI raw agreement: {raw*100:.1f}%")
    print(f"  human-AI Cohen's kappa: {kappa:.3f}")

    print("\n  Confusion matrix (rows=human, cols=GPT):")
    present = [c for c in CATS if c in set(m[label_col]) | set(m["gpt_label"])]
    cm = confusion_matrix(m[label_col], m["gpt_label"], labels=present)
    print(pd.DataFrame(cm, index=present, columns=present).to_string())

    # per-category agreement (where do they diverge?)
    print("\n  Per-category recall (human label -> matched by GPT):")
    for c in present:
        sub = m[m[label_col] == c]
        if len(sub):
            print(f"    {c:20s} n={len(sub):3d}  matched {(sub['gpt_label']==c).mean()*100:5.1f}%")
    m.to_csv(f"{OUT_DIR}/rq4_validation_pairs.csv", index=False)


def findings():
    samp = pd.read_csv(SAMPLE)
    gpt = pd.read_csv(GPT_SAMPLE)
    # pick a key present in BOTH files
    if "row_key" in samp.columns and "row_key" in gpt.columns:
        key = "row_key"
    elif "comment_id" in samp.columns and "comment_id" in gpt.columns:
        key = "comment_id"
    else:
        raise SystemExit(f"No shared key. sample cols={list(samp.columns)}; "
                         f"gpt cols={list(gpt.columns)}")
    samp = samp[[key, "group", "agent"]].drop_duplicates(key)
    gpt = gpt.drop_duplicates(key, keep="last")
    df = samp.merge(gpt, on=key, how="inner")
    df["gpt_label"] = df["gpt_label"].str.upper().str.strip()
    print(f"\n[findings] {len(df)} classified sample comments "
          f"(sample rows={len(samp)}, gpt rows={len(gpt)}, key={key})")

    # distribution of comment types per group (row-normalized %)
    ct = pd.crosstab(df["group"], df["gpt_label"])
    ct = ct.reindex(columns=[c for c in CATS if c in ct.columns], fill_value=0)
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    print("\n=== Comment-type distribution by group (%) ===")
    print(pct.to_string())
    ct.to_csv(f"{OUT_DIR}/rq4_counts_by_group.csv")
    pct.to_csv(f"{OUT_DIR}/rq4_pct_by_group.csv")

    # non-technical share per group (SOCIAL+UNDERSTANDING+KNOWLEDGE_TRANSFER+REVIEW_TOOL)
    nontech = ["SOCIAL", "UNDERSTANDING", "KNOWLEDGE_TRANSFER", "REVIEW_TOOL"]
    nt = pct[[c for c in nontech if c in pct.columns]].sum(axis=1)
    print("\n=== Non-technical comment share by group (%) ===")
    print(nt.round(2).to_string())


if __name__ == "__main__":
    validation()
    findings()