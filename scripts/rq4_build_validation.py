#!/usr/bin/env python3
"""
RQ5 human-validation sample builder.

Draws a stratified random sample of judged suggestions (default 100) across the
three labels, so a human can check whether the LLM's IMPLEMENTED /
NOT_IMPLEMENTED / UNCLEAR judgment is correct. Produces a review-friendly CSV
with the suggestion, the diff, the LLM label, and an empty human_label column.

Stratify so all labels are represented (UNCLEAR too, to check those aren't
hiding real signal). Fixed seed for reproducibility.
"""
import os
import pandas as pd

OUT_DIR = "rq5_outputs"
JUDG = f"{OUT_DIR}/rq5_judgments.csv"          # sugg_id, label, agent, diff_status
DIFFS = f"{OUT_DIR}/rq5_sample_api_diffs.csv"   # sugg_id, body, file_diff, path, ...
OUT = f"{OUT_DIR}/rq5_validation_sample.csv"
N = 100
SEED = 42


def main():
    j = pd.read_csv(JUDG)
    d = pd.read_csv(DIFFS)
    df = j.merge(d[["sugg_id", "agent", "path", "body", "file_diff", "diff_status"]],
                 on="sugg_id", how="left", suffixes=("", "_d"))

    # stratified by label, proportional but ensure each label present
    parts = []
    for lab, g in df.groupby("label"):
        k = max(10, round(N * len(g) / len(df)))   # at least 10 per label
        parts.append(g.sample(n=min(k, len(g)), random_state=SEED))
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # show the SAME diff length the judge saw (12k), not a short 2k preview,
    # so human labeling is comparable to the LLM's judgment
    sample["file_diff_full"] = sample["file_diff"].fillna("").str.slice(0, 12000)
    sample["human_label"] = ""      # you fill: IMPLEMENTED / NOT_IMPLEMENTED / UNCLEAR
    sample["notes"] = ""

    cols = ["sugg_id", "agent", "path", "diff_status",
            "body", "file_diff_full", "label", "human_label", "notes"]
    sample[cols].to_csv(OUT, index=False)
    print(f"[done] {OUT}  ({len(sample)} rows to review)")
    print("\nlabel mix in the validation sample:")
    print(sample["label"].value_counts().to_string())
    print("\nColumns to review: read `body` (suggestion) + `file_diff_short` (diff),")
    print("then fill `human_label` with IMPLEMENTED / NOT_IMPLEMENTED / UNCLEAR.")
    print("`label` is the LLM's judgment (compare against your human_label afterward).")


if __name__ == "__main__":
    main()