#!/usr/bin/env python3
"""
RQ4 sampling (approved capped per-agent design).

Groups, sampled per authoring agent to balance agents within each group:
  same author-reviewer  : up to CAP_SAME per agent   (reviewer same vendor as author)
  diff author-reviewer  : up to CAP_DIFF per agent   (reviewer different vendor)
  human reviewer        : up to CAP_HUMAN per agent

Caps intentionally balance agents so no single agent dominates a group. Where an
agent has fewer than the cap, we take all available (this is why group totals are
not exact multiples of the cap). OpenAI Codex has no same-tool comments (it does
not self-review), so it is absent from the same author-reviewer group.

Reads rq4_all_comments_labeled.csv (deduped, non-empty) and records comment_ids
for reproducibility.
"""

import os
import pandas as pd

OUT_DIR = "rq4_outputs"
LABELED_INDEX = f"{OUT_DIR}/rq4_all_comments_labeled.csv"
CAP_SAME = 100
CAP_DIFF = 300
CAP_HUMAN = 300
SEED = 42


def cap_sample(df, cap):
    """Sample up to `cap` comments per authoring agent."""
    parts = []
    for agent, g in df.groupby("agent"):
        parts.append(g.sample(n=min(cap, len(g)), random_state=SEED))
    return pd.concat(parts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(LABELED_INDEX)
    # correct dedup: (comment_id, comment_text), since comment_id is a review id
    # shared across distinct inline comments
    df = df.drop_duplicates(["comment_id", "comment_text"])
    if "comment_text" in df.columns:
        df = df[df["comment_text"].fillna("").str.strip() != ""]
    # add a stable unique key per distinct comment (id may repeat)
    df = df.reset_index(drop=True)
    df["row_key"] = df.index.astype(str)
    print(f"[clean] {len(df):,} unique non-empty comments")

    caps = {"same_tool": CAP_SAME, "diff_tool": CAP_DIFF, "human": CAP_HUMAN}
    samples = []
    print("\n=== sampled per group x agent ===")
    for grp, cap in caps.items():
        sub = df[df["group"] == grp]
        s = cap_sample(sub, cap)
        samples.append(s)
        print(f"\n{grp} (cap {cap}/agent) -> {len(s)} comments")
        print(s["agent"].value_counts().to_string())

    sample = pd.concat(samples).reset_index(drop=True)
    sample.to_csv(f"{OUT_DIR}/rq4_classification_sample.csv", index=False)

    print("\n=== group totals ===")
    print(sample["group"].value_counts().to_string())
    print(f"total: {len(sample)}")

    print("\n=== full population sizes (before sampling) ===")
    print(df.groupby(["group"]).size().to_string())
    print("\nfull population by group x agent:")
    print(pd.crosstab(df["agent"], df["group"]).to_string())


if __name__ == "__main__":
    main()