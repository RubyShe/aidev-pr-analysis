#!/usr/bin/env python3
"""
RQ5 step 1: sample AI-reviewer suggestions and attach the relevant code diffs.

A "suggestion" = an inline review comment (has a file path) made by an AI reviewer
(strict 19-account allowlist) on an AI-authored PR.

For each sampled suggestion we attach the diff(s) of the SAME FILE from that PR's
commits (pr_commit_details.patch, matched on pr_id + filename). The suggestion
text + these diffs go to the LLM judge in step 2.

Sampling: up to 300 per authoring agent (stratified), matching the paper.
Dedup: (comment_id, comment_text) -- comment_id is a review id shared by
distinct inline comments (learned in RQ4).
"""

import os
import pandas as pd

HF_BASE = "hf://datasets/hao-li/AIDev-full"
ALLOWLIST_CSV = "../files/llm_bot_allowlist_1.csv"
OUT_DIR = "rq5_outputs"
CAP_PER_AGENT = 300
SEED = 42

AUTHOR_VENDOR = {
    "Claude_Code": "claude", "Copilot": "copilot", "Cursor": "cursor",
    "Devin": "devin", "OpenAI_Codex": "openai",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    al = pd.read_csv(ALLOWLIST_CSV)
    al["is_llm_based"] = al["is_llm_based"].astype(str).str.strip().str.lower() == "true"
    ai_reviewers = set(al.loc[al["is_llm_based"], "user"])

    pr = pd.read_parquet(f"{HF_BASE}/pull_request.parquet")[["id", "agent", "html_url"]]
    comments = pd.read_parquet(f"{HF_BASE}/pr_review_comments.parquet")[
        ["pull_request_url", "user", "user_type", "body", "path", "diff_hunk"]
    ]

    # keep only AI-reviewer inline comments with a file path and non-empty text
    c = comments.copy()
    c = c[c["user"].isin(ai_reviewers)]
    c = c[c["path"].fillna("") != ""]
    c = c[c["body"].fillna("").str.strip() != ""]

    # link comment -> PR id (via html_url; fallback repo+number)
    url_to_id = pr.set_index("html_url")["id"]
    c["pr_id"] = c["pull_request_url"].map(url_to_id)
    if c["pr_id"].notna().sum() == 0:
        prn = pr.copy()
        prn["num"] = prn["html_url"].str.extract(r"/pull/(\d+)").astype("Int64")
        prn["repo"] = prn["html_url"].str.extract(r"github\.com/([^/]+/[^/]+)/pull/")
        c["num"] = c["pull_request_url"].str.extract(r"/pulls?/(\d+)").astype("Int64")
        c["repo"] = c["pull_request_url"].str.extract(r"repos/([^/]+/[^/]+)/pulls?/")
        c["pr_id"] = c.set_index(["repo", "num"]).index.map(prn.set_index(["repo", "num"])["id"])
    c = c.dropna(subset=["pr_id"])
    c["pr_id"] = c["pr_id"].astype(pr["id"].dtype)

    # attach authoring agent, keep AI-authored only, dedup
    c = c.merge(pr[["id", "agent"]], left_on="pr_id", right_on="id", how="left")
    c = c.dropna(subset=["agent"])
    c = c.drop_duplicates(["pull_request_url", "body", "path"])
    print(f"[suggestions] {len(c):,} AI-reviewer inline suggestions on AI-authored PRs")
    print(c["agent"].value_counts().to_string())

    # stratified sample: up to CAP per authoring agent
    parts = []
    for agent, g in c.groupby("agent"):
        parts.append(g.sample(n=min(CAP_PER_AGENT, len(g)), random_state=SEED))
    sample = pd.concat(parts).reset_index(drop=True)
    sample["sugg_id"] = sample.index.astype(str)   # stable unique key
    print(f"\n[sample] {len(sample)} suggestions (cap {CAP_PER_AGENT}/agent)")
    print(sample["agent"].value_counts().to_string())

    # attach diffs: pr_commit_details.patch for the SAME file (pr_id + filename)
    pr_ids = set(sample["pr_id"])
    print(f"\n[diffs] loading commit details for {len(pr_ids):,} PRs ...")
    details = pd.read_parquet(
        f"{HF_BASE}/pr_commit_details.parquet",
        columns=["pr_id", "filename", "patch", "status", "additions", "deletions"],
    )
    details = details[details["pr_id"].isin(pr_ids)]
    print(f"[diffs] {len(details):,} file-diff rows for sampled PRs")

    # for each suggestion, concatenate patches of the SAME file in that PR
    def get_diff(row):
        d = details[(details["pr_id"] == row["pr_id"]) & (details["filename"] == row["path"])]
        if len(d) == 0:
            return ""      # no diff for this file -> UNCLEAR downstream
        patches = [p for p in d["patch"].fillna("").tolist() if p.strip()]
        return "\n---\n".join(patches)[:8000]   # cap length for the LLM

    # (do the join efficiently)
    det_idx = details.groupby(["pr_id", "filename"])["patch"].apply(
        lambda s: "\n---\n".join([p for p in s.fillna("") if str(p).strip()])[:8000]
    )
    sample["file_diff"] = sample.apply(
        lambda r: det_idx.get((r["pr_id"], r["path"]), ""), axis=1
    )
    sample["has_diff"] = sample["file_diff"].str.strip() != ""
    print(f"\n[diffs] suggestions with a matching file diff: {sample['has_diff'].sum()} / {len(sample)}")

    keep = ["sugg_id", "pr_id", "agent", "user", "path", "body", "diff_hunk",
            "file_diff", "has_diff"]
    sample[keep].to_csv(f"{OUT_DIR}/rq5_sample_with_diffs.csv", index=False)
    print(f"\n[save] {OUT_DIR}/rq5_sample_with_diffs.csv")


if __name__ == "__main__":
    main()