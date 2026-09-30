#!/usr/bin/env python3
"""
RQ4 step 1: count the population of review comments in each of the three groups,
so we can sample proportionally.

Groups (comments on AI-authored PRs only):
  same_tool  : comment by an AI reviewer of the SAME vendor as the authoring agent
  diff_tool  : comment by an AI reviewer of a DIFFERENT vendor
  human      : comment by a human reviewer

We count inline review comments AND formal review bodies (both carry reviewer
text). Non-AI bots (CI, static analysis) are excluded. Strict 19-account allowlist.

Output: total count per group, overall and per authoring agent, so we can decide
proportional sample sizes.
"""

import os
import re
from collections import Counter
import pandas as pd

HF_BASE = "hf://datasets/hao-li/AIDev-full"
PR_TABLE = "pull_request"
REVIEWS_TABLE = "pr_reviews"
COMMENTS_TABLE = "pr_review_comments"
ALLOWLIST_CSV = "../files/llm_bot_allowlist_1.csv"   # adjust if needed
OUT_DIR = "rq4_outputs"

REVIEWER_VENDOR = {
    "Copilot": "copilot", "copilot-pull-request-reviewer[bot]": "copilot",
    "copilot-swe-agent[bot]": "copilot", "cursor[bot]": "cursor",
    "claude[bot]": "claude", "claudeai-v1[bot]": "claude",
    "devin-ai-integration[bot]": "devin", "windsurf-bot[bot]": "windsurf",
    "openhands-ai[bot]": "openhands", "rover-app[bot]": "rover",
    "lingma-agents[bot]": "lingma", "coderabbitai[bot]": "coderabbit",
    "gemini-code-assist[bot]": "gemini", "sourcery-ai[bot]": "sourcery",
    "greptile-apps[bot]": "greptile", "qodo-merge-pro[bot]": "qodo",
    "qodo-merge-for-open-source[bot]": "qodo", "bito-code-review[bot]": "bito",
    "chatgpt-codex-connector[bot]": "openai",
}
AUTHOR_VENDOR = {
    "Claude_Code": "claude", "Copilot": "copilot", "Cursor": "cursor",
    "Devin": "devin", "OpenAI_Codex": "openai",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    al = pd.read_csv(ALLOWLIST_CSV)
    al["is_llm_based"] = al["is_llm_based"].astype(str).str.strip().str.lower() == "true"
    ai_reviewers = set(al.loc[al["is_llm_based"], "user"])

    pr = pd.read_parquet(f"{HF_BASE}/{PR_TABLE}.parquet")
    reviews = pd.read_parquet(f"{HF_BASE}/{REVIEWS_TABLE}.parquet")[["pr_id", "user", "user_type", "body"]]
    comments = pd.read_parquet(f"{HF_BASE}/{COMMENTS_TABLE}.parquet")[
        ["pull_request_url", "user", "user_type", "body"]
    ]

    # link both to PR id
    rev = reviews.rename(columns={"pr_id": "id"})
    rev["kind"] = "review"
    com = comments.copy()
    url_to_id = pr.set_index("html_url")["id"]
    com["id"] = com["pull_request_url"].map(url_to_id)
    if com["id"].notna().sum() == 0:
        prn = pr.copy()
        prn["num"] = prn["html_url"].str.extract(r"/pull/(\d+)").astype("Int64")
        prn["repo"] = prn["html_url"].str.extract(r"github\.com/([^/]+/[^/]+)/pull/")
        com["num"] = com["pull_request_url"].str.extract(r"/pulls?/(\d+)").astype("Int64")
        com["repo"] = com["pull_request_url"].str.extract(r"repos/([^/]+/[^/]+)/pulls?/")
        com["id"] = com.set_index(["repo", "num"]).index.map(prn.set_index(["repo", "num"])["id"])
    com = com.dropna(subset=["id"])
    com["id"] = com["id"].astype(rev["id"].dtype)
    com["kind"] = "comment"

    ev = pd.concat([rev[["id", "user", "user_type", "body", "kind"]],
                    com[["id", "user", "user_type", "body", "kind"]]], ignore_index=True)

    # keep only comments with non-empty text (classifiable)
    ev = ev[ev["body"].fillna("").str.strip() != ""]
    # dedup on (id, text): comment_id is a REVIEW id shared by multiple distinct
    # inline comments, so id alone would wrongly collapse different comments.
    # (id, text) removes only true duplicates (identical id AND text).
    ev = ev.drop_duplicates(["id", "body"])

    # classify reviewer type
    ev["is_ai"] = ev["user"].isin(ai_reviewers)
    ev["is_human"] = ev["user_type"].astype(str).str.lower() == "user"
    ev = ev[ev["is_ai"] | ev["is_human"]]          # drop non-AI bots

    # attach authoring agent + vendors
    ev = ev.join(pr.set_index("id")["agent"], on="id").dropna(subset=["agent"])
    ev["author_vendor"] = ev["agent"].map(AUTHOR_VENDOR)
    ev["reviewer_vendor"] = ev["user"].map(REVIEWER_VENDOR)

    def group(r):
        if r["is_human"]:
            return "human"
        return "same_tool" if r["reviewer_vendor"] == r["author_vendor"] else "diff_tool"
    ev["group"] = ev.apply(group, axis=1)

    print("=== RQ4 comment population sizes ===")
    print("\nOverall by group:")
    print(ev["group"].value_counts().to_string())

    print("\nBy authoring agent x group:")
    print(pd.crosstab(ev["agent"], ev["group"]).to_string())

    # proportional sample suggestion for a target total (edit as desired)
    TARGET = 3000
    counts = ev["group"].value_counts()
    prop = (counts / counts.sum() * TARGET).round().astype(int)
    print(f"\nProportional sample for target total={TARGET}:")
    print(prop.to_string())

    ev[["id", "user", "group", "agent", "kind", "body"]].rename(
        columns={"id": "comment_id", "body": "comment_text"}
    ).to_csv(f"{OUT_DIR}/rq4_all_comments_labeled.csv", index=False)
    print(f"\nSaved full labeled comment index to {OUT_DIR}/rq4_all_comments_labeled.csv")
    print("(this lets us draw the proportional sample next, and merge the 383 by id)")


if __name__ == "__main__":
    main()