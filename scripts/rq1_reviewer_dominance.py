#!/usr/bin/env python3
"""
RQ1 Findings #3 and #4.

Finding #3 (event dominance): when review occurs, how much of the review
ACTIVITY (events = reviews + inline comments) is produced by AI vs human
reviewers, per authoring agent, and how many UNIQUE reviewers each side has.
This is the "few AI reviewers generate most events" result.

Finding #4 (reviewer structure): for each authoring agent, break the AI review
events into:
    self     : reviewer is the SAME vendor as the authoring agent
    cross-AI : reviewer is a DIFFERENT vendor's AI
    human    : human reviewer
and list the top reviewer accounts per authoring agent. This separates genuine
cross-vendor checking from an agent reviewing its own output (closed loop).

Numbers here are strict-allowlist (19 accounts), matching Findings #1-#2.
"""

import os
from collections import Counter

import pandas as pd

HF_BASE = "hf://datasets/hao-li/AIDev-full"
PR_TABLE = "pull_request"
REVIEWS_TABLE = "pr_reviews"
COMMENTS_TABLE = "pr_review_comments"
ALLOWLIST_CSV = "llm_bot_allowlist_1.csv"
OUT_DIR = "rq1_outputs"

# --------------------------------------------------------------------------- #
# Vendor mapping: collapse reviewer logins AND authoring-agent labels into a
# common vendor space so "self-review" (reviewer vendor == author vendor) can
# be detected. Authoring labels in the dataset: Claude_Code, Copilot, Cursor,
# Devin, OpenAI_Codex.
# --------------------------------------------------------------------------- #
REVIEWER_VENDOR = {
    "Copilot": "copilot",
    "copilot-pull-request-reviewer[bot]": "copilot",
    "copilot-swe-agent[bot]": "copilot",
    "cursor[bot]": "cursor",
    "claude[bot]": "claude",
    "claudeai-v1[bot]": "claude",
    "devin-ai-integration[bot]": "devin",
    "windsurf-bot[bot]": "windsurf",
    "openhands-ai[bot]": "openhands",
    "rover-app[bot]": "rover",
    "lingma-agents[bot]": "lingma",
    "coderabbitai[bot]": "coderabbit",
    "gemini-code-assist[bot]": "gemini",
    "sourcery-ai[bot]": "sourcery",
    "greptile-apps[bot]": "greptile",
    "qodo-merge-pro[bot]": "qodo",
    "qodo-merge-for-open-source[bot]": "qodo",
    "bito-code-review[bot]": "bito",
    "chatgpt-codex-connector[bot]": "openai",   # ChatGPT/Codex connector
}
AUTHOR_VENDOR = {
    "Claude_Code": "claude",
    "Copilot": "copilot",
    "Cursor": "cursor",
    "Devin": "devin",
    "OpenAI_Codex": "openai",
}


def load_allowlist(path):
    al = pd.read_csv(path)
    al["is_llm_based"] = al["is_llm_based"].astype(str).str.strip().str.lower() == "true"
    return set(al.loc[al["is_llm_based"], "user"])


def load_tables():
    pr = pd.read_parquet(f"{HF_BASE}/{PR_TABLE}.parquet")
    reviews = pd.read_parquet(f"{HF_BASE}/{REVIEWS_TABLE}.parquet")
    comments = pd.read_parquet(f"{HF_BASE}/{COMMENTS_TABLE}.parquet")
    print(f"[load] pr={len(pr):,} reviews={len(reviews):,} comments={len(comments):,}")
    return pr, reviews, comments


def build_events(pr, reviews, comments, ai_reviewers):
    """One row per review event, tagged with reviewer type and PR's authoring agent."""
    rev = reviews[["pr_id", "user", "user_type"]].rename(columns={"pr_id": "id"})
    rev["event"] = "review"

    com = comments[["pull_request_url", "user", "user_type"]].copy()
    # robust join: try html_url, else repo+PR-number
    url_to_id = pr.set_index("html_url")["id"]
    com["id"] = com["pull_request_url"].map(url_to_id)
    if com["id"].notna().sum() == 0:
        prn = pr.copy()
        prn["num"] = prn["html_url"].str.extract(r"/pull/(\d+)").astype("Int64")
        prn["repo"] = prn["html_url"].str.extract(r"github\.com/([^/]+/[^/]+)/pull/")
        com["num"] = com["pull_request_url"].str.extract(r"/pulls?/(\d+)").astype("Int64")
        com["repo"] = com["pull_request_url"].str.extract(r"repos/([^/]+/[^/]+)/pulls?/")
        key = prn.set_index(["repo", "num"])["id"]
        com["id"] = com.set_index(["repo", "num"]).index.map(key)
    com = com.dropna(subset=["id"])
    com["id"] = com["id"].astype(pr["id"].dtype)
    com["event"] = "comment"

    events = pd.concat([rev[["id", "user", "user_type", "event"]],
                        com[["id", "user", "user_type", "event"]]], ignore_index=True)

    # classify each reviewer event
    events["is_ai"] = events["user"].isin(ai_reviewers)
    events["is_human"] = events["user_type"].astype(str).str.lower() == "user"
    events["is_other_bot"] = (~events["is_ai"]) & (~events["is_human"])
    events = events[~events["is_other_bot"]].copy()          # drop CI/static/PR-mgmt bots

    # attach authoring agent + vendors
    events = events.join(pr.set_index("id")["agent"], on="id")
    events["author_vendor"] = events["agent"].map(AUTHOR_VENDOR)
    events["reviewer_vendor"] = events["user"].map(REVIEWER_VENDOR)   # NaN for humans

    def rtype(r):
        if r["is_human"]:
            return "human"
        if pd.notna(r["reviewer_vendor"]) and r["reviewer_vendor"] == r["author_vendor"]:
            return "self"
        return "cross-AI"
    events["reviewer_type"] = events.apply(rtype, axis=1)
    print(f"[events] {len(events):,} AI/human events "
          f"({int(events['is_ai'].sum()):,} AI, {int(events['is_human'].sum()):,} human)")
    return events


def finding3(events):
    """Per-agent: AI vs human share of review EVENTS, and share of UNIQUE reviewers."""
    rows = []
    for agent, g in events.groupby("agent"):
        n_events = len(g)
        ai_ev = int(g["is_ai"].sum())
        hu_ev = int(g["is_human"].sum())
        uniq = g.groupby("user")["is_ai"].first()
        n_uniq = uniq.shape[0]
        ai_uniq = int(uniq.sum())
        hu_uniq = n_uniq - ai_uniq
        rows.append({
            "agent": agent,
            "events_total": n_events,
            "AI_event_%": round(ai_ev / n_events * 100, 2),
            "human_event_%": round(hu_ev / n_events * 100, 2),
            "unique_reviewers": n_uniq,
            "AI_unique_%": round(ai_uniq / n_uniq * 100, 2),
            "human_unique_%": round(hu_uniq / n_uniq * 100, 2),
        })
    return pd.DataFrame(rows).set_index("agent")


def finding4_split(events):
    """Per-agent breakdown of events into self / cross-AI / human."""
    ct = pd.crosstab(events["agent"], events["reviewer_type"])
    for c in ["self", "cross-AI", "human"]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[["self", "cross-AI", "human"]]
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    pct.columns = [f"{c} %" for c in pct.columns]
    return ct.join(pct)


def finding4_top_reviewers(events, top_n=6):
    """Top reviewer accounts per authoring agent, with self/cross-AI/human tag."""
    out = {}
    for agent, g in events.groupby("agent"):
        top = (g.groupby(["user", "reviewer_type"]).size()
                 .reset_index(name="events")
                 .sort_values("events", ascending=False)
                 .head(top_n))
        out[agent] = top
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ai_reviewers = load_allowlist(ALLOWLIST_CSV)
    pr, reviews, comments = load_tables()
    events = build_events(pr, reviews, comments, ai_reviewers)

    f3 = finding3(events)
    f4 = finding4_split(events)
    tops = finding4_top_reviewers(events)

    f3.to_csv(f"{OUT_DIR}/rq1_finding3_event_dominance.csv")
    f4.to_csv(f"{OUT_DIR}/rq1_finding4_self_cross_human.csv")
    with open(f"{OUT_DIR}/rq1_finding4_top_reviewers.txt", "w") as fh:
        for agent, top in tops.items():
            fh.write(f"\n=== {agent} ===\n{top.to_string(index=False)}\n")

    print("\n########## FINDING #3: event dominance (per agent) ##########")
    print(f3)
    print("\n########## FINDING #4: reviewer split self / cross-AI / human ##########")
    print(f4)
    print("\n########## FINDING #4: top reviewers per authoring agent ##########")
    for agent, top in tops.items():
        print(f"\n--- {agent} ---")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()