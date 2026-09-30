#!/usr/bin/env python3
"""
RQ1: What review configurations occur for AI-authored pull requests,
     and how are AI and human reviewers involved?

Framing (per supervisor): AI-reviewing-AI is the phenomenon of interest;
human review is the baseline. Outputs are structured so the headline can be
the AI-AI configuration and cross-tool review, not the raw "no review" rate.

This script produces, from a SINGLE source of truth:
  1. Overall configuration table   (No review / AI review / Human review / AI+Human)
  2. Per-agent configuration table  (same four categories, one row per authoring agent)
  3. TWO distinct involvement metrics, kept separate and labelled:
       (a) PR-share       : fraction of PRs in each configuration
       (b) event-share    : fraction of review EVENTS produced by AI vs human
  4. A statistical test with effect size for "AI reviewers are involved more
     often than humans", reported at both the PR level and the event level.

All counts derive from one classification pass, so every number in the paper
must trace back to the CSVs this script writes. No figure/text mismatches.

Reviewer identification uses the curated allowlist (is_llm_based == True).
The dataset's own `user_type` flag is ALSO computed so the methodology/threats
section can report the delta between the two (construct validity).
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HF_BASE = "hf://datasets/hao-li/AIDev-full"   # FULL scope: ~932k PRs + review tables
# NOTE: table name `pull_request` means different things across repos:
#   hao-li/AIDev       -> pull_request = 33,596 (pop subset, >100 stars)
#   hao-li/AIDev-full  -> pull_request = ~932k  (full population)  <-- this one
# AIDev-full carries full-scale review tables, so RQ1 can run on the full
# population that matches the paper's reported 931,785.
PR_TABLE = "pull_request"                  # ~932k in AIDev-full
REVIEWS_TABLE = "pr_reviews"               # ~137k
COMMENTS_TABLE = "pr_review_comments"      # ~141k (no _v2 suffix in this repo)
ALLOWLIST_CSV = "llm_bot_allowlist_1.csv"
OUT_DIR = "rq1_outputs"


def load_allowlist(path):
    """Return the set of reviewer logins classified as AI (is_llm_based == True)."""
    al = pd.read_csv(path)
    al["is_llm_based"] = al["is_llm_based"].astype(str).str.strip().str.lower() == "true"
    ai_reviewers = set(al.loc[al["is_llm_based"], "user"])
    print(f"[allowlist] {len(al)} accounts; {len(ai_reviewers)} classified AI "
          f"({Counter(al.loc[al['is_llm_based'], 'category'])})")
    return ai_reviewers


def load_wide_allowlist(path):
    """
    Sensitivity set: the strict 19 PLUS every account categorized as an
    llm_code_review bot that was excluded only because it is not in Zhong et
    al.'s validated list. Non-AI bots (static analysis, CI, PR mgmt) stay out.
    """
    al = pd.read_csv(path)
    is_true = al["is_llm_based"].astype(str).str.strip().str.lower() == "true"
    is_extra_review = (~is_true) & (al["category"] == "llm_code_review")
    wide = set(al.loc[is_true | is_extra_review, "user"])
    print(f"[wide allowlist] {len(wide)} accounts "
          f"({int(is_true.sum())} strict + {int(is_extra_review.sum())} extra llm_code_review)")
    return wide


def load_tables():
    """Load the three tables RQ1 needs, directly as parquet from the Hub."""
    pr = pd.read_parquet(f"{HF_BASE}/{PR_TABLE}.parquet")
    reviews = pd.read_parquet(f"{HF_BASE}/{REVIEWS_TABLE}.parquet")
    comments = pd.read_parquet(f"{HF_BASE}/{COMMENTS_TABLE}.parquet")
    print(f"[load] {PR_TABLE}={len(pr):,}  {REVIEWS_TABLE}={len(reviews):,}  "
          f"{COMMENTS_TABLE}={len(comments):,}")
    print(f"[load] pull_request columns: {list(pr.columns)}")
    print(f"[load] pr_reviews columns:   {list(reviews.columns)}")
    print(f"[load] {COMMENTS_TABLE} columns: {list(comments.columns)}")
    return pr, reviews, comments


def build_review_events(pr, reviews, comments, ai_reviewers):
    """
    Collapse formal reviews + inline comments into one long table of review
    EVENTS, one row per (pr_id, reviewer, event). Each event is tagged:
      is_ai_allowlist : reviewer login is in the curated AI allowlist
      is_bot_flag     : dataset user_type == 'Bot'  (for the delta comparison)

    General PR-level comments are excluded (following Zhong et al.): we keep
    only formal review submissions and inline code comments.
    """
    # --- formal review submissions: link via pr_id -> pull_request.id
    rev = reviews[["pr_id", "user", "user_type"]].copy()
    rev = rev.rename(columns={"pr_id": "id"})
    rev["event"] = "review"

    # --- inline review comments: link via pull_request_url -> pull request id.
    #     pull_request_url may be an API url (.../pulls/N) while pr has html_url
    #     (.../pull/N). Try a direct url match first; if it drops everything,
    #     fall back to matching on the trailing PR number.
    com = comments[["pull_request_url", "user", "user_type"]].copy()
    url_to_id = pr.set_index("html_url")["id"]
    com["id"] = com["pull_request_url"].map(url_to_id)

    if com["id"].notna().sum() == 0:
        # fallback: extract trailing integer (PR number) from both sides
        pr_num = pr.copy()
        pr_num["num"] = pr_num["html_url"].str.extract(r"/pull/(\d+)").astype("Int64")
        # need repo to disambiguate numbers across repos
        pr_num["repo"] = pr_num["html_url"].str.extract(r"github\.com/([^/]+/[^/]+)/pull/")
        com["num"] = com["pull_request_url"].str.extract(r"/pulls?/(\d+)").astype("Int64")
        com["repo"] = com["pull_request_url"].str.extract(r"repos/([^/]+/[^/]+)/pulls?/")
        key = pr_num.set_index(["repo", "num"])["id"]
        com["id"] = com.set_index(["repo", "num"]).index.map(key)
        print("[events] used PR-number fallback for comment->PR join")

    com = com.drop(columns=["pull_request_url"], errors="ignore")
    com["event"] = "comment"
    com = com.dropna(subset=["id"])
    com["id"] = com["id"].astype(pr["id"].dtype)

    events = pd.concat([rev, com], ignore_index=True)
    events["is_ai_allowlist"] = events["user"].isin(ai_reviewers)
    events["is_bot_flag"] = events["user_type"].astype(str).str.lower() == "bot"
    # Human = dataset user_type 'User'. AI = allowlist. Everything else is a
    # non-AI bot (static analysis, CI, PR management) and is EXCLUDED: it counts
    # as neither AI nor human review.
    events["is_human"] = (events["user_type"].astype(str).str.lower() == "user")
    events["is_other_bot"] = (~events["is_ai_allowlist"]) & (~events["is_human"])

    n_before = len(events)
    n_other = int(events["is_other_bot"].sum())
    events = events[~events["is_other_bot"]].copy()
    print(f"[events] kept {len(events):,} AI/human review events "
          f"(excluded {n_other:,} non-AI bot events: static analysis / CI / PR mgmt)")
    print(f"[events]   of kept: {int(events['is_ai_allowlist'].sum()):,} AI, "
          f"{int(events['is_human'].sum()):,} human")
    return events


def classify_prs(pr, events):
    """
    Assign each AI-authored PR to exactly one of four configurations using the
    ALLOWLIST definition of an AI reviewer:
        No review     : no review events at all
        AI review     : >=1 AI reviewer, 0 human reviewers
        Human review  : >=1 human reviewer, 0 AI reviewers
        AI+Human      : >=1 of each
    A "human reviewer" = a reviewer event with user_type 'User'. Non-AI bots
    were already dropped in build_review_events, so they cannot make a PR count
    as reviewed. A PR touched only by a CI/static-analysis bot -> "No review".
    """
    # per-PR flags: did any AI reviewer / any human reviewer act on this PR
    per_pr = events.groupby("id").agg(
        any_ai=("is_ai_allowlist", "max"),
        any_human=("is_human", "max"),
    )
    cfg = pd.DataFrame(index=pr["id"].unique())
    cfg = cfg.join(per_pr)
    cfg["any_ai"] = cfg["any_ai"].fillna(False).astype(bool)
    cfg["any_human"] = cfg["any_human"].fillna(False).astype(bool)

    def label(row):
        if not row["any_ai"] and not row["any_human"]:
            return "No review"
        if row["any_ai"] and not row["any_human"]:
            return "AI review"
        if not row["any_ai"] and row["any_human"]:
            return "Human review"
        return "AI+Human review"

    cfg["config"] = cfg.apply(label, axis=1)
    # attach authoring agent for the per-agent breakdown
    cfg = cfg.join(pr.set_index("id")["agent"], how="left")
    return cfg.reset_index().rename(columns={"index": "id"})


def overall_table(cfg):
    order = ["No review", "AI review", "Human review", "AI+Human review"]
    t = cfg["config"].value_counts().reindex(order).fillna(0).astype(int)
    out = pd.DataFrame({"count": t, "percent": (t / t.sum() * 100).round(2)})
    return out


def per_agent_table(cfg):
    order = ["No review", "AI review", "Human review", "AI+Human review"]
    ct = pd.crosstab(cfg["agent"], cfg["config"]).reindex(columns=order).fillna(0).astype(int)
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    pct.columns = [f"{c} %" for c in pct.columns]
    return ct.join(pct)


def involvement_metrics(cfg, events):
    """
    Metric (a) PR-share : among REVIEWED PRs, share touched by AI vs by humans.
    Metric (b) event-share : among all review EVENTS, share produced by AI vs human.
    These answer 'are AI reviewers more involved?' in two different senses; the
    draft conflated them, so both are reported explicitly.
    """
    reviewed = cfg[cfg["config"] != "No review"]
    pr_share = {
        "reviewed_prs": len(reviewed),
        "prs_with_AI": int(reviewed["any_ai"].sum()),
        "prs_with_human": int(reviewed["any_human"].sum()),
        "pct_reviewed_with_AI": round(reviewed["any_ai"].mean() * 100, 2),
        "pct_reviewed_with_human": round(reviewed["any_human"].mean() * 100, 2),
    }
    # events already contain only AI and human rows (non-AI bots dropped)
    n_ai_events = int(events["is_ai_allowlist"].sum())
    n_hu_events = int(events["is_human"].sum())
    event_share = {
        "total_events": len(events),
        "AI_events": n_ai_events,
        "human_events": n_hu_events,
        "pct_events_AI": round(n_ai_events / len(events) * 100, 2),
        "pct_events_human": round(n_hu_events / len(events) * 100, 2),
    }
    return pr_share, event_share


def stat_tests(pr_share, event_share):
    """
    Two-proportion z-tests with Cohen's h effect size. At N this large p-values
    are near-zero by construction, so the EFFECT SIZE is what the paper reports.
    Cohen's h: 0.2 small, 0.5 medium, 0.8 large.
    """
    def cohens_h(p1, p2):
        return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))

    results = {}

    # PR-level: among reviewed PRs, AI-touched vs human-touched
    n = pr_share["reviewed_prs"]
    c_ai, c_hu = pr_share["prs_with_AI"], pr_share["prs_with_human"]
    stat, p = proportions_ztest([c_ai, c_hu], [n, n])
    results["PR_level"] = {
        "p_ai": c_ai / n, "p_human": c_hu / n,
        "z": round(stat, 3), "p_value": p,
        "cohens_h": round(cohens_h(c_ai / n, c_hu / n), 3),
    }

    # Event-level: AI events vs human events out of all events
    N = event_share["total_events"]
    c_ai, c_hu = event_share["AI_events"], event_share["human_events"]
    stat, p = proportions_ztest([c_ai, c_hu], [N, N])
    results["event_level"] = {
        "p_ai": c_ai / N, "p_human": c_hu / N,
        "z": round(stat, 3), "p_value": p,
        "cohens_h": round(cohens_h(c_ai / N, c_hu / N), 3),
    }
    return results


def allowlist_vs_userflag_delta(events):
    """
    Construct-validity check: how many review events are classified differently
    by the curated allowlist vs the dataset's own user_type=='Bot' flag.
    Reported in Threats to Validity to justify the curated list.
    """
    ct = pd.crosstab(events["is_ai_allowlist"], events["is_bot_flag"],
                     rownames=["allowlist_AI"], colnames=["dataset_bot_flag"])
    disagree = int(ct.loc[True, False] if (True in ct.index and False in ct.columns) else 0) \
        + int(ct.loc[False, True] if (False in ct.index and True in ct.columns) else 0)
    return ct, disagree


def run_analysis(pr, reviews, comments, ai_reviewers, out_dir, tag):
    """Full RQ1 pipeline for one AI-reviewer definition. Returns key numbers."""
    os.makedirs(out_dir, exist_ok=True)
    events = build_review_events(pr, reviews, comments, ai_reviewers)
    cfg = classify_prs(pr, events)

    overall = overall_table(cfg)
    per_agent = per_agent_table(cfg)
    pr_share, event_share = involvement_metrics(cfg, events)
    tests = stat_tests(pr_share, event_share)

    overall.to_csv(f"{out_dir}/rq1_overall_config_{tag}.csv")
    per_agent.to_csv(f"{out_dir}/rq1_per_agent_config_{tag}.csv")
    pd.DataFrame([pr_share]).to_csv(f"{out_dir}/rq1_prshare_{tag}.csv", index=False)
    pd.DataFrame([event_share]).to_csv(f"{out_dir}/rq1_eventshare_{tag}.csv", index=False)
    pd.DataFrame(tests).T.to_csv(f"{out_dir}/rq1_stat_tests_{tag}.csv")

    print(f"\n########## RESULTS [{tag}] ##########")
    print("=== OVERALL CONFIGURATION ===")
    print(overall)
    print("\n=== PER-AGENT CONFIGURATION ===")
    print(per_agent)
    print("\n=== INVOLVEMENT (a) PR-share among reviewed PRs ===")
    print(pr_share)
    print("\n=== INVOLVEMENT (b) event-share ===")
    print(event_share)
    print("\n=== STATISTICAL TESTS (Cohen's h) ===")
    for lvl, r in tests.items():
        print(f"  {lvl}: h={r['cohens_h']}  p_ai={r['p_ai']:.4f}  p_human={r['p_human']:.4f}")

    return {
        "overall": overall,
        "pr_share": pr_share,
        "event_share": event_share,
        "tests": tests,
    }


def print_sensitivity_comparison(strict, wide):
    """Side-by-side strict vs wide, so the reader sees how much the numbers move."""
    print("\n\n################ SENSITIVITY: strict (19) vs wide (55) ################")
    so, wo = strict["overall"]["percent"], wide["overall"]["percent"]
    comp = pd.DataFrame({"strict_%": so, "wide_%": wo, "delta_pp": (wo - so).round(2)})
    print("\n--- Configuration distribution ---")
    print(comp)

    print("\n--- 'AI more involved than human' (event-share % AI) ---")
    print(f"  strict: {strict['event_share']['pct_events_AI']}%   "
          f"wide: {wide['event_share']['pct_events_AI']}%")
    print("\n--- Effect size (event-level Cohen's h) ---")
    print(f"  strict: {strict['tests']['event_level']['cohens_h']}   "
          f"wide: {wide['tests']['event_level']['cohens_h']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", default=ALLOWLIST_CSV)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also run the wide (55-bot) AI set and compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ai_reviewers = load_allowlist(args.allowlist)
    pr, reviews, comments = load_tables()
    print(f"[check] AI-authored PRs (paper reports ~931,785): {pr['id'].nunique():,}")
    print(f"[check] authoring agents: {dict(Counter(pr['agent']))}")

    strict = run_analysis(pr, reviews, comments, ai_reviewers, args.out, tag="strict")

    # construct-validity diagnostic (uses strict events)
    ev = build_review_events(pr, reviews, comments, ai_reviewers)
    delta_ct, disagree = allowlist_vs_userflag_delta(ev)
    delta_ct.to_csv(f"{args.out}/rq1_allowlist_vs_userflag.csv")
    print(f"\n=== ALLOWLIST vs dataset user_type flag: {disagree:,} events disagree ===")
    print(delta_ct)

    if args.sensitivity:
        wide_reviewers = load_wide_allowlist(args.allowlist)
        wide = run_analysis(pr, reviews, comments, wide_reviewers, args.out, tag="wide")
        print_sensitivity_comparison(strict, wide)


if __name__ == "__main__":
    main()