#!/usr/bin/env python3
"""
RQ2: Do different AI combinations result in more suggested changes?

Scope: AI-to-AI pull requests only (PRs authored by an AI agent AND reviewed
by at least one AI reviewer, with no human reviewer -- i.e. the "AI review"
configuration from RQ1).

Within those, each PR is labeled:
    self      : every AI reviewer is the SAME vendor as the authoring agent
    cross     : at least one AI reviewer is a DIFFERENT vendor
(PRs whose reviewers are a mix of self and cross count as cross, since a
different-vendor reviewer is present.)

We compare self vs cross on THREE activity metrics per PR:
    (1) n_reviews   : formal review submissions
    (2) n_inline    : inline review comments (the "suggested changes")
    (3) n_events    : reviews + inline comments (total volume)

Statistics: distributions are counts and heavily skewed, so we use
Mann-Whitney U (two-sided) and report Cliff's delta as the effect size,
plus medians and means for each group. Strict 19-account allowlist.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HF_BASE = "hf://datasets/hao-li/AIDev-full"
PR_TABLE = "pull_request"
REVIEWS_TABLE = "pr_reviews"
COMMENTS_TABLE = "pr_review_comments"
ALLOWLIST_CSV = "../files/llm_bot_allowlist_1.csv"
OUT_DIR = "rq2_outputs"

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


def cliffs_delta(a, b):
    """Cliff's delta via the U statistic. Sign: positive => a tends > b."""
    a, b = np.asarray(a), np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    # use rank-based computation through Mann-Whitney U
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    delta = (2.0 * u) / (n * m) - 1.0
    return delta


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
    """Per review event: PR id, reviewer, is_ai, is_human, event type."""
    rev = reviews[["pr_id", "user", "user_type"]].rename(columns={"pr_id": "id"})
    rev["event"] = "review"

    com = comments[["pull_request_url", "user", "user_type"]].copy()
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
    events["is_ai"] = events["user"].isin(ai_reviewers)
    events["is_human"] = events["user_type"].astype(str).str.lower() == "user"
    events["is_other_bot"] = (~events["is_ai"]) & (~events["is_human"])
    events = events[~events["is_other_bot"]].copy()   # drop CI/static/PR-mgmt bots
    events = events.join(pr.set_index("id")["agent"], on="id")
    events["author_vendor"] = events["agent"].map(AUTHOR_VENDOR)
    events["reviewer_vendor"] = events["user"].map(REVIEWER_VENDOR)
    return events


def select_groups(events):
    """Partition AI-authored PRs into RQ2 groups.
    human   : >=1 human reviewer, 0 AI reviewers   (baseline)
    ai_self : AI-to-AI, all AI reviewers same vendor as author
    ai_cross: AI-to-AI, >=1 AI reviewer a different vendor
    hybrid  : >=1 AI reviewer AND >=1 human reviewer
    (AI-to-AI = >=1 AI reviewer, 0 human reviewers.)"""
    per_pr = events.groupby("id").agg(any_ai=("is_ai", "max"),
                                      any_human=("is_human", "max"))
    human_ids = set(per_pr[(~per_pr["any_ai"]) & (per_pr["any_human"])].index)
    ai_ids = set(per_pr[(per_pr["any_ai"]) & (~per_pr["any_human"])].index)
    hybrid_ids = set(per_pr[(per_pr["any_ai"]) & (per_pr["any_human"])].index)
    return human_ids, ai_ids, hybrid_ids


def label_self_cross(events, ai_ids):
    """For each AI-to-AI PR: self if all AI reviewers share the author's vendor,
    cross if any AI reviewer is a different vendor."""
    ev = events[events["id"].isin(ai_ids) & events["is_ai"]].copy()
    ev["is_cross"] = ev["reviewer_vendor"] != ev["author_vendor"]
    per_pr = ev.groupby("id")["is_cross"].max()   # any cross-vendor reviewer -> cross
    label = per_pr.map({True: "cross", False: "self"})
    return label   # Series indexed by PR id


def per_pr_activity(events, ids, reviewer_mask):
    """Count reviews, inline comments, total per PR, restricted to the given PR
    ids and to events matching reviewer_mask ('is_ai' or 'is_human')."""
    ev = events[events["id"].isin(ids) & events[reviewer_mask]]
    g = ev.groupby("id")["event"].value_counts().unstack(fill_value=0)
    for c in ["review", "comment"]:
        if c not in g.columns:
            g[c] = 0
    g = g.rename(columns={"review": "n_reviews", "comment": "n_inline"})
    g["n_events"] = g["n_reviews"] + g["n_inline"]
    return g[["n_reviews", "n_inline", "n_events"]]


def compare_three(df):
    """Kruskal-Wallis across the three groups, then pairwise Mann-Whitney vs the
    human baseline with Cliff's delta. df has columns: metric cols + 'group'
    with values human / ai_self / ai_cross."""
    from scipy.stats import kruskal
    rows = []
    for metric in ["n_reviews", "n_inline", "n_events"]:
        h = df.loc[df["group"] == "human", metric].values
        s = df.loc[df["group"] == "ai_self", metric].values
        c = df.loc[df["group"] == "ai_cross", metric].values

        kw_stat, kw_p = kruskal(h, s, c)

        # pairwise vs human baseline; delta sign positive => group > human
        u_s, p_s = mannwhitneyu(s, h, alternative="two-sided")
        u_c, p_c = mannwhitneyu(c, h, alternative="two-sided")
        d_s = cliffs_delta(s, h)
        d_c = cliffs_delta(c, h)
        # and self vs cross
        u_sc, p_sc = mannwhitneyu(s, c, alternative="two-sided")
        d_sc = cliffs_delta(s, c)

        rows.append({
            "metric": metric,
            "human_median": np.median(h), "self_median": np.median(s), "cross_median": np.median(c),
            "human_mean": round(np.mean(h), 3), "self_mean": round(np.mean(s), 3), "cross_mean": round(np.mean(c), 3),
            "KW_p": kw_p,
            "self_vs_human_d": round(d_s, 3), "self_vs_human_p": p_s,
            "cross_vs_human_d": round(d_c, 3), "cross_vs_human_p": p_c,
            "self_vs_cross_d": round(d_sc, 3), "self_vs_cross_p": p_sc,
        })
    return pd.DataFrame(rows).set_index("metric")


def hybrid_decomposition(events, hybrid_ids):
    """Within hybrid PRs (both AI and human reviewers), count AI-produced and
    human-produced activity separately, per PR. Returns paired per-PR table and
    a Wilcoxon signed-rank test (paired, since both sides act on the same PR)."""
    from scipy.stats import wilcoxon
    ev = events[events["id"].isin(hybrid_ids)]
    ai = per_pr_activity(ev, hybrid_ids, "is_ai").add_prefix("ai_")
    hu = per_pr_activity(ev, hybrid_ids, "is_human").add_prefix("hu_")
    both = ai.join(hu, how="outer").fillna(0)

    rows = []
    for m in ["n_reviews", "n_inline", "n_events"]:
        a = both[f"ai_{m}"].values
        h = both[f"hu_{m}"].values
        # paired test: AI vs human on the SAME PRs
        try:
            w, p = wilcoxon(a, h, zero_method="wilcox")
        except ValueError:
            w, p = np.nan, np.nan
        rows.append({
            "metric": m,
            "ai_median": np.median(a), "human_median": np.median(h),
            "ai_mean": round(np.mean(a), 3), "human_mean": round(np.mean(h), 3),
            "wilcoxon_W": w, "p_value": p,
            "pct_PRs_AI_ge_human": round((a >= h).mean() * 100, 1),
        })
    return both, pd.DataFrame(rows).set_index("metric")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ai_reviewers = load_allowlist(ALLOWLIST_CSV)
    pr, reviews, comments = load_tables()
    events = build_events(pr, reviews, comments, ai_reviewers)

    human_ids, ai_ids, hybrid_ids = select_groups(events)
    print(f"[scope] human-review PRs (baseline): {len(human_ids):,}")
    print(f"[scope] AI-to-AI PRs: {len(ai_ids):,}")
    print(f"[scope] hybrid (AI+human) PRs: {len(hybrid_ids):,}")

    sc_label = label_self_cross(events, ai_ids)
    print(f"[split] AI-to-AI: {dict(Counter(sc_label))}")

    act_human = per_pr_activity(events, human_ids, "is_human").assign(group="human")
    act_ai = per_pr_activity(events, ai_ids, "is_ai").join(sc_label.rename("sc")).assign(
        group=lambda d: np.where(d["sc"] == "self", "ai_self", "ai_cross")
    ).drop(columns=["sc"])

    df = pd.concat([act_human, act_ai[["n_reviews", "n_inline", "n_events", "group"]]])
    stats = compare_three(df)

    # compute hybrid stats FIRST (needed for the joint correction)
    hyb_pairs, hyb_stats = hybrid_decomposition(events, hybrid_ids)

    # ---- multiple-comparison correction across the whole RQ2 family (BH/FDR) ----
    from statsmodels.stats.multitest import multipletests
    pcols_three = ["self_vs_human_p", "cross_vs_human_p", "self_vs_cross_p"]
    all_p = []
    for m in stats.index:
        for c in pcols_three:
            all_p.append(stats.loc[m, c])
    for m in hyb_stats.index:
        all_p.append(hyb_stats.loc[m, "p_value"])
    reject, p_adj, _, _ = multipletests(all_p, method="fdr_bh")
    i = 0
    for m in stats.index:
        for c in pcols_three:
            stats.loc[m, c + "_adj"] = p_adj[i]; i += 1
    for m in hyb_stats.index:
        hyb_stats.loc[m, "p_value_adj"] = p_adj[i]; i += 1

    # ---- save + print ----
    df.to_csv(f"{OUT_DIR}/rq2_per_pr_activity.csv")
    stats.to_csv(f"{OUT_DIR}/rq2_stats.csv")
    hyb_pairs.to_csv(f"{OUT_DIR}/rq2_hybrid_pairs.csv")
    hyb_stats.to_csv(f"{OUT_DIR}/rq2_hybrid_stats.csv")

    print("\n=== RQ2a: activity by group (human baseline vs AI self vs AI cross) ===")
    print(f"group sizes: human={len(act_human):,}  "
          f"ai_self={(act_ai['group']=='ai_self').sum():,}  "
          f"ai_cross={(act_ai['group']=='ai_cross').sum():,}")
    print(stats.to_string())

    print("\n=== RQ2b: within HYBRID PRs, AI vs human activity (paired, same PRs) ===")
    print(hyb_stats.to_string())

    print("\nCliff's delta sign: positive => that group has MORE activity than the comparison.")
    print("|d|<0.147 negligible, <0.33 small, <0.474 medium, else large.")


if __name__ == "__main__":
    main()