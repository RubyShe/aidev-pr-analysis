#!/usr/bin/env python3
"""
RQ5 step 1b: fetch PR file patches from the GitHub REST API for each sampled
suggestion, cache them to disk (for the replication package), and attach the
patch of the SAME file the suggestion is about.

Auth: reads GITHUB_TOKEN from the environment (personal access token).
Rate limit: 5,000 req/hour authenticated; the script paginates PR files and
sleeps when the remaining budget runs low.

Reproducibility: every fetched PR's file list is cached to
rq5_outputs/pr_files_cache.jsonl, so the analysis can be re-run offline from the
cache and the cache shipped in the replication package.
"""

import os
import re
import json
import time
import requests
import pandas as pd

OUT_DIR = "rq5_outputs"
SAMPLE = f"{OUT_DIR}/rq5_sample_with_diffs.csv"   # from rq5_sample.py (has pr_id, path, body, agent)
CACHE = f"{OUT_DIR}/pr_files_cache.jsonl"
OUT = f"{OUT_DIR}/rq5_sample_api_diffs.csv"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

# We need owner/repo/number per suggestion. The sample was built from
# pull_request_url; if the sample lacks it, we reconstruct from the AIDev PR table.
HF_BASE = "hf://datasets/hao-li/AIDev-full"


def load_pr_urls(sample):
    """Ensure each row has owner/repo/number. Join AIDev pull_request on pr_id
    to get html_url, then parse owner/repo/number."""
    pr = pd.read_parquet(f"{HF_BASE}/pull_request.parquet")[["id", "html_url"]]
    m = sample.merge(pr, left_on="pr_id", right_on="id", how="left")
    parts = m["html_url"].str.extract(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
    m["owner"], m["repo"], m["number"] = parts[0], parts[1], parts[2]
    return m


def fetch_pr_files(owner, repo, number, cache_seen):
    """Fetch the list of files (with patches) for a PR, with pagination.
    Returns list of {filename, patch, status, additions, deletions}."""
    key = f"{owner}/{repo}#{number}"
    if key in cache_seen:
        return cache_seen[key]
    files = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
        r = requests.get(url, headers=HEADERS, params={"per_page": 100, "page": page})
        if r.status_code == 404:
            files = None  # PR/repo gone
            break
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(5, reset - int(time.time()) + 2)
            print(f"  rate limited; sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        batch = r.json()
        for f in batch:
            files.append({
                "filename": f.get("filename"),
                "patch": f.get("patch", ""),      # unified diff text (may be absent for huge/binary)
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
            })
        if len(batch) < 100:
            break
        page += 1
    cache_seen[key] = files
    with open(CACHE, "a") as fh:
        fh.write(json.dumps({"key": key, "files": files}) + "\n")
    return files


def main():
    if not TOKEN:
        raise SystemExit("Set GITHUB_TOKEN in your environment first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    sample = pd.read_csv(SAMPLE)
    sample = load_pr_urls(sample)
    sample = sample.dropna(subset=["owner", "repo", "number"])

    # load any existing cache (resume)
    cache_seen = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try:
                rec = json.loads(line)
                cache_seen[rec["key"]] = rec["files"]
            except Exception:
                pass
        print(f"[cache] {len(cache_seen)} PRs already cached")

    file_diffs, statuses = [], []
    uniq_prs = sample.drop_duplicates(["owner", "repo", "number"])
    print(f"[fetch] {len(uniq_prs)} unique PRs to fetch")

    for i, (_, r) in enumerate(sample.iterrows()):
        files = fetch_pr_files(r["owner"], r["repo"], r["number"], cache_seen)
        if files is None:
            file_diffs.append(""); statuses.append("pr_unavailable"); continue
        # find the file matching the suggestion's path
        match = [f for f in files if f["filename"] == r["path"]]
        if not match:
            file_diffs.append(""); statuses.append("file_not_changed"); continue
        patch = "\n---\n".join([f["patch"] for f in match if f.get("patch")])
        if not patch.strip():
            file_diffs.append(""); statuses.append("changed_no_patch_text")
        else:
            file_diffs.append(patch[:12000]); statuses.append("ok")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sample)} suggestions processed")

    sample["file_diff"] = file_diffs
    sample["diff_status"] = statuses
    keep = ["sugg_id", "pr_id", "agent", "user", "path", "body", "diff_hunk",
            "owner", "repo", "number", "file_diff", "diff_status"]
    keep = [k for k in keep if k in sample.columns]
    sample[keep].to_csv(OUT, index=False)

    print(f"\n[done] {OUT}")
    print(sample["diff_status"].value_counts().to_string())


if __name__ == "__main__":
    main()