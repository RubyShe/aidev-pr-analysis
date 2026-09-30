# Replication Package: When Agents Write and Agents Review

Replication package for the TOSEM submission "When Agents Write and Agents Review: Where Humans Still Shape the Code." This repository contains the analysis code, derived artifacts, and cached results needed to reproduce the tables and figures in the paper.

## Dataset

We analyze the AIDev-full dataset (a fork of AIDev providing full-scope commit and review data), pulled directly from HuggingFace at runtime: hf://datasets/hao-li/AIDev-full. It is not bundled here.

Our analysis covers 931,785 AI-authored pull requests from five coding agents (OpenAI Codex, GitHub Copilot, Cursor, Devin, Claude Code) across 116,003 repositories.

## Structure

- scripts/  : analysis scripts (one group per research question)
- figures/  : plotting scripts that produce the paper figures
- data/     : curated inputs (AI-reviewer allowlist, reviewer sanity-check)
- cached/   : precomputed results, so figures/tables can be regenerated offline

## Setup

Install dependencies:

    pip install -r requirements.txt

RQ3 and RQ4 use the OpenAI API (GPT-4.1) for classification/judgment; set OPENAI_API_KEY in your environment if you run the full path.

## Reproduction

### Quick path (from cached results, no API or dataset download)

Regenerate the paper's figures from cached/:

    python figures/rq1_figure.py          # RQ1 composition
    python figures/rq2_figure.py          # RQ2 activity
    python figures/rq3_figure.py          # RQ3 feedback types
    python figures/rq4_figure_overall.py  # RQ4 overall implementation
    python figures/rq4_figure_by_type.py  # RQ4 implementation by type

Tables (RQ1 configurations, RQ3 populations) are read from the corresponding CSVs in cached/.

### Full path (re-run from raw data)

- RQ1: python scripts/rq1_config_distribution.py ; python scripts/rq1_reviewer_dominance.py
- RQ2: python scripts/rq2_review_activity.py
- RQ3: python scripts/rq3_sampling.py ; python scripts/rq3_classify.py ; python scripts/rq3_results.py ; python scripts/rq3_population.py ; python scripts/rq3_build_validation.py
- RQ4: python scripts/rq4_sampling.py ; python scripts/rq4_fetch_diffs.py ; python scripts/rq4_judge.py ; python scripts/rq4_bytype.py ; python scripts/rq4_build_validation.py ; python scripts/rq4_validation_agreement.py

## AI reviewer allowlist

data/llm_bot_allowlist.csv is the curated list of 19 AI reviewer accounts (11 AI coding agents + 8 LLM-based review tools), built by manual inspection of the 89 bot accounts appearing as reviewers. See Section 4.4 of the paper.

## Large artifacts

Two large files are archived separately (on Zenodo, DOI to be added for the camera-ready) rather than in this repository:

- rq4_all_comments_labeled.csv (~232 MB): full labeled comment set
- pr_files_cache.jsonl (~200 MB): cached GitHub API diffs for RQ4

The quick path does not require them; the full path uses them to avoid re-fetching from the GitHub API.

## Note on file naming

Script prefixes follow the paper's research questions: rq1_, rq2_, rq3_ (feedback types), rq4_ (implementation). Some cached CSVs retain earlier internal names (rq4_*, rq5_*); these correspond to the paper's RQ3 and RQ4 respectively.
