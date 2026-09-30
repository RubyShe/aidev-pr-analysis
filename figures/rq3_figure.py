#!/usr/bin/env python3
"""
RQ4 figure: comment-type distribution across the three reviewer configurations.
Grouped bar chart, 10 taxonomy categories x 3 groups. Reads rq4_pct_by_group.csv.
Colors matched to the paper's palette (human = grey, AI groups = red/blue family).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# larger fonts for readability (Zhenlan's comment)
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

CSV = "rq4_outputs/rq4_pct_by_group.csv"
OUT = "../outputs/rq4_bb_three_way_comparison.pdf"   # adjust to your figure folder
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(CSV, index_col=0)   # rows = group, cols = categories

CATS = ["DEFECT", "CODE_IMPROVEMENT", "UNDERSTANDING", "KNOWLEDGE_TRANSFER",
        "TESTING", "EXTERNAL_IMPACT", "SOCIAL", "REVIEW_TOOL", "SETUP", "MISC"]
CATS = [c for c in CATS if c in df.columns]
LABELS = {"CODE_IMPROVEMENT": "Code\nImprov.", "KNOWLEDGE_TRANSFER": "Knowl.\nTransfer",
          "EXTERNAL_IMPACT": "External\nImpact", "REVIEW_TOOL": "Review\nTool",
          "UNDERSTANDING": "Under-\nstanding"}

# order groups: same_tool, diff_tool, human
group_order = ["same_tool", "diff_tool", "human"]
group_disp = {"same_tool": "Self-review",
              "diff_tool": "Cross-vendor",
              "human": "Human"}
colors = {"same_tool": "#c44", "diff_tool": "#48a", "human": "#999"}

x = np.arange(len(CATS))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 3.6))
for i, g in enumerate(group_order):
    if g not in df.index:
        continue
    vals = [df.loc[g, c] for c in CATS]
    ax.bar(x + (i - 1) * w, vals, w, label=group_disp[g], color=colors[g])
ax.set_xticks(x)
ax.set_xticklabels([LABELS.get(c, c.title()) for c in CATS], fontsize=11)
ax.set_ylabel("Share of comments (%)")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")