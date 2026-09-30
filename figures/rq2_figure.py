import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "../outputs/rq2_activity.pdf"   # adjust to your figure folder
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# medians from the two RQ2 outputs
stats = pd.read_csv("../outputs/rq2_outputs/rq2_stats.csv", index_col=0)
hyb   = pd.read_csv("../outputs/rq2_outputs/rq2_hybrid_stats.csv", index_col=0)

metrics = ["n_reviews", "n_inline", "n_events"]
labels  = ["Formal reviews", "Inline comments", "Total events"]

# build median matrix: rows = groups, cols = metrics
groups = {
    "Human":       [stats.loc[m, "human_median"] for m in metrics],
    "AI self":     [stats.loc[m, "self_median"]  for m in metrics],
    "AI cross":    [stats.loc[m, "cross_median"] for m in metrics],
    "Hybrid: AI":  [hyb.loc[m, "ai_median"]      for m in metrics],
    "Hybrid: human":[hyb.loc[m, "human_median"]  for m in metrics],
}
colors = ["#999", "#e6a0a0", "#c44", "#48a", "#bcd"]

x = np.arange(len(metrics))
n = len(groups)
w = 0.16

fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (name, vals) in enumerate(groups.items()):
    ax.bar(x + (i - (n-1)/2)*w, vals, w, label=name, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Median count per PR")
ax.legend(fontsize=8, ncol=2, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"saved {OUT_PATH}")