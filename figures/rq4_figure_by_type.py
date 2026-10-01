#!/usr/bin/env python3
"""RQ5 Figure 2: conditional implementation rate by suggestion type.
Reads rq5_rate_by_type.csv. Shows n per bar and de-emphasizes small-n types."""
import os
import pandas as pd
import matplotlib.pyplot as plt

# larger fonts for readability (Zhenlan's comment)
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

OUT = "../outputs/rq5_technical_by_type.pdf"   # adjust folder; rename rq3->rq5
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df = pd.read_csv("rq5_outputs/rq5_rate_by_type.csv", index_col=0)

# reliability threshold: bars with n>=40 are solid, else faded
RELIABLE_N = 50
df = df.sort_values("rate_pct", ascending=True)
labels = [t.replace("_", "\n") for t in df.index]
rates = df["rate_pct"].values
ns = df["n"].values
colors = ["#3a9a5a" if n >= RELIABLE_N else "#ccc9b8" for n in ns]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.barh(labels, rates, color=colors)
for b, r, n in zip(bars, rates, ns):
    ax.text(r + 1, b.get_y()+b.get_height()/2, f"{r:.1f}%  (n={n})",
            va="center", fontsize=11)
ax.set_xlabel("Conditional implementation rate (%)")
ax.set_xlim(0, max(rates)*1.4)
# legend explaining fading
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#3a9a5a", label=f"n \u2265 {RELIABLE_N} (reliable)"),
                   Patch(color="#ccc9b8", label=f"n < {RELIABLE_N} (interpret with caution)")],
          loc="lower right")
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")