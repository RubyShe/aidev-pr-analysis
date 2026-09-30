#!/usr/bin/env python3
"""RQ5 Figure 1: overall technical acceptance (IMPLEMENTED/NOT/UNCLEAR)."""
import os
import pandas as pd
import matplotlib.pyplot as plt

OUT = "../outputs/rq5_technical_overall.pdf"   # adjust to your figure folder
os.makedirs(os.path.dirname(OUT), exist_ok=True)

j = pd.read_csv("rq5_outputs/rq5_judgments.csv")
counts = j["label"].value_counts()
order = ["IMPLEMENTED", "NOT_IMPLEMENTED", "UNCLEAR"]
vals = [counts.get(k, 0) for k in order]
labels = ["Implemented", "Not implemented", "Unclear"]
colors = ["#3a9a5a", "#c0392b", "#999"]   # green / red / grey

fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+5, f"{v}\n({v/sum(vals)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of suggestions")
ax.set_ylim(0, max(vals)*1.18)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}  counts={dict(zip(order,vals))}")