import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("rq1_outputs/rq1_finding4_self_cross_human.csv", index_col=0)
# order agents largest-sample first, rename for display
order = ["OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code"]
disp = {"OpenAI_Codex":"OpenAI Codex","Claude_Code":"Claude Code"}
df = df.loc[order]
pct = df[["self %", "cross-AI %", "human %"]]
fig, ax = plt.subplots(figsize=(7, 3.2))
bottom = [0]*len(df)
colors = {"self %":"#c44", "cross-AI %":"#48a", "human %":"#999"}
labels = {"self %":"Self (closed loop)", "cross-AI %":"Cross-vendor AI", "human %":"Human"}
for col in ["self %", "cross-AI %", "human %"]:
    ax.barh([disp.get(a,a) for a in df.index], pct[col], left=bottom,
            color=colors[col], label=labels[col])
    bottom = [b+v for b,v in zip(bottom, pct[col])]
ax.set_xlabel("Share of review events (%)")
ax.set_xlim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("rq1_finding4_split.pdf", bbox_inches="tight")
print("saved rq1_finding4_split.pdf")