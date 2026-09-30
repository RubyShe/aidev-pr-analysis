#!/usr/bin/env python3
"""
RQ5 validation: agreement between human labels and the LLM judge on the
100-item validation sample. Reports raw agreement, Cohen's kappa, a confusion
matrix, and per-label recall. Also reports agreement on the binary
IMPLEMENTED-vs-not question (often the number that matters most).
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CSV = "rq5_outputs/rq5_validation_sample_adjudicated.csv"   # after you fill human_label
LABELS = ["IMPLEMENTED", "NOT_IMPLEMENTED", "UNCLEAR"]

df = pd.read_csv(CSV)
df["human_label"] = df["human_label"].astype(str).str.upper().str.strip()
df["label"] = df["label"].astype(str).str.upper().str.strip()
df = df[df["human_label"].isin(LABELS)]   # only rows you labeled
print(f"[validation] {len(df)} labeled rows")

raw = (df["human_label"] == df["label"]).mean()
kappa = cohen_kappa_score(df["human_label"], df["label"])
print(f"\n3-way (IMPLEMENTED/NOT/UNCLEAR):")
print(f"  raw agreement: {raw*100:.1f}%")
print(f"  Cohen's kappa: {kappa:.3f}")

print("\nConfusion matrix (rows=human, cols=LLM):")
present = [l for l in LABELS if l in set(df["human_label"]) | set(df["label"])]
cm = confusion_matrix(df["human_label"], df["label"], labels=present)
print(pd.DataFrame(cm, index=present, columns=present).to_string())

# binary view: IMPLEMENTED vs (NOT+UNCLEAR)
b_h = (df["human_label"] == "IMPLEMENTED")
b_l = (df["label"] == "IMPLEMENTED")
raw_b = (b_h == b_l).mean()
kappa_b = cohen_kappa_score(b_h, b_l)
print(f"\nBinary (IMPLEMENTED vs not):")
print(f"  raw agreement: {raw_b*100:.1f}%")
print(f"  Cohen's kappa: {kappa_b:.3f}")

print("\nPer-label agreement (human label -> matched by LLM):")
for l in present:
    sub = df[df["human_label"] == l]
    if len(sub):
        print(f"  {l:18s} n={len(sub):3d}  matched {(sub['label']==l).mean()*100:5.1f}%")