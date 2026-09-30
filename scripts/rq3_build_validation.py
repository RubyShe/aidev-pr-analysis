#!/usr/bin/env python3
"""
RQ4: build the gold-standard human-labeled validation set and report
inter-rater agreement.

Inputs:
  file1 (384): comment_id, comment_text, author_agent, reviewer_type, human_label
  file2 (174): comment_id, comment_text, author_agent, reviewer_type,
               my_label, prof_label, final_label   (the disagreement subset)

Outputs:
  gold_384.csv  : 384 comments with a single FINAL human label. For the 174
                  disagreement comments we take file2.final_label; for the rest
                  we keep file1.human_label.
  Inter-rater agreement (Cohen's kappa + raw %) reconstructed over all 384:
     - the 174 disagreement rows use my_label vs prof_label from file2
     - the remaining 210 rows are agreements, so both annotators = final label
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

FILE1 = "../files/validation_labeled.csv"    # adjust names/paths
FILE2 = "../files/validation_disagreement.csv" 
OUT = "rq4_outputs/gold_384.csv"


def main():
    f1 = pd.read_csv(FILE1)
    f2 = pd.read_csv(FILE2)
    print(f"[load] file1={len(f1)} rows, file2={len(f2)} rows")

    # --- build final label ---
    final_map = dict(zip(f2["comment_id"], f2["final_label"]))
    gold = f1.copy()
    gold["final_label"] = gold.apply(
        lambda r: final_map.get(r["comment_id"], r["human_label"]), axis=1
    )
    n_updated = gold["comment_id"].isin(f2["comment_id"]).sum()
    print(f"[merge] updated {n_updated} disagreement rows with reconciled labels "
          f"({len(gold)-n_updated} kept from file1)")
    gold.to_csv(OUT, index=False)
    print(f"[save] {OUT}  ({len(gold)} comments, one final label each)")

    # --- reconstruct both annotators' labels over all 384 for inter-rater kappa ---
    dis_ids = set(f2["comment_id"])
    a_labels, b_labels = [], []
    f2i = f2.set_index("comment_id")
    for _, r in gold.iterrows():
        cid = r["comment_id"]
        if cid in dis_ids:
            a_labels.append(f2i.loc[cid, "my_label"])
            b_labels.append(f2i.loc[cid, "prof_label"])
        else:
            # agreement row: both annotators gave the same (final) label
            a_labels.append(r["final_label"])
            b_labels.append(r["final_label"])

    raw_agree = sum(a == b for a, b in zip(a_labels, b_labels)) / len(a_labels)
    kappa = cohen_kappa_score(a_labels, b_labels)
    print("\n=== Inter-rater agreement (over all 384) ===")
    print(f"  raw agreement: {raw_agree*100:.1f}%")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  (disagreements reconciled: {len(dis_ids)})")

    # label distribution sanity check
    print("\n=== Final label distribution ===")
    print(gold["final_label"].value_counts().to_string())


if __name__ == "__main__":
    main()