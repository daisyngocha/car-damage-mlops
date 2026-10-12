"""So nhãn tự gán trên Label Studio với nhãn gốc Kaggle."""

from pathlib import Path

import pandas as pd

ANN = Path("data/annotations")
key = pd.read_csv(ANN / "answer_key.csv")
ls = pd.read_csv(ANN / "ls_export_v1.csv")
ls["sample_id"] = ls["image"].str.rsplit("/", n=1).str[-1]
df = key.merge(ls[["sample_id", "damage"]], on="sample_id", how="left")

print(f"Đã gán: {df['damage'].notna().sum()}/{len(df)}")
df = df.dropna(subset=["damage"])
p_obs = (df["true_label"] == df["damage"]).mean()
p_true = df["true_label"].value_counts(normalize=True)
p_mine = df["damage"].value_counts(normalize=True)
p_exp = (p_true * p_mine).sum()
kappa = (p_obs - p_exp) / (1 - p_exp)
print(f"Agreement: {p_obs:.1%}   Cohen's kappa: {kappa:.2f}")
print(pd.crosstab(df["true_label"], df["damage"], margins=True))
df[df["true_label"] != df["damage"]].to_csv(ANN / "disagreements.csv", index=False)
