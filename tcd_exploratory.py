"""Exploratory diagnostics on the frozen test (NOT part of the decision rules). Run from campaign/."""
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tcd_frozen")
import numpy as np, pandas as pd
import tcd_frozen as T
P = pd.read_csv("tcd_frozen/tcd_pf.csv")
jr = T.jr_factory()
MASK = np.load("fields.npz")["mask"]
X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True)
TR = X[X.family != "laminate"].reset_index(drop=True)
TR = TR.merge(pd.DataFrame([dict(id=int(i), **jr(MASK[int(i)])) for i in TR.id]), on="id")
cols = T.E1 + ["J_1.5"]
m, s = TR[cols].mean(), TR[cols].std()
A = np.column_stack([np.ones(len(TR)), ((TR[cols] - m) / s).values])
c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ TR.R_sigma.values)
out = {}
for st in ("C1", "C2"):
    te = P[(P.set == st) & P.separated]
    pred = np.column_stack([np.ones(len(te)), ((te[cols] - m) / s).values]) @ c
    y = te.R_sigma.values; r = y - pred
    out[st] = dict(n=len(te), std_y=float(y.std(ddof=1)), mean_resid=float(r.mean()), rmse=float(np.sqrt(np.mean(r**2))),
                   R2_after_offset=float(1 - np.sum((r - r.mean())**2) / np.sum((y - y.mean())**2)),
                   J15_range_test=[float(te["J_1.5"].min()), float(te["J_1.5"].max())],
                   by_family={f: float(r[te.family.values == f].mean()) for f in te.family.unique()})
out["design_J15_range"] = [float(TR["J_1.5"].min()), float(TR["J_1.5"].max())]
out["design_std_R_sigma_inc"] = float(TR.R_sigma.std())
json.dump(out, open("tcd_frozen/tcd_exploratory.json", "w"), indent=1)
print(json.dumps(out, indent=1))
