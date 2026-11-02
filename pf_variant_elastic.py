"""Elastic (pre-critical) descriptors for the robustness variants:
orient (transposed masks), contrast3 (E_i/E_m=3), ell040 (hazard length 0.04).
Writes pf_var/<variant>/elastic.csv"""
import os, json
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, lame, analyze_state

df = pd.read_csv("campaign_results.csv")
MASK = np.load("fields.npz")["mask"]
N = MASK.shape[1]
S = json.load(open("pf_var_subsets.json"))
EBAR = (1.0, 0.0, 0.0)
sol = FFTElastic2D(N)
CFG = {"orient": (list(range(len(df))), True, 10.0, 0.025),
       "contrast3": (S["s61"], False, 3.0, 0.025),
       "ell040": (S["s61"], False, 10.0, 0.040)}
lam_m, mu_m = lame(1.0, 0.3)
for var, (ids, tr, Ei, ell) in CFG.items():
    lam_i, mu_i = lame(Ei, 0.3)
    rows = []
    for i in ids:
        mask = MASK[i].T.copy() if tr else MASK[i]
        lam = np.where(mask, lam_i, lam_m)
        mu = np.where(mask, mu_i, mu_m)
        st, ss, it, err = sol.solve(lam, mu, EBAR, tol=1e-7, maxit=4000)
        o = analyze_state(mask, st, ss, (lam_m, mu_m, lam_i, mu_i), EBAR, m=20, ell=ell)
        o.pop("_sbar", None)
        o["id"] = i
        rows.append(o)
    out = pd.DataFrame(rows)
    os.makedirs(f"pf_var/{var}", exist_ok=True)
    out.to_csv(f"pf_var/{var}/elastic.csv", index=False)
    print(var, len(out), "done", flush=True)
