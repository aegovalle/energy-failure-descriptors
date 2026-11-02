"""Sensitivity of the realized crack-path cost to the extraction settings (v0.7).
Reprocesses the stored baseline damage fields (no new phase-field solves):
damage threshold d_c in {0.85, 0.90, 0.95} and 16- vs 8-neighbour stencils.
P_min is recomputed with the same stencil. Run from campaign/:
    python path_sensitivity.py      -> path_sensitivity.csv, path_sensitivity.json
"""
import json
import numpy as np
import pandas as pd
from pf_fft import PhaseFieldRVE
from descr_tools import min_path, omega, HALF_OFFS, HALF_OFFS8

rng = np.random.default_rng(31)
X = pd.read_csv("v06_desc.csv")[["id", "A_lig"]].merge(pd.read_csv("pf_descriptors.csv"), on="id")
X = X.sort_values("id").reset_index(drop=True)
X = X.rename(columns={"R_Gpath": "R_path_base"})
M = np.load("fields.npz")["mask"]
D = np.load("pf_fields.npz")["d"]
d_hom = np.load("pf_out/hom_d.npy")
N = M.shape[1]
G_hom = PhaseFieldRVE(np.zeros((N, N), bool)).Gc
GC = [PhaseFieldRVE(m).Gc for m in M]
E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LV = {"G0": ["f"], "G1": ["f", "rhoI"], "GL": ["f", "rhoI", "A_lig"], "G2": ["f", "rhoI", "lnPm"],
      "E0": ["f", "e11_m", "e22_m", "e12_m"], "E1": E1}
lam = (X.family == "laminate").values
lam_v = lam & (X.orient == "v").values
lam_h = lam & (X.orient == "h").values

rows, summ = [], {}
for st_name, offs in (("16", HALF_OFFS), ("8", HALF_OFFS8)):
    pm_hom = min_path(G_hom, np.ones((N, N), bool), offs)
    X["lnPm"] = [np.log(min_path(g, np.ones((N, N), bool), offs) / pm_hom) for g in GC]
    for thr in (0.85, 0.90, 0.95):
        tag = f"s{st_name}_d{thr:.2f}"
        cp_hom = min_path(G_hom, d_hom >= thr, offs)
        Rp = np.array([min_path(GC[i], D[i] >= thr, offs) for i in range(len(X))]) / cp_hom
        X[tag] = np.log(Rp)
        ok = np.isfinite(X[tag].values)
        f = X.f.values
        s = {"n_nan": int((~ok).sum()), "C_hom": float(cp_hom),
             "spearman_vs_base": float(pd.Series(X[tag][ok]).corr(pd.Series(X.R_path_base[ok]), method="spearman")),
             "max_abs_diff_vs_base": float(np.nanmax(np.abs(X[tag] - X.R_path_base))),
             "lam_h_max_relerr_1p2f": float(np.nanmax(np.abs(np.exp(X[tag][lam_h]) / (1 + 2 * f[lam_h]) - 1))),
             "lam_v_max_abs": float(np.nanmax(np.abs(X[tag][lam_v]))),
             "Pmin_violations": int((X[tag][ok] < X.lnPm[ok] - 0.01).sum())}
        for scope in ("all", "inc"):
            Dd = X[ok] if scope == "all" else X[ok & ~lam]
            R = Dd[tag].values
            for lv, cols in LV.items():
                o = omega(Dd[cols].values, R, 0.01, 300, rng)
                rows.append(dict(setting=tag, scope=scope, level=lv, ratio=o["ratio"], n=len(Dd)))
                s[f"{scope}_{lv}"] = o["ratio"]
        summ[tag] = s
        print(tag, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}), flush=True)
pd.DataFrame(rows).to_csv("path_sensitivity.csv", index=False)
json.dump(summ, open("path_sensitivity.json", "w"), indent=1)
