r"""v0.9 re-analysis (no new simulations). Run from campaign/ after v09_fields.py.

Outputs
  v09_levels.csv      Omega ratio (1e4 response permutations), p_perm, n_q, within-cell and
                      translation-copy pairs among the selected pairs, LOCO R^2, McShane
                      e*(L) at L in {0, 1, 3, 10}, and L needed to reach the noise floor
  v09_incremental.csv conditional permutation tests for added blocks (block weights
                      0.5, 1, 2), Holm-adjusted within each target family
  v09_mcshane.csv     e*(L) curves
  v09_dupcoord.csv    redundant-coordinate control (V4)
  v09_noise.json      noise ceiling for R_path, misc. checks (pair distances, n_q, ...)
"""
import json
import numpy as np
import pandas as pd
from descr_tools import surrogate_cv, omega_subsample_ci
from v09_stats import zscore, omega_sel, mcshane, lipschitz_needed, cond_perm, holm, pair_index

rng = np.random.default_rng(20261102)
NPERM = 10000
BCOND = 2000

X = (pd.read_csv("v06_desc.csv")[["id", "lig_min", "A_lig"]]
     .merge(pd.read_csv("pf_descriptors.csv"), on="id")
     .merge(pd.read_csv("v09_fields.csv"), on="id")
     .rename(columns={"R_Gpath": "R_path"}).sort_values("id").reset_index(drop=True))
F = np.load("fields.npz")
SBAR, MASK = F["sbar"], F["mask"]
N = MASK.shape[1]
for m in (2, 5, 10, 40):
    X[f"RH_m{m}"] = [m * np.log(SBAR[i][~MASK[i]].max())
                     + np.log(np.sum((SBAR[i][~MASK[i]] / SBAR[i][~MASK[i]].max()) ** m) / N ** 2)
                     for i in range(len(X))]
X["lnWm"] = np.log(X.Wph_m)                       # positive control: matrix-phase energy

# MKS baseline: PCA of the cropped two-point autocovariance maps (unsupervised, all cells)
S2 = np.load("v09_S2.npy").reshape(len(X), -1)
S2c = S2 - S2.mean(0)
U, s, Vt = np.linalg.svd(S2c, full_matrices=False)
PCS = U * s
for k in range(20):
    X[f"pc{k}"] = PCS[:, k]
expl = (s ** 2 / np.sum(s ** 2))

E0 = ["f", "e11_m", "e22_m", "e12_m"]
E1 = E0 + ["Jgap_m", "Jgap_i"]
COV = [f"{c}_{t}" for t in ("m", "i") for c in ("c11", "c22", "c12", "c1122", "c1112", "c2212")]
SPL = [f"{c}_{t}" for t in ("m", "i") for c in ("trp_mu", "trp_sd", "dev_mu", "dev_sd")]
LEVELS = {
    "G0": ["f"], "G1": ["f", "rhoI"], "GL": ["f", "rhoI", "A_lig"], "G2": ["f", "rhoI", "lnPmin"],
    "Gpsi": ["f", "rhoI", "lnPpsi"],
    "E0": E0, "E1": E1, "E0+cov": E0 + COV, "E0+split": E0 + SPL,
    "E1+L": E1 + ["A_lig"], "T1": E1 + ["I"], "Tpsi": E1 + ["J_psi"], "E1+H": E1 + ["lnB"],
    "MKS5": [f"pc{k}" for k in range(5)], "MKS10": [f"pc{k}" for k in range(10)],
    "MKS20": [f"pc{k}" for k in range(20)],
}
TARGETS = {"R_E": "W_hom", "R_H": "lnB", "R_sigma": "R_sigma", "R_path": "R_path",
           "W_m (control)": "lnWm", "R_H m=2": "RH_m2", "R_H m=5": "RH_m5", "R_H m=10": "RH_m10",
           "R_H m=40": "RH_m40"}
NOISE = {"R_E": 0.006, "R_H": 0.5, "R_sigma": 0.01, "R_path": 0.04}
SCOPES = {"all": np.ones(len(X), bool), "inc": (X.family != "laminate").values,
          "n>1": ((X.family != "laminate") & (X.n > 1)).values}
LGRID = np.r_[0, np.geomspace(0.01, 100, 81)]

rows, mc_rows = [], []
for sc, msk in SCOPES.items():
    D = X[msk].reset_index(drop=True)
    II, JJ = pair_index(len(D))
    same_cell = (D.cell.values[II] == D.cell.values[JJ])
    tcopy = same_cell & (D.n.values[II] == 1) & (D.family.values[II] == "disks")
    for tn, col in TARGETS.items():
        if sc != "all" and tn == "R_E":
            pass
        R = D[col].values.astype(float)
        for lv, cols in LEVELS.items():
            if tn == "R_H" and lv in ("T1", "E1+H"):
                continue
            Z = zscore(D[cols].values)
            o = omega_sel(Z, R, 0.01, NPERM, rng, II=II, JJ=JJ)
            sl, _ = surrogate_cv(D[cols].values, R, D.cell.values, 1, 5, rng)
            e = mcshane(Z, R, [0, 1, 3, 10])
            r = dict(scope=sc, n=len(D), target=tn, level=lv, ratio=o["ratio"], p_perm=o["p_perm"],
                     null_q05=o["null_q05"], nq=o["nq"], npairs=o["npairs"],
                     within_cell=int(np.sum(o["sel"] & same_cell)),
                     transl_copies=int(np.sum(o["sel"] & tcopy)), R2_loco=sl["R2"],
                     e_L0=e[0], e_L1=e[1], e_L3=e[2], e_L10=e[3])
            if tn in NOISE:
                r["L_needed"] = lipschitz_needed(Z, R, NOISE[tn])
            rows.append(r)
            if tn in NOISE and lv in ("G1", "GL", "G2", "Gpsi", "E0", "E1", "E0+cov", "E0+split",
                                     "T1", "E1+H", "MKS10"):
                for L, ev in zip(LGRID, mcshane(Z, R, LGRID)):
                    mc_rows.append(dict(scope=sc, target=tn, level=lv, L=L, e_star=ev))
        print(sc, tn, flush=True)
LV = pd.DataFrame(rows)
LV.to_csv("v09_levels.csv", index=False)
pd.DataFrame(mc_rows).to_csv("v09_mcshane.csv", index=False)

# subsampling intervals for the n>1 scope, main targets
ci_rows = []
D = X[SCOPES["n>1"]].reset_index(drop=True)
for tn in ("R_H", "R_sigma", "R_path"):
    R = D[TARGETS[tn]].values.astype(float)
    for lv in ("G1", "GL", "G2", "Gpsi", "E0", "E1", "E0+cov", "E0+split", "T1", "Tpsi", "E1+H", "MKS10"):
        if tn == "R_H" and lv in ("T1", "E1+H"):
            continue
        lo, hi = omega_subsample_ci(D[LEVELS[lv]].values, R, 0.01, rng, nsub=100, frac=0.8, nperm=60)
        ci_rows.append(dict(target=tn, level=lv, ci_lo=lo, ci_hi=hi))
pd.DataFrame(ci_rows).to_csv("v09_ci_n1.csv", index=False)
print("ci done", flush=True)

# ---------------------------------------------------------------- incremental tests
INC = [  # (target, base level, added columns, label)
    ("R_H", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"), ("R_sigma", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"),
    ("R_path", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"),
    ("W_m (control)", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"),
    ("R_H m=2", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"), ("R_H m=5", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"),
    ("R_H m=10", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"), ("R_H m=40", "E0", ["Jgap_m", "Jgap_i"], "E0->E1"),
    ("R_H", "E0", COV, "E0->E0+cov"), ("R_sigma", "E0", COV, "E0->E0+cov"), ("R_path", "E0", COV, "E0->E0+cov"),
    ("W_m (control)", "E0", COV, "E0->E0+cov"),
    ("R_H", "E0", SPL, "E0->E0+split"), ("R_sigma", "E0", SPL, "E0->E0+split"),
    ("R_path", "E0", SPL, "E0->E0+split"),
    ("R_H", "G1", ["A_lig"], "G1->GL"), ("R_sigma", "G1", ["A_lig"], "G1->GL"),
    ("R_path", "G1", ["A_lig"], "G1->GL"),
    ("R_path", "G1", ["lnPmin"], "G1->G2"), ("R_sigma", "G1", ["lnPmin"], "G1->G2"),
    ("R_path", "G1", ["lnPpsi"], "G1->Gpsi"), ("R_sigma", "G1", ["lnPpsi"], "G1->Gpsi"),
    ("R_sigma", "E1", ["I"], "E1->T1"), ("R_sigma", "E1", ["lnB"], "E1->E1+H"),
    ("R_path", "E1", ["I"], "E1->T1"),
    ("R_H", "E1", ["A_lig"], "E1->E1+L"), ("R_sigma", "E1", ["A_lig"], "E1->E1+L"),
]
inc_rows = []
for sc in ("all", "n>1"):
    D = X[SCOPES[sc]].reset_index(drop=True)
    for tn, base, add, lab in INC:
        R = D[TARGETS[tn]].values.astype(float)
        Zk = zscore(D[LEVELS[base]].values)
        for w in (0.5, 1.0, 2.0):
            B = BCOND if w == 1.0 else 500
            c = cond_perm(Zk, D[add].values, R, 0.01, B, rng, k_nn=5, w=w)
            # ordinary ratio of the enlarged level at this weight (response-permutation null)
            Zw = np.hstack([Zk, zscore(D[add].values) * w])
            o = omega_sel(Zw, R, 0.01, 2000, rng)
            ob = omega_sel(Zk, R, 0.01, 2000, rng)
            inc_rows.append(dict(scope=sc, target=tn, step=lab, w=w, ratio_base=ob["ratio"],
                                 ratio_new=o["ratio"], Omega_base=ob["Omega"], Omega_new=c["Omega"],
                                 Omega_cnull=c["Omega_cnull"], ratio_c=c["ratio_c"], p_cond=c["p_cond"],
                                 B=B))
        print("inc", sc, tn, lab, flush=True)
INCD = pd.DataFrame(inc_rows)
INCD["p_holm"] = np.nan
for (sc, w), g in INCD.groupby(["scope", "w"]):
    INCD.loc[g.index, "p_holm"] = holm(g.p_cond.values)
INCD.to_csv("v09_incremental.csv", index=False)

# ---------------------------------------------------------------- V4 redundant coordinate
dup_rows = []
for sc in ("all", "n>1"):
    D = X[SCOPES[sc]].reset_index(drop=True)
    for tn in ("R_H", "R_sigma", "R_path"):
        R = D[TARGETS[tn]].values.astype(float)
        for lv in ("G1", "E0", "E1", "GL"):
            Z = zscore(D[LEVELS[lv]].values)
            base = omega_sel(Z, R, 0.01, 4000, rng)["ratio"]
            for j in range(Z.shape[1]):
                Zd = np.hstack([Z, Z[:, [j]]])
                dup_rows.append(dict(scope=sc, target=tn, level=lv, dup_col=LEVELS[lv][j], ratio=base,
                                     ratio_dup=omega_sel(Zd, R, 0.01, 4000, rng)["ratio"]))
pd.DataFrame(dup_rows).to_csv("v09_dupcoord.csv", index=False)

# ---------------------------------------------------------------- misc checks and noise ceiling
misc = {"pca_explained_10": float(expl[:10].sum()), "pca_explained_20": float(expl[:20].sum()),
        "min_tr_all": float(X.min_tr.min()),
        "nq": {"q0.005_all": int(np.ceil(0.005 * 23220 - 1e-9)), "q0.01_all": int(np.ceil(0.01 * 23220 - 1e-9)),
               "q0.01_inc": int(np.ceil(0.01 * 16110 - 1e-9)), "q0.01_n>1": int(np.ceil(0.01 * 11175 - 1e-9))}}
for sc in ("all", "inc", "n>1"):
    D = X[SCOPES[sc]].reset_index(drop=True)
    Z = zscore(D[E1].values)
    idx = {int(v): k for k, v in enumerate(D.id.values)}
    dd = {}
    for a, b in ((54, 59), (49, 73)):
        if a in idx and b in idx:
            dd[f"{a}-{b}"] = float(np.linalg.norm(Z[idx[a]] - Z[idx[b]]))
    misc[f"E1_pair_dist_{sc}"] = dd
# constant surrogate: midrange attains half the range exactly
misc["half_range"] = {"R_E": float(np.ptp(X.W_hom)) / 2, "R_H": float(np.ptp(X.lnB)) / 2}
# noise ceiling for R_path among n>1 inclusions
D = X[SCOPES["n>1"]]
var_path = float(D.R_path.var())
n1 = X[(X.family == "disks") & (X.n == 1)]
sd_transl = float(np.sqrt(n1.groupby("cell").R_path.var().mean()))
res = pd.read_csv("pf_resolution_summary.csv")
res_inc = res[res.family != "laminate"]
sd_res = float(np.sqrt(np.mean(res_inc.dR_path ** 2)))
ps = pd.read_csv("path_sensitivity.csv")
misc["path_sensitivity_columns"] = ps.columns.tolist()
misc["R_path_noise"] = {"var_n>1": var_path, "sd_n>1": float(np.sqrt(var_path)),
                        "sd_translation": sd_transl, "sd_resolution_rms": sd_res,
                        "R2max_translation": 1 - sd_transl ** 2 / var_path,
                        "R2max_resolution": 1 - sd_res ** 2 / var_path,
                        "R2max_combined": 1 - (sd_transl ** 2 + sd_res ** 2) / var_path}
json.dump(misc, open("v09_noise.json", "w"), indent=1)
print(json.dumps(misc, indent=1))
