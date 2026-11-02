r"""
Additional robustness checks (introduced in Draft v0.8, retained in v0.9):
1. Hazard exponent sweep m in {2, 5, 10, 20, 40} on the 216 elastic fields (Point 2.7).
2. Translation-copy removal: repeating inclusion comparisons on the 150 multi-inclusion
   cells (n > 1, zero translation copies) and on the 153 deduplicated inclusion cells
   (only 1 realization per n=1 nominal cell) (Point 2.6).
3. Paired bootstrap/subsampling 90 % intervals for level differences
   Delta(Omega/Omega_null) = ratio(level_2) - ratio(level_1) on the same subsamples (Point 2.6).
Outputs: v08_checks.json, v08_m_sweep.csv, v08_n1_control.csv, v08_paired_diff.csv
"""
import json
import numpy as np
import pandas as pd
from descr_tools import omega, omega_subsample_ci, surrogate_cv

rng = np.random.default_rng(202608)

X = pd.read_csv("v06_desc.csv")[["id", "lig_min", "A_lig", "A_ell"]].merge(
    pd.read_csv("pf_descriptors.csv"), on="id").rename(columns={"R_Gpath": "R_path"}).sort_values("id").reset_index(drop=True)
F = np.load("fields.npz")
SBAR, MASK = F["sbar"], F["mask"]
N = MASK.shape[1]

E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LEVELS = {
    "G0": ["f"],
    "G1": ["f", "rhoI"],
    "GL": ["f", "rhoI", "A_lig"],
    "G2": ["f", "rhoI", "lnPmin"],
    "E0": ["f", "e11_m", "e22_m", "e12_m"],
    "E1": E1,
    "E1+L": E1 + ["A_lig"],
    "T1": E1 + ["I"],
    "Tpsi": E1 + ["J_psi"],
    "E1+H": E1 + ["lnB"],
}

# ==============================================================================
# 1. Exponent sweep m in {2, 5, 10, 20, 40} for R_H^(m) = ln int_{Omega_m} s^m dx
# ==============================================================================
MS = (2, 5, 10, 20, 40)
RH_m = {}
I_m = {}
for m in MS:
    vals = np.empty(len(X))
    intens = np.empty(len(X))
    for i in range(len(X)):
        sm = SBAR[i][~MASK[i]]
        sp = sm.max()
        intens[i] = m * np.log(sp)
        vals[i] = intens[i] + np.log(np.sum((sm / sp) ** m) / (N * N))
    RH_m[m] = vals
    I_m[m] = intens

m_rows = []
for m in MS:
    R_all = RH_m[m]
    for scope, mask_sc in (("all", np.ones(len(X), bool)),
                           ("inc_180", (X.family != "laminate").values),
                           ("inc_n>1", ((X.family != "laminate") & (X.n > 1)).values)):
        D = X[mask_sc].reset_index(drop=True)
        R = R_all[mask_sc]
        for lv in ("G1", "GL", "E0", "E1"):
            cols = LEVELS[lv]
            o = omega(D[cols].values, R, 0.01, 500, rng)
            sl, _ = surrogate_cv(D[cols].values, R, D["cell"].values, 1, 5, rng)
            m_rows.append(dict(m=m, scope=scope, level=lv, ratio=o["ratio"],
                               p_perm=o["p_perm"], R2_loco=sl["R2"]))
    # Also record counterexample pair (54, 59) scaled initiation strain ratio exp(-dR_H/m)
df_m = pd.DataFrame(m_rows)
df_m.to_csv("v08_m_sweep.csv", index=False)

cx_m = {int(m): dict(dR_H=float(RH_m[m][54] - RH_m[m][59]),
                     strain_ratio=float(np.exp(-(RH_m[m][54] - RH_m[m][59]) / m)))
        for m in MS}

# ==============================================================================
# 2. Translation-copy control: 180 inclusions vs 150 multi-inclusion (n>1)
#    vs 153 deduplicated inclusions (only real==0 kept for n==1)
# ==============================================================================
SCOPES = {
    "inc_180": (X.family != "laminate").values,
    "inc_150_n>1": ((X.family != "laminate") & (X.n > 1)).values,
    "inc_153_dedup": ((X.family != "laminate") & ((X.n > 1) | (X.id % 10 == 6))).values,
}
# wait: check which realizations exist for n==1
n1_first_ids = X[(X.family == "disks") & (X.n == 1)].groupby("cell").id.first().values
SCOPES["inc_153_dedup"] = ((X.family != "laminate") & ((X.n > 1) | (X.id.isin(n1_first_ids)))).values

ctrl_rows = []
for sc_name, sc_mask in SCOPES.items():
    D = X[sc_mask].reset_index(drop=True)
    for tn, col in (("R_H", "lnB"), ("R_sigma", "R_sigma"), ("R_path", "R_path")):
        R = D[col].values.astype(float)
        for lv in ("G0", "G1", "GL", "G2", "E0", "E1", "E1+L", "T1", "Tpsi", "E1+H"):
            if tn == "R_H" and lv in ("T1", "E1+H"):
                continue
            cols = LEVELS[lv]
            o1 = omega(D[cols].values, R, 0.01, 500, rng)
            o2 = omega(D[cols].values, R, 0.02, 300, rng)
            lo, hi = omega_subsample_ci(D[cols].values, R, 0.01, rng, nsub=100, frac=0.8, nperm=60)
            sl, _ = surrogate_cv(D[cols].values, R, D["cell"].values, 1, 5, rng)
            ctrl_rows.append(dict(scope=sc_name, n=len(D), target=tn, level=lv,
                                  ratio_q001=o1["ratio"], p_perm=o1["p_perm"],
                                  ci_lo=lo, ci_hi=hi, ratio_q002=o2["ratio"],
                                  R2_loco=sl["R2"]))
df_ctrl = pd.DataFrame(ctrl_rows)
df_ctrl.to_csv("v08_n1_control.csv", index=False)

# ==============================================================================
# 3. Paired subsampling intervals for level differences Delta(ratio) = ratio(B) - ratio(A)
# ==============================================================================
def paired_diff_ci(D, col, lv_a, lv_b, q=0.01, nsub=200, frac=0.8, nperm=60):
    R = D[col].values.astype(float)
    Xa = D[LEVELS[lv_a]].values
    Xb = D[LEVELS[lv_b]].values
    full_a = omega(Xa, R, q, 500, rng)["ratio"]
    full_b = omega(Xb, R, q, 500, rng)["ratio"]
    n = len(R)
    m = int(round(frac * n))
    diffs = np.empty(nsub)
    for b in range(nsub):
        idx = rng.choice(n, m, replace=False)
        ra = omega(Xa[idx], R[idx], q, nperm, rng)["ratio"]
        rb = omega(Xb[idx], R[idx], q, nperm, rng)["ratio"]
        diffs[b] = rb - ra
    return dict(ratio_a=float(full_a), ratio_b=float(full_b),
                diff=float(full_b - full_a),
                diff_q05=float(np.quantile(diffs, 0.05)),
                diff_q50=float(np.quantile(diffs, 0.50)),
                diff_q95=float(np.quantile(diffs, 0.95)),
                prob_b_lt_a=float(np.mean(diffs < 0)))

COMPS = [
    ("all", "R_H", "lnB", "E0", "E1"),
    ("inc_180", "R_H", "lnB", "E0", "E1"),
    ("inc_150_n>1", "R_H", "lnB", "E0", "E1"),
    ("inc_180", "R_H", "lnB", "G1", "GL"),
    ("inc_150_n>1", "R_H", "lnB", "G1", "GL"),
    ("inc_180", "R_sigma", "R_sigma", "E0", "E1"),
    ("inc_150_n>1", "R_sigma", "R_sigma", "E0", "E1"),
    ("inc_180", "R_sigma", "R_sigma", "E1", "E1+H"),
    ("inc_150_n>1", "R_sigma", "R_sigma", "E1", "E1+H"),
    ("all", "R_path", "R_path", "G1", "G2"),
    ("inc_180", "R_path", "R_path", "G1", "G2"),
    ("inc_150_n>1", "R_path", "R_path", "G1", "G2"),
]

p_rows = []
for sc_name, tn, col, la, lb in COMPS:
    D = X if sc_name == "all" else X[SCOPES[sc_name]].reset_index(drop=True)
    res = paired_diff_ci(D, col, la, lb, q=0.01)
    p_rows.append(dict(scope=sc_name, target=tn, comp=f"{la}->{lb}", **res))
df_p = pd.DataFrame(p_rows)
df_p.to_csv("v08_paired_diff.csv", index=False)

out = {"cx_m": cx_m, "m_sweep": m_rows, "paired_diff": p_rows}
with open("v08_checks.json", "w") as f:
    json.dump(out, f, indent=2)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print("=== M SWEEP ===")
print(df_m.pivot_table(index=["scope", "level"], columns="m", values="ratio").round(3))
print("=== M SWEEP R2_LOCO ===")
print(df_m.pivot_table(index=["scope", "level"], columns="m", values="R2_loco").round(3))
print("=== N>1 CONTROL (q=1%) ===")
print(df_ctrl.pivot_table(index=["target", "level"], columns="scope", values=["ratio_q001", "ci_lo", "ci_hi", "R2_loco"]).round(3))
print("=== PAIRED DIFFERENCES ===")
print(df_p.round(3).to_string(index=False))
