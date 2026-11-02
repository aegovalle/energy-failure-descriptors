"""Blind validation analysis (v0.7), following blind/blind_protocol.md exactly.
Run from campaign/:  python blind_analyze.py
Outputs: blind/blind_results.json, blind/blind_omega.csv, blind/blind_pf.csv,
         ../figures/fig_blind.pdf|png
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from descr_tools import omega, omega_subsample_ci, min_path
from pf_fft import PhaseFieldRVE

rng = np.random.default_rng(20261008)
SIG_HOM, E0_HOM, C_HOM = 4.452871, 3.304542, 0.98
E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LV = {"G0": ["f"], "G1": ["f", "rhoI"], "Gd": ["f", "rhoI", "lig_min"], "GL": ["f", "rhoI", "A_lig"],
      "G2": ["f", "rhoI", "lnPmin"], "E0": ["f", "e11_m", "e22_m", "e12_m"], "E1": E1,
      "E1+L": E1 + ["A_lig"], "E1+H": E1 + ["lnB"], "Tpsi": E1 + ["J_psi"], "H": ["lnB"]}

# ------------------------------------------------------------------ data
TR = pd.read_csv("v06_desc.csv")[["id", "lig_min", "A_lig"]].merge(
    pd.read_csv("pf_descriptors.csv"), on="id").rename(columns={"R_Gpath": "R_path"})
LAM = TR[TR.family == "laminate"].reset_index(drop=True)
TR = TR[TR.family != "laminate"].reset_index(drop=True)          # 180 original inclusion RVEs
BL = pd.read_csv("blind/blind_elastic.csv")
BL["R_H"] = BL["lnB"]
TR["R_H"] = TR["lnB"]
LAM["R_H"] = LAM["lnB"]

pf_rows = []
for i in BL.bid[BL.pf]:
    fj = f"blind/pf_out/b_{i:03d}.json"
    if not os.path.exists(fj):
        continue
    o = json.load(open(fj))
    d = np.load(f"blind/pf_out/b_{i:03d}_d.npy")
    m = np.load("blind/blind_masks.npz")["mask"][i]
    cp = min_path(PhaseFieldRVE(m).Gc, d >= 0.9)
    pf_rows.append(dict(bid=i, R_sigma=np.log(o["sigma_peak"] / SIG_HOM), R_path=np.log(cp / C_HOM),
                        J_psi=-np.log(o["e0"] / E0_HOM), separated=o["separated"],
                        eps_ratio=o["eps_peak"] / o["e0"], time_s=o["time_s"]))
PF = pd.DataFrame(pf_rows)
if len(PF):
    PF = BL.merge(PF, on="bid")
    PF.to_csv("blind/blind_pf.csv", index=False)


def fit_predict(Xtr, ytr, Xte, ridge=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    P = lambda X: np.column_stack([np.ones(len(X)), (X - mu) / sd])
    A = P(Xtr)
    c = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ ytr)
    return P(Xte) @ c


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


rows = []


def ladder(D, tag, target, levels, q, train=None, ci=True):
    out = {}
    nan = int((~np.isfinite(D[target].values.astype(float))).sum())
    D = D[np.isfinite(D[target].values.astype(float))].reset_index(drop=True)
    if train is not None:
        train = train[np.isfinite(train[target].values.astype(float))]
    R = D[target].values.astype(float)
    for lv in levels:
        X = D[LV[lv]].values.astype(float)
        o = omega(X, R, q, 500, rng)
        lo, hi = omega_subsample_ci(X, R, q, rng, nsub=100, frac=0.8, nperm=40) if ci else (np.nan, np.nan)
        r2o = np.nan
        if train is not None:
            r2o = r2(R, fit_predict(train[LV[lv]].values.astype(float), train[target].values.astype(float), X))
        row = dict(set=tag, target=target, level=lv, q=q, n=len(D), ratio=o["ratio"], ci_lo=lo, ci_hi=hi,
                   p_perm=o["p_perm"], null_q05=o["null_q05"], R2_oos=r2o)
        rows.append(row)
        out[lv] = row
    return out


res = {}
A = BL[BL.set == "A"].reset_index(drop=True)
O = BL[BL.set == "O"].reset_index(drop=True)
LVS_H = ["G0", "G1", "Gd", "GL", "G2", "E0", "E1", "E1+L"]
a1 = ladder(A, "A", "R_H", LVS_H, 0.01, TR)
a2 = ladder(A, "A", "R_H", ["G1", "GL"], 0.02, TR, ci=False)
res["H1"] = dict(ratio_G1=a1["G1"]["ratio"], ratio_GL=a1["GL"]["ratio"],
                 ci_GL=[a1["GL"]["ci_lo"], a1["GL"]["ci_hi"]], ci_G1=[a1["G1"]["ci_lo"], a1["G1"]["ci_hi"]],
                 confirmed=bool(a1["GL"]["ratio"] <= 0.8 * a1["G1"]["ratio"]),
                 q2_ratio_G1=a2["G1"]["ratio"], q2_ratio_GL=a2["GL"]["ratio"],
                 q2_holds=bool(a2["GL"]["ratio"] <= 0.8 * a2["G1"]["ratio"]))
res["H2"] = dict(R2_G1=a1["G1"]["R2_oos"], R2_GL=a1["GL"]["R2_oos"], R2_E1=a1["E1"]["R2_oos"],
                 confirmed=bool(a1["GL"]["R2_oos"] - a1["G1"]["R2_oos"] >= 0.05))
res["A_corr_Alig_RH"] = float(np.corrcoef(A.A_lig, A.R_H)[0, 1])
o1 = ladder(O, "O", "R_H", LVS_H, 0.05, TR)
res["O_exploratory"] = dict(ratio_G1=o1["G1"]["ratio"], ratio_GL=o1["GL"]["ratio"],
                            R2_G1=o1["G1"]["R2_oos"], R2_GL=o1["GL"]["R2_oos"],
                            corr_Alig_RH=float(np.corrcoef(O.A_lig, O.R_H)[0, 1]))

if len(PF):
    B = PF[PF.set == "A"].reset_index(drop=True)
    OP = PF[PF.set == "O"].reset_index(drop=True)
    res["B_n"], res["B_separated"] = int(len(B)), int(B.separated.sum())
    LVS_F = ["G0", "G1", "Gd", "GL", "G2", "E0", "E1", "E1+L", "Tpsi", "H", "E1+H"]
    b_s = ladder(B, "B", "R_sigma", LVS_F, 0.05, TR)
    ladder(B, "B", "R_sigma", ["G1", "GL", "E1", "E1+H"], 0.01, TR, ci=False)
    res["H3"] = dict(ratio_GL=b_s["GL"]["ratio"], R2_GL=b_s["GL"]["R2_oos"], ratio_G1=b_s["G1"]["ratio"],
                     R2_G1=b_s["G1"]["R2_oos"], ratio_E1H=b_s["E1+H"]["ratio"], R2_E1H=b_s["E1+H"]["R2_oos"],
                     confirmed=bool(b_s["GL"]["ratio"] >= 0.5 and b_s["GL"]["R2_oos"] <= 0.4))
    b_p = ladder(B, "B", "R_path", LVS_F, 0.05, TR)
    pool = pd.concat([B, LAM], ignore_index=True)
    p_p = ladder(pool, "B+lam", "R_path", ["G0", "G1", "G2", "E0", "E1"], 0.05, None)
    viol = int((B.R_path < B.lnPmin - 0.01).sum())
    res["H4"] = dict(a_ratio_G1=p_p["G1"]["ratio"], a_ratio_G2=p_p["G2"]["ratio"],
                     a_confirmed=bool(p_p["G2"]["ratio"] <= 0.5 * p_p["G1"]["ratio"]),
                     b_ratio_G2=b_p["G2"]["ratio"], b_R2_G2=b_p["G2"]["R2_oos"],
                     b_confirmed=bool(b_p["G2"]["ratio"] >= 0.5 and b_p["G2"]["R2_oos"] <= 0.4),
                     c_violations=viol, c_confirmed=viol == 0)
    res["B_corr_Alig"] = {t: float(np.corrcoef(B.A_lig, B[t])[0, 1]) for t in ("R_H", "R_sigma", "R_path")}
    res["B_eps_ratio_mean"] = float(B.eps_ratio.mean())
    res["B_std"] = {t: float(B[t].std()) for t in ("R_sigma", "R_path")}
    if len(OP):
        res["O_pf"] = dict(n=int(len(OP)), corr_Alig={t: float(np.corrcoef(OP.A_lig, OP[t])[0, 1])
                                                     for t in ("R_H", "R_sigma", "R_path")},
                           R2_oos_GL_sigma=r2(OP.R_sigma.values, fit_predict(TR[LV["GL"]].values, TR.R_sigma.values, OP[LV["GL"]].values)),
                           violations=int((OP.R_path < OP.lnPmin - 0.01).sum()))
    res["pf_cpu_h"] = float(PF.time_s.sum() / 3600)

om = pd.DataFrame(rows)
om.to_csv("blind/blind_omega.csv", index=False)
json.dump(res, open("blind/blind_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(om.round(3).to_string(index=False))
print(json.dumps(res, indent=1, default=float))

# ------------------------------------------------------------------ figure
nP = 3 if len(PF) else 1
fig, ax = plt.subplots(1, nP, figsize=(2.6 * nP, 2.6))
ax = np.atleast_1d(ax)
for a_, D, t, lab in [(ax[0], None, "R_H", "$R_H$")] + ([(ax[1], None, "R_sigma", "$R_\\sigma$"),
                                                          (ax[2], None, "R_path", "$R_{\\rm path}$")] if len(PF) else []):
    src_tr = TR
    src_bl = A if t == "R_H" else PF[PF.set == "A"]
    src_o = O if t == "R_H" else PF[PF.set == "O"]
    a_.scatter(src_tr.A_lig, src_tr[t], s=6, c="0.7", label="design set (180)")
    a_.scatter(src_bl.A_lig, src_bl[t], s=10, c="#A63D40", label="blind, in-distribution")
    a_.scatter(src_o.A_lig, src_o[t], s=10, marker="^", c="#10324F", label="blind, aspect ratio 6")
    a_.set_xlabel(r"$A_\ell$ (frozen)", fontsize=9); a_.set_ylabel(lab, fontsize=9)
    a_.tick_params(labelsize=8)
ax[0].legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(r"..\figures\fig_blind.pdf", bbox_inches="tight")
fig.savefig(r"..\figures\fig_blind.png", dpi=200, bbox_inches="tight")
print("figure written")
