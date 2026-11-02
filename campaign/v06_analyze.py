r"""
v0.6 analysis: insufficiency ladders for four targets with the geometric
ligament descriptor, q-sensitivity, subsampling intervals, and random K-fold
versus leave-one-nominal-cell-out (LOCO) surrogates.
Inputs : pf_descriptors.csv (from pf_analyze.py), fields.npz
Outputs: v06_desc.csv, v06_omega.csv, v06_surrogates.csv, v06_summary.json,
         ../figures/fig_v06_ligament.pdf|png, ../figures/fig_pf_omega.pdf|png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from descr_tools import ligament, ligament_amp, omega, omega_subsample_ci, surrogate_cv

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 10, "legend.fontsize": 8})
FIG = r"..\figures"
rng = np.random.default_rng(2026)

X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True)
M = np.load("fields.npz")["mask"]
G = np.array([ligament(m) for m in M])
A = np.array([ligament_amp(m, k=10.0, ell=0.025) for m in M])
X["lig_min"], X["lig_ell"] = G[:, 0], G[:, 1]
X["A_lig"], X["A_ell"] = A[:, 0], A[:, 1]
X = X.rename(columns={"R_Gpath": "R_path"})
X[["id", "family", "n", "f", "cell", "lig_min", "lig_ell", "A_lig", "A_ell"]].to_csv("v06_desc.csv", index=False)

E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LEVELS = {
    "control": ["e11_g", "e22_g", "e12_g"],
    "G0": ["f"],
    "G1": ["f", "rhoI"],
    "Gd": ["f", "rhoI", "lig_min"],          # raw gap width
    "GL": ["f", "rhoI", "A_lig"],            # ligament amplification
    "G2": ["f", "rhoI", "lnPmin"],
    "GL+P": ["f", "rhoI", "A_lig", "lnPmin"],
    "E0": ["f", "e11_m", "e22_m", "e12_m"],
    "E1": E1,
    "E1+L": E1 + ["A_lig"],
    "T1": E1 + ["I"],
    "Tpsi": E1 + ["J_psi"],
    "H": ["lnB"],
    "E1+H": E1 + ["lnB"],
}
TARGETS = {"R_E": "W_hom", "R_H": "lnB", "R_sigma": "R_sigma", "R_path": "R_path"}
SKIP = {"R_E": {"H", "E1+H"}, "R_H": {"H", "E1+H", "T1"}}   # trivial/identity levels
QS = (0.005, 0.01, 0.02, 0.05)

om_rows, sur_rows = [], []
for scope in ("all", "inc"):
    D = X if scope == "all" else X[X.family != "laminate"].reset_index(drop=True)
    for tn, col in TARGETS.items():
        R = D[col].values.astype(float)
        for lv, cols in LEVELS.items():
            if lv in SKIP.get(tn, ()):
                continue
            Xd = D[cols].values
            row = dict(scope=scope, target=tn, level=lv, dim=len(cols))
            for q in QS:
                o = omega(Xd, R, q, 500, rng, return_pair=(q == 0.01))
                row[f"ratio_q{q}"] = o["ratio"]
                if q == 0.01:
                    row.update(Omega=o["Omega"], Omega_null=o["Omega_null"], npairs=o["npairs"],
                               p_perm=o["p_perm"], null_q05=o["null_q05"], null_q95=o["null_q95"],
                               pair_i=int(D.id[o["pair"][0]]), pair_j=int(D.id[o["pair"][1]]))
            lo, hi = omega_subsample_ci(Xd, R, 0.01, rng, nsub=100, frac=0.8, nperm=40)
            row.update(ci_lo=lo, ci_hi=hi)
            om_rows.append(row)
            for deg in (1, 2):
                sk, _ = surrogate_cv(Xd, R, None, deg, 5, rng)
                sl, _ = surrogate_cv(Xd, R, D["cell"].values, deg, 5, rng)
                sur_rows.append(dict(scope=scope, target=tn, level=lv, deg=deg,
                                     R2_kfold=sk["R2"], einf_kfold=sk["einf"],
                                     R2_loco=sl["R2"], einf_loco=sl["einf"], RMSE_loco=sl["RMSE"]))
        print(scope, tn, "done", flush=True)
om = pd.DataFrame(om_rows); om.to_csv("v06_omega.csv", index=False)
sur = pd.DataFrame(sur_rows); sur.to_csv("v06_surrogates.csv", index=False)

inc = X[X.family != "laminate"]
summ = {"corr_inc": {c: {t: float(np.corrcoef(inc[c], inc[TARGETS[t]])[0, 1]) for t in TARGETS}
                     for c in ("lig_min", "A_lig", "A_ell", "I", "J_psi", "lnB", "lnPmin")},
        "pair_49_73": {c: [float(X.loc[49, c]), float(X.loc[73, c])]
                       for c in ("lig_min", "A_lig", "A_ell", "I", "J_psi", "R_sigma", "lnB")},
        "pair_54_59": {c: [float(X.loc[54, c]), float(X.loc[59, c])]
                       for c in ("lig_min", "A_lig", "A_ell", "I", "lnB", "R_sigma")}}
json.dump(summ, open("v06_summary.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(om[["scope", "target", "level", "ratio_q0.005", "ratio_q0.01", "ratio_q0.02", "ratio_q0.05",
          "ci_lo", "ci_hi", "pair_i", "pair_j"]].round(3).to_string(index=False))
print(sur.round(3).to_string(index=False))
print(json.dumps(summ, indent=1))

# ---------------------------------------------------------------- figures
SHORT = {"control": "control", "G0": "$f$", "G1": "$f,\\rho_I$", "Gd": "$f,\\rho_I,\\delta_{\\min}$",
         "GL": "$f,\\rho_I,A_\\ell$", "G2": "$f,\\rho_I,P_{\\min}$", "GL+P": "$f,\\rho_I,A_\\ell,P_{\\min}$",
         "E0": "E0", "E1": "E1", "E1+L": "E1$+A_\\ell$", "T1": "E1$+I$", "Tpsi": "E1$+J_\\psi$",
         "H": "$R_H$", "E1+H": "E1$+R_H$"}
order = ["control", "G0", "G1", "Gd", "GL", "G2", "GL+P", "E0", "E1", "E1+L", "T1", "Tpsi", "H", "E1+H"]
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
for k, (tn, title) in enumerate((("R_sigma", r"$R_\sigma=\ln(\sigma_{\rm peak}/\sigma_{\rm peak}^{\rm hom})$"),
                                 ("R_path", r"$R_{\rm path}=\ln(C_{\rm path}/C_{\rm path}^{\rm hom})$"))):
    x = np.arange(len(order))
    for off, scope, colr, lab in ((-0.2, "all", "#10324F", "all 216 RVEs"),
                                  (0.2, "inc", "#A63D40", "inclusions only (180)")):
        s = om[(om.target == tn) & (om.scope == scope)].set_index("level").loc[order]
        ax[k].bar(x + off, s["ratio_q0.01"], 0.4, color=colr, label=lab)
        ax[k].errorbar(x + off, s["ratio_q0.01"],
                       yerr=[np.clip(s["ratio_q0.01"] - s.ci_lo, 0, None), np.clip(s.ci_hi - s["ratio_q0.01"], 0, None)],
                       fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
    ax[k].set_xticks(x); ax[k].set_xticklabels([SHORT[l] for l in order], rotation=60, ha="right", fontsize=9)
    ax[k].tick_params(axis="y", labelsize=9)
    ax[k].axhline(1, color="gray", ls=":", lw=1)
    ax[k].set_title(title, fontsize=10); ax[k].set_yscale("log")
ax[0].set_ylabel(r"$\Omega(q)/\Omega^{\rm null}$ at $q=1\%$", fontsize=10)
fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           fontsize=8.5, bbox_to_anchor=(0.52, 1.04))
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pf_omega.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_pf_omega.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
for fam, mk in (("disks", "o"), ("ellipses", "^")):
    s = inc.family == fam
    for a_, t in zip(ax, ("lnB", "R_sigma", "R_path")):
        a_.scatter(inc.A_lig[s], inc[t][s], s=8, marker=mk, alpha=0.6, label=fam)
for a_, t, lab in zip(ax, ("lnB", "R_sigma", "R_path"), ("$R_H$", "$R_\\sigma$", "$R_{\\rm path}$")):
    a_.scatter(X.A_lig[[49, 73]], X[t][[49, 73]], s=40, facecolors="none", edgecolors="k", lw=1.0)
    for i, nm in ((49, "A"), (73, "B")):
        a_.annotate(nm, (X.A_lig[i], X[t][i]), xytext=(4, 2), textcoords="offset points", fontsize=7)
    r = np.corrcoef(inc.A_lig, inc[t])[0, 1]
    a_.set_xlabel(r"ligament amplification $A_\ell$"); a_.set_ylabel(lab)
    a_.set_title(f"inclusions: $r$ = {r:.2f}", fontsize=8)
ax[0].legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_v06_ligament.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_v06_ligament.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("figures written")
