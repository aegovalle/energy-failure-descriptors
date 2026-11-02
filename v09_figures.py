"""Figures for Draft v0.9 (run from campaign/ after v09_analysis.py and v09_postprocess.py)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v09_stats import zscore, pair_index
from scipy.spatial.distance import pdist

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9})
FIG = "../figures/"
L = pd.read_csv("v09_levels.csv")

# ------------------------------------------------------------------ hazard-exponent sweep
ms = [2, 5, 10, 20, 40]
tmap = {2: "R_H m=2", 5: "R_H m=5", 10: "R_H m=10", 20: "R_H", 40: "R_H m=40"}
lv_style = {"E0": ("E0: phase means", "C0", "o-"), "E1": ("E1: + Jensen gaps", "C1", "s--"),
            "E0+cov": ("E0 + full covariance", "C4", "^:"), "G1": (r"G1: $(f,\rho_I)$", "0.5", "v-"),
            "GL": (r"GL: G1 + $A_\ell$", "C3", "D-")}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for sc, a in (("n>1", ax[0]),):
    for lv, (lab, c, st) in lv_style.items():
        y = [L[(L.scope == sc) & (L.target == tmap[m]) & (L.level == lv)].ratio.values[0] for m in ms]
        a.plot(ms, y, st, color=c, label=lab, ms=5)
    a.set_xscale("log"); a.set_xticks(ms); a.set_xticklabels(ms)
    a.set_xlabel(r"hazard exponent $\beta$"); a.set_ylabel(r"$\Omega/\Omega^{\rm null}$ at $q=1\%$")
    a.set_title("worst-case insufficiency (150 cells, n>1)", fontsize=10)
for lv, (lab, c, st) in lv_style.items():
    y = [L[(L.scope == "n>1") & (L.target == tmap[m]) & (L.level == lv)].R2_loco.values[0] for m in ms]
    ax[1].plot(ms, y, st, color=c, label=lab, ms=5)
ax[1].set_xscale("log"); ax[1].set_xticks(ms); ax[1].set_xticklabels(ms)
ax[1].set_xlabel(r"hazard exponent $\beta$"); ax[1].set_ylabel(r"$R^2_{\rm LOCO}$ (linear)")
ax[1].set_title("average prediction", fontsize=10)
ax[1].legend(loc="lower left", fontsize=8)
g1 = np.array([L[(L.scope == "n>1") & (L.target == tmap[m]) & (L.level == "G1")].R2_loco.values[0] for m in ms])
gl = np.array([L[(L.scope == "n>1") & (L.target == tmap[m]) & (L.level == "GL")].R2_loco.values[0] for m in ms])
e0 = np.array([L[(L.scope == "n>1") & (L.target == tmap[m]) & (L.level == "E0")].R2_loco.values[0] for m in ms])
e1 = np.array([L[(L.scope == "n>1") & (L.target == tmap[m]) & (L.level == "E1")].R2_loco.values[0] for m in ms])
ax[2].plot(ms, gl - g1, "D-", color="C3", label=r"$G1\to GL$ (ligament)")
ax[2].plot(ms, e1 - e0, "s--", color="C1", label=r"$E0\to E1$ (Jensen gaps)")
ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_xscale("log"); ax[2].set_xticks(ms); ax[2].set_xticklabels(ms)
ax[2].set_xlabel(r"hazard exponent $\beta$"); ax[2].set_ylabel(r"$\Delta R^2_{\rm LOCO}$")
ax[2].set_title("gain from the added block", fontsize=10); ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG + "fig_m_sweep.pdf"); fig.savefig(FIG + "fig_m_sweep.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ strength decomposition
D = pd.read_csv("v09_strength_decomp.csv")
X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True)
XV = X
fam = D.family.values
mk = {"laminate": ("s", "C0"), "disks": ("o", "C1"), "ellipses": ("^", "C2")}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for f, (m_, c) in mk.items():
    s = fam == f
    ax[0].scatter(D.onset_part[s], D.R_sigma[s], marker=m_, s=12, color=c, label=f)
    if f != "laminate":
        ax[1].scatter(XV.lnVeff[s], D.lnGamma[s], marker=m_, s=12, color=c, label=f)
    ax[2].scatter(np.log(X.P_min[s] / 0.98), X.R_Gpath[s], marker=m_, s=12, color=c, label=f)
lim = np.array([D.onset_part.min(), D.onset_part.max()])
ax[0].plot(lim, lim, "k:", lw=1, label="failure at onset")
ax[0].set_xlabel(r"onset part  $\ln R_E - J_\psi$"); ax[0].set_ylabel(r"$R_\sigma$")
ax[0].legend(fontsize=8)
ax[0].set_title(r"$R_\sigma=\ln R_E-J_\psi+\ln(\Gamma_\sigma/\Gamma_\sigma^{\rm hom})$", fontsize=10)
inc = fam != "laminate"
r = np.corrcoef(XV.lnVeff[inc], D.lnGamma[inc])[0, 1]
ax[1].set_xlabel(r"tail width  $\ln V_{\rm eff}$ (elastic)"); ax[1].set_ylabel(r"growth term  $\ln(\Gamma_\sigma/\Gamma_\sigma^{\rm hom})$")
ax[1].set_title(f"inclusions: correlation {r:.2f}", fontsize=10)
x = np.log(X.P_min / 0.98)
ax[2].plot([0, x.max()], [0, x.max()], "k:", lw=1, label="identity (lower bound)")
ax[2].set_xlabel(r"$\ln(P_{\min}/P_{\min}^{\rm hom})$"); ax[2].set_ylabel(r"$R_{\rm path}$")
ax[2].legend(fontsize=8, loc="upper left")
ins = ax[2].inset_axes([0.55, 0.08, 0.42, 0.42])
for f, (m_, c) in mk.items():
    s = (fam == f) & (fam != "laminate")
    ins.scatter(x[s], X.R_Gpath[s], marker=m_, s=6, color=c)
ins.plot([0, 0.06], [0, 0.06], "k:", lw=0.8)
ins.set_title("inclusions", fontsize=7); ins.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(FIG + "fig_v09_strength_path.pdf"); fig.savefig(FIG + "fig_v09_strength_path.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ McShane curves (dimension-normalized metric)
Xall = (pd.read_csv("v06_desc.csv")[["id", "A_lig"]].merge(pd.read_csv("pf_descriptors.csv"), on="id")
        .merge(pd.read_csv("v09_fields.csv"), on="id").rename(columns={"R_Gpath": "R_path"})
        .sort_values("id").reset_index(drop=True))
E0 = ["f", "e11_m", "e22_m", "e12_m"]; E1 = E0 + ["Jgap_m", "Jgap_i"]
COV = [f"{c}_{t}" for t in ("m", "i") for c in ("c11", "c22", "c12", "c1122", "c1112", "c2212")]
lev = {"G1": ["f", "rhoI"], "GL": ["f", "rhoI", "A_lig"], "E0": E0, "E1": E1, "E0+cov": E0 + COV,
       "E1+H": E1 + ["lnB"]}
sty = {"G1": ("0.5", "-"), "GL": ("C3", "-"), "E0": ("C0", "-"), "E1": ("C1", "--"), "E0+cov": ("C4", ":"),
       "E1+H": ("C2", "-.")}
Dn = Xall[(Xall.family != "laminate") & (Xall.n > 1)].reset_index(drop=True)
Ls = np.r_[0, np.geomspace(0.01, 100, 120)]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, (tn, col, noise) in zip(ax, (("R_H", "lnB", 0.5), ("R_sigma", "R_sigma", 0.01))):
    R = Dn[col].values
    II, JJ = pair_index(len(R))
    dR = np.abs(R[II] - R[JJ])
    for lv, cols in lev.items():
        if tn == "R_H" and lv == "E1+H":
            continue
        Z = zscore(Dn[cols].values)
        d = pdist(Z) / np.sqrt(Z.shape[1])
        e = [0.5 * np.max(np.maximum(dR - Lv * d, 0)) for Lv in Ls]
        c, st = sty[lv]
        a.plot(Ls[1:], e[1:], st, color=c, label=lv)
    a.axhline(noise, color="k", lw=0.6, ls=":")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"Lipschitz constant $\Lambda$ (per RMS standardized unit)")
    a.set_ylabel(r"$e^*(\Lambda)$")
    a.set_title(("hazard $R_H$" if tn == "R_H" else r"strength $R_\sigma$") + " (150 cells, n>1)", fontsize=10)
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG + "fig_mcshane.pdf"); fig.savefig(FIG + "fig_mcshane.png", dpi=200)
plt.close(fig)
print("figures done")
