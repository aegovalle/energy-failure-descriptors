r"""
Analysis of the descriptor-discovery campaign.
Outputs: summary.json, omega_table.csv, surrogate_table.csv, figures/*.pdf|png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 10, "legend.fontsize": 8})
FIG = r"..\figures"
M = 20
ALPHA = 0.01 ** (1.0 / M)

df = pd.read_csv("campaign_results.csv")
F = np.load("fields.npz")
SBAR, MASK = F["sbar"], F["mask"]
N = MASK.shape[1]

E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LEVELS = {
    "C_glob": ["e11_g", "e22_g", "e12_g"],
    "G0": ["f"],
    "G1": ["f", "rhoI"],
    "E0": ["f", "e11_m", "e22_m", "e12_m"],
    "E1": E1,
    "E1+G": E1 + ["rhoI"],
    "T1": E1 + ["I"],
    "T2": E1 + ["I", "lnValpha"],
}
LABEL = {"C_glob": "global means\n(control)", "G0": "$f$", "G1": "$f,\\rho_I$",
         "E0": "phase means\n(E0)", "E1": "+Jensen gaps\n(E1)", "E1+G": "E1$+\\rho_I$",
         "T1": "E1$+I$", "T2": "E1$+I+\\ln V_\\alpha$"}
TARGETS = {"R_E": "W_hom", "R_H": "lnB"}
QS = (0.005, 0.01, 0.02, 0.05)
rng = np.random.default_rng(1)


def standardize(X):
    X = np.asarray(X, float)
    s = X.std(0)
    keep = s > 1e-9  # descriptors are normalized by Ebar; below this is round-off
    Z = np.zeros_like(X)
    Z[:, keep] = (X[:, keep] - X[:, keep].mean(0)) / s[keep]
    return Z, keep


n = len(df)
II, JJ = np.triu_indices(n, 1)
NTOT = len(II)


def omega_q(cols, R, nperm=300):
    Z, _ = standardize(df[cols].values)
    d = pdist(Z) if Z.shape[1] > 0 else np.zeros(NTOT)
    order = np.argsort(d, kind="stable")
    dR = np.abs(R[II] - R[JJ])
    perms = [rng.permutation(R) for _ in range(nperm)]
    out = {}
    for q in QS:
        nq = int(np.ceil(q * NTOT))
        dcut = d[order[nq - 1]]
        sel = d <= dcut + 1e-12
        om = dR[sel].max()
        om95 = np.quantile(dR[sel], 0.95)
        nulls = np.array([np.abs(p[II] - p[JJ])[sel].max() for p in perms])
        nulls95 = np.array([np.quantile(np.abs(p[II] - p[JJ])[sel], 0.95) for p in perms[:100]])
        k = np.argmax(np.where(sel, dR, -1))
        out[q] = dict(Omega=om, Omega_null=nulls.mean(), ratio=om / nulls.mean(),
                      Omega95=om95, ratio95=om95 / nulls95.mean(),
                      npairs=int(sel.sum()), dcut=dcut, pair=(int(II[k]), int(JJ[k])))
    return out


def quad_feats(Z):
    cols = [np.ones(len(Z))]
    p = Z.shape[1]
    for a in range(p):
        cols.append(Z[:, a])
    for a in range(p):
        for b in range(a, p):
            cols.append(Z[:, a] * Z[:, b])
    return np.column_stack(cols)


def surrogate(cols, R, k=5):
    Z, keep = standardize(df[cols].values)
    Z = Z[:, keep]
    Phi = quad_feats(Z)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    pred = np.zeros(n)
    for fo in folds:
        tr = np.setdiff1d(idx, fo)
        A = Phi[tr]
        c = np.linalg.solve(A.T @ A + 1e-8 * np.eye(A.shape[1]), A.T @ R[tr])
        pred[fo] = Phi[fo] @ c
    A = Phi
    c = np.linalg.solve(A.T @ A + 1e-8 * np.eye(A.shape[1]), A.T @ R)
    fit = Phi @ c
    ss = np.sum((R - R.mean()) ** 2)
    return dict(R2_cv=1 - np.sum((R - pred) ** 2) / ss,
                RMSE_cv=np.sqrt(np.mean((R - pred) ** 2)),
                einf_cv=np.max(np.abs(R - pred)),
                R2_fit=1 - np.sum((R - fit) ** 2) / ss,
                einf_fit=np.max(np.abs(R - fit)), pred=pred)


summary = {"n_states": int(n), "n_pairs": int(NTOT),
           "families": df.family.value_counts().to_dict()}

# ---------------- verification ----------------
lam_m, mu_m = 1.0 * 0.3 / (1.3 * 0.4), 1.0 / 2.6
lam_i, mu_i = 10 * lam_m, 10 * mu_m


def lam_analytic(f, orient):
    Mm, Mi = lam_m + 2 * mu_m, lam_i + 2 * mu_i
    fa = np.array([1 - f, f]); L = np.array([lam_m, lam_i]); Mv = np.array([Mm, Mi])
    if orient == "h":
        s = np.sum(fa * L / Mv) / np.sum(fa / Mv)
        sig11 = Mv + L * (s - L) / Mv
        W = 0.5 * np.sum(fa * sig11)
        sm = sig11[0]; s1m = max(sig11[0], s)
    else:
        s = 1.0 / np.sum(fa / Mv)
        W = 0.5 * s
        s1m = s
    return W / (0.5 * Mm)


lam_df = df[df.family == "laminate"]
an = np.array([lam_analytic(f, o) for f, o in zip(lam_df.f, lam_df.orient)])
summary["laminate_max_rel_err_W"] = float(np.max(np.abs(lam_df.W_hom.values / an - 1)))
summary["hill_mandel_max_rel_err"] = float(np.max(np.abs(df.W_HM / df.W_hom - 1)))
summary["laminate_max_Jgap"] = float(lam_df[["Jgap_m", "Jgap_i"]].abs().values.max())
summary["max_eq_err"] = float(df.eq_err.max())
summary["max_iters"] = int(df.iters.max())

# ---------------- nominal-cell exact barriers and noise floor ----------------
cells = {}
for tname, col in TARGETS.items():
    R = df[col].values
    g = df.groupby("cell")[col]
    rng_cell = (g.max() - g.min())
    fam = df.groupby("cell").family.first()
    nn = df.groupby("cell").n.first()
    cells[tname] = {
        "Omega0_lam": float(rng_cell[fam == "laminate"].max()),
        "Omega0_disk_n1_noise": float(rng_cell[(fam == "disks") & (nn == 1)].max()),
        "Omega0_disks_n>1": float(rng_cell[(fam == "disks") & (nn > 1)].max()),
        "Omega0_ellipses": float(rng_cell[fam == "ellipses"].max()),
        "range_R": float(R.max() - R.min()),
    }
summary["nominal_cells"] = cells

# ---------------- Omega hierarchy and surrogates ----------------
om_rows, sur_rows, preds = [], [], {}
for tname, col in TARGETS.items():
    R = df[col].values
    for lv, cols in LEVELS.items():
        o = omega_q(cols, R)
        s = surrogate(cols, R)
        preds[(tname, lv)] = s.pop("pred")
        for q, v in o.items():
            om_rows.append(dict(target=tname, level=lv, q=q, **{k: v[k] for k in
                           ("Omega", "Omega_null", "ratio", "Omega95", "ratio95", "npairs", "dcut")},
                                pair_i=v["pair"][0], pair_j=v["pair"][1]))
        sur_rows.append(dict(target=tname, level=lv, dim=len(cols), **s,
                             half_Omega_1pct=o[0.01]["Omega"] / 2))
om = pd.DataFrame(om_rows); om.to_csv("omega_table.csv", index=False)
sur = pd.DataFrame(sur_rows); sur.to_csv("surrogate_table.csv", index=False)

# ---------------- resolution ----------------
try:
    res = pd.read_csv("resolution.csv")
    summary["resolution"] = {k: float(np.max(np.abs(res[k + "_255"] - res[k + "_127"])))
                             for k in ("W_hom", "lnB", "I", "lnVeff", "e_alpha", "Jgap_m")}
    summary["resolution_mean_abs"] = {k: float(np.mean(np.abs(res[k + "_255"] - res[k + "_127"])))
                                      for k in ("W_hom", "lnB", "I", "lnVeff", "e_alpha")}
    summary["resolution_n"] = int(len(res))
except FileNotFoundError:
    pass

# ---------------- counterexample pair (E1, R_H, q=1%) ----------------
row = om[(om.target == "R_H") & (om.level == "E1") & (om.q == 0.01)].iloc[0]
a, b = int(row.pair_i), int(row.pair_j)
cols_show = E1 + ["rhoI", "W_hom", "lnB", "I", "lnVeff", "lnValpha", "e_alpha", "phi_alpha", "family", "f_nom", "n"]
pair = df.loc[[a, b], cols_show]
summary["counterexample_E1"] = {
    "ids": [a, b], "dist_E1_std": float(row.dcut),
    "states": pair.to_dict(orient="records"),
    "dR_H": float(df.lnB[a] - df.lnB[b]), "dI": float(df.I[a] - df.I[b]),
    "dlnVeff": float(df.lnVeff[a] - df.lnVeff[b]),
    "dW": float(df.W_hom[a] - df.W_hom[b]),
}
summary["identity_R_eq_phi_minus_e"] = float(np.max(np.abs(df.lnB - (df.phi_alpha - df.e_alpha))))
summary["e_alpha_stats"] = {"mean": float(df.e_alpha.mean()), "std": float(df.e_alpha.std()),
                            "min": float(df.e_alpha.min()), "max": float(df.e_alpha.max())}

with open("summary.json", "w") as fh:
    json.dump(summary, fh, indent=2, default=float)

print(json.dumps(summary, indent=1, default=float)[:4000])
print(om[om.q == 0.01][["target", "level", "Omega", "Omega_null", "ratio", "ratio95", "npairs"]].to_string(index=False))
print(sur[["target", "level", "dim", "R2_cv", "RMSE_cv", "einf_cv", "einf_fit", "half_Omega_1pct"]].to_string(index=False))

# ================= FIGURES =================
# 1) gallery
ex = [df[(df.family == "laminate") & (df.orient == "h")].index[2],
      df[(df.family == "disks") & (df.n == 9) & (df.f_nom == 0.2)].index[0],
      df[(df.family == "ellipses") & (df.n == 9) & (df.f_nom == 0.2)].index[0]]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.9))
vmax = max(SBAR[i].max() for i in ex)
for c, i in enumerate(ex):
    ax[0, c].imshow(MASK[i].T, origin="lower", cmap="Greys", interpolation="nearest")
    ax[0, c].set_title(f"{df.family[i]}  $f$={df.f[i]:.2f}", fontsize=9)
    im = ax[1, c].imshow(np.where(MASK[i], np.nan, SBAR[i]).T, origin="lower", cmap="inferno",
                         vmin=0, vmax=vmax, interpolation="nearest")
    ax[1, c].set_title(f"$\\ln B_\\ell$={df.lnB[i]:.2f}", fontsize=9)
for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(im, ax=ax[1, :], shrink=0.8, label=r"$\bar\sigma_1/\sigma_{\rm ref}$ (matrix)")
fig.savefig(f"{FIG}/fig_campaign_gallery.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_campaign_gallery.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 2) Omega ratio by level
lv_order = list(LEVELS.keys())
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for k, (tname, title) in enumerate((("R_E", r"$R_E=W_{\rm hom}$ (energy)"),
                                     ("R_H", r"$R_H=\ln B_\ell$ (hazard, $m=20$)"))):
    sub = om[(om.target == tname) & (om.q == 0.01)].set_index("level").loc[lv_order]
    sub95 = sub.ratio95
    x = np.arange(len(lv_order))
    ax[k].bar(x - 0.2, sub.ratio, 0.4, color="#10324F", label=r"$\Omega(q)/\Omega^{\rm null}$")
    ax[k].bar(x + 0.2, sub95, 0.4, color="#A63D40", label=r"95th-pct version")
    ax[k].set_xticks(x); ax[k].set_xticklabels([LABEL[l] for l in lv_order], rotation=60, fontsize=7)
    ax[k].axhline(1, color="gray", ls=":", lw=1)
    ax[k].set_title(title, fontsize=9)
    ax[k].set_yscale("log")
ax[0].set_ylabel(r"normalized insufficiency at $q=1\%$")
ax[0].legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_omega_levels.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_omega_levels.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 3) counterexample pair
fig = plt.figure(figsize=(7.4, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
vmax = max(SBAR[a].max(), SBAR[b].max())
for c, i in enumerate((a, b)):
    axx = fig.add_subplot(gs[0, c])
    axx.imshow(np.where(MASK[i], np.nan, SBAR[i]).T, origin="lower", cmap="inferno",
               vmin=0, vmax=vmax, interpolation="nearest")
    axx.contour(MASK[i].T.astype(float), levels=[0.5], colors="w", linewidths=0.4)
    axx.set_xticks([]); axx.set_yticks([])
    axx.set_title(f"state {'AB'[c]}: {df.family[i]}, $n$={df.n[i]}\n$\\ln B_\\ell$={df.lnB[i]:.2f}", fontsize=8)
axs = fig.add_subplot(gs[0, 2])
t = np.linspace(0.5, 1.0, 400)
for c, (i, colr) in enumerate(((a, "#10324F"), (b, "#A63D40"))):
    sm = SBAR[i][~MASK[i]]
    sp = sm.max()
    V = np.array([np.sum(sm >= tt * sp) for tt in t]) / N ** 2
    with np.errstate(divide="ignore"):
        axs.plot(t, M * np.log(sp) + np.log(V), color=colr, lw=1.8, label=f"state {'AB'[c]}")
axs.axvline(ALPHA, color="gray", ls="--", lw=0.8)
axs.set_xlabel(r"$t=\bar\sigma_1/\sigma_{\rm peak}$")
axs.set_ylabel(r"$I+\ln V(t)$")
axs.legend(fontsize=7)
axs.set_title(r"tail spectrum", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_counterexample.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_counterexample.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 4) surrogate scatter
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
for k, (tname, lv) in enumerate((("R_E", "E0"), ("R_H", "E1"), ("R_H", "T2"))):
    R = df[TARGETS[tname]].values
    p = preds[(tname, lv)]
    for fam, mk in (("laminate", "s"), ("disks", "o"), ("ellipses", "^")):
        sel = df.family == fam
        ax[k].scatter(R[sel], p[sel], s=8, marker=mk, alpha=0.7, label=fam)
    lo, hi = R.min(), R.max()
    ax[k].plot([lo, hi], [lo, hi], "k:", lw=0.8)
    r = sur[(sur.target == tname) & (sur.level == lv)].iloc[0]
    tmath = f"${tname}$"
    ax[k].set_title(f"{tmath} | {lv}: $R^2_{{cv}}$={r.R2_cv:.3f}\n$e_\\infty^{{cv}}$={r.einf_cv:.3g}", fontsize=8)
    ax[k].set_xlabel("FFT"); ax[k].set_ylabel("surrogate (CV)")
ax[0].legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_surrogates.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_surrogates.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("figures written")
