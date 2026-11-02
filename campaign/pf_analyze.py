r"""
Analysis of the in-house AT1 phase-field campaign (Phase D').
Inputs : campaign_results.csv, fields.npz (elastic campaign, N=127),
         pf_results.csv, pf_fields.npz, pf_hist.npz, pf_hom.json, pf_out/hom_d.npy,
         pf_resolution_255.csv (optional)
Outputs: pf_summary.json, pf_omega_table.csv, pf_surrogate_table.csv,
         pf_descriptors.csv, ../figures/fig_pf_*.pdf|png
"""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pf_fft import PhaseFieldRVE, lame

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 10, "legend.fontsize": 8})
FIG = r"..\figures"
rng = np.random.default_rng(7)

df = pd.read_csv("campaign_results.csv")
MASK = np.load("fields.npz")["mask"]
N = MASK.shape[1]
h = 1.0 / N
pf = pd.read_csv("pf_results.csv").sort_values("id").reset_index(drop=True)
D = np.load("pf_fields.npz")["d"]
HIST = np.load("pf_hist.npz")
hom = json.load(open("pf_hom.json"))
d_hom = np.load("pf_out/hom_d.npy")
n = len(df)
assert len(pf) == n


# ---------------------------------------------------------------- paths
HALF_OFFS = [(1, 0), (0, 1), (1, 1), (-1, 1), (1, 2), (-1, 2), (2, 1), (-2, 1)]


def min_path(Gc, allowed, sources_all=False):
    """Minimum of int Gc ds over closed paths wrapping once in y (periodic in x),
    restricted to 'allowed' cells; 16-neighbour stencil. Returns (value, path-cells)."""
    Nn = Gc.shape[0]
    h = 1.0 / Nn
    J = np.arange(Nn + 1)
    I = np.arange(Nn)
    II, JJ = np.meshgrid(I, J, indexing="ij")          # (N, N+1)
    A = allowed[II, JJ % Nn]
    G = Gc[II, JJ % Nn]
    nid = JJ * Nn + II
    rows, cols, w = [], [], []
    for di, dj in HALF_OFFS:
        for sgn in (1,):
            i2 = (II + di) % Nn
            j2 = JJ + dj
            ok = (j2 >= 0) & (j2 <= Nn)
            j2c = np.clip(j2, 0, Nn)
            ok &= A & A[i2, j2c]
            L = np.hypot(di, dj) * h
            rows.append(nid[ok]); cols.append((j2c * Nn + i2)[ok])
            w.append(L * 0.5 * (G[ok] + G[i2, j2c][ok]))
    Mx = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Nn * (Nn + 1),) * 2)
    src = np.flatnonzero(A[:, 0])
    if len(src) == 0:
        return np.nan, None
    dist, pred = dijkstra(Mx, directed=False, indices=src, return_predecessors=True)
    tgt = Nn * Nn + src                                 # same x index on the copy layer
    vals = dist[np.arange(len(src)), tgt]
    k = int(np.argmin(vals))
    if not np.isfinite(vals[k]):
        return np.nan, None
    path = []
    node = tgt[k]
    while node >= 0 and node != src[k]:
        path.append(node)
        node = pred[k, node]
    path.append(src[k])
    path = np.array(path)
    return float(vals[k]), np.c_[path % Nn, (path // Nn) % Nn]


def gc_field(mask):
    return PhaseFieldRVE(mask).Gc


THR = 0.9
rows = []
paths = {}
Gp_hom, _ = min_path(gc_field(np.zeros((N, N), bool)), d_hom >= THR)
Pmin_hom, _ = min_path(gc_field(np.zeros((N, N), bool)), np.ones((N, N), bool))
for i in range(n):
    Gc = gc_field(MASK[i])
    gp, pth = min_path(Gc, D[i] >= THR)
    pm, _ = min_path(Gc, np.ones((N, N), bool))
    paths[i] = pth
    rows.append(dict(id=i, Gamma_path=gp, P_min=pm))
P = pd.DataFrame(rows)
pf = pf.merge(P, on="id")
pf["R_sigma"] = np.log(pf.sigma_peak / hom["sigma_peak"])
pf["R_Gpath"] = np.log(pf.Gamma_path / Gp_hom)
pf["R_Gdiss"] = np.log(pf.Gamma / hom["Gamma"])
pf["J_psi"] = -np.log(pf.e0 / hom["e0"])              # onset intensity (undamaged solution)
pf["lnPmin"] = np.log(pf.P_min / Pmin_hom)
X = df.merge(pf[["id", "R_sigma", "R_Gpath", "R_Gdiss", "J_psi", "lnPmin", "Gamma_path",
                 "P_min", "sigma_peak", "Gamma", "e0", "eps_peak", "separated", "n_steps",
                 "time_s", "M0"]], on="id")
X.to_csv("pf_descriptors.csv", index=False)

summary = {"n": int(n), "separated": int(X.separated.sum()),
           "hom": {k: hom[k] for k in ("e0", "sigma_peak", "Gamma", "M0")},
           "Gamma_path_hom": Gp_hom, "P_min_hom": Pmin_hom,
           "sigma_c_AT1_1D": float(np.sqrt(3 * 0.98 * (lame(1, .3)[0] + 2 * lame(1, .3)[1]) / (8 * 0.025))),
           "time_s_total": float(X.time_s.sum()), "time_s_max": float(X.time_s.max()),
           "n_path_nan": int(X.Gamma_path.isna().sum())}

# ---------------------------------------------------------------- laminate controls
lm, mm = lame(1.0, 0.3)
li, mi = lame(10.0, 0.3)


def lam_onset(f, orient):
    """Analytic undamaged laminate under Ebar = e1(x)e1 (unit), AT1 onset."""
    Mm, Mi = lm + 2 * mm, li + 2 * mi
    if orient == "h":      # layers normal to y: eps11 uniform, sigma22 continuous, <eps22> = 0
        # sigma22 = lam*1 + M*e22 equal in both, (1-f) e22m + f e22i = 0
        A = np.array([[Mm, -Mi], [1 - f, f]])
        b = np.array([li - lm, 0.0])
        e22m, e22i = np.linalg.solve(A, b)
        em = (1.0, e22m); ei = (1.0, e22i)
        s11 = (1 - f) * (Mm * 1 + lm * e22m) + f * (Mi * 1 + li * e22i)
    else:                  # layers normal to x: sigma11 continuous, eps22 = 0, <eps11> = 1
        s = 1.0 / ((1 - f) / Mm + f / Mi)
        em = (s / Mm, 0.0); ei = (s / Mi, 0.0)
        s11 = s

    def psi(e, lam_, mu_):
        K = lam_ + mu_                     # plane-strain 2D bulk used in pf_fft: lam + 2mu/3 (3D K)
        K = lam_ + 2 * mu_ / 3
        tr = e[0] + e[1]
        dev2 = e[0] ** 2 + e[1] ** 2 - tr ** 2 / 3
        return 0.5 * K * max(tr, 0) ** 2 + mu_ * dev2
    psic_m = 3 * 1.0 / (16 * 0.025)
    psic_i = 3 * 3.0 / (16 * 0.025)
    e0 = min(np.sqrt(psic_m * 0.98 / psi(em, lm, mm)), np.sqrt(psic_i * 0.98 / psi(ei, li, mi)))
    return e0, s11 * e0, s11


lam = X[X.family == "laminate"].copy()
an = np.array([lam_onset(f, o) for f, o in zip(lam.f, lam.orient)])
lam["e0_an"] = an[:, 0]
lam["sig_onset_an"] = an[:, 1]
lam["Gpath_ratio_expected"] = np.where(lam.orient == "h", 1 + 2 * lam.f, 1.0)
lam["Gpath_ratio"] = lam.Gamma_path / Gp_hom
summary["laminate"] = {
    "e0_max_rel_err": float(np.max(np.abs(lam.e0 / lam.e0_an - 1))),
    "v_max_abs_R_sigma": float(lam[lam.orient == "v"].R_sigma.abs().max()),
    "v_max_abs_R_Gpath": float(lam[lam.orient == "v"].R_Gpath.abs().max()),
    "h_max_rel_err_Gpath_vs_1p2f": float(np.max(np.abs(lam[lam.orient == "h"].Gpath_ratio
                                                       / lam[lam.orient == "h"].Gpath_ratio_expected - 1))),
    "h_peak_over_onset_max": float(np.max(lam[lam.orient == "h"].sigma_peak
                                          / lam[lam.orient == "h"].sig_onset_an)),
    "h_peak_over_onset_min": float(np.min(lam[lam.orient == "h"].sigma_peak
                                          / lam[lam.orient == "h"].sig_onset_an)),
    "v_max_abs_R_Gdiss": float(lam[lam.orient == "v"].R_Gdiss.abs().max()),
    "h_R_Gdiss_vs_ln1p2f_maxerr": float(np.max(np.abs(lam[lam.orient == "h"].R_Gdiss
                                                      - np.log(1 + 2 * lam[lam.orient == "h"].f)))),
}
lam[["id", "orient", "n", "f", "e0", "e0_an", "sigma_peak", "sig_onset_an", "R_sigma",
     "Gpath_ratio", "Gpath_ratio_expected", "R_Gdiss"]].to_csv("pf_laminate_check.csv", index=False)

# ---------------------------------------------------------------- Omega machinery
II, JJ = np.triu_indices(n, 1)
NTOT = len(II)
QS = (0.005, 0.01, 0.02, 0.05)


def standardize(A):
    A = np.asarray(A, float)
    s = A.std(0)
    keep = s > 1e-9
    Z = np.zeros_like(A)
    Z[:, keep] = (A[:, keep] - A[:, keep].mean(0)) / s[keep]
    return Z, keep


def omega_q(cols, R, nperm=300):
    ok = np.isfinite(R)
    Z, _ = standardize(X[cols].values)
    dd = pdist(Z)
    sel_ok = ok[II] & ok[JJ]
    order = np.argsort(np.where(sel_ok, dd, np.inf), kind="stable")
    nvalid = int(sel_ok.sum())
    dR = np.abs(R[II] - R[JJ])
    Rv = R.copy()
    perms = []
    for _ in range(nperm):
        p = Rv.copy(); p[ok] = rng.permutation(Rv[ok]); perms.append(p)
    out = {}
    for q in QS:
        nq = int(np.ceil(q * nvalid))
        dcut = dd[order[nq - 1]]
        sel = sel_ok & (dd <= dcut + 1e-12)
        om = dR[sel].max()
        nulls = np.array([np.abs(p[II] - p[JJ])[sel].max() for p in perms])
        k = np.argmax(np.where(sel, dR, -1))
        out[q] = dict(Omega=om, Omega_null=nulls.mean(), ratio=om / nulls.mean(),
                      npairs=int(sel.sum()), dcut=dcut, pair=(int(II[k]), int(JJ[k])))
    return out


def quad_feats(Z):
    cols = [np.ones(len(Z))]
    p = Z.shape[1]
    cols += [Z[:, a] for a in range(p)]
    cols += [Z[:, a] * Z[:, b] for a in range(p) for b in range(a, p)]
    return np.column_stack(cols)


def surrogate(cols, R, k=5, deg=2):
    ok = np.isfinite(R)
    Z, keep = standardize(X[cols].values)
    Zk = Z[:, keep]
    Phi = (quad_feats(Zk) if deg == 2 else np.column_stack([np.ones(len(Zk)), Zk]))[ok]
    Rk = R[ok]
    m = len(Rk)
    idx = rng.permutation(m)
    pred = np.zeros(m)
    for fo in np.array_split(idx, k):
        tr = np.setdiff1d(idx, fo)
        A = Phi[tr]
        c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Rk[tr])
        pred[fo] = Phi[fo] @ c
    ss = np.sum((Rk - Rk.mean()) ** 2)
    full = np.full(len(R), np.nan); full[ok] = pred
    return dict(R2_cv=1 - np.sum((Rk - pred) ** 2) / ss,
                RMSE_cv=float(np.sqrt(np.mean((Rk - pred) ** 2))),
                einf_cv=float(np.max(np.abs(Rk - pred)))), full


E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LEVELS = {
    "C_glob": ["e11_g", "e22_g", "e12_g"],
    "G0": ["f"],
    "G1": ["f", "rhoI"],
    "G2": ["f", "rhoI", "lnPmin"],
    "E0": ["f", "e11_m", "e22_m", "e12_m"],
    "E1": E1,
    "E1+P": E1 + ["lnPmin"],
    "T1": E1 + ["I"],
    "Tpsi": E1 + ["J_psi"],
    "Tpsi+P": E1 + ["J_psi", "lnPmin"],
    "H": ["lnB"],
    "E1+H": E1 + ["lnB"],
}
LABEL = {"C_glob": "global means\n(control)", "G0": "$f$", "G1": "$f,\\rho_I$",
         "G2": "$f,\\rho_I,P_{\\min}$", "E0": "phase means\n(E0)", "E1": "+Jensen gaps\n(E1)",
         "E1+P": "E1$+P_{\\min}$", "T1": "E1$+I$", "Tpsi": "E1$+J_\\psi$",
         "Tpsi+P": "E1$+J_\\psi+P_{\\min}$", "H": "$R_H$ only", "E1+H": "E1$+R_H$"}
TARGETS = {"R_sigma": "R_sigma", "R_Gpath": "R_Gpath"}

om_rows, sur_rows, preds = [], [], {}
for tname, col in TARGETS.items():
    R = X[col].values.astype(float)
    for lv, cols in LEVELS.items():
        o = omega_q(cols, R)
        s, pr = surrogate(cols, R)
        sl, _ = surrogate(cols, R, deg=1)
        preds[(tname, lv)] = pr
        for q, v in o.items():
            om_rows.append(dict(target=tname, level=lv, q=q, Omega=v["Omega"],
                                Omega_null=v["Omega_null"], ratio=v["ratio"],
                                npairs=v["npairs"], dcut=v["dcut"],
                                pair_i=v["pair"][0], pair_j=v["pair"][1]))
        sur_rows.append(dict(target=tname, level=lv, dim=len(cols), **s,
                             R2_cv_lin=sl["R2_cv"], einf_cv_lin=sl["einf_cv"],
                             half_Omega_1pct=o[0.01]["Omega"] / 2))
om = pd.DataFrame(om_rows); om.to_csv("pf_omega_table.csv", index=False)
sur = pd.DataFrame(sur_rows); sur.to_csv("pf_surrogate_table.csv", index=False)

# inclusion-only ladder (laminates excluded: their range dominates R_Gpath)
Xfull = X
X = Xfull[Xfull.family != "laminate"].reset_index(drop=True)
II, JJ = np.triu_indices(len(X), 1)
inc_rows = []
for tname, col in TARGETS.items():
    R = X[col].values.astype(float)
    for lv, cols in LEVELS.items():
        o = omega_q(cols, R)
        s, _ = surrogate(cols, R)
        sl, _ = surrogate(cols, R, deg=1)
        inc_rows.append(dict(target=tname, level=lv, Omega=o[0.01]["Omega"],
                             Omega_null=o[0.01]["Omega_null"], ratio=o[0.01]["ratio"],
                             npairs=o[0.01]["npairs"], R2_cv_quad=s["R2_cv"],
                             R2_cv_lin=sl["R2_cv"], einf_cv_lin=sl["einf_cv"],
                             RMSE_cv_lin=sl["RMSE_cv"]))
inc = pd.DataFrame(inc_rows); inc.to_csv("pf_omega_inclusions.csv", index=False)
summary["inclusions_only_range"] = {t: float(X[t].max() - X[t].min()) for t in TARGETS}
summary["inclusions_only_std"] = {t: float(X[t].std()) for t in TARGETS}
X = Xfull
II, JJ = np.triu_indices(n, 1)
print(inc.to_string(index=False))

# nominal cells / floors
cells = {}
for tname, col in TARGETS.items():
    g = X.groupby("cell")[col]
    rc = g.max() - g.min()
    fam = X.groupby("cell").family.first()
    nn = X.groupby("cell").n.first()
    cells[tname] = {"Omega0_lam": float(rc[fam == "laminate"].max()),
                    "Omega0_disk_n1": float(rc[(fam == "disks") & (nn == 1)].max()),
                    "Omega0_disks_n>1": float(rc[(fam == "disks") & (nn > 1)].max()),
                    "Omega0_ellipses": float(rc[fam == "ellipses"].max()),
                    "range": float(np.nanmax(X[col]) - np.nanmin(X[col])),
                    "std": float(np.nanstd(X[col]))}
summary["nominal_cells"] = cells
summary["corr"] = {
    "R_sigma~J_psi": float(np.corrcoef(X.R_sigma, X.J_psi)[0, 1]),
    "R_sigma~I": float(np.corrcoef(X.R_sigma, X.I)[0, 1]),
    "R_Gpath~lnPmin": float(X[["R_Gpath", "lnPmin"]].corr().iloc[0, 1]),
    "R_Gpath~f": float(X[["R_Gpath", "f"]].corr().iloc[0, 1]),
    "R_sigma~R_Gpath": float(X[["R_sigma", "R_Gpath"]].corr().iloc[0, 1]),
    "R_sigma~lnB": float(np.corrcoef(X.R_sigma, X.lnB)[0, 1]),
}

# resolution
try:
    rs = pd.read_csv("pf_resolution_255.csv")
    from pf_resolution import geoms_all
    from fft_rve import rasterize
    GEO = geoms_all()
    N2 = 255
    d_h2 = np.load("pf_res/hom_d.npy")
    Gp_hom2, _ = min_path(gc_field(np.zeros((N2, N2), bool)), d_h2 >= THR)
    rs = rs[rs.key != "hom"].copy()
    rs["id"] = rs.key.str[4:].astype(int)
    gp2 = []
    for i in rs.id:
        m2 = rasterize(GEO[i], N2)
        gp2.append(min_path(gc_field(m2), np.load(f"pf_res/rve_{i:03d}_d.npy") >= THR)[0])
    rs["R_Gpath"] = np.log(np.array(gp2) / Gp_hom2)
    m = rs.merge(X[["id", "R_sigma", "R_Gpath", "R_Gdiss", "family"]], on="id", suffixes=("_255", "_127"))
    summary["resolution_255"] = {
        "ids": m.id.tolist(),
        "Gamma_path_hom_255": Gp_hom2,
        "max_abs_dR_sigma": float(np.max(np.abs(m.R_sigma_255 - m.R_sigma_127))),
        "mean_abs_dR_sigma": float(np.mean(np.abs(m.R_sigma_255 - m.R_sigma_127))),
        "max_abs_dR_Gpath": float(np.max(np.abs(m.R_Gpath_255 - m.R_Gpath_127))),
        "max_abs_dR_Gdiss": float(np.max(np.abs(m.R_Gamma - m.R_Gdiss))),
        "rows": m[["id", "family", "R_sigma_127", "R_sigma_255", "R_Gpath_127", "R_Gpath_255",
                   "R_Gdiss", "R_Gamma"]].to_dict(orient="records"),
    }
except FileNotFoundError:
    pass

# counterexample pairs at E1 for both targets
cx = {}
for tname in TARGETS:
    r = om[(om.target == tname) & (om.level == "E1") & (om.q == 0.01)].iloc[0]
    a, b = int(r.pair_i), int(r.pair_j)
    cx[tname] = {"ids": [a, b], "dcut_E1": float(r.dcut), "dR": float(X[tname][a] - X[tname][b]),
                 "families": [X.family[a], X.family[b]], "n": [int(X.n[a]), int(X.n[b])],
                 "f": [float(X.f[a]), float(X.f[b])],
                 "J_psi": [float(X.J_psi[a]), float(X.J_psi[b])],
                 "I": [float(X.I[a]), float(X.I[b])],
                 "lnPmin": [float(X.lnPmin[a]), float(X.lnPmin[b])],
                 "R_sigma": [float(X.R_sigma[a]), float(X.R_sigma[b])],
                 "R_Gpath": [float(X.R_Gpath[a]), float(X.R_Gpath[b])]}
summary["counterexamples_E1"] = cx
# the elastic-campaign counterexample (ids 54, 59)
summary["pair_54_59"] = {k: [float(X[k][54]), float(X[k][59])] for k in
                         ("R_sigma", "R_Gpath", "R_Gdiss", "J_psi", "I", "lnB", "lnPmin")}

with open("pf_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:6000])
print(om[om.q == 0.01][["target", "level", "Omega", "Omega_null", "ratio", "npairs"]].to_string(index=False))
print(sur[["target", "level", "dim", "R2_cv", "RMSE_cv", "einf_cv", "half_Omega_1pct"]].to_string(index=False))

# ================================================================ FIGURES
# 1) crack gallery + stress-strain curves
ex = [int(X[(X.family == "laminate") & (X.orient == "h") & (X.n == 2)].index[2]),
      int(X[(X.family == "disks") & (X.n == 9) & (X.f_nom == 0.2)].index[0]),
      int(X[(X.family == "ellipses") & (X.n == 9) & (X.f_nom == 0.2)].index[0])]
fig = plt.figure(figsize=(7.4, 2.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
for c, i in enumerate(ex):
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(D[i].T, origin="lower", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(MASK[i].T.astype(float), levels=[0.5], colors="c", linewidths=0.5)
    if paths[i] is not None:
        ax.plot(paths[i][:, 0], paths[i][:, 1], ",", color="w", ms=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{X.family[i]}, $f$={X.f[i]:.2f}\n$R_\\sigma$={X.R_sigma[i]:.2f}, "
                 f"$R_{{\\rm path}}$={X.R_Gpath[i]:.2f}", fontsize=7)
ax = fig.add_subplot(gs[0, 3])
hh = HIST["hom"]
ax.plot(np.r_[0, hh[:, 0]] / hom["e0"], np.r_[0, hh[:, 1]] / hom["sigma_peak"], "k-", lw=1.2, label="homogeneous")
for i, colr in zip(ex, ("#10324F", "#A63D40", "#4C8C4A")):
    hi_ = HIST[f"rve_{i:03d}"]
    ax.plot(np.r_[0, hi_[:, 0]] / hom["e0"], np.r_[0, hi_[:, 1]] / hom["sigma_peak"], color=colr, lw=1.2,
            label=X.family[i])
ax.set_xlabel(r"$\bar\varepsilon/\varepsilon_0^{\rm hom}$")
ax.set_ylabel(r"$\bar\sigma_{11}/\sigma_{\rm peak}^{\rm hom}$")
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pf_gallery.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_pf_gallery.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 2) Omega ratios (superseded by v06_analyze.py, which writes fig_pf_omega)
if False:
    lv_order = list(LEVELS)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for k, (tname, title) in enumerate((("R_sigma", r"$R_\sigma=\ln(\sigma_{\rm peak}/\sigma_{\rm peak}^{\rm hom})$"),
                                        ("R_Gpath", r"$R_\Gamma=\ln(\Gamma_{\rm path}/\Gamma_{\rm path}^{\rm hom})$"))):
        sub = om[(om.target == tname) & (om.q == 0.01)].set_index("level").loc[lv_order]
        subi = inc[inc.target == tname].set_index("level").loc[lv_order]
        x = np.arange(len(lv_order))
        ax[k].bar(x - 0.2, sub.ratio, 0.4, color="#10324F", label="all 216 RVEs")
        ax[k].bar(x + 0.2, subi.ratio, 0.4, color="#A63D40", label="inclusions only (180)")
        SHORT = {"C_glob": "control", "G0": "$f$", "G1": "$f,\\rho_I$", "G2": "$f,\\rho_I,P_{\\min}$",
                 "E0": "E0", "E1": "E1", "E1+P": "E1$+P_{\\min}$", "T1": "E1$+I$", "Tpsi": "E1$+J_\\psi$",
                 "Tpsi+P": "E1$+J_\\psi+P_{\\min}$", "H": "$R_H$", "E1+H": "E1$+R_H$"}
        ax[k].set_xticks(x); ax[k].set_xticklabels([SHORT[l] for l in lv_order], rotation=60, ha="right", fontsize=7)
        ax[k].axhline(1, color="gray", ls=":", lw=1)
        ax[k].set_title(title, fontsize=9)
        ax[k].set_yscale("log")
    ax[0].set_ylabel(r"$\Omega(q)/\Omega^{\rm null}$ at $q=1\%$")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    pass
    pass
    plt.close(fig)

# 3) counterexample (strength, E1)
a, b = cx["R_sigma"]["ids"]
fig = plt.figure(figsize=(7.4, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
for c, i in enumerate((a, b)):
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(D[i].T, origin="lower", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(MASK[i].T.astype(float), levels=[0.5], colors="c", linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"state {'AB'[c]}: {X.family[i]}, $n$={X.n[i]}, $f$={X.f[i]:.3f}\n"
                 f"$R_\\sigma$={X.R_sigma[i]:.3f}, $J_\\psi$={X.J_psi[i]:.3f}", fontsize=7)
ax = fig.add_subplot(gs[0, 2])
for i, colr in ((a, "#10324F"), (b, "#A63D40")):
    hi_ = HIST[f"rve_{i:03d}"]
    ax.plot(np.r_[0, hi_[:, 0]] / hom["e0"], np.r_[0, hi_[:, 1]] / hom["sigma_peak"], color=colr, lw=1.5,
            label=f"state {'AB'[(i == b) * 1]}")
ax.set_xlabel(r"$\bar\varepsilon/\varepsilon_0^{\rm hom}$")
ax.set_ylabel(r"$\bar\sigma_{11}/\sigma_{\rm peak}^{\rm hom}$")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pf_counterexample.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_pf_counterexample.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 4) scatter: R_sigma vs J_psi, R_Gpath vs lnPmin
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for fam, mk in (("laminate", "s"), ("disks", "o"), ("ellipses", "^")):
    s_ = X.family == fam
    ax[0].scatter(X.J_psi[s_], X.R_sigma[s_], s=9, marker=mk, alpha=0.7, label=fam)
    ax[1].scatter(X.lnPmin[s_], X.R_Gpath[s_], s=9, marker=mk, alpha=0.7, label=fam)
lo = min(X.J_psi.min(), 0); hi = X.J_psi.max()
ax[0].plot([lo, hi], [-lo, -hi], "k:", lw=0.8, label=r"$R_\sigma=-J_\psi$")
ax[0].set_xlabel(r"onset intensity $J_\psi$"); ax[0].set_ylabel(r"$R_\sigma$")
lo = X.lnPmin.min(); hi = X.lnPmin.max()
ax[1].plot([lo, hi], [lo, hi], "k:", lw=0.8, label="identity")
ax[1].set_xlabel(r"$\ln(P_{\min}/P_{\min}^{\rm hom})$"); ax[1].set_ylabel(r"$R_{\rm path}$")
ax[0].legend(fontsize=6); ax[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pf_scatter.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_pf_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("figures written")
