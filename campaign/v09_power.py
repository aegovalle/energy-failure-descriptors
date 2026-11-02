r"""v0.9 power and metric-invariance analysis (re-analysis only).

1. Conditional tests for a pre-specified primary family of six incremental steps, with
   three statistics computed on the n_q closest pairs of the enlarged descriptor
     max   : Omega(q) (worst case, as in the paper)
     mean  : mean |dR| (local mean, delta-test type; uses all selected pairs)
     p95   : 95th percentile of |dR|
   and two conditional samplers for eta | Psi_k
     swap  : local swaps between near neighbours (as in v09_analysis)
     FL    : Freedman--Lane residual permutation, eta = g(Psi_k) + r with g a quadratic
             ridge fit, residuals permuted globally
   Holm adjustment within the primary family for each (statistic, sampler).
2. Mahalanobis (whitened, pseudo-inverse) metric: invariant under invertible linear
   reparameterizations of Psi and under duplication of components. Main ladders and the
   primary steps are repeated with it.
Outputs: v09_power.csv, v09_mahal_levels.csv, v09_mahal_dup.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from descr_tools import surrogate_cv
from v09_stats import zscore, pair_index, holm

rng = np.random.default_rng(20261103)
B = 2000
Q = 0.01

X = (pd.read_csv("v06_desc.csv")[["id", "A_lig"]]
     .merge(pd.read_csv("pf_descriptors.csv"), on="id")
     .merge(pd.read_csv("v09_fields.csv"), on="id")
     .rename(columns={"R_Gpath": "R_path"}).sort_values("id").reset_index(drop=True))
X["lnWm"] = np.log(X.Wph_m)
E0 = ["f", "e11_m", "e22_m", "e12_m"]
E1 = E0 + ["Jgap_m", "Jgap_i"]
COV = [f"{c}_{t}" for t in ("m", "i") for c in ("c11", "c22", "c12", "c1122", "c1112", "c2212")]
SPL = [f"{c}_{t}" for t in ("m", "i") for c in ("trp_mu", "trp_sd", "dev_mu", "dev_sd")]
LEV = {"G0": ["f"], "G1": ["f", "rhoI"], "GL": ["f", "rhoI", "A_lig"], "G2": ["f", "rhoI", "lnPmin"],
       "E0": E0, "E1": E1, "E0+cov": E0 + COV, "E0+split": E0 + SPL, "T1": E1 + ["I"], "E1+H": E1 + ["lnB"]}
TG = {"R_H": "lnB", "R_sigma": "R_sigma", "R_path": "R_path", "W_m (control)": "lnWm"}
SC = {"all": np.ones(len(X), bool), "n>1": ((X.family != "laminate") & (X.n > 1)).values}

PRIMARY = [  # (label, scope, target, base, added)
    ("P1 gaps, hazard", "n>1", "R_H", "E0", ["Jgap_m", "Jgap_i"]),
    ("P2 covariance, hazard", "n>1", "R_H", "E0", COV),
    ("P3 gaps, strength", "n>1", "R_sigma", "E0", ["Jgap_m", "Jgap_i"]),
    ("P4 hazard added, strength", "n>1", "R_sigma", "E1", ["lnB"]),
    ("P5 ligament, hazard", "n>1", "R_H", "G1", ["A_lig"]),
    ("P6 minimum cut, path cost", "all", "R_path", "G1", ["lnPmin"]),
]
CONTROL = [("C positive control", "n>1", "W_m (control)", "E0", ["Jgap_m", "Jgap_i"]),
           ("C positive control", "all", "W_m (control)", "E0", ["Jgap_m", "Jgap_i"])]


def whiten(A, tol=1e-8):
    A = np.asarray(A, float)
    A = A - A.mean(0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > tol * s.max()
    return U[:, keep] * np.sqrt(len(A))


def stats_sel(Z, R, II, JJ):
    d = pdist(Z)
    nq = int(np.ceil(Q * len(d) - 1e-9))
    dcut = np.partition(d, nq - 1)[nq - 1]
    sel = d <= dcut + 1e-12
    dR = np.abs(R[II] - R[JJ])[sel]
    return np.array([dR.max(), dR.mean(), np.quantile(dR, 0.95)])


def null_ratio(Z, R, II, JJ, nperm=2000):
    obs = stats_sel(Z, R, II, JJ)
    d = pdist(Z)
    nq = int(np.ceil(Q * len(d) - 1e-9))
    sel = d <= np.partition(d, nq - 1)[nq - 1] + 1e-12
    Is, Js = II[sel], JJ[sel]
    P = np.argsort(rng.random((nperm, len(R))), axis=1)
    Rp = R[P]
    nl = np.abs(Rp[:, Is] - Rp[:, Js])
    nulls = np.stack([nl.max(1), nl.mean(1), np.quantile(nl, 0.95, axis=1)], 1)
    return obs / nulls.mean(0), (1 + (nulls <= obs).sum(0)) / (1 + nperm)


def sampler_swap(Zk, eta):
    n = len(eta)
    D = np.sqrt(((Zk[:, None] - Zk[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    nbrs = [np.unique(np.r_[np.flatnonzero(D[i] <= 1e-12), np.argsort(D[i], kind="stable")[:5]]) for i in range(n)]

    def draw():
        perm = np.arange(n); free = np.ones(n, bool)
        for i in rng.permutation(n):
            if not free[i]:
                continue
            c = [j for j in nbrs[i] if free[j] and j != i]
            if not c:
                continue
            j = c[rng.integers(len(c))]; free[i] = free[j] = False
            if rng.random() < 0.5:
                perm[i], perm[j] = j, i
        return eta[perm]
    return draw


def sampler_fl(Zk, eta):
    p = Zk.shape[1]
    cols = [np.ones(len(Zk))] + [Zk[:, a] for a in range(p)] + [Zk[:, a] * Zk[:, b] for a in range(p) for b in range(a, p)]
    Phi = np.column_stack(cols)
    C = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ eta)
    fit = Phi @ C
    res = eta - fit

    def draw():
        return fit + res[rng.permutation(len(eta))]
    return draw


def cond_test(D, tgt, base, add, metric):
    R = D[TG[tgt]].values.astype(float)
    II, JJ = pair_index(len(R))
    Xk = D[LEV[base]].values; Ea = D[add].values.astype(float)
    if metric == "std":
        Zk = zscore(Xk); emb = lambda e: np.hstack([Zk, zscore(e)])
    else:
        Zk = whiten(Xk); emb = lambda e: whiten(np.hstack([Xk, e]))
    obs = stats_sel(emb(Ea), R, II, JJ)
    rows = []
    for sname, mk in (("swap", sampler_swap), ("FL", sampler_fl)):
        draw = mk(zscore(Xk), Ea)
        nulls = np.array([stats_sel(emb(draw()), R, II, JJ) for _ in range(B)])
        for k, st in enumerate(("max", "mean", "p95")):
            rows.append(dict(statistic=st, sampler=sname, obs=obs[k], cnull=nulls[:, k].mean(),
                             rho_c=obs[k] / nulls[:, k].mean(),
                             p_c=(1 + np.sum(nulls[:, k] <= obs[k])) / (1 + B)))
    return rows


out = []
for metric in ("std", "mahal"):
    for lab, sc, tgt, base, add in PRIMARY + CONTROL:
        D = X[SC[sc]].reset_index(drop=True)
        for r in cond_test(D, tgt, base, add, metric):
            out.append(dict(metric=metric, test=lab, scope=sc, target=tgt, step=f"{base}+{len(add)}", **r))
        print(metric, lab, sc, flush=True)
P = pd.DataFrame(out)
P["primary"] = ~P.test.str.startswith("C")
P["p_holm"] = np.nan
for key, g in P[P.primary].groupby(["metric", "statistic", "sampler"]):
    P.loc[g.index, "p_holm"] = holm(g.p_c.values)
P.to_csv("v09_power.csv", index=False)

# Mahalanobis ladders
lv_rows = []
for sc in ("all", "n>1"):
    D = X[SC[sc]].reset_index(drop=True)
    II, JJ = pair_index(len(D))
    for tgt in ("R_H", "R_sigma", "R_path"):
        R = D[TG[tgt]].values.astype(float)
        for lv in ("G1", "GL", "G2", "E0", "E1", "E0+cov", "E0+split", "T1", "E1+H"):
            if tgt == "R_H" and lv in ("T1", "E1+H"):
                continue
            rs, ps = null_ratio(zscore(D[LEV[lv]].values), R, II, JJ)
            rm, pm = null_ratio(whiten(D[LEV[lv]].values), R, II, JJ)
            lv_rows.append(dict(scope=sc, target=tgt, level=lv, ratio_std=rs[0], ratio_mahal=rm[0],
                                p_std=ps[0], p_mahal=pm[0], mean_ratio_std=rs[1], mean_ratio_mahal=rm[1]))
pd.DataFrame(lv_rows).to_csv("v09_mahal_levels.csv", index=False)

# duplication invariance of the Mahalanobis metric
dup = []
D = X[SC["n>1"]].reset_index(drop=True); II, JJ = pair_index(len(D))
for tgt in ("R_H", "R_sigma", "R_path"):
    R = D[TG[tgt]].values.astype(float)
    for lv in ("G1", "E1"):
        A = D[LEV[lv]].values
        base = stats_sel(whiten(A), R, II, JJ)[0]
        for j in range(A.shape[1]):
            dup.append(dict(target=tgt, level=lv, col=LEV[lv][j], omega=base,
                            omega_dup=stats_sel(whiten(np.hstack([A, A[:, [j]]])), R, II, JJ)[0],
                            omega_scaled=stats_sel(whiten(A * np.r_[np.ones(A.shape[1] - 1), 7.3]), R, II, JJ)[0]))
pd.DataFrame(dup).to_csv("v09_mahal_dup.csv", index=False)
print("done")
