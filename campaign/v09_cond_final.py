r"""Final conditional tests for Draft v0.9 (n>1 scope only, homogeneous population).

Sampler (sensitivity analysis only; it failed the realistic size check): eta resampled as quadratic ridge
fit of eta on Psi_k plus permuted residuals, residuals permuted within morphology family (disks / ellipses).
Inference uses the local-swap sampler (v09_r3_power.py, run_test(..., "swap", ...)).
Statistics on the n_q closest pairs of the enlarged descriptor: max (= Omega(q)) and mean.
Metrics: standardized Euclidean and Mahalanobis (whitened, pseudo-inverse).
Primary family (Holm): P1 gaps->hazard, P2 covariance->hazard, P3 gaps->strength,
P4 hazard->strength, P5 ligament->hazard. Exploratory: positive control, split moments,
covariance/split for strength and path, gaps across hazard exponents.
Calibration under a realistic synthetic null: the added block is permuted within nominal
cells (which preserves its family- and cell-level structure while breaking its link with
the realization-level response); rejection rates at 5% over 60 repetitions.
Outputs: v09_cond_final.csv, v09_cond_calib.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from v09_stats import zscore, pair_index, holm

rng = np.random.default_rng(20261105)
Q = 0.01
X = (pd.read_csv("v06_desc.csv")[["id", "A_lig"]]
     .merge(pd.read_csv("pf_descriptors.csv"), on="id")
     .merge(pd.read_csv("v09_fields.csv"), on="id")
     .rename(columns={"R_Gpath": "R_path"}).sort_values("id").reset_index(drop=True))
X["lnWm"] = np.log(X.Wph_m)
F = np.load("fields.npz"); S, M = F["sbar"], F["mask"]; N = M.shape[1]
for m in (2, 5, 10, 40):
    X[f"RH{m}"] = [m * np.log(S[i][~M[i]].max()) + np.log(np.sum((S[i][~M[i]] / S[i][~M[i]].max()) ** m) / N ** 2)
                   for i in range(len(X))]
X["RH20"] = X.lnB
D = X[(X.family != "laminate") & (X.n > 1)].reset_index(drop=True)
STRATA = D.family.values
CELL = D.cell.values
II, JJ = pair_index(len(D))
E0 = ["f", "e11_m", "e22_m", "e12_m"]; E1 = E0 + ["Jgap_m", "Jgap_i"]; GAP = ["Jgap_m", "Jgap_i"]
COV = [f"{c}_{t}" for t in ("m", "i") for c in ("c11", "c22", "c12", "c1122", "c1112", "c2212")]
SPL = [f"{c}_{t}" for t in ("m", "i") for c in ("trp_mu", "trp_sd", "dev_mu", "dev_sd")]
G1 = ["f", "rhoI"]

TESTS = [("P1", "gaps -> hazard", "RH20", E0, GAP, True),
         ("P2", "covariance -> hazard", "RH20", E0, COV, True),
         ("P3", "gaps -> strength", "R_sigma", E0, GAP, True),
         ("P4", "hazard -> strength", "R_sigma", E1, ["lnB"], True),
         ("P5", "ligament -> hazard", "RH20", G1, ["A_lig"], True),
         ("X1", "positive control: gaps -> matrix energy", "lnWm", E0, GAP, False),
         ("X2", "split -> hazard", "RH20", E0, SPL, False),
         ("X3", "split -> strength", "R_sigma", E0, SPL, False),
         ("X4", "covariance -> strength", "R_sigma", E0, COV, False),
         ("X5", "gaps -> path", "R_path", E0, GAP, False),
         ("X6", "minimum cut -> path", "R_path", G1, ["lnPmin"], False),
         ("X7", "tail I -> strength", "R_sigma", E1, ["I"], False)]
TESTS += [(f"M{m}", f"gaps -> hazard, m={m}", f"RH{m}", E0, GAP, False) for m in (2, 5, 10, 40)]


def whiten(A, tol=1e-8):
    A = np.asarray(A, float); A = A - A.mean(0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * s.max()] * np.sqrt(len(A))


def stats_sel(Z, R):
    d = pdist(Z); nq = int(np.ceil(Q * len(d) - 1e-9))
    sel = d <= np.partition(d, nq - 1)[nq - 1] + 1e-12
    dR = np.abs(R[II] - R[JJ])[sel]
    return np.array([dR.max(), dR.mean()])


def fl_parts(Xk, eta):
    Zk = zscore(Xk); p = Zk.shape[1]
    Phi = np.column_stack([np.ones(len(Zk))] + [Zk[:, a] for a in range(p)]
                          + [Zk[:, a] * Zk[:, b] for a in range(p) for b in range(a, p)])
    C = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ eta)
    fit = Phi @ C
    return fit, eta - fit


def strat_perm(groups):
    idx = np.arange(len(groups))
    for g in np.unique(groups):
        k = np.flatnonzero(groups == g)
        idx[k] = k[rng.permutation(len(k))]
    return idx


def test(Xk, eta, R, metric, B):
    emb = (lambda e: np.hstack([zscore(Xk), zscore(e)])) if metric == "std" else (lambda e: whiten(np.hstack([Xk, e])))
    fit, res = fl_parts(Xk, eta)
    obs = stats_sel(emb(eta), R)
    nl = np.array([stats_sel(emb(fit + res[strat_perm(STRATA)]), R) for _ in range(B)])
    return obs / nl.mean(0), (1 + (nl <= obs).sum(0)) / (1 + B)


rows = []
for code, lab, tg, base, add, prim in TESTS:
    R = D[tg].values.astype(float); Xk = D[base].values; eta = D[add].values.astype(float)
    for metric in ("std", "mahal"):
        rho, p = test(Xk, eta, R, metric, 2000)
        rows.append(dict(code=code, test=lab, primary=prim, metric=metric, rho_max=rho[0], p_max=p[0],
                         rho_mean=rho[1], p_mean=p[1]))
    print(code, flush=True)
T = pd.DataFrame(rows)
for col in ("p_max", "p_mean"):
    T[col.replace("p_", "holm_")] = np.nan
    for metric in ("std", "mahal"):
        g = T[(T.primary) & (T.metric == metric)]
        T.loc[g.index, col.replace("p_", "holm_")] = holm(g[col].values)
T.to_csv("v09_cond_final.csv", index=False)

cal = []
for code, lab, tg, base, add, prim in TESTS[:7]:
    R = D[tg].values.astype(float); Xk = D[base].values; eta = D[add].values.astype(float)
    for metric in ("std", "mahal"):
        ps = np.array([test(Xk, eta[strat_perm(CELL)], R, metric, 200)[1] for _ in range(60)])
        cal.append(dict(code=code, metric=metric, rej05_max=float(np.mean(ps[:, 0] <= 0.05)),
                        rej05_mean=float(np.mean(ps[:, 1] <= 0.05))))
    print("calib", code, flush=True)
C = pd.DataFrame(cal); C.to_csv("v09_cond_calib.csv", index=False)
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False)); print(C.to_string(index=False))
