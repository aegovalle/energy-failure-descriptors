r"""(a) Calibration of the conditional samplers under a true conditional null: the added block is
replaced by an independent draw from its own Freedman--Lane model (quadratic fit in Psi_k plus
permuted residuals), which is conditionally independent of R given Psi_k up to the model.
The rejection rate at 5% over repeated draws should be close to 0.05.
(b) Extra conditional tests for the split moments (E0 -> E0+split), both metrics, FL sampler.
Outputs: v09_power_calib.csv, v09_power_split.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from v09_stats import zscore, pair_index

rng = np.random.default_rng(20261104)
Q = 0.01
X = (pd.read_csv("v06_desc.csv")[["id", "A_lig"]]
     .merge(pd.read_csv("pf_descriptors.csv"), on="id")
     .merge(pd.read_csv("v09_fields.csv"), on="id")
     .rename(columns={"R_Gpath": "R_path"}).sort_values("id").reset_index(drop=True))
E0 = ["f", "e11_m", "e22_m", "e12_m"]; E1 = E0 + ["Jgap_m", "Jgap_i"]
COV = [f"{c}_{t}" for t in ("m", "i") for c in ("c11", "c22", "c12", "c1122", "c1112", "c2212")]
SPL = [f"{c}_{t}" for t in ("m", "i") for c in ("trp_mu", "trp_sd", "dev_mu", "dev_sd")]
LEV = {"G1": ["f", "rhoI"], "E0": E0, "E1": E1}
TG = {"R_H": "lnB", "R_sigma": "R_sigma", "R_path": "R_path"}
SC = {"all": np.ones(len(X), bool), "n>1": ((X.family != "laminate") & (X.n > 1)).values}
PRIMARY = [("P1", "n>1", "R_H", "E0", ["Jgap_m", "Jgap_i"]), ("P2", "n>1", "R_H", "E0", COV),
           ("P3", "n>1", "R_sigma", "E0", ["Jgap_m", "Jgap_i"]), ("P4", "n>1", "R_sigma", "E1", ["lnB"]),
           ("P5", "n>1", "R_H", "G1", ["A_lig"]), ("P6", "all", "R_path", "G1", ["lnPmin"])]


def whiten(A, tol=1e-8):
    A = np.asarray(A, float); A = A - A.mean(0)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * s.max()] * np.sqrt(len(A))


def stats_sel(Z, R, II, JJ):
    d = pdist(Z); nq = int(np.ceil(Q * len(d) - 1e-9))
    sel = d <= np.partition(d, nq - 1)[nq - 1] + 1e-12
    dR = np.abs(R[II] - R[JJ])[sel]
    return np.array([dR.max(), dR.mean()])


def fl_model(Zk, eta):
    p = Zk.shape[1]
    Phi = np.column_stack([np.ones(len(Zk))] + [Zk[:, a] for a in range(p)]
                          + [Zk[:, a] * Zk[:, b] for a in range(p) for b in range(a, p)])
    C = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ eta)
    fit = Phi @ C
    return fit, eta - fit


def fl_test(Xk, Ea, R, B, metric):
    II, JJ = pair_index(len(R))
    emb = (lambda e: np.hstack([zscore(Xk), zscore(e)])) if metric == "std" else (lambda e: whiten(np.hstack([Xk, e])))
    fit, res = fl_model(zscore(Xk), Ea)
    obs = stats_sel(emb(Ea), R, II, JJ)
    nl = np.array([stats_sel(emb(fit + res[rng.permutation(len(R))]), R, II, JJ) for _ in range(B)])
    return obs / nl.mean(0), (1 + (nl <= obs).sum(0)) / (1 + B), fit, res


# (a) calibration
cal = []
for lab, sc, tg, base, add in PRIMARY:
    D = X[SC[sc]].reset_index(drop=True)
    R = D[TG[tg]].values.astype(float); Xk = D[LEV[base]].values; Ea = D[add].values.astype(float)
    fit, res = fl_model(zscore(Xk), Ea)
    for metric in ("std", "mahal"):
        ps = []
        for r in range(60):
            eta0 = fit + res[rng.permutation(len(R))]          # synthetic null block
            _, p, _, _ = fl_test(Xk, eta0, R, 200, metric)
            ps.append(p)
        ps = np.array(ps)
        cal.append(dict(test=lab, metric=metric, rej05_max=float(np.mean(ps[:, 0] <= 0.05)),
                        rej05_mean=float(np.mean(ps[:, 1] <= 0.05)), n=len(ps)))
    print("calib", lab, flush=True)
pd.DataFrame(cal).to_csv("v09_power_calib.csv", index=False)

# (b) split moments
sp = []
for sc in ("all", "n>1"):
    D = X[SC[sc]].reset_index(drop=True)
    for tg in ("R_H", "R_sigma", "R_path"):
        R = D[TG[tg]].values.astype(float)
        for metric in ("std", "mahal"):
            rho, p, _, _ = fl_test(D[E0].values, D[SPL].values.astype(float), R, 2000, metric)
            sp.append(dict(scope=sc, target=tg, metric=metric, rho_max=rho[0], p_max=p[0], rho_mean=rho[1], p_mean=p[1]))
pd.DataFrame(sp).to_csv("v09_power_split.csv", index=False)
print(pd.DataFrame(cal).to_string(index=False)); print(pd.DataFrame(sp).round(3).to_string(index=False))
