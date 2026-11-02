"""Add the critical-distance level E1+J_{1.5 ell} to v09_levels.csv and v09_ci_n1.csv."""
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE
from descr_tools import surrogate_cv, omega_subsample_ci
from v09_stats import zscore, omega_sel, pair_index

rng = np.random.default_rng(20261109)
N = 127; R15 = 1.5 * 0.025
lm, mm = lame(1.0, 0.3); li, mi = lame(10.0, 0.3)
sol = FFTElastic2D(N)
x = (np.arange(N) + 0.5) / N
X1, X2 = np.meshgrid(x, x, indexing="ij")
d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
K = ((d1 ** 2 + d2 ** 2) <= R15 ** 2).astype(float); K /= K.sum(); KH = np.fft.fft2(K)
pf0 = PhaseFieldRVE(np.zeros((N, N), bool))
H0 = 0.5 * np.log((pf0.psi_plus((np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)))) / pf0.psic).max())
M = np.load("fields.npz")["mask"]
J = []
for m in M:
    lam = np.where(m, li, lm); mu = np.where(m, mi, mm)
    (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
    pf = PhaseFieldRVE(m)
    r = pf.psi_plus((e11, e22, e12)) / pf.psic
    J.append(0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(r) * KH)).max()) - H0)
X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True).rename(columns={"R_Gpath": "R_path"})
X["J15"] = J
pd.DataFrame({"id": X.id, "J15": J}).to_csv("v09_J15.csv", index=False)
E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
cols = E1 + ["J15"]
L = pd.read_csv("v09_levels.csv"); C = pd.read_csv("v09_ci_n1.csv")
L = L[L.level != "E1+J15"]; C = C[C.level != "E1+J15"]
new, newc = [], []
for sc, msk in (("all", np.ones(len(X), bool)), ("n>1", ((X.family != "laminate") & (X.n > 1)).values)):
    D = X[msk].reset_index(drop=True)
    II, JJ = pair_index(len(D))
    same = D.cell.values[II] == D.cell.values[JJ]
    for tn in ("R_sigma", "R_path"):
        R = D[tn].values.astype(float)
        o = omega_sel(zscore(D[cols].values), R, 0.01, 10000, rng, II=II, JJ=JJ)
        sl, _ = surrogate_cv(D[cols].values, R, D.cell.values, 1, 5, rng)
        new.append(dict(scope=sc, n=len(D), target=tn, level="E1+J15", ratio=o["ratio"], p_perm=o["p_perm"],
                        null_q05=o["null_q05"], nq=o["nq"], npairs=o["npairs"], within_cell=int(np.sum(o["sel"] & same)),
                        transl_copies=0, R2_loco=sl["R2"]))
        if sc == "n>1":
            lo, hi = omega_subsample_ci(D[cols].values, R, 0.01, rng, nsub=100, frac=0.8, nperm=60)
            newc.append(dict(target=tn, level="E1+J15", ci_lo=lo, ci_hi=hi))
pd.concat([L, pd.DataFrame(new)]).to_csv("v09_levels.csv", index=False)
pd.concat([C, pd.DataFrame(newc)]).to_csv("v09_ci_n1.csv", index=False)
print(pd.DataFrame(new)[["scope", "target", "ratio", "p_perm", "R2_loco"]].round(3)); print(pd.DataFrame(newc).round(2))
