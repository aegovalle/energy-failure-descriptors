r"""Round-3 analyses (no new fracture solves).
(a) variance structure of the strength decomposition (inclusions, n>1, within f)
(b) mechanism check: growth term vs tail width ln V_eff and other independent variables
(c) nonlocal (critical-distance) onset intensity J_psi^r averaged over a disk of radius r
Outputs: v09_r3_decomp.json, v09_tcd.csv, v09_r3_tcd.json
"""
import json
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE
from descr_tools import surrogate_cv
from v09_stats import zscore, omega_sel

rng = np.random.default_rng(20261106)
X = (pd.read_csv("v06_desc.csv")[["id", "A_lig"]].merge(pd.read_csv("pf_descriptors.csv"), on="id")
     .sort_values("id").reset_index(drop=True))
DEC = pd.read_csv("v09_strength_decomp.csv")
X = X.merge(DEC[["id", "lnRE", "lnGamma", "onset_part"]], on="id")
X["minusJ"] = -X.J_psi
out = {}


def cov3(D):
    V = D[["lnRE", "minusJ", "lnGamma"]]
    return {"cov": V.cov().round(5).values.tolist(), "corr": V.corr().round(3).values.tolist(),
            "var_R_sigma": float(D.R_sigma.var()),
            "corr_Rsigma_onset": float(np.corrcoef(D.R_sigma, D.onset_part)[0, 1]),
            "slope_Rsigma_on_onset": float(np.polyfit(D.onset_part, D.R_sigma, 1)[0]),
            "corr_Rsigma_minusJ": float(np.corrcoef(D.R_sigma, D.minusJ)[0, 1]),
            "slope_Rsigma_on_minusJ": float(np.polyfit(D.minusJ, D.R_sigma, 1)[0]),
            "corr_Rsigma_lnRE": float(np.corrcoef(D.R_sigma, D.lnRE)[0, 1])}


inc = X[X.family != "laminate"]
n1 = inc[inc.n > 1]
out["inc180"] = cov3(inc)
out["n>1"] = cov3(n1)
out["within_f_n>1"] = {str(f): cov3(g) for f, g in n1.groupby("f_nom")}
# null expectation: corr(onset, growth) if R_sigma were independent of the onset part
vo, vr = n1.onset_part.var(), n1.R_sigma.var()
out["null_corr_if_independent"] = float(-np.sqrt(vo) / np.sqrt(vo + vr))
# (b) mechanism: growth vs independent variables
for nm, D in (("inc180", inc), ("n>1", n1)):
    out[f"mech_{nm}"] = {c: float(np.corrcoef(D.lnGamma, D[c])[0, 1]) for c in ("lnVeff", "I", "lnB", "A_lig", "f", "lnRE")}
    out[f"mech_{nm}_within_f"] = {str(f): {c: float(np.corrcoef(g.lnGamma, g[c])[0, 1]) for c in ("lnVeff", "I", "A_lig")}
                                  for f, g in D.groupby("f_nom")}

# (c) nonlocal onset intensity
N = 127
lam_m, mu_m = lame(1.0, 0.3); lam_i, mu_i = lame(10.0, 0.3)
MASK = np.load("fields.npz")["mask"]
sol = FFTElastic2D(N)


def disk_hat(r):
    x = (np.arange(N) + 0.5) / N
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
    K = ((d1 ** 2 + d2 ** 2) <= r ** 2).astype(float); K /= K.sum()
    return np.fft.fft2(K)


RADII = {"r0": 0.0, "r_half": 0.0125, "r_ell": 0.025, "r_2ell": 0.05}
KH = {k: disk_hat(r) for k, r in RADII.items() if r > 0}
rows = []
for i, mask in enumerate(MASK):
    lam = np.where(mask, lam_i, lam_m); mu = np.where(mask, mu_i, mu_m)
    (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
    pf = PhaseFieldRVE(mask)
    ratio = pf.psi_plus((e11, e22, e12)) / pf.psic
    r = {"id": i, "r0": 0.5 * np.log(ratio.max())}
    for k, Kh in KH.items():
        avg = np.real(np.fft.ifft2(np.fft.fft2(ratio) * Kh))
        r[k] = 0.5 * np.log(avg.max())
    rows.append(r)
T = pd.DataFrame(rows)
pf0 = PhaseFieldRVE(np.zeros((N, N), bool))
from fft_rve import FFTElastic2D as _F
hom_ratio = 0.5 * np.log((pf0.psi_plus((np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)))) / pf0.psic).max())
for k in RADII:
    T[f"J_{k}"] = T[k] - hom_ratio            # same normalization as J_psi (relative to the homogeneous cell)
T.to_csv("v09_tcd.csv", index=False)
X = X.merge(T[["id"] + [f"J_{k}" for k in RADII]], on="id")
D = X[(X.family != "laminate") & (X.n > 1)].reset_index(drop=True)
E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
res = {"check_J_r0_vs_Jpsi_maxabs": float(np.abs(X.J_r0 - X.J_psi).max())}
for k in ["J_psi"] + [f"J_{k}" for k in RADII] + ["lnB"]:
    o = omega_sel(zscore(D[E1 + [k]].values), D.R_sigma.values, 0.01, 4000, rng)
    sl, _ = surrogate_cv(D[E1 + [k]].values, D.R_sigma.values, D.cell.values, 1, 5, rng)
    s1, _ = surrogate_cv(D[[k]].values, D.R_sigma.values, D.cell.values, 1, 5, rng)
    res[k] = {"corr_with_Rsigma": float(np.corrcoef(D[k], D.R_sigma)[0, 1]),
              "ratio_E1_plus": o["ratio"], "R2_loco_E1_plus": sl["R2"], "R2_loco_alone": s1["R2"]}
json.dump(res, open("v09_r3_tcd.json", "w"), indent=1)
json.dump(out, open("v09_r3_decomp.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print(json.dumps(res, indent=1))
