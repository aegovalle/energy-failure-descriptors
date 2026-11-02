r"""Mechanism test for the critical-distance onset intensity: does the optimal averaging radius scale
with the regularization length? Uses the stored model variants on the 61-cell subset
(baseline l=0.025L, l=0.040L, and contrast E_i/E_m=3). Output: v09_tcd_ell.csv"""
import json, os
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE
from descr_tools import surrogate_cv

rng = np.random.default_rng(20261108)
N = 127
R_ABS = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.1]
x = (np.arange(N) + 0.5) / N
X1, X2 = np.meshgrid(x, x, indexing="ij")
d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
KH = {}
for r in R_ABS[1:]:
    K = ((d1 ** 2 + d2 ** 2) <= r ** 2).astype(float); K /= K.sum(); KH[r] = np.fft.fft2(K)
sol = FFTElastic2D(N)
MASK = np.load("fields.npz")["mask"]
s61 = json.load(open("pf_var_subsets.json"))["s61"]
base = pd.read_csv("pf_descriptors.csv").set_index("id")


def jr(mask, Ei, ell):
    lm, mm = lame(1.0, 0.3); li, mi = lame(Ei, 0.3)
    lam = np.where(mask, li, lm); mu = np.where(mask, mi, mm)
    (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
    pf = PhaseFieldRVE(mask, Ei=Ei, ell=ell)
    ratio = pf.psi_plus((e11, e22, e12)) / pf.psic
    out = {0.0: 0.5 * np.log(ratio.max())}
    for r, Kh in KH.items():
        out[r] = 0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(ratio) * Kh)).max())
    return out


rows = []
for case, Ei, ell in (("baseline", 10.0, 0.025), ("ell040", 10.0, 0.040), ("contrast3", 3.0, 0.025)):
    if case == "baseline":
        sig = {i: base.sigma_peak[i] for i in s61}; hom = json.load(open("pf_hom.json"))["sigma_peak"]
    else:
        sig = {i: json.load(open(f"pf_var/{case}/rve_{i:03d}.json"))["sigma_peak"] for i in s61}
        hom = json.load(open(f"pf_var/{case}/hom.json"))["sigma_peak"]
    for i in s61:
        if base.family[i] == "laminate":
            continue
        J = jr(MASK[i], Ei, ell)
        r = dict(case=case, id=i, cell=base.cell[i], f=base.f[i], R_sigma=np.log(sig[i] / hom))
        r.update({f"J_{k:g}": v for k, v in J.items()})
        rows.append(r)
    print(case, flush=True)
T = pd.DataFrame(rows)
T.to_csv("v09_tcd_ell_raw.csv", index=False)
res = []
for case, g in T.groupby("case"):
    g = g.reset_index(drop=True)
    for r in R_ABS:
        c = f"J_{r:g}"
        s, _ = surrogate_cv(g[["f", c]].values, g.R_sigma.values, g.cell.values, 1, 5, rng)
        res.append(dict(case=case, r=r, corr=float(np.corrcoef(g[c], g.R_sigma)[0, 1]), R2_loco_f_J=s["R2"], n=len(g)))
R = pd.DataFrame(res); R.to_csv("v09_tcd_ell.csv", index=False)
print(R.pivot_table(index="r", columns="case", values=["corr", "R2_loco_f_J"]).round(3))
