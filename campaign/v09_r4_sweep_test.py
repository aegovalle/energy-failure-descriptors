r"""Round 4: conditional test for E1 -> E1 + J_r that accounts for the radius sweep.
The radius was chosen as the one maximizing R2_LOCO of E1+J_r over FAC (in units of ell). Here the
whole selection is repeated inside each local-swap draw (all J_r columns swapped jointly):
  (a) 'selected': statistic at the radius that maximizes R2_LOCO in that draw (mimics the actual choice);
  (b) 'min over r': minimum over the radii of the nearest-pair statistic (max-T type global test).
Also: J_r at N = 255 for the 20 resolution cells (sensitivity to resolution, compared with J_psi).
Outputs: v09_r4_sweep.json, v09_r4_res255.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
import pandas as pd

exec(open("v09_r3_power.py", encoding="utf-8").read().split("SETUPS =")[0])   # D, samplers, stats, run_test
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE
from descr_tools import surrogate_cv

ELL = 0.025
FAC = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
lam_m, mu_m = lame(1.0, 0.3); lam_i, mu_i = lame(10.0, 0.3)


def jr_all(mask):
    n = mask.shape[0]
    sol = FFTElastic2D(n)
    x = (np.arange(n) + 0.5) / n
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    d1 = (X1 - 0.5 / n + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / n + 0.5) % 1.0 - 0.5
    lam = np.where(mask, lam_i, lam_m); mu = np.where(mask, mu_i, mu_m)
    (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
    pf = PhaseFieldRVE(mask)
    ratio = pf.psi_plus((e11, e22, e12)) / pf.psic
    pf0 = PhaseFieldRVE(np.zeros((n, n), bool))
    H0 = 0.5 * np.log((pf0.psi_plus((np.ones((n, n)), np.zeros((n, n)), np.zeros((n, n)))) / pf0.psic).max())
    out = {}
    for a in FAC:
        if a == 0:
            out["J_0"] = 0.5 * np.log(ratio.max()) - H0
            continue
        K = ((d1 ** 2 + d2 ** 2) <= (a * ELL) ** 2).astype(float); K /= K.sum()
        out[f"J_{a:g}"] = 0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(ratio) * np.fft.fft2(K))).max()) - H0
    return out


if __name__ == "__main__":
    rg = np.random.default_rng(20261108)
    JC = [f"J_{a:g}" for a in FAC]
    DJ = pd.DataFrame([dict(id=int(i), **jr_all(M[int(i)])) for i in D.id])
    Dn = D.merge(DJ, on="id")
    assert len(Dn) == len(D)
    Xk = Dn[E1].values; R = Dn.R_sigma.values.astype(float); ETA = Dn[JC].values.astype(float)
    cell = Dn.cell.values
    Zk = zscore(Xk)

    def r2loco(eta):
        return np.array([surrogate_cv(np.column_stack([Xk, eta[:, k]]), R, cell, 1, 5, rg)[0]["R2"]
                         for k in range(eta.shape[1])])

    def stat_all(eta):
        return np.array([stats_sel(np.hstack([Zk, zscore(eta[:, [k]])]), R) for k in range(eta.shape[1])])

    r2o = r2loco(ETA); so = stat_all(ETA)
    kb = int(np.argmax(r2o))
    obs_sel = so[kb]; obs_min = so.min(0)
    draw = swap_draw_factory(Xk, ETA)
    B = 2000
    nsel, nmin, nfix = [], [], []
    k15 = JC.index("J_1.5")
    for b in range(B):
        e = draw(rg)
        r2b = r2loco(e); sb = stat_all(e)
        nsel.append(sb[int(np.argmax(r2b))]); nmin.append(sb.min(0)); nfix.append(sb[k15])
        if b % 200 == 0:
            print(b, flush=True)
    nsel, nmin, nfix = map(np.array, (nsel, nmin, nfix))
    p = lambda nl, ob: ((1 + (nl <= ob).sum(0)) / (1 + B)).tolist()
    res = {"radii_ell": FAC, "R2_LOCO_obs": r2o.tolist(), "best": JC[kb],
           "p_fixed_1.5_[max,mean]": p(nfix, so[k15]),
           "p_selected_by_R2LOCO_[max,mean]": p(nsel, obs_sel),
           "p_min_over_r_[max,mean]": p(nmin, obs_min),
           "rho_selected_[max,mean]": (obs_sel / nsel.mean(0)).tolist()}
    json.dump(res, open("v09_r4_sweep.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
