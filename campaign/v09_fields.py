r"""v0.9 re-analysis, step A0: re-solve the 216 undamaged elastic cells from the stored
phase maps (fields.npz) and extract the additional field statistics requested in review:

  * min tr(eps) over each cell (hybrid AT1 = isotropic AT1 if tr(eps) >= 0 everywhere)
  * full phase-wise strain covariance (3 x 3 symmetric in Voigt-like components)
  * phase-wise mean / std of <tr eps>_+ and |eps_dev| (the Amor-split alphabet)
  * phase-wise mean energy <W>_alpha (positive control target)
  * Amor psi^+ field -> field-weighted minimum cut P_psi = min int Gc / psi^+_n ds
  * two-point autocorrelation of the phase indicator (MKS baseline)

Outputs: v09_fields.csv (one row per cell), v09_S2.npy (cropped 2-point maps).
Run from campaign/:  python v09_fields.py
"""
import time
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE
from descr_tools import min_path

N = 127
TOL = 1e-7
lam_m, mu_m = lame(1.0, 0.3)
lam_i, mu_i = lame(10.0, 0.3)
EBAR = (1.0, 0.0, 0.0)
CROP = 31                                  # half-width of the cropped 2-point map

F = np.load("fields.npz")
MASK = F["mask"]
sol = FFTElastic2D(N)


def psi_amor(lam, mu, e11, e22, e12):
    K = lam + 2.0 * mu / 3.0
    tr = e11 + e22
    dev2 = e11 ** 2 + e22 ** 2 + 2 * e12 ** 2 - tr ** 2 / 3.0
    return 0.5 * K * np.maximum(tr, 0.0) ** 2 + mu * dev2


# homogeneous-matrix reference value of psi^+ under the same load
PSI_HOM = float(psi_amor(lam_m, mu_m, np.array(1.0), np.array(0.0), np.array(0.0)))

rows, S2 = [], []
t0 = time.time()
for i, mask in enumerate(MASK):
    lam = np.where(mask, lam_i, lam_m)
    mu = np.where(mask, mu_i, mu_m)
    (e11, e22, e12), (s11, s22, s12), it, err = sol.solve(lam, mu, EBAR, tol=TOL, maxit=4000)
    tr = e11 + e22
    trp = np.maximum(tr, 0.0)
    edev = np.sqrt(np.maximum(e11 ** 2 + e22 ** 2 + 2 * e12 ** 2 - tr ** 2 / 3.0, 0.0))
    w = 0.5 * (s11 * e11 + s22 * e22 + 2 * s12 * e12)
    r = {"id": i, "min_tr": float(tr.min()), "frac_tr_neg": float(np.mean(tr < 0))}
    for tag, ph in (("m", ~mask), ("i", mask)):
        if ph.sum() == 0:
            for k in ("c11", "c22", "c12", "c1122", "c1112", "c2212", "trp_mu", "trp_sd",
                      "dev_mu", "dev_sd", "Wph"):
                r[f"{k}_{tag}"] = 0.0
            continue
        a, b, c = e11[ph], e22[ph], e12[ph]
        r[f"c11_{tag}"] = float(a.var())
        r[f"c22_{tag}"] = float(b.var())
        r[f"c12_{tag}"] = float(c.var())
        r[f"c1122_{tag}"] = float(np.mean((a - a.mean()) * (b - b.mean())))
        r[f"c1112_{tag}"] = float(np.mean((a - a.mean()) * (c - c.mean())))
        r[f"c2212_{tag}"] = float(np.mean((b - b.mean()) * (c - c.mean())))
        r[f"trp_mu_{tag}"] = float(trp[ph].mean())
        r[f"trp_sd_{tag}"] = float(trp[ph].std())
        r[f"dev_mu_{tag}"] = float(edev[ph].mean())
        r[f"dev_sd_{tag}"] = float(edev[ph].std())
        r[f"Wph_{tag}"] = float(w[ph].mean() / (0.5 * (lam_m + 2 * mu_m)))
    # field-weighted minimum cut: Gc / (psi^+ / psi_hom), same Gc map (incl. strip) as the PF model
    pf = PhaseFieldRVE(mask)
    psin = np.maximum(psi_amor(lam, mu, e11, e22, e12) / PSI_HOM, 1e-6)
    r["P_psi"] = min_path(pf.Gc / psin, np.ones_like(mask, bool))
    # two-point autocovariance of the phase indicator (translation invariant)
    x = mask.astype(float) - mask.mean()
    ac = np.real(np.fft.ifft2(np.abs(np.fft.fft2(x)) ** 2)) / x.size
    ac = np.fft.fftshift(ac)
    c0 = N // 2
    S2.append(ac[c0 - CROP:c0 + CROP + 1, c0 - CROP:c0 + CROP + 1].astype(np.float32))
    rows.append(r)
    if i % 20 == 0:
        print(i, f"it={it} min_tr={r['min_tr']:.3e} t={time.time() - t0:.0f}s", flush=True)

# homogeneous reference for P_psi
pf_h = PhaseFieldRVE(np.zeros((N, N), bool))
P_psi_hom = min_path(pf_h.Gc, np.ones((N, N), bool))
df = pd.DataFrame(rows)
df["lnPpsi"] = np.log(df.P_psi / P_psi_hom)
df.to_csv("v09_fields.csv", index=False)
np.save("v09_S2.npy", np.array(S2))
print("done", time.time() - t0, "s; global min tr", df.min_tr.min(), "P_psi_hom", P_psi_hom, flush=True)
