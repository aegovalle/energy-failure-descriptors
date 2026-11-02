r"""V3 check: origin of the excess regularized dissipation in the homogeneous AT1 cell.
Hypothesis: the fully damaged plateau forms after the instability, when the broken band,
held only by the residual stiffness k_r, carries the imposed macroscopic strain; it should
therefore depend on k_r and on the strain overshoot of the separating load step, and not
disappear with grid refinement. Runs: k_r in {1e-2, 1e-3, 1e-4} x dE_rel in {0.01, 0.001}
at N=127, and k_r in {1e-3, 1e-4} at N=255. Output: v09_dissipation_check.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool


def work(task):
    N, kr, dE = task
    from pf_fft import PhaseFieldRVE
    pf = PhaseFieldRVE(np.zeros((N, N), bool), kres=kr)
    out, hist, d = pf.run(dE_rel=dE)
    c = d.mean(1)
    h = 1.0 / N
    ell = 0.025
    return dict(N=N, k_r=kr, dE_rel=dE, n_steps=out["n_steps"], eps_sep_over_e0=out["eps_sep"] / out["e0"],
                dissipation=out["Gamma"], plateau_over_ell=(c > 0.99).sum() * h / ell,
                support_over_ell=(c > 1e-6).sum() * h / ell,
                excess_pred_plateau=0.375 * 0.98 * (c > 0.99).sum() * h / ell)


if __name__ == "__main__":
    tasks = [(127, kr, dE) for kr in (1e-2, 1e-3, 1e-4) for dE in (0.01, 0.001)] + \
            [(255, kr, 0.01) for kr in (1e-3, 1e-4)]
    with Pool(len(tasks)) as p:
        res = p.map(work, tasks)
    df = pd.DataFrame(res)
    df.to_csv("v09_dissipation_check.csv", index=False)
    print(df.round(4).to_string(index=False))
