r"""Residual-stiffness check: re-run the homogeneous cell and the two counterexample pairs
(49-73 strength, 54-59 elastic) plus two further inclusion cells with k_r = 1e-4 instead of 1e-3
(N = 127, all other settings unchanged). Output: v09_kres_check.csv"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool

IDS = [49, 73, 54, 59, 96, 167]


def work(task):
    key, kr = task
    from pf_fft import PhaseFieldRVE
    from descr_tools import min_path
    M = np.load("fields.npz")["mask"]
    mask = np.zeros((127, 127), bool) if key == "hom" else M[key]
    pf = PhaseFieldRVE(mask, kres=kr)
    out, hist, d = pf.run()
    cp = min_path(pf.Gc, d >= 0.9)
    return dict(key=key, k_r=kr, sigma_peak=out["sigma_peak"], C_path=cp, Gamma=out["Gamma"])


if __name__ == "__main__":
    tasks = [("hom", 1e-4)] + [(i, 1e-4) for i in IDS]
    with Pool(len(tasks)) as p:
        res = pd.DataFrame(p.map(work, tasks))
    h = res[res.key == "hom"].iloc[0]
    res["R_sigma"] = np.log(res.sigma_peak / h.sigma_peak)
    res["R_path"] = np.log(res.C_path / h.C_path)
    base = pd.read_csv("pf_descriptors.csv").set_index("id")
    res["R_sigma_base"] = [0.0 if k == "hom" else base.R_sigma[k] for k in res.key]
    res["R_path_base"] = [0.0 if k == "hom" else base.R_Gpath[k] for k in res.key]
    res["dR_sigma"] = res.R_sigma - res.R_sigma_base
    res["dR_path"] = res.R_path - res.R_path_base
    res.to_csv("v09_kres_check.csv", index=False)
    print(res.round(4).to_string(index=False))
