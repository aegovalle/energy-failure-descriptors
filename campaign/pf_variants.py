"""Sensitivity and robustness campaigns for the AT1 study.

Variants (relative to the baseline of pf_campaign.py):
  orient    : all 216 RVEs loaded along e2 (implemented by transposing the masks)
  contrast3 : E_i/E_m = 3 (G_c^i/G_c^m = 3 kept), 61-RVE stratified subset
  ell040    : ell = 0.040 L, same subset
  dE0005    : load increment 0.005 eps0, 16-RVE subset
  tol3e4    : staggered tolerance 3e-4, 16-RVE subset
  strip095  : symmetry-breaking strip at 5 % (instead of 2 %), 16-RVE subset
  strip099  : strip at 1 %, 16-RVE subset
Each variant also runs its own homogeneous cell. Results are cached in
pf_var/<variant>/ (resumable).   Usage: python pf_variants.py [n_workers]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time
import numpy as np
import pandas as pd

ROOT = "pf_var"


def subsets(df):
    lam = df[(df.family == "laminate") & (df.n.isin([1, 4])) &
             (np.round(df.f, 3).isin([0.126, 0.252, 0.378]))].id.tolist()
    inc = []
    for fam, ns in (("disks", (4, 9, 16)), ("ellipses", (4, 9))):
        for f in (0.1, 0.2, 0.3):
            for n in ns:
                inc += df[(df.family == fam) & (df.f_nom == f) & (df.n == n) &
                          (df.real.isin([0, 3, 8]))].id.tolist()
    n1 = df[(df.family == "disks") & (df.n == 1) & (df.real == 0)].id.tolist()
    s61 = sorted(set(lam + inc + n1 + [73]))
    s16 = sorted(set([13, 16, 1, 28, 49, 73, 54, 59, 96, 100, 116, 130, 158, 170, 190, 205]))
    return s61, s16


VARIANTS = {
    "orient":    dict(sub="all", transpose=True, model={}, run={}),
    "contrast3": dict(sub="s61", transpose=False, model={"Ei": 3.0}, run={}),
    "ell040":    dict(sub="s61", transpose=False, model={"ell": 0.040}, run={}),
    "dE0005":    dict(sub="s16", transpose=False, model={}, run={"dE_rel": 0.005}),
    "tol3e4":    dict(sub="s16", transpose=False, model={}, run={"tol_d": 3e-4}),
    "strip095":  dict(sub="s16", transpose=False, model={"strip_fac": 0.95}, run={}),
    "strip099":  dict(sub="s16", transpose=False, model={"strip_fac": 0.99}, run={}),
    "var_iso":   dict(sub="s16", transpose=False, model={"split": False}, run={}),
    "miehe":     dict(sub="s16", transpose=False, model={"split": "miehe"}, run={}),
}


def work(task):
    var, key, mask, model, runkw = task
    from pf_fft import PhaseFieldRVE
    d = os.path.join(ROOT, var)
    fj = os.path.join(d, f"{key}.json")
    if os.path.exists(fj):
        return var, key, "cached"
    t0 = time.time()
    out, hist, dd = PhaseFieldRVE(mask, **model).run(**runkw)
    out["time_s"] = time.time() - t0
    np.save(os.path.join(d, f"{key}_d.npy"), dd.astype(np.float32))
    np.save(os.path.join(d, f"{key}_hist.npy"), hist)
    with open(fj, "w") as f:
        json.dump(out, f)
    return var, key, f"{out['time_s']:.0f}s sep={out['separated']}"


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    df = pd.read_csv("campaign_results.csv")
    MASK = np.load("fields.npz")["mask"]
    N = MASK.shape[1]
    s61, s16 = subsets(df)
    json.dump({"s61": s61, "s16": s16}, open("pf_var_subsets.json", "w"))
    tasks = []
    for var, cfg in VARIANTS.items():
        os.makedirs(os.path.join(ROOT, var), exist_ok=True)
        ids = {"all": list(range(len(df))), "s61": s61, "s16": s16}[cfg["sub"]]
        tasks.append((var, "hom", np.zeros((N, N), bool), cfg["model"], cfg["run"]))
        for i in ids:
            m = MASK[i].T.copy() if cfg["transpose"] else MASK[i]
            tasks.append((var, f"rve_{i:03d}", m, cfg["model"], cfg["run"]))
    # expensive (inclusion-rich, small increments) first
    def cost(t):
        c = t[2].mean() + (t[2].mean() > 0) * 0.5
        return -(c * (2.0 if t[4].get("dE_rel") == 0.005 else 1.0))
    tasks.sort(key=cost)
    print(len(tasks), "tasks", flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nw) as pool:
        for k, (var, key, msg) in enumerate(pool.imap_unordered(work, tasks)):
            print(f"[{k+1}/{len(tasks)}] {var}/{key} {msg} ({time.time()-t0:.0f}s)", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
