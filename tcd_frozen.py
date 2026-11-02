"""Frozen prospective test of the critical-distance onset intensity J_{1.5 ell} (v0.9, round 4).
Protocol: tcd_frozen/tcd_protocol.md (SHA-256 of protocol and of this script in tcd_frozen/tcd_protocol.sha256).
Run from campaign/:
    python tcd_frozen/tcd_frozen.py generate      # new cells (new seed) + elastic solution + descriptors
    python tcd_frozen/tcd_frozen.py pf 24         # AT1 phase field (frozen defaults), resumable
    python tcd_frozen/tcd_frozen.py analyze       # decision rules H1-H3 exactly as frozen
Outputs in tcd_frozen/: tcd_elastic.csv, tcd_masks.npz, pf_out/, tcd_pf.csv, tcd_results.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time
sys.path.insert(0, os.getcwd())          # solvers live in campaign/ (run from there)
import numpy as np
import pandas as pd

HERE = "tcd_frozen"
OUT = os.path.join(HERE, "pf_out")
SEED = 20_261_102            # new seed, never used before
SEED_BOOT = 20_261_103
N = 127
ELL = 0.025
RADIUS = 1.5 * ELL           # frozen radius (chosen on the design set in v0.9)
SIG_HOM = 4.452871           # homogeneous-cell peak stress (as in blind_analyze.py)
E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
# set, family, aspect ratio, n values, f values, realizations
PLAN = (("C1", "disks", 1.0, (4, 9, 16), (0.1, 0.2, 0.3), 2),
        ("C1", "ellipses", 3.0, (4, 9), (0.1, 0.2, 0.3), 2),
        ("C2", "disks", 1.0, (25,), (0.1, 0.2, 0.3), 2),
        ("C2", "ellipses", 3.0, (16,), (0.1, 0.2, 0.3), 2))


def jr_factory():
    from fft_rve import FFTElastic2D, lame
    from pf_fft import PhaseFieldRVE
    lam_m, mu_m = lame(1.0, 0.3); lam_i, mu_i = lame(10.0, 0.3)
    sol = FFTElastic2D(N)
    x = (np.arange(N) + 0.5) / N
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
    K = ((d1 ** 2 + d2 ** 2) <= RADIUS ** 2).astype(float); K /= K.sum(); KH = np.fft.fft2(K)
    pf0 = PhaseFieldRVE(np.zeros((N, N), bool))
    H0 = 0.5 * np.log((pf0.psi_plus((np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)))) / pf0.psic).max())

    def jr(mask):
        lam = np.where(mask, lam_i, lam_m); mu = np.where(mask, mu_i, mu_m)
        (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
        pf = PhaseFieldRVE(mask)
        ratio = pf.psi_plus((e11, e22, e12)) / pf.psic        # both phases, local psi_c
        return {"J_0": 0.5 * np.log(ratio.max()) - H0,
                "J_1.5": 0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(ratio) * KH)).max()) - H0}
    return jr


def generate():
    from fft_rve import FFTElastic2D, gen_inclusions
    import campaign_run as cr
    rng = np.random.default_rng(SEED)
    sol = FFTElastic2D(N)
    jr = jr_factory()
    rows, masks, k = [], [], 0
    for st, fam, ar, ns, fs, nr in PLAN:
        for f in fs:
            for n in ns:
                for r in range(nr):
                    g = gen_inclusions(f, n, rng, ar=ar)
                    out, mask = cr.run(g, N, sol)
                    out.pop("_sbar", None)
                    out.update(cid=k, set=st, family=fam, ar=ar, f_nom=f, n=n, real=r, cell=f"{fam}_{f}_{n}")
                    out.update(jr(mask))
                    rows.append(out); masks.append(mask); k += 1
                    print(k, st, fam, f, n, r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "tcd_elastic.csv"), index=False)
    np.savez_compressed(os.path.join(HERE, "tcd_masks.npz"), mask=np.array(masks))
    print("saved", len(rows), "cells")


def work(task):
    key, mask = task
    from pf_fft import PhaseFieldRVE
    fj = os.path.join(OUT, f"{key}.json")
    if os.path.exists(fj):
        return key, "cached"
    t0 = time.time()
    out, hist, d = PhaseFieldRVE(mask).run()           # frozen defaults
    out["time_s"] = time.time() - t0
    np.save(os.path.join(OUT, f"{key}_d.npy"), d.astype(np.float32))
    json.dump(out, open(fj, "w"))
    return key, f"{out['time_s']:.0f}s sep={out['separated']}"


def pf(nw):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    M = np.load(os.path.join(HERE, "tcd_masks.npz"))["mask"]
    tasks = sorted([(f"c_{i:03d}", M[i]) for i in range(len(M))], key=lambda t: -t[1].mean())
    t0 = time.time()
    with Pool(nw) as pool:
        for k, (key, msg) in enumerate(pool.imap_unordered(work, tasks)):
            print(f"[{k+1}/{len(tasks)}] {key} {msg} ({time.time()-t0:.0f}s)", flush=True)


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def analyze():
    EL = pd.read_csv(os.path.join(HERE, "tcd_elastic.csv"))
    rows = []
    for i in EL.cid:
        o = json.load(open(os.path.join(OUT, f"c_{i:03d}.json")))
        rows.append(dict(cid=i, R_sigma=np.log(o["sigma_peak"] / SIG_HOM), separated=bool(o["separated"])))
    T = EL.merge(pd.DataFrame(rows), on="cid")
    T.to_csv(os.path.join(HERE, "tcd_pf.csv"), index=False)
    # training set: the 180 design inclusion cells (as in v09_tcd_blind.py)
    jr = jr_factory()
    MASK = np.load("fields.npz")["mask"]
    X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True)
    TR = X[X.family != "laminate"].reset_index(drop=True)
    TR = TR.merge(pd.DataFrame([dict(id=int(i), **jr(MASK[int(i)])) for i in TR.id]), on="id")
    models = {"E1+J_1.5": E1 + ["J_1.5"], "E1+R_H": E1 + ["lnB"], "E1+J_psi": E1 + ["J_0"], "E1": E1,
              "J_1.5 alone": ["J_1.5"]}

    def fit_pred(cols, te):
        m, s = TR[cols].mean(), TR[cols].std()
        A = np.column_stack([np.ones(len(TR)), ((TR[cols] - m) / s).values])
        c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ TR.R_sigma.values)
        return np.column_stack([np.ones(len(te)), ((te[cols] - m) / s).values]) @ c

    res = {"n_cells": {s: int((T.set == s).sum()) for s in ("C1", "C2")},
           "n_separated": {s: int(((T.set == s) & T.separated).sum()) for s in ("C1", "C2")}}
    rng = np.random.default_rng(SEED_BOOT)
    for st in ("C1", "C2"):
        te = T[(T.set == st) & T.separated].reset_index(drop=True)
        y = te.R_sigma.values
        P = {nm: fit_pred(cols, te) for nm, cols in models.items()}
        res[st] = {nm: r2(y, p) for nm, p in P.items()}
        res[st]["corr_J_1.5"] = float(np.corrcoef(te["J_1.5"], y)[0, 1])
        if st == "C1":
            B = {"vs_E1+R_H": [], "vs_E1+J_psi": []}
            for _ in range(2000):
                idx = rng.integers(0, len(y), len(y))
                if np.var(y[idx]) == 0:
                    continue
                a = r2(y[idx], P["E1+J_1.5"][idx])
                B["vs_E1+R_H"].append(a - r2(y[idx], P["E1+R_H"][idx]))
                B["vs_E1+J_psi"].append(a - r2(y[idx], P["E1+J_psi"][idx]))
            res["C1_boot_lower5"] = {k: float(np.quantile(v, 0.05)) for k, v in B.items()}
    frac = res["n_separated"]["C1"] / res["n_cells"]["C1"]
    if frac < 0.8:
        res["H1"] = res["H2"] = "inconclusive (fewer than 80% of C1 cells separated)"
    else:
        res["H1"] = "confirmed" if res["C1"]["E1+J_1.5"] >= 0.80 else "refuted"
        res["H2"] = "confirmed" if min(res["C1_boot_lower5"].values()) > 0 else "refuted"
    res["H3"] = "confirmed" if res["C2"]["E1+J_1.5"] >= 0.70 else "refuted"
    json.dump(res, open(os.path.join(HERE, "tcd_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "generate":
        generate()
    elif cmd == "pf":
        pf(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    elif cmd == "analyze":
        analyze()
