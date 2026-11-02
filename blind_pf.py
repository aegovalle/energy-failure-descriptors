"""Blind validation set (v0.7): AT1 phase-field runs on the PF subset (Sets B and O-PF).
Frozen settings: PhaseFieldRVE(mask) and run() defaults (see blind/blind_protocol.md).
Run from campaign/:  python blind_pf.py [n_workers]
Per-RVE cache in blind/pf_out/ (resumable).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time
import numpy as np
import pandas as pd

OUT = os.path.join("blind", "pf_out")


def work(task):
    key, mask = task
    from pf_fft import PhaseFieldRVE
    fj = os.path.join(OUT, f"{key}.json")
    if os.path.exists(fj):
        return key, "cached"
    t0 = time.time()
    out, hist, d = PhaseFieldRVE(mask).run()
    out["time_s"] = time.time() - t0
    np.save(os.path.join(OUT, f"{key}_d.npy"), d.astype(np.float32))
    with open(fj, "w") as f:
        json.dump(out, f)
    return key, f"{out['time_s']:.0f}s sep={out['separated']} steps={out['n_steps']}"


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join("blind", "blind_elastic.csv"))
    M = np.load(os.path.join("blind", "blind_masks.npz"))["mask"]
    ids = df.bid[df.pf].tolist()
    tasks = sorted([(f"b_{i:03d}", M[i]) for i in ids], key=lambda t: -t[1].mean())
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nw) as pool:
        for k, (key, msg) in enumerate(pool.imap_unordered(work, tasks)):
            print(f"[{k+1}/{len(tasks)}] {key} {msg} ({time.time()-t0:.0f}s)", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
