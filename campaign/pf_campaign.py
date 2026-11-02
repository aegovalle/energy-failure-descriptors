"""In-house AT1 phase-field FFT campaign on the 216 RVEs of the elastic campaign.

Usage:  python pf_campaign.py [n_workers]
Per-RVE results are cached in pf_out/ (resumable); merged outputs:
pf_results.csv, pf_fields.npz (final damage fields), pf_hist.npz.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time
import numpy as np
import pandas as pd

OUT = "pf_out"


def work(task):
    key, mask = task
    from pf_fft import PhaseFieldRVE
    fj = os.path.join(OUT, f"{key}.json")
    if os.path.exists(fj):
        return key, "cached"
    t0 = time.time()
    pf = PhaseFieldRVE(mask)
    out, hist, d = pf.run()
    out["time_s"] = time.time() - t0
    np.save(os.path.join(OUT, f"{key}_d.npy"), d.astype(np.float32))
    np.save(os.path.join(OUT, f"{key}_hist.npy"), hist)
    with open(fj, "w") as f:
        json.dump(out, f)
    return key, f"{out['time_s']:.1f}s sep={out['separated']} steps={out['n_steps']}"


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("campaign_results.csv")
    MASK = np.load("fields.npz")["mask"]
    N = MASK.shape[1]
    tasks = [("hom", np.zeros((N, N), bool))] + [(f"rve_{i:03d}", MASK[i]) for i in range(len(df))]
    # longest (inclusion-rich) first for load balance
    tasks = [tasks[0]] + sorted(tasks[1:], key=lambda t: -t[1].mean())
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nw) as pool:
        for k, (key, msg) in enumerate(pool.imap_unordered(work, tasks)):
            print(f"[{k+1}/{len(tasks)}] {key} {msg}  ({time.time()-t0:.0f}s)", flush=True)
    # merge
    rows, D, Hs = [], [], {}
    for i in range(len(df)):
        key = f"rve_{i:03d}"
        r = json.load(open(os.path.join(OUT, f"{key}.json")))
        r["id"] = i
        rows.append(r)
        D.append(np.load(os.path.join(OUT, f"{key}_d.npy")))
        Hs[key] = np.load(os.path.join(OUT, f"{key}_hist.npy"))
    hom = json.load(open(os.path.join(OUT, "hom.json")))
    res = pd.DataFrame(rows)
    res["R_sigma"] = np.log(res["sigma_peak"] / hom["sigma_peak"])
    res["R_Gamma"] = np.log(res["Gamma"] / hom["Gamma"])
    res.to_csv("pf_results.csv", index=False)
    np.savez_compressed("pf_fields.npz", d=np.array(D))
    np.savez_compressed("pf_hist.npz", hom=np.load(os.path.join(OUT, "hom_hist.npy")), **Hs)
    with open("pf_hom.json", "w") as f:
        json.dump(hom, f, indent=1)
    print("done", time.time() - t0, "s; separated:", res["separated"].sum(), "/", len(res))


if __name__ == "__main__":
    main()
