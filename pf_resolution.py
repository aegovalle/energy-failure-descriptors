"""Resolution check of the AT1 campaign: selected RVEs regenerated at N=255
(same physical ell and strip width), compared with the N=127 results."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, time
import numpy as np
import pandas as pd


def geoms_all():
    from fft_rve import gen_inclusions
    import campaign_run as cr
    rng = np.random.default_rng(20261008)
    geoms = []
    for nrow in (16, 24, 32, 40, 48, 64):
        for orient in ("h", "v"):
            for nl in (1, 2, 4):
                geoms.append(cr.gen_laminate_rows(nrow, nl, orient))
    for fam, ar, ns in (("disks", 1.0, (1, 4, 9, 16)), ("ellipses", 3.0, (4, 9))):
        for f in (0.1, 0.2, 0.3):
            for n in ns:
                for r in range(10):
                    geoms.append(gen_inclusions(f, n, rng, ar=ar))
    return geoms


def work(task):
    key, mask = task
    from pf_fft import PhaseFieldRVE
    t0 = time.time()
    out, hist, d = PhaseFieldRVE(mask).run()
    out["time_s"] = time.time() - t0
    out["key"] = key
    np.save(f"pf_res/{key}_d.npy", d.astype(np.float32))
    return out


def main():
    from fft_rve import rasterize
    os.makedirs("pf_res", exist_ok=True)
    ids = [int(s) for s in sys.argv[1].split(",")]
    Ng = int(sys.argv[2]) if len(sys.argv) > 2 else 255
    geoms = geoms_all()
    MASK = np.load("fields.npz")["mask"]
    tasks = [("hom", np.zeros((Ng, Ng), bool))]
    for i in ids:
        m127 = rasterize(geoms[i], 127)
        assert np.array_equal(m127, MASK[i]), f"regeneration mismatch {i}"
        tasks.append((f"rve_{i:03d}", rasterize(geoms[i], Ng)))
    from multiprocessing import Pool
    with Pool(min(len(tasks), 24)) as pool:
        res = pool.map(work, tasks)
    df = pd.DataFrame(res)
    hom = df[df.key == "hom"].iloc[0]
    df["R_sigma"] = np.log(df.sigma_peak / hom.sigma_peak)
    df["R_Gamma"] = np.log(df.Gamma / hom.Gamma)
    df.to_csv(f"pf_resolution_{Ng}{sys.argv[3] if len(sys.argv) > 3 else ''}.csv", index=False)
    print(df[["key", "sigma_peak", "Gamma", "R_sigma", "R_Gamma", "separated", "time_s"]])


if __name__ == "__main__":
    main()
