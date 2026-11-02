"""Blind validation set (v0.7): generation, elastic solution and a priori descriptors.
Protocol frozen in blind/blind_protocol.md (hash in blind/blind_protocol.sha256).
Run from campaign/:  python blind_generate.py
Outputs: blind/blind_elastic.csv, blind/blind_masks.npz
"""
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, rasterize, gen_inclusions
import campaign_run as cr
from descr_tools import ligament, ligament_amp, min_path
from pf_fft import PhaseFieldRVE

SEED = 7_202_610
N = 127
P_MIN_HOM = 0.98


def main():
    rng = np.random.default_rng(SEED)
    sol = FFTElastic2D(N)
    plan = []
    for fam, ar, ns, fs, nr, st in (("disks", 1.0, (4, 9, 16), (0.1, 0.2, 0.3), 6, "A"),
                                    ("ellipses", 3.0, (4, 9), (0.1, 0.2, 0.3), 6, "A"),
                                    ("ellipses6", 6.0, (4, 9, 16), (0.1, 0.2), 5, "O")):
        for f in fs:
            for n in ns:
                for r in range(nr):
                    plan.append((st, fam, ar, f, n, r))
    rows, masks = [], []
    for k, (st, fam, ar, f, n, r) in enumerate(plan):
        g = gen_inclusions(f, n, rng, ar=ar)
        out, mask = cr.run(g, N, sol)
        out.pop("_sbar", None)
        Gc = PhaseFieldRVE(mask).Gc
        out.update(bid=k, set=st, family=fam, ar=ar, f_nom=f, n=n, real=r,
                   cell=f"{fam}_{f}_{n}", pf=(r < (3 if st == "A" else 2)),
                   lig_min=ligament(mask)[0], A_lig=ligament_amp(mask, k=10.0)[0],
                   lnPmin=np.log(min_path(Gc, np.ones((N, N), bool)) / P_MIN_HOM))
        rows.append(out)
        masks.append(mask)
        if k % 10 == 0:
            print(k, st, fam, f, n, r, f"iters={out['iters']} lnB={out['lnB']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("blind/blind_elastic.csv", index=False)
    np.savez_compressed("blind/blind_masks.npz", mask=np.array(masks))
    print("saved", len(df), "states; PF subset", int(df.pf.sum()), "max iters", df.iters.max())


if __name__ == "__main__":
    main()
