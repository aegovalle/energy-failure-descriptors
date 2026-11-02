r"""
Phase A'/B'/C campaign: laminates (exact control), disks and ellipses
(controlled loss), plus resolution study. Writes campaign_results.csv,
fields.npz and resolution.csv.
"""
import time
import numpy as np
import pandas as pd
from fft_rve import (FFTElastic2D, lame, rasterize, gen_inclusions,
                     analyze_state)

N = 127
NREF = 255
ELL = 0.025
M = 20
TOL = 1e-7
lam_m, mu_m = lame(1.0, 0.3)
lam_i, mu_i = lame(10.0, 0.3)
MATS = (lam_m, mu_m, lam_i, mu_i)
EBAR = (1.0, 0.0, 0.0)


def gen_laminate_rows(rows, nl, orient, Ngrid=N):
    r = rows // nl
    layers = [(np.floor(k * Ngrid / nl) / Ngrid, r / Ngrid) for k in range(nl)]
    return {"type": "laminate", "orient": orient, "layers": layers}


def run(geom, Ngrid, sol, keep=False):
    mask = rasterize(geom, Ngrid)
    lam = np.where(mask, lam_i, lam_m)
    mu = np.where(mask, mu_i, mu_m)
    st, ss, it, err = sol.solve(lam, mu, EBAR, tol=TOL, maxit=4000)
    out = analyze_state(mask, st, ss, MATS, EBAR, m=M, ell=ELL, keep_field=keep)
    out["iters"] = it
    out["eq_err"] = err
    return out, mask


def main():
    rng = np.random.default_rng(20261008)
    sol = FFTElastic2D(N)
    rows, fields, masks, geoms = [], [], [], []
    t0 = time.time()

    # Laminates: exact positive control
    for nrow in (16, 24, 32, 40, 48, 64):
        for orient in ("h", "v"):
            for nl in (1, 2, 4):
                g = gen_laminate_rows(nrow, nl, orient)
                out, mask = run(g, N, sol, keep=True)
                out.update(family="laminate", f_nom=round(nrow / N, 4), n=nl,
                           orient=orient, real=nl,
                           cell=f"lam_{nrow}_{orient}")
                rows.append(out); geoms.append(g)

    # Disks and ellipses: controlled loss of sufficiency
    for fam, ar, ns in (("disks", 1.0, (1, 4, 9, 16)), ("ellipses", 3.0, (4, 9))):
        for f in (0.1, 0.2, 0.3):
            for n in ns:
                for r in range(10):
                    g = gen_inclusions(f, n, rng, ar=ar)
                    out, mask = run(g, N, sol, keep=True)
                    out.update(family=fam, f_nom=f, n=n, orient="-", real=r,
                               cell=f"{fam}_{f}_{n}")
                    rows.append(out); geoms.append(g)
                print(f"{fam} f={f} n={n} done  t={time.time()-t0:.0f}s", flush=True)

    for k, o in enumerate(rows):
        fields.append(o.pop("_sbar"))
    df = pd.DataFrame(rows)
    df.insert(0, "id", np.arange(len(df)))
    df.to_csv("campaign_results.csv", index=False)
    masks = np.array([rasterize(g, N) for g in geoms])
    np.savez_compressed("fields.npz", sbar=np.array(fields), mask=masks)
    print("saved", len(df), "states; max iters", df.iters.max(),
          "max eq_err", df.eq_err.max(), flush=True)

    # Resolution study (same analytic geometry, ell fixed in physical units)
    solr = FFTElastic2D(NREF)
    res = []
    sel = df[(df.family != "laminate") & (df.real < 2)].id.values
    for i in sel:
        o_ref, _ = run(geoms[i], NREF, solr)
        o_ref.pop("_sbar", None)
        rr = {"id": i, "family": df.family[i], "f_nom": df.f_nom[i], "n": df.n[i]}
        for k in ("W_hom", "lnB", "I", "lnVeff", "lnValpha", "e_alpha", "Jgap_m", "Jgap_i",
                  "e11_m", "e22_m"):
            rr[k + "_127"] = df[k][i]
            rr[k + "_255"] = o_ref[k]
        res.append(rr)
        print("res", i, f"dlnB={o_ref['lnB']-df.lnB[i]:+.4f}", flush=True)
    pd.DataFrame(res).to_csv("resolution.csv", index=False)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
