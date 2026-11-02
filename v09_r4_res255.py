r"""Round 4: resolution sensitivity of the critical-distance onset J_r (same physical radius) on the
20 cells solved at N = 255, compared with the pointwise onset J_0 (= J_psi from the elastic field).
Output: v09_r4_res255.csv, v09_r4_res255.json"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
import pandas as pd
from fft_rve import rasterize
from pf_resolution import geoms_all
from v09_r4_sweep_test import jr_all, FAC

if __name__ == "__main__":
    S = pd.read_csv("v09_resolution_all20.csv")
    geoms = geoms_all()
    rows = []
    for i in S.id:
        a = jr_all(rasterize(geoms[int(i)], 127)); b = jr_all(rasterize(geoms[int(i)], 255))
        r = {"id": int(i), "family": S.set_index("id").family[i]}
        for k in a:
            r[f"{k}_127"], r[f"{k}_255"], r[f"d{k}"] = a[k], b[k], b[k] - a[k]
        rows.append(r); print(i, flush=True)
    T = pd.DataFrame(rows); T.to_csv("v09_r4_res255.csv", index=False)
    inc = T.family != "laminate"
    out = {}
    for a in FAC:
        k = f"J_{a:g}"
        out[k] = {"max_abs_d_all": float(T[f"d{k}"].abs().max()), "mean_abs_d_all": float(T[f"d{k}"].abs().mean()),
                  "max_abs_d_incl": float(T[f"d{k}"][inc].abs().max()),
                  "mean_abs_d_incl": float(T[f"d{k}"][inc].abs().mean()),
                  "spearman_incl": float(T[inc][f"{k}_127"].corr(T[inc][f"{k}_255"], method="spearman"))}
    out["n_incl"] = int(inc.sum())
    json.dump(out, open("v09_r4_res255.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
