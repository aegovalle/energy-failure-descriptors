import json
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, rasterize, gen_inclusions
import campaign_run as cr

# Regenerate geometries with the same RNG sequence to recover the pair
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

df = pd.read_csv("campaign_results.csv")
S = json.load(open("summary.json"))
a, b = S["counterexample_E1"]["ids"]
out = {}
for Ng in (127, 255, 381):
    sol = FFTElastic2D(Ng)
    vals = []
    for i in (a, b):
        o, _ = cr.run(geoms[i], Ng, sol)
        o.pop("_sbar", None)
        vals.append(o)
        if Ng == 127:
            assert abs(o["lnB"] - df.lnB[i]) < 1e-8, "geometry regeneration mismatch"
    out[Ng] = {k: [v[k] for v in vals] for k in ("lnB", "I", "lnVeff", "W_hom", "e11_m", "Jgap_m")}
    print(Ng, {k: (round(v[0], 4), round(v[1], 4), round(v[0] - v[1], 4)) for k, v in out[Ng].items()}, flush=True)
json.dump({str(k): v for k, v in out.items()}, open("counterexample_resolution.json", "w"), indent=1)
