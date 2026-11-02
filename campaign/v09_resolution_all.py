"""Complete the N=255 resolution summary with the three RVEs (2, 30, 160) that were
solved at N=255 but missing from pf_resolution_summary.csv. Run from campaign/."""
import json
import numpy as np
import pandas as pd
from fft_rve import rasterize
from pf_fft import PhaseFieldRVE
from descr_tools import min_path
from pf_resolution import geoms_all

NG = 255
base = pd.read_csv("pf_descriptors.csv").set_index("id")
geoms = geoms_all()
R = pd.concat([pd.read_csv(f) for f in ("pf_resolution_255.csv", "pf_resolution_255_b.csv")])
hom = R[R.key == "hom"].iloc[0]
R = R[R.key != "hom"].drop_duplicates("key")
cp_hom = min_path(PhaseFieldRVE(np.zeros((NG, NG), bool)).Gc, np.load("pf_res/hom_d.npy") >= 0.9)
rows = []
for _, r in R.iterrows():
    i = int(r.key[4:])
    m = rasterize(geoms[i], NG)
    cp = min_path(PhaseFieldRVE(m).Gc, np.load(f"pf_res/{r.key}_d.npy") >= 0.9)
    rows.append(dict(id=i, family=base.family[i], f=base.f[i], n=base.n[i],
                     R_sigma_127=base.R_sigma[i], R_sigma_255=np.log(r.sigma_peak / hom.sigma_peak),
                     R_path_127=base.R_Gpath[i], R_path_255=np.log(cp / cp_hom)))
S = pd.DataFrame(rows).sort_values("id")
S["dR_sigma"] = S.R_sigma_255 - S.R_sigma_127
S["dR_path"] = S.R_path_255 - S.R_path_127
S.to_csv("v09_resolution_all20.csv", index=False)
inc = S.family != "laminate"
out = {"n": len(S), "by_family": S.family.value_counts().to_dict(),
       "max_abs_dR_sigma": float(S.dR_sigma.abs().max()), "mean_abs_dR_sigma": float(S.dR_sigma.abs().mean()),
       "max_abs_dR_path": float(S.dR_path.abs().max()),
       "n_dR_path_le_0.01": int((S.dR_path.abs() <= 0.01).sum()),
       "rms_dR_path_incl": float(np.sqrt(np.mean(S.dR_path[inc] ** 2))),
       "spearman_Rs_incl": float(S[inc].R_sigma_127.corr(S[inc].R_sigma_255, method="spearman")),
       "spearman_Rp_incl": float(S[inc].R_path_127.corr(S[inc].R_path_255, method="spearman"))}
json.dump(out, open("v09_resolution_all20.json", "w"), indent=1)
print(S.round(4).to_string(index=False)); print(out)
