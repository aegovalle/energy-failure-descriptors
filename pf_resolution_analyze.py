"""Summary of the phase-field resolution check (N=127 vs N=255) for both resolution sets.
Run from campaign/ after pf_resolution.py.  Output: pf_resolution_summary.csv/.json"""
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
R = pd.concat([pd.read_csv(f) for f in ("pf_resolution_255.csv", "pf_resolution_255_b.csv")], ignore_index=True)
hom = R[R.key == "hom"].iloc[0]
cp_hom = min_path(PhaseFieldRVE(np.zeros((NG, NG), bool)).Gc, np.load("pf_res/hom_d.npy") >= 0.9)
rows = []
for _, r in R[R.key != "hom"].drop_duplicates("key").iterrows():
    i = int(r.key[4:])
    m = rasterize(geoms[i], NG)
    cp = min_path(PhaseFieldRVE(m).Gc, np.load(f"pf_res/{r.key}_d.npy") >= 0.9)
    rows.append(dict(id=i, family=base.family[i], f=base.f[i], n=base.n[i],
                     R_sigma_127=base.R_sigma[i], R_sigma_255=np.log(r.sigma_peak / hom.sigma_peak),
                     R_path_127=base.R_Gpath[i], R_path_255=np.log(cp / cp_hom),
                     eps_ratio_255=r.eps_peak / r.e0, time_s=r.time_s))
S = pd.DataFrame(rows).sort_values("id")
S["dR_sigma"] = S.R_sigma_255 - S.R_sigma_127
S["dR_path"] = S.R_path_255 - S.R_path_127
S.to_csv("pf_resolution_summary.csv", index=False)
inc = S.family != "laminate"
summ = {"n": int(len(S)), "n_by_family": S.family.value_counts().to_dict(), "C_path_hom_255": float(cp_hom),
        "max_abs_dR_sigma": float(S.dR_sigma.abs().max()), "mean_abs_dR_sigma": float(S.dR_sigma.abs().mean()),
        "max_abs_dR_path": float(S.dR_path.abs().max()), "mean_abs_dR_path": float(S.dR_path.abs().mean()),
        "max_abs_dR_sigma_ellipses": float(S.dR_sigma[S.family == "ellipses"].abs().max()),
        "max_abs_dR_path_ellipses": float(S.dR_path[S.family == "ellipses"].abs().max()),
        "pair_49_73": {"dRs_127": float(base.R_sigma[49] - base.R_sigma[73]),
                       "dRs_255": float(S.set_index("id").R_sigma_255.get(49, np.nan) - S.set_index("id").R_sigma_255.get(73, np.nan))},
        "spearman_R_sigma_inc": float(S[inc].R_sigma_127.corr(S[inc].R_sigma_255, method="spearman")),
        "cpu_h_255": float(S.time_s.sum() / 3600)}
json.dump(summ, open("pf_resolution_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(S.round(4).to_string(index=False))
print(json.dumps(summ, indent=1, default=float))
