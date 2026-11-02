r"""
Robustness analysis of the AT1 sensitivity variants (pf_var/<variant>/).
  orient    : full ladder on all 216 RVEs loaded along e2 (transposed masks)
  contrast3 : E_i/E_m = 3,  ell040 : ell = 0.04 L   (61-RVE subset, q = 5 %)
  dE0005, tol3e4, strip095, strip099 : numerical variants on 16 RVEs
Outputs: v06_variants.csv (ladders), v06_variants_num.csv, v06_variants.json
"""
import json, os
import numpy as np
import pandas as pd
from pf_fft import PhaseFieldRVE
from descr_tools import min_path, ligament, ligament_amp, omega, surrogate_cv

rng = np.random.default_rng(11)
THR = 0.9
df = pd.read_csv("campaign_results.csv")
MASK = np.load("fields.npz")["mask"]
N = MASK.shape[1]
S = json.load(open("pf_var_subsets.json"))
BASE = pd.read_csv("v06_desc.csv")[["id", "lig_min", "A_lig"]].merge(
    pd.read_csv("pf_descriptors.csv"), on="id").rename(columns={"R_Gpath": "R_path"})
META = df[["id", "family", "n", "cell", "f_nom", "orient"]]
MODEL = {"orient": {}, "contrast3": {"Ei": 3.0}, "ell040": {"ell": 0.040}, "dE0005": {},
         "tol3e4": {}, "strip095": {"strip_fac": 0.95}, "strip099": {"strip_fac": 0.99},
         "var_iso": {"split": False}, "miehe": {"split": "miehe"}}
KCON = {"contrast3": 3.0}


def load_variant(var):
    d = f"pf_var/{var}"
    hom = json.load(open(f"{d}/hom.json"))
    Gh = PhaseFieldRVE(np.zeros((N, N), bool), **MODEL[var]).Gc
    Cp_hom = min_path(Gh, np.load(f"{d}/hom_d.npy") >= THR)
    Pm_hom = min_path(Gh, np.ones((N, N), bool))
    rows = []
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("rve_") and fn.endswith(".json")):
            continue
        i = int(fn[4:7])
        o = json.load(open(f"{d}/{fn}"))
        m = MASK[i].T.copy() if var == "orient" else MASK[i]
        Gc = PhaseFieldRVE(m, **MODEL[var]).Gc
        cp = min_path(Gc, np.load(f"{d}/rve_{i:03d}_d.npy") >= THR)
        pm = min_path(Gc, np.ones((N, N), bool))
        lg = ligament(m)[0]
        al = ligament_amp(m, k=KCON.get(var, 10.0))[0]
        rows.append(dict(id=i, R_sigma=np.log(o["sigma_peak"] / hom["sigma_peak"]),
                         R_path=np.log(cp / Cp_hom), J_psi=-np.log(o["e0"] / hom["e0"]),
                         lnPmin=np.log(pm / Pm_hom), lig_min=lg, A_lig=al,
                         separated=o["separated"], eps_ratio=o["eps_peak"] / o["e0"]))
    R = pd.DataFrame(rows)
    el = f"{d}/elastic.csv"
    if os.path.exists(el):
        E = pd.read_csv(el)
    else:                                   # numerical variants: elastic state unchanged
        E = df.drop(columns=["family", "f_nom", "n", "orient", "real", "cell"])
    X = R.merge(E, on="id").merge(META, on="id").sort_values("id").reset_index(drop=True)
    return X, hom


E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
LEVELS = {"control": ["e11_g", "e22_g", "e12_g"], "G0": ["f"], "G1": ["f", "rhoI"],
          "GL": ["f", "rhoI", "A_lig"], "G2": ["f", "rhoI", "lnPmin"],
          "E0": ["f", "e11_m", "e22_m", "e12_m"], "E1": E1, "E1+L": E1 + ["A_lig"],
          "T1": E1 + ["I"], "Tpsi": E1 + ["J_psi"], "H": ["lnB"], "E1+H": E1 + ["lnB"]}
TG = {"R_H": "lnB", "R_sigma": "R_sigma", "R_path": "R_path"}


def ladder(X, tag, q):
    out = []
    for scope in ("all", "inc"):
        D = X if scope == "all" else X[X.family != "laminate"].reset_index(drop=True)
        for tn, col in TG.items():
            Dk = D[np.isfinite(D[col].values.astype(float))].reset_index(drop=True)
            R = Dk[col].values.astype(float)
            for lv, cols in LEVELS.items():
                if tn == "R_H" and lv in ("T1", "H", "E1+H"):
                    continue
                o = omega(Dk[cols].values, R, q, 300, rng)
                s, _ = surrogate_cv(Dk[cols].values, R, None, 1, 5, rng)
                sl, _ = surrogate_cv(Dk[cols].values, R, Dk["cell"].values, 1, 5, rng)
                out.append(dict(case=tag, scope=scope, q=q, n=len(Dk), target=tn, level=lv,
                                ratio=o["ratio"], R2_kfold=s["R2"], R2_loco=sl["R2"]))
    return out


summary, rows = {}, []
# ---------------- orientation: full ladder, all 216
Xo, ho = load_variant("orient")
summary["orient"] = {"n": len(Xo), "separated": int(Xo.separated.sum()), "hom": ho,
                     "spearman_vs_base": {}, "Pmin_bound_violations": int((Xo.R_path < Xo.lnPmin - 0.01).sum())}
B = BASE.set_index("id").loc[Xo.id]
for t in ("R_sigma", "R_path"):
    summary["orient"]["spearman_vs_base"][t] = float(pd.Series(Xo[t].values).corr(pd.Series(B[t].values), method="spearman"))
rows += ladder(Xo, "orient", 0.01)
rows += ladder(BASE, "base", 0.01)

# ---------------- contrast / ell on s61, q = 5 %
Bs = BASE[BASE.id.isin(S["s61"])].reset_index(drop=True)
rows += ladder(Bs, "base_s61", 0.05)
for var in ("contrast3", "ell040"):
    Xv, hv = load_variant(var)
    rows += ladder(Xv, var, 0.05)
    inc = Xv[Xv.family != "laminate"]
    bb = BASE.set_index("id").loc[Xv.id]
    summary[var] = {"n": len(Xv), "separated": int(Xv.separated.sum()), "hom": hv,
                    "spearman_vs_base": {t: float(pd.Series(Xv[t].values).corr(pd.Series(bb[t].values), method="spearman"))
                                         for t in ("R_sigma", "R_path")},
                    "Pmin_bound_violations": int((Xv.R_path < Xv.lnPmin - 0.01).sum()),
                    "eps_ratio_inc_mean": float(inc.eps_ratio.mean()),
                    "std_inc": {t: float(inc[t].std()) for t in ("R_sigma", "R_path", "lnB")},
                    "corr_inc_A_lig": {t: float(np.corrcoef(inc.A_lig, inc[TG[t]])[0, 1]) for t in TG}}
    if {49, 73} <= set(Xv.id):
        a, b = Xv.set_index("id").loc[49], Xv.set_index("id").loc[73]
        summary[var]["pair_49_73_dRsigma"] = float(a.R_sigma - b.R_sigma)

# ---------------- numerical variants on s16
num = []
for var in ("dE0005", "tol3e4", "strip095", "strip099", "var_iso", "miehe"):
    Xv, hv = load_variant(var)
    bb = BASE.set_index("id").loc[Xv.id]
    dS = Xv.R_sigma.values - bb.R_sigma.values
    dP = Xv.R_path.values - bb.R_path.values
    lam = (Xv.family == "laminate").values
    num.append(dict(variant=var, n=len(Xv), sigma_hom=hv["sigma_peak"],
                    max_dRsigma=float(np.abs(dS).max()), mean_dRsigma=float(np.abs(dS).mean()),
                    max_dRpath=float(np.abs(dP).max()), mean_dRpath=float(np.abs(dP).mean()),
                    max_dRpath_inc=float(np.abs(dP[~lam]).max()),
                    pair_49_73_dRsigma=float(Xv.set_index("id").loc[49].R_sigma - Xv.set_index("id").loc[73].R_sigma),
                    pair_54_59_dRsigma=float(Xv.set_index("id").loc[54].R_sigma - Xv.set_index("id").loc[59].R_sigma),
                    spearman_Rsigma=float(pd.Series(Xv.R_sigma.values).corr(pd.Series(bb.R_sigma.values), method="spearman")),
                    spearman_Rpath=float(pd.Series(Xv.R_path.values).corr(pd.Series(bb.R_path.values), method="spearman"))))
pd.DataFrame(num).to_csv("v06_variants_num.csv", index=False)
L = pd.DataFrame(rows)
L.to_csv("v06_variants.csv", index=False)
json.dump(summary, open("v06_variants.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
P = L.pivot_table(index=["scope", "target", "level"], columns="case", values="ratio").round(3)
print(P.to_string())
print(L.pivot_table(index=["scope", "target", "level"], columns="case", values="R2_loco").round(2).to_string())
print(pd.DataFrame(num).round(4).to_string(index=False))
print(json.dumps(summary, indent=1, default=float))
