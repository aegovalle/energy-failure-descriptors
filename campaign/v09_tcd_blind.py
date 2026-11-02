r"""Critical-distance onset intensity J_r: radius scan on the design set, conditional test with the
local-swap sampler, and out-of-sample check on the blind phase-field cells (Set B, Set O), with the
radius chosen on the design set only. Outputs: v09_tcd_scan.csv, v09_tcd_blind.json"""
import json
import numpy as np
import pandas as pd
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE
from descr_tools import surrogate_cv
from v09_stats import zscore, omega_sel

rng = np.random.default_rng(20261107)
N = 127; ELL = 0.025
lam_m, mu_m = lame(1.0, 0.3); lam_i, mu_i = lame(10.0, 0.3)
sol = FFTElastic2D(N)
FAC = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
x = (np.arange(N) + 0.5) / N
X1, X2 = np.meshgrid(x, x, indexing="ij")
d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
KH = {}
for a in FAC[1:]:
    K = ((d1 ** 2 + d2 ** 2) <= (a * ELL) ** 2).astype(float); K /= K.sum(); KH[a] = np.fft.fft2(K)
pf0 = PhaseFieldRVE(np.zeros((N, N), bool))
H0 = 0.5 * np.log((pf0.psi_plus((np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)))) / pf0.psic).max())


def jr(mask):
    lam = np.where(mask, lam_i, lam_m); mu = np.where(mask, mu_i, mu_m)
    (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
    pf = PhaseFieldRVE(mask)
    ratio = pf.psi_plus((e11, e22, e12)) / pf.psic
    out = {"J_0": 0.5 * np.log(ratio.max()) - H0}
    for a, Kh in KH.items():
        out[f"J_{a:g}"] = 0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(ratio) * Kh)).max()) - H0
    return out


MASK = np.load("fields.npz")["mask"]
DJ = pd.DataFrame([dict(id=i, **jr(m)) for i, m in enumerate(MASK)])
X = (pd.read_csv("pf_descriptors.csv").merge(DJ, on="id").sort_values("id").reset_index(drop=True))
JC = [f"J_{a:g}" for a in FAC]
E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
D = X[(X.family != "laminate") & (X.n > 1)].reset_index(drop=True)
rows = []
for c in JC + ["lnB", "J_psi"]:
    o = omega_sel(zscore(D[E1 + [c]].values), D.R_sigma.values, 0.01, 4000, rng)
    s1, _ = surrogate_cv(D[E1 + [c]].values, D.R_sigma.values, D.cell.values, 1, 5, rng)
    s0, _ = surrogate_cv(D[[c]].values, D.R_sigma.values, D.cell.values, 1, 5, rng)
    rows.append(dict(desc=c, corr=float(np.corrcoef(D[c], D.R_sigma)[0, 1]), ratio_E1plus=o["ratio"],
                     R2_E1plus=s1["R2"], R2_alone=s0["R2"]))
S = pd.DataFrame(rows); S.to_csv("v09_tcd_scan.csv", index=False)
best = S[S.desc.isin(JC)].sort_values("R2_E1plus", ascending=False).desc.iloc[0]

# conditional test, local-swap sampler (valid sampler), E1 -> E1 + J_best, n>1
exec(open("v09_r3_power.py", encoding="utf-8").read().split("SETUPS =")[0].split("exec(open(")[0])
src = open("v09_cond_final.py", encoding="utf-8").read().split("rows = []")[0]
ns = {}
exec(src, ns)
exec(open("v09_r3_power.py", encoding="utf-8").read().split("SETUPS =")[0].split('exec(open("v09_cond_final.py"')[1].split("\n", 1)[1], ns)
Dn = ns["D"].merge(DJ, on="id")
assert len(Dn) == len(D)
ns["D"] = Dn
p = ns["run_test"](Dn[E1].values, Dn[[best]].values.astype(float), Dn.R_sigma.values.astype(float), "swap", "std", 2000, rng)
pm = ns["run_test"](Dn[E1].values, Dn[[best]].values.astype(float), Dn.R_sigma.values.astype(float), "swap", "mahal", 2000, rng)
emb = ns["emb_factory"](Dn[E1].values, "std")
cond = {"best": best, "p_swap_std_max": float(p[0]), "p_swap_std_mean": float(p[1]),
        "p_swap_mahal_max": float(pm[0]), "p_swap_mahal_mean": float(pm[1])}

# blind out-of-sample (surrogates trained on the 180 design inclusion cells, as in the frozen protocol)
BM = np.load("blind/blind_masks.npz")["mask"]
BP = pd.read_csv("blind/blind_pf.csv")
BP = BP[BP.separated].reset_index(drop=True)
BJ = pd.DataFrame([dict(bid=int(b), **jr(BM[int(b)])) for b in BP.bid])
BP = BP.merge(BJ, on="bid")
TR = X[X.family != "laminate"].reset_index(drop=True)


def oos(cols, te):
    Ztr_m, Ztr_s = TR[cols].mean(), TR[cols].std()
    A = np.column_stack([np.ones(len(TR)), ((TR[cols] - Ztr_m) / Ztr_s).values])
    c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ TR.R_sigma.values)
    B = np.column_stack([np.ones(len(te)), ((te[cols] - Ztr_m) / Ztr_s).values])
    pred = B @ c
    y = te.R_sigma.values
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


blind = {}
for st in ("A", "O"):
    te = BP[BP.set == st]
    blind[st] = {"n": int(len(te))}
    for nm, cols in (("E1", E1), ("E1+R_H", E1 + ["lnB"]), ("E1+J_psi", E1 + ["J_0"]), (f"E1+{best}", E1 + [best]),
                     (f"{best} alone", [best])):
        blind[st][nm] = oos(cols, te)
    blind[st]["corr_best"] = float(np.corrcoef(te[best], te.R_sigma)[0, 1])
json.dump({"scan": rows, "cond": cond, "blind": blind}, open("v09_tcd_blind.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(S.round(3).to_string(index=False)); print(cond); print(json.dumps(blind, indent=1))
