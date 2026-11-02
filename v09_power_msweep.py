import numpy as np, pandas as pd
exec(open("v09_power_calib.py", encoding="utf-8").read().split("# (a) calibration")[0])
F = np.load("fields.npz"); S, M = F["sbar"], F["mask"]; N = M.shape[1]
for m in (2, 5, 10, 40):
    X[f"RH{m}"] = [m * np.log(S[i][~M[i]].max()) + np.log(np.sum((S[i][~M[i]] / S[i][~M[i]].max()) ** m) / N ** 2) for i in range(len(X))]
X["RH20"] = X.lnB
rows = []
for sc in ("all", "n>1"):
    D = X[SC[sc]].reset_index(drop=True)
    for m in (2, 5, 10, 20, 40):
        R = D[f"RH{m}"].values
        for metric in ("std", "mahal"):
            rho, p, _, _ = fl_test(D[E0].values, D[["Jgap_m", "Jgap_i"]].values, R, 2000, metric)
            rows.append(dict(scope=sc, m=m, metric=metric, rho_max=rho[0], p_max=p[0], rho_mean=rho[1], p_mean=p[1]))
df = pd.DataFrame(rows); df.to_csv("v09_power_msweep.csv", index=False); print(df.round(3).to_string(index=False))
