import numpy as np, pandas as pd
import sys
sys.argv = ["x"]
exec(open("v09_power_calib.py", encoding="utf-8").read().split("# (a) calibration")[0])
cal = []
for sc in ("all", "n>1"):
    D = X[SC[sc]].reset_index(drop=True)
    for tg in ("R_H", "R_sigma"):
        R = D[TG[tg]].values.astype(float); Xk = D[E0].values; Ea = D[SPL].values.astype(float)
        fit, res = fl_model(zscore(Xk), Ea)
        for metric in ("std", "mahal"):
            ps = np.array([fl_test(Xk, fit + res[rng.permutation(len(R))], R, 200, metric)[1] for _ in range(60)])
            cal.append(dict(scope=sc, target=tg, metric=metric, rej05_max=float(np.mean(ps[:, 0] <= 0.05)),
                            rej05_mean=float(np.mean(ps[:, 1] <= 0.05))))
            print(cal[-1], flush=True)
pd.DataFrame(cal).to_csv("v09_power_calib_split.csv", index=False)
