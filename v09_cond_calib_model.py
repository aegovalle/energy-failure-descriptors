import numpy as np, pandas as pd
src = open("v09_cond_final.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)
cal = []
for code, lab, tg, base, add, prim in TESTS[:5]:
    R = D[tg].values.astype(float); Xk = D[base].values; eta = D[add].values.astype(float)
    fit, res = fl_parts(Xk, eta)
    for metric in ("std", "mahal"):
        ps = np.array([test(Xk, fit + res[strat_perm(STRATA)], R, metric, 200)[1] for _ in range(60)])
        cal.append(dict(code=code, metric=metric, rej05_max=float(np.mean(ps[:, 0] <= 0.05)),
                        rej05_mean=float(np.mean(ps[:, 1] <= 0.05))))
        print(cal[-1], flush=True)
pd.DataFrame(cal).to_csv("v09_cond_calib_model.csv", index=False)
