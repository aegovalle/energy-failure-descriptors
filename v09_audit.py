"""V6 audit: regenerate the numeric cells of Tables 2 and 3 of Draft v0.9 from the v0.9 outputs
and compare them with the LaTeX source."""
import re
import pandas as pd
from v09_stats import holm

L = pd.read_csv("v09_levels.csv"); C = pd.read_csv("v09_ci_n1.csv")
tex = open("../Draft_v09_Energy_Failure_Descriptors.tex", encoding="utf-8").read()


def fmt_p(p):
    return r"\(\le10^{-3}\)" if p <= 1e-3 + 1e-12 else (f"{p:.2f}" if p >= 0.01 else f"{p:.3f}")


def row(target, levels):
    out = {}
    pn = [L[(L.scope == "n>1") & (L.target == target) & (L.level == lv)].p_perm.values[0] for lv in levels]
    ph = dict(zip(levels, holm(pn)))
    for lv in levels:
        a = L[(L.scope == "all") & (L.target == target) & (L.level == lv)].iloc[0]
        n = L[(L.scope == "n>1") & (L.target == target) & (L.level == lv)].iloc[0]
        c = C[(C.target == target) & (C.level == lv)]
        ci = f" [{c.ci_lo.values[0]:.2f}, {c.ci_hi.values[0]:.2f}]" if len(c) else ""
        out[lv] = (f"{a.ratio:.3f}", f"{n.ratio:.2f}{ci}", fmt_p(ph[lv]), f"{n.R2_loco:.2f}")
    return out


bad = 0
for target, levels in (("R_sigma", ["G0", "G1", "GL", "G2", "Gpsi", "E0", "E1", "E0+cov", "E0+split", "MKS10", "T1", "Tpsi", "E1+H", "E1+J15"]),
                       ("R_path", ["G0", "G1", "GL", "G2", "Gpsi", "E0", "E1", "E0+cov", "E0+split", "MKS10", "T1", "Tpsi", "E1+H", "E1+J15"]),
                       ("R_H", ["G0", "G1", "E0", "E1", "E0+cov", "E0+split", "MKS10"])):
    for lv, cells in row(target, levels).items():
        for cval in cells:
            if cval not in tex:
                bad += 1
                print("MISSING", target, lv, cval)
print("audit done; missing cells:", bad)
