"""Redraw fig_omega_levels (elastic ladder, energy vs hazard) from omega_table.csv
with larger labels (v0.7). Values are unchanged."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10})
FIG = r"..\figures"
LV = ["C_glob", "G0", "G1", "E0", "E1", "E1+G", "T1", "T2"]
LABEL = {"C_glob": "global means\n(control)", "G0": "$f$", "G1": "$f,\\rho_I$",
         "E0": "phase means\n(E0)", "E1": "+Jensen gaps\n(E1)", "E1+G": "E1$+\\rho_I$",
         "T1": "E1$+I$", "T2": "E1$+I+\\ln V_\\alpha$"}
om = pd.read_csv("omega_table.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
for k, (tn, title) in enumerate((("R_E", r"$R_E=W_{\rm hom}$ (energy)"),
                                 ("R_H", r"$R_H=\ln B_\ell$ (hazard, $\beta=20$)"))):
    s = om[(om.target == tn) & (om.q == 0.01)].set_index("level").loc[LV]
    x = np.arange(len(LV))
    ax[k].bar(x - 0.2, s.ratio, 0.4, color="#10324F", label=r"$\Omega(q)/\Omega^{\rm null}$")
    ax[k].bar(x + 0.2, s.ratio95, 0.4, color="#A63D40", label="95th-pct version")
    ax[k].set_xticks(x); ax[k].set_xticklabels([LABEL[l] for l in LV], rotation=60, ha="right", fontsize=9)
    ax[k].tick_params(axis="y", labelsize=9)
    ax[k].axhline(1, color="gray", ls=":", lw=1)
    ax[k].set_title(title, fontsize=10); ax[k].set_yscale("log")
ax[0].set_ylabel(r"normalized insufficiency at $q=1\%$", fontsize=10)
ax[0].legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_omega_levels.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig_omega_levels.png", dpi=200, bbox_inches="tight")
print("ok")
