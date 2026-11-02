"""Exploratory (post hoc) figure: the pair that sets the GL insufficiency in blind Set A.
Recomputes the elastic fields of the two blind RVEs. Run from campaign/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fft_rve import FFTElastic2D, analyze_state
import campaign_run as cr

B = pd.read_csv("blind/blind_elastic.csv")
M = np.load("blind/blind_masks.npz")["mask"]
sol = FFTElastic2D(127)
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.9))
for a_, i, nm in zip(ax, (37, 40), ("C", "D")):
    m = M[i]
    lam = np.where(m, cr.lam_i, cr.lam_m); mu = np.where(m, cr.mu_i, cr.mu_m)
    st, ss, _, _ = sol.solve(lam, mu, cr.EBAR, tol=cr.TOL, maxit=4000)
    o = analyze_state(m, st, ss, cr.MATS, cr.EBAR, m=cr.M, ell=cr.ELL, keep_field=True)
    s = np.ma.masked_where(m, o["_sbar"])
    im = a_.imshow(s.T, origin="lower", cmap="magma", extent=(0, 1, 0, 1), vmin=0, vmax=float(s.max()))
    a_.contour(np.linspace(0, 1, 127), np.linspace(0, 1, 127), m.T.astype(float), [0.5], colors="c", linewidths=0.6)
    k = np.unravel_index(np.argmax(np.where(m, -1, o["_sbar"])), m.shape)
    a_.plot((k[0] + 0.5) / 127, (k[1] + 0.5) / 127, "wx", ms=6)
    r = B.loc[i]
    a_.set_title(f"{nm}: $A_\\ell$={r.A_lig:.2f}, $R_H$={r.lnB:.1f}", fontsize=8.5)
    a_.set_xticks([]); a_.set_yticks([])
    plt.colorbar(im, ax=a_, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(r"..\figures\fig_blind_pair.pdf", bbox_inches="tight")
fig.savefig(r"..\figures\fig_blind_pair.png", dpi=200, bbox_inches="tight")
print("ok")
