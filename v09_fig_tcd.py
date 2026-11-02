"""Figure: critical-distance onset intensity (radius scan, scaling with ell, blind prediction)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fft_rve import FFTElastic2D, lame
from pf_fft import PhaseFieldRVE

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8.5})
N = 127; ELL = 0.025; RB = 1.5 * ELL
S = pd.read_csv("v09_tcd_scan.csv")
E = pd.read_csv("v09_tcd_ell.csv")
fac = {"J_0": 0, "J_0.5": 0.5, "J_1": 1, "J_1.5": 1.5, "J_2": 2, "J_2.5": 2.5, "J_3": 3, "J_4": 4}
S = S[S.desc.isin(fac)].copy(); S["a"] = S.desc.map(fac)

lm, mm = lame(1.0, 0.3); li, mi = lame(10.0, 0.3)
sol = FFTElastic2D(N)
x = (np.arange(N) + 0.5) / N
X1, X2 = np.meshgrid(x, x, indexing="ij")
d1 = (X1 - 0.5 / N + 0.5) % 1.0 - 0.5; d2 = (X2 - 0.5 / N + 0.5) % 1.0 - 0.5
K = ((d1 ** 2 + d2 ** 2) <= RB ** 2).astype(float); K /= K.sum(); KH = np.fft.fft2(K)
pf0 = PhaseFieldRVE(np.zeros((N, N), bool))
H0 = 0.5 * np.log((pf0.psi_plus((np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)))) / pf0.psic).max())


def j15(mask):
    lam = np.where(mask, li, lm); mu = np.where(mask, mi, mm)
    (e11, e22, e12), _, _, _ = sol.solve(lam, mu, (1.0, 0.0, 0.0), tol=1e-7, maxit=4000)
    pf = PhaseFieldRVE(mask)
    r = pf.psi_plus((e11, e22, e12)) / pf.psic
    return 0.5 * np.log(np.real(np.fft.ifft2(np.fft.fft2(r) * KH)).max()) - H0


E1 = ["f", "e11_m", "e22_m", "e12_m", "Jgap_m", "Jgap_i"]
X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True)
M = np.load("fields.npz")["mask"]
TR = X[X.family != "laminate"].reset_index(drop=True)
TR["J15"] = [j15(M[i]) for i in TR.id]
BM = np.load("blind/blind_masks.npz")["mask"]
BP = pd.read_csv("blind/blind_pf.csv"); BP = BP[BP.separated].reset_index(drop=True)
BP["J15"] = [j15(BM[int(b)]) for b in BP.bid]
cols = E1 + ["J15"]
mu_, sd_ = TR[cols].mean(), TR[cols].std()
A = np.column_stack([np.ones(len(TR)), ((TR[cols] - mu_) / sd_).values])
c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ TR.R_sigma.values)
BP["pred"] = np.column_stack([np.ones(len(BP)), ((BP[cols] - mu_) / sd_).values]) @ c

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ax[0].plot(S.a, S.R2_E1plus, "o-", color="C3", label=r"$E1+J_r$")
ax[0].plot(S.a, S.R2_alone, "s--", color="C0", label=r"$J_r$ alone")
ax[0].axhline(0.50, color="C2", ls=":", label=r"$E1+R_H$")
ax[0].set_xlabel(r"averaging radius $r/\ell$"); ax[0].set_ylabel(r"$R^2_{\rm LOCO}$ for $R_\sigma$")
ax[0].set_title("design set, 150 cells with n>1", fontsize=10); ax[0].legend()
ellc = {"baseline": 0.025, "ell040": 0.040, "contrast3": 0.025}
labc = {"baseline": r"$\ell=0.025L$, $\kappa=10$", "ell040": r"$\ell=0.040L$, $\kappa=10$", "contrast3": r"$\ell=0.025L$, $\kappa=3$"}
for k, (case, g) in enumerate(E.groupby("case")):
    ax[1].plot(g.r / ellc[case], g.R2_loco_f_J, "o-", color=f"C{k}", label=labc[case])
ax[1].set_xlabel(r"averaging radius $r/\ell$"); ax[1].set_ylabel(r"$R^2_{\rm LOCO}$ ($f+J_r$)")
ax[1].set_title("model variants, 49 inclusion cells", fontsize=10); ax[1].legend()
for st, mk, col, lab in (("A", "o", "C3", "blind Set B"), ("O", "^", "C0", "blind Set O (aspect 6)")):
    g = BP[BP.set == st]
    ax[2].scatter(g.R_sigma, g.pred, marker=mk, color=col, s=18, label=lab)
lim = [BP[["R_sigma", "pred"]].min().min() - 0.02, BP[["R_sigma", "pred"]].max().max() + 0.02]
ax[2].plot(lim, lim, "k:", lw=1)
ax[2].set_xlabel(r"$R_\sigma$ (phase field)"); ax[2].set_ylabel(r"predicted from $E1+J_{1.5\ell}$")
ax[2].set_title("blind cells (radius fixed on design set)", fontsize=10); ax[2].legend()
fig.tight_layout()
fig.savefig("../figures/fig_tcd.pdf"); fig.savefig("../figures/fig_tcd.png", dpi=200)
print("ok")
