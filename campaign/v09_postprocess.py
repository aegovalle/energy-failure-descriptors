r"""v0.9 post-processing of stored phase-field outputs (no new solves).

B1  Regularized dissipation at the separation step: crack core vs diffuse background
    (homogeneous cell, N = 127 and 255; all 216 RVEs at N = 127).
B2  Onset intensity J_psi and eps_peak/e0 at N = 127 vs 255 for the 17 re-solved RVEs.
B3  Homogeneous cell and series laminates: sigma_onset, sigma_peak, nucleation column,
    phase of the weakened strip.
B4  Strength decomposition  R_sigma = ln R_E - J_psi + ln(Gamma_s / Gamma_s^hom),
    Gamma_s = sigma_peak / sigma_onset, sigma_onset = M0 * e0 (undamaged secant).
Outputs: v09_post.json, v09_strength_decomp.csv
"""
import json
import numpy as np
import pandas as pd
from pf_fft import PhaseFieldRVE

out = {}
ELL = 0.025


def split_dissipation(d, mask=None):
    N = d.shape[0]
    pf = PhaseFieldRVE(np.zeros((N, N), bool) if mask is None else mask)
    h = 1.0 / N
    gx = (np.roll(d, -1, 0) - d) * N
    gy = (np.roll(d, -1, 1) - d) * N
    dens = 0.375 * (pf.Gc * d / ELL + ELL * (pf.Gx * gx ** 2 + pf.Gy * gy ** 2))
    col = d.mean(1)                                   # crack is a band x = const
    xc = int(np.argmax(col))
    x = np.arange(N)
    dist = np.abs((x - xc + N // 2) % N - N // 2) * h
    core = (dist <= 2 * ELL + 1.5 * h)[:, None] * np.ones((1, N), bool)
    tot = float(dens.sum() * h * h)
    cor = float(dens[core].sum() * h * h)
    bg = d[~core]
    return dict(total=tot, core=cor, background=tot - cor, x_crack=float((xc + 0.5) * h),
                d_bg_mean=float(bg.mean()), d_bg_max=float(bg.max()),
                frac_area_d_gt_0=float(np.mean(d > 1e-6)))


# ---------------------------------------------------------------- B1
d127 = np.load("pf_out/hom_d.npy").astype(float)
d255 = np.load("pf_res/hom_d.npy").astype(float)
out["B1_hom"] = {"N127": split_dissipation(d127), "N255": split_dissipation(d255),
                 "Gc_L_strip": 0.98}
# 1D AT1 profile check: integral of the optimal profile gives Gc exactly
D = np.load("pf_fields.npz")["d"]
MASK = np.load("fields.npz")["mask"]
X = pd.read_csv("pf_descriptors.csv").sort_values("id").reset_index(drop=True)
rows = []
for i in range(len(X)):
    s = split_dissipation(D[i].astype(float), MASK[i])
    s["id"] = i
    rows.append(s)
B1 = pd.DataFrame(rows)
inc = (X.family != "laminate").values
out["B1_all"] = {"bg_share_mean_incl": float((B1.background / B1.total)[inc].mean()),
                 "bg_share_hom": out["B1_hom"]["N127"]["background"] / out["B1_hom"]["N127"]["total"]}

# ---------------------------------------------------------------- B2
R = pd.concat([pd.read_csv(f) for f in ("pf_resolution_255.csv", "pf_resolution_255_b.csv")])
hom255 = R[R.key == "hom"].iloc[0]
R = R[R.key != "hom"].drop_duplicates("key")
e0_hom127 = json.load(open("pf_hom.json"))["e0"]
b2 = []
for _, r in R.iterrows():
    i = int(r.key[4:])
    b2.append(dict(id=i, family=X.family[i],
                   J_psi_127=float(-np.log(X.e0[i] / e0_hom127)),
                   J_psi_255=float(-np.log(r.e0 / hom255.e0)),
                   epsr_127=float(X.eps_peak[i] / X.e0[i]),
                   epsr_255=float(r.eps_peak / r.e0)))
B2 = pd.DataFrame(b2).sort_values("id")
B2["dJ"] = B2.J_psi_255 - B2.J_psi_127
B2["d_epsr"] = B2.epsr_255 - B2.epsr_127
pair = B2.set_index("id")
out["B2"] = {"rows": B2.round(4).to_dict("records"),
             "max_abs_dJ": float(B2.dJ.abs().max()), "mean_abs_dJ": float(B2.dJ.abs().mean()),
             "max_abs_d_epsr": float(B2.d_epsr.abs().max()),
             "pair_49_73_dJ": {"N127": float(pair.J_psi_127[49] - pair.J_psi_127[73]),
                               "N255": float(pair.J_psi_255[49] - pair.J_psi_255[73])},
             "pair_54_59_dJ": {"N127": float(pair.J_psi_127[54] - pair.J_psi_127[59]),
                               "N255": float(pair.J_psi_255[54] - pair.J_psi_255[59])}}

# ---------------------------------------------------------------- B3
hom = json.load(open("pf_hom.json"))
N = 127
xs = (np.arange(N) + 0.5) / N
strip_cols = np.flatnonzero(np.abs((xs - 0.95 + 0.5) % 1.0 - 0.5) < 0.5 * 2 / 127 + 1e-12)
b3 = [dict(id="hom", sigma_onset=hom["M0"] * hom["e0"], sigma_peak=hom["sigma_peak"],
           x_nuc=out["B1_hom"]["N127"]["x_crack"], strip_phase="matrix")]
for i in range(36):
    if X.orient[i] != "v":            # 'v': lamination normal e1 (layers in series)
        continue
    m = MASK[i]
    ph = "inclusion" if m[strip_cols].all() else ("matrix" if not m[strip_cols].any() else "mixed")
    s = split_dissipation(D[i].astype(float), m)
    b3.append(dict(id=i, sigma_onset=float(X.M0[i] * X.e0[i]), sigma_peak=float(X.sigma_peak[i]),
                   R_sigma=float(X.R_sigma[i]), x_nuc=s["x_crack"], strip_phase=ph,
                   eps_peak_over_e0=float(X.eps_peak[i] / X.e0[i])))
out["B3"] = b3

# ---------------------------------------------------------------- B4
C = pd.read_csv("campaign_results.csv").sort_values("id").reset_index(drop=True)
sig_on = X.M0 * X.e0
sig_on_h = hom["M0"] * hom["e0"]
Gs = X.sigma_peak / sig_on
Gs_h = hom["sigma_peak"] / sig_on_h
dec = pd.DataFrame({"id": X.id, "family": X.family, "n": X.n, "R_sigma": X.R_sigma,
                    "lnRE": np.log(C.W_hom), "lnM0": np.log(X.M0 / hom["M0"]),
                    "J_psi": X.J_psi, "lnGamma": np.log(Gs / Gs_h)})
dec["onset_part"] = dec.lnM0 - dec.J_psi
dec["resid"] = dec.R_sigma - dec.onset_part - dec.lnGamma
dec.to_csv("v09_strength_decomp.csv", index=False)
inc = dec.family != "laminate"
out["B4"] = {"max_abs_resid": float(dec.resid.abs().max()),
             "max_abs_lnRE_minus_lnM0": float((dec.lnRE - dec.lnM0).abs().max()),
             "incl_var_R_sigma": float(dec.R_sigma[inc].var()),
             "incl_var_onset": float(dec.onset_part[inc].var()),
             "incl_var_lnGamma": float(dec.lnGamma[inc].var()),
             "incl_cov_onset_lnGamma": float(np.cov(dec.onset_part[inc], dec.lnGamma[inc])[0, 1]),
             "incl_corr_onset_lnGamma": float(np.corrcoef(dec.onset_part[inc], dec.lnGamma[inc])[0, 1]),
             "incl_lnGamma_mean": float(dec.lnGamma[inc].mean()),
             "incl_lnGamma_range": [float(dec.lnGamma[inc].min()), float(dec.lnGamma[inc].max())],
             "Gamma_s_hom": float(Gs_h)}
json.dump(out, open("v09_post.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "B2"}, indent=1, default=str))
print(B2.round(4).to_string(index=False))
