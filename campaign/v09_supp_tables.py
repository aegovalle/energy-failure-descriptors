"""Write LaTeX tables for Supplementary Section S4 of Draft v0.9 from the v0.9 outputs."""
import math
import pandas as pd

T = pd.read_csv("v09_cond_final.csv")
I = pd.read_csv("v09_incremental.csv")
CM = pd.read_csv("v09_cond_calib_model.csv")
L = pd.read_csv("v09_levels.csv")
U = pd.read_csv("v09_dupcoord.csv")
MH = pd.read_csv("v09_mahal_levels.csv")
DC = pd.read_csv("v09_dissipation_check.csv")
KR = pd.read_csv("v09_kres_check.csv")
out = []

swapkey = {"P1": ("R_H", "E0->E1"), "P2": ("R_H", "E0->E0+cov"), "P3": ("R_sigma", "E0->E1"),
           "P4": ("R_sigma", "E1->E1+H"), "P5": ("R_H", "G1->GL"), "X1": ("W_m (control)", "E0->E1"),
           "X2": ("R_H", "E0->E0+split"), "X3": ("R_sigma", "E0->E0+split"), "X4": ("R_sigma", "E0->E0+cov"),
           "X5": ("R_path", "E0->E1"), "X6": ("R_path", "G1->G2"), "X7": ("R_sigma", "E1->T1"),
           "M2": ("R_H m=2", "E0->E1"), "M5": ("R_H m=5", "E0->E1"), "M10": ("R_H m=10", "E0->E1"),
           "M40": ("R_H m=40", "E0->E1")}
out.append(r"\begin{table}[ht]\centering\scriptsize\setlength{\tabcolsep}{2.5pt}")
out.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{llccccccc}\toprule")
out.append(r" & & \multicolumn{3}{c}{resampling sampler (sensitivity only), standardized} & \multicolumn{2}{c}{resampling, Mahalanobis} & \multicolumn{2}{c}{local swap (inference)}\\")
out.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
out.append(r"Code & Step & $\rho_{\rm c}$ ($p_{\rm c}$) & Holm & mean: $\rho_{\rm c}$ ($p_{\rm c}$) & $\rho_{\rm c}$ ($p_{\rm c}$) & mean: $\rho_{\rm c}$ ($p_{\rm c}$) & $\rho_{\rm c}$ & $p_{\rm c}$\\\midrule")
for code in T.code.unique():
    a = T[(T.code == code) & (T.metric == "std")].iloc[0]
    b = T[(T.code == code) & (T.metric == "mahal")].iloc[0]
    tg, st = swapkey[code]
    sw = I[(I.scope == "n>1") & (I.target == tg) & (I.step == st) & (I.w == 1.0)]
    swc = (f"{sw.ratio_c.values[0]:.2f} & {sw.p_cond.values[0]:.3f}") if len(sw) else "--- & ---"
    hol = f"{a.holm_max:.3f}" if a.primary else "---"
    name = a.test.replace("->", r"$\to$").replace("m=", r"$\beta$=")
    out.append(f"{code} & {name} & {a.rho_max:.2f} ({a.p_max:.3f}) & {hol} & "
               f"{a.rho_mean:.2f} ({a.p_mean:.3f}) & {b.rho_max:.2f} ({b.p_max:.3f}) & {b.rho_mean:.2f} ({b.p_mean:.3f}) & {swc}\\\\")
out.append(r"\bottomrule\end{tabular}}")
out.append(r"\caption{Conditional permutation tests on the 150 multi-inclusion cells at $q=1\%$ (2000 resamples). "
           r"P1--P5: primary family (Holm adjustment of the worst-case statistic shown); X, M: exploratory. "
           r"Resampling sampler: the added block is resampled from a fitted quadratic conditional model (residuals permuted "
           r"within morphology family); it is anti-conservative in the realistic synthetic null of Table~\ref{tab:S-power} and is shown "
           r"only as a sensitivity analysis. Local swap: exchange between near neighbours, used for inference. "
           r"``mean'': mean $|R_i-R_j|$ over the closest pairs. Under its own model the resampling sampler rejects at the 5\% "
           r"level in " + f"{CM.rej05_max.min():.2f}--{CM.rej05_max.max():.2f}" + r" (worst case) and "
           + f"{CM.rej05_mean.min():.2f}--{CM.rej05_mean.max():.2f}" +
           r" (mean) of 60 model-generated nulls per primary test.}\label{tab:S-cond}\end{table}")

out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{llcccc}\toprule Target & Level & ratio, standardized & ratio, Mahalanobis & $p$, std. & $p$, Mahal.\\\midrule")
tl = {"R_H": r"$R_H$", "R_sigma": r"$R_\sigma$", "R_path": r"$R_{\rm path}$"}
for _, r in MH[MH.scope == "n>1"].iterrows():
    out.append(f"{tl[r.target]} & {r.level} & {r.ratio_std:.3f} & {r.ratio_mahal:.3f} & {r.p_std:.3f} & {r.p_mahal:.3f}\\\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Equal-coverage ratios on the 150 multi-inclusion cells with the standardized Euclidean and the "
           r"Mahalanobis (pseudo-inverse, whitened) metric; 2000 response permutations. The Mahalanobis metric is invariant "
           r"under invertible linear reparameterizations and under duplication of components (verified to machine precision).}"
           r"\label{tab:S-mahal}\end{table}")

lab = {"R_H": r"$R_H$", "R_sigma": r"$R_\sigma$", "R_path": r"$R_{\rm path}$"}
out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{llccccc}\toprule Scope & Level & $\beta=2$ & 5 & 10 & 20 & 40\\\midrule")
tm = {2: "R_H m=2", 5: "R_H m=5", 10: "R_H m=10", 20: "R_H", 40: "R_H m=40"}
for sc, scl in (("all", "all"), ("n>1", "$n>1$")):
    for lv in ("G1", "GL", "E0", "E1", "E0+cov", "E0+split"):
        r = [L[(L.scope == sc) & (L.target == tm[m]) & (L.level == lv)].iloc[0] for m in tm]
        out.append(f"{scl} & {lv} & " + " & ".join(f"{x.ratio:.3f} ({x.R2_loco:.2f})" for x in r) + r"\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Hazard exponent sweep: $\Omega/\Omega^{\rm null}$ at $q=1\%$ ($10^4$ permutations) and, in brackets, "
           r"linear $R^2_{\rm LOCO}$.}\label{tab:S-msweep}\end{table}")

out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{llccc}\toprule Scope & Target & Level & base ratio & with one duplicated component (min--max)\\\midrule")
for (sc, tg, lv), g in U.groupby(["scope", "target", "level"]):
    out.append(f"{'all' if sc == 'all' else '$n>1$'} & {lab[tg]} & {lv} & {g.ratio.iloc[0]:.3f} & "
               f"{g.ratio_dup.min():.3f}--{g.ratio_dup.max():.3f}\\\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Redundant-coordinate control with the standardized Euclidean metric: each standardized component is "
           r"duplicated in turn (no information is added). With the Mahalanobis metric the change is zero.}\label{tab:S-dup}\end{table}")

out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{cccccc}\toprule $N$ & $k_{\rm r}$ & load step & dissipation & plateau$/\ell$ & support$/\ell$\\\midrule")
for _, r in DC.iterrows():
    e = int(round(math.log10(r.k_r)))
    out.append(f"{int(r.N)} & $10^{{{e}}}$ & ${r.dE_rel:g}\\,\\varepsilon_0$ & {r.dissipation:.3f} & {r.plateau_over_ell:.2f} & {r.support_over_ell:.2f}\\\\")
out.append(r"\midrule \multicolumn{6}{l}{Effect of $k_{\rm r}=10^{-4}$ (instead of $10^{-3}$) on the targets, $N=127$:}\\")
for _, r in KR[KR.key != "hom"].iterrows():
    out.append(f"\\multicolumn{{2}}{{c}}{{cell {r.key}}} & \\multicolumn{{2}}{{c}}{{$\\Delta R_\\sigma={r.dR_sigma:+.4f}$}} & \\multicolumn{{2}}{{c}}{{$\\Delta R_{{\\rm path}}={r.dR_path:+.4f}$}}\\\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Regularized dissipation of the homogeneous cell at separation ($G_cL=0.98$) for three residual "
           r"stiffnesses and two load steps, and effect of the residual stiffness on strength and path cost.}\label{tab:S-diss}\end{table}")

# --- power control
import numpy as np
PW = pd.read_csv("v09_r3_power.csv")
SW = pd.read_csv("v09_r3_swapcal.csv")
G = PW.groupby(["code", "noise", "lam", "sampler"]).agg(share=("share", "mean"), pmax=("p_max", lambda x: np.mean(x <= 0.05)),
                                                      pmean=("p_mean", lambda x: np.mean(x <= 0.05))).reset_index()
out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{lcccccc}\toprule & & & \multicolumn{2}{c}{local swap} & \multicolumn{2}{c}{resampling}\\\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
out.append(r"Step & noise & $\lambda$ (share of variance) & worst case & mean & worst case & mean\\\midrule")
nm = {"P1": r"gaps$\to$hazard", "P2": r"covariance$\to$hazard", "P3": r"gaps$\to$strength"}
for (code, noise, lam), g in G.groupby(["code", "noise", "lam"]):
    a = g[g.sampler == "swap"].iloc[0]; b = g[g.sampler == "res"].iloc[0]
    out.append(f"{nm[code]} & {noise:g} & {lam:g} ({a.share:.3f}) & {a.pmax:.2f} & {a.pmean:.2f} & {b.pmax:.2f} & {b.pmean:.2f}\\\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Graduated power control on the 150 multi-inclusion cells: rejection rate at the 5\% level (25 repetitions) "
           r"for a synthetic target equal to the base-level part of the response plus $\lambda$ standard deviations of the "
           r"part of the added block not predictable from the base level, plus Gaussian noise. $\lambda=0$ is a realistic "
           r"conditional null: the local-swap sampler keeps its size (0--0.08), the resampling sampler does not (up to 0.52). "
           r"Under nulls generated from the fitted conditional model the local-swap sampler rejects in "
           + f"{SW.rej05_max.min():.2f}--{SW.rej05_max.max():.2f}" + r" of cases.}\label{tab:S-power}\end{table}")

# --- strength decomposition
import json
DJ = json.load(open("v09_r3_decomp.json"))
out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}\toprule Scope & var stiff. & var onset & var growth & $\rho$(st,on) & $\rho$(st,gr) & $\rho$(on,gr) & corr($R_\sigma$, onset part) & slope\\\midrule")
for key, lab in (("inc180", "180 inclusions"), ("n>1", "150, $n>1$")):
    v = DJ[key]; cv = np.array(v["cov"]); cr = np.array(v["corr"])
    out.append(f"{lab} & {cv[0,0]:.4f} & {cv[1,1]:.4f} & {cv[2,2]:.4f} & {cr[0,1]:.2f} & {cr[0,2]:.2f} & {cr[1,2]:.2f} & {v['corr_Rsigma_onset']:.2f} & {v['slope_Rsigma_on_onset']:.2f}\\\\")
for fk, v in DJ["within_f_n>1"].items():
    cv = np.array(v["cov"]); cr = np.array(v["corr"])
    out.append(f"$n>1$, $f={fk}$ & {cv[0,0]:.4f} & {cv[1,1]:.4f} & {cv[2,2]:.4f} & {cr[0,1]:.2f} & {cr[0,2]:.2f} & {cr[1,2]:.2f} & {v['corr_Rsigma_onset']:.2f} & {v['slope_Rsigma_on_onset']:.2f}\\\\")
out.append(r"\bottomrule\end{tabular}}")
out.append(r"\caption{Variance structure of $R_\sigma=\ln R_E-J_\psi+\ln(\Gamma_\sigma/\Gamma^{\rm hom}_\sigma)$ (stiffness, onset, growth). "
           r"If $R_\sigma$ were independent of the onset part, the onset--growth correlation on the 150 cells would be "
           + f"{DJ['null_corr_if_independent']:.2f}" + r".}\label{tab:S-decomp}\end{table}")

# --- critical-distance onset
TS = pd.read_csv("v09_tcd_scan.csv"); TE = pd.read_csv("v09_tcd_ell.csv"); TB = json.load(open("v09_tcd_blind.json"))
out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{lcccc}\toprule Descriptor & corr with $R_\sigma$ & ratio of $E1+$descriptor & $R^2_{\rm LOCO}$ ($E1+$) & $R^2_{\rm LOCO}$ (alone)\\\midrule")
lab = {"J_0": r"$J_0=J_\psi$"}
for _, r in TS.iterrows():
    d = r.desc
    if d == "J_psi":
        continue
    name = lab.get(d, r"$J_{" + d[2:] + r"\ell}$" if d.startswith("J_") else r"$R_H$")
    out.append(f"{name} & {r['corr']:.2f} & {r.ratio_E1plus:.2f} & {r.R2_E1plus:.2f} & {r.R2_alone:.2f}\\\\")
out.append(r"\midrule \multicolumn{5}{l}{Held-out $R^2_{\rm oos}$ (trained on the 180 design inclusion cells; radius chosen without these cells):}\\")
for st, nmst in (("A", "Set B (45)"), ("O", "Set O (12)")):
    b = TB["blind"][st]
    out.append(f"\\multicolumn{{5}}{{l}}{{{nmst}: $E1$ {b['E1']:.2f}; $E1+R_H$ {b['E1+R_H']:.2f}; $E1+J_\\psi$ {b['E1+J_psi']:.2f}; $E1+J_{{1.5\\ell}}$ {b['E1+J_1.5']:.2f}}}\\\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Critical-distance onset intensity $J_r$ on the 150 multi-inclusion cells (radius in units of $\ell$) and held-out check. "
           r"Conditional test of $E1\to E1+J_{1.5\ell}$ (local swap): $p_{\rm c}=" + f"{TB['cond']['p_swap_std_max']:.4f}" + r"$.}\label{tab:S-tcd}\end{table}")
out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\begin{tabular}{lcccccccc}\toprule Variant & \multicolumn{8}{c}{$R^2_{\rm LOCO}$ of $f+J_r$ for absolute radius $r/L$}\\")
rs = sorted(TE.r.unique())
out.append(r" & " + " & ".join(f"{r:g}" for r in rs) + r"\\\midrule")
vl = {"baseline": r"$\ell=0.025L$, $\kappa=10$", "ell040": r"$\ell=0.040L$, $\kappa=10$", "contrast3": r"$\ell=0.025L$, $\kappa=3$"}
for case, g in TE.groupby("case"):
    g = g.set_index("r")
    out.append(vl[case] + " & " + " & ".join(f"{g.R2_loco_f_J[r]:.2f}" for r in rs) + r"\\")
out.append(r"\bottomrule\end{tabular}")
out.append(r"\caption{Shift of the optimal averaging radius with the regularization length (49 inclusion cells of the 61-cell subset).}\label{tab:S-tcdell}\end{table}")
# --- round 4: sweep-adjusted test, resolution of J_r, frozen prospective test
SW = json.load(open("v09_r4_sweep.json")); RS = json.load(open("v09_r4_res255.json"))
TF = json.load(open("tcd_frozen/tcd_results.json")); TX = json.load(open("tcd_frozen/tcd_exploratory.json"))
out.append(r"\begin{table}[ht]\centering\small")
out.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccc}\toprule")
out.append(r"\multicolumn{5}{l}{(a) Conditional test $E1\to E1+J_r$ (local swap, 2000 draws), radius selection repeated in each draw}\\\midrule")
out.append(r" & fixed $r=1.5\ell$ & selected by $R^2_{\rm LOCO}$ & minimum over $r$ & \\")
out.append(f"$p_{{\\rm c}}$ (max / mean) & {SW['p_fixed_1.5_[max,mean]'][0]:.4f} / {SW['p_fixed_1.5_[max,mean]'][1]:.4f} & {SW['p_selected_by_R2LOCO_[max,mean]'][0]:.4f} / {SW['p_selected_by_R2LOCO_[max,mean]'][1]:.4f} & {SW['p_min_over_r_[max,mean]'][0]:.4f} / {SW['p_min_over_r_[max,mean]'][1]:.4f} & \\\\")
out.append(r"\midrule \multicolumn{5}{l}{(b) Change between $N=127$ and $N=255$ (20 cells; 14 inclusion cells)}\\\midrule")
out.append(r" & max $|\Delta|$ (all) & max $|\Delta|$ (incl.) & mean $|\Delta|$ (incl.) & Spearman (incl.)\\")
for k, nm in (("J_0", r"$J_0=J_\psi$"), ("J_1", r"$J_{1\ell}$"), ("J_1.5", r"$J_{1.5\ell}$"), ("J_2", r"$J_{2\ell}$")):
    v = RS[k]
    out.append(f"{nm} & {v['max_abs_d_all']:.3f} & {v['max_abs_d_incl']:.3f} & {v['mean_abs_d_incl']:.3f} & {v['spearman_incl']:.3f}\\\\")
out.append(r"\midrule \multicolumn{5}{l}{(c) Frozen prospective test: $R^2_{\rm oos}$ (trained on the 180 design inclusion cells)}\\\midrule")
out.append(r" & $E1+J_{1.5\ell}$ & $E1+R_H$ & $E1+J_\psi$ & $E1$\\")
for st, nm in (("C1", "C1 (30 cells)"), ("C2", "C2 (12 cells)")):
    v = TF[st]
    out.append(f"{nm} & {v['E1+J_1.5']:.2f} & {v['E1+R_H']:.2f} & {v['E1+J_psi']:.2f} & {v['E1']:.2f}\\\\")
b = TF["C1_boot_lower5"]
out.append(r"\multicolumn{5}{l}{Decisions: H1 " + TF["H1"] + r"; H2 " + TF["H2"] + f" (5\\% bootstrap bounds {b['vs_E1+R_H']:.2f} and {b['vs_E1+J_psi']:.2f}); H3 " + TF["H3"] + r".}\\")
out.append(r"\multicolumn{5}{l}{Exploratory: RMSE " + f"{TX['C1']['rmse']:.3f} (C1), {TX['C2']['rmse']:.3f} (C2); mean residual {TX['C1']['mean_resid']:+.3f} (C1), {TX['C2']['mean_resid']:+.3f} (C2); std of $R_\\sigma$ {TX['C1']['std_y']:.3f}, {TX['C2']['std_y']:.3f}." + r"}\\")
out.append(r"\bottomrule\end{tabular}}")
out.append(r"\caption{Critical-distance onset intensity: (a) conditional test accounting for the radius sweep; (b) resolution sensitivity at the same physical radius; (c) prospective test under the frozen protocol of Section~S2.}\label{tab:S-tcdr4}\end{table}")
open("../supp_v09_tables.tex", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok", len(out))
