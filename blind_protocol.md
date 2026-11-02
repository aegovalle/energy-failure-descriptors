# Prospective blind validation of the ligament amplification A_ell (Draft v0.7)

Frozen on 2026-10-08, **before** any blind microstructure was generated or solved.
The SHA-256 of this file is written to `blind_protocol.sha256` together with the
time stamp, before `blind_generate.py` is run. Nothing below may be changed after
the blind data exist; any deviation must be reported as such in the manuscript.

## 1. Frozen definitions (identical to Draft v0.6)

| Item | Frozen choice |
|---|---|
| Grid, cell | N = 127, unit periodic cell, plane strain, E_m = 1, E_i = 10, nu = 0.3 |
| Load | macroscopic strain e1 (x) e1 |
| Elastic solver | `fft_rve.FFTElastic2D`, tol 1e-7 |
| Hazard target | R_H = lnB, m = 20, nonlocal disk ell = 0.025, matrix only (`analyze_state`) |
| A_ell | `descr_tools.ligament_amp(mask, k=10.0)[0]`: ln max_y max_gaps (cbar+delta)/(cbar/k+delta) along the load direction; no transverse averaging |
| delta_min | `descr_tools.ligament(mask)[0]` |
| P_min | `descr_tools.min_path(Gc, all cells)`, 16-neighbour stencil; ln(P_min / 0.98) |
| Levels | G0=(f); G1=(f, rhoI); GL=G1+A_ell; Gd=G1+delta_min; G2=G1+lnPmin; E0=(f, e11_m, e22_m, e12_m); E1=E0+(Jgap_m, Jgap_i); E1+L; E1+R_H |
| Statistic | `descr_tools.omega`: standardized Euclidean metric on the evaluated set, equal coverage q, all ties kept, mean of 500 permutations as null |
| Intervals | `omega_subsample_ci`, 80 % subsamples, 100 draws, 40 permutations, 5-95 % |
| Surrogates | linear, ridge 1e-3, features standardized with the **training** mean/std; trained on the 180 original inclusion RVEs; out-of-sample R^2 = 1 - SSE/SST on the blind set |
| Phase field | `pf_fft.PhaseFieldRVE(mask)` with all defaults (AT1 hybrid, ell = 0.025, G_c^m = 1, G_c^i = 3, strip at x = 0.95, factor 0.98), `run()` defaults (dE_rel 0.01, tol_d 1e-3, separation at 10 % secant stiffness) |
| PF targets | R_sigma = ln(sigma_peak / 4.452871); R_path = ln(C_path / 0.98), C_path = `min_path(Gc, d >= 0.9)`; J_psi = -ln(e0 / 3.304542) |

## 2. Blind microstructures (new seed 7_202_610, never used before)

* **Set A (in-distribution, elastic):** disks (n = 4, 9, 16) and ellipses of aspect ratio 3
  (n = 4, 9), f = 0.1, 0.2, 0.3, six realizations per nominal cell: 15 x 6 = 90 RVEs.
  Same generator `gen_inclusions` as the original campaign (single-disk cells are excluded
  because their realizations are translation copies).
* **Set B (in-distribution, phase field):** realizations 0, 1, 2 of every nominal cell of
  Set A: 45 RVEs.
* **Set O (out-of-distribution, exploratory):** ellipses of aspect ratio 6, n = 4, 9, 16,
  f = 0.1, 0.2, five realizations: 30 RVEs (elastic); realizations 0, 1 solved with the
  phase field: 12 RVEs.

## 3. Hypotheses and decision rules (fixed in advance)

* **H1 (primary, Set A, R_H, q = 1 %):** Omega/Omega_null(GL) <= 0.8 x Omega/Omega_null(G1).
  Secondary check: the same inequality at q = 2 %.
* **H2 (Set A, R_H):** R^2_oos(GL) - R^2_oos(G1) >= 0.05.
* **H3 (Set B, R_sigma, q = 5 %):** A_ell does not close strength:
  Omega/Omega_null(GL) >= 0.5 **and** R^2_oos(GL) <= 0.4.
* **H4 (Set B, R_path):**
  (a) between families, Set B pooled with the 36 original laminates (exact controls,
      not blind), q = 5 %: Omega/Omega_null(G2) <= 0.5 x Omega/Omega_null(G1);
  (b) within blind inclusions, q = 5 %: Omega/Omega_null(G2) >= 0.5 **and** R^2_oos(G2) <= 0.4;
  (c) lower bound: no blind RVE with R_path < lnPmin - 0.01.

q = 5 % for Set B (45 states, 990 pairs) gives about 50 selected pairs, the same choice
made for the 61-cell variant study; q = 1 % (10 pairs) is reported as secondary.
Each hypothesis is reported as confirmed or not confirmed; a failure leads to the
corresponding claim being reduced in the manuscript, not to a change of definitions.
Set O is exploratory and is reported separately; it cannot confirm or refute H1-H4.
