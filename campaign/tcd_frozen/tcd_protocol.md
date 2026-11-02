# Frozen protocol: prospective test of the critical-distance onset intensity J_{1.5ℓ}

Manuscript: *Energy and failure select different descriptors* (A.E. Gómez-Ovalle, C. Ramírez Ovalle), Draft v0.9.
This protocol and the analysis script `tcd_frozen.py` are hashed (SHA-256, file `tcd_protocol.sha256`) and committed to
the public repository https://github.com/aegovalle/energy-failure-descriptors **before** any cell of this test is
generated. The public commit date is the verifiable time stamp. No cell, elastic field or phase-field result of this
test exists at the time of hashing. The generator was only checked for feasibility of the planned (f, n) combinations
with a different, throw-away seed (placement only; no solve; outputs discarded). The script was dry-run with that
throw-away seed and fake phase-field outputs, and on the already analysed blind Sets B and O to check that it reproduces
the published values.

## 1. What is tested

The critical-distance onset intensity
J_r = ½ ln max_x ⟨ψ⁺/ψ_c⟩_{D_r(x)} − ½ ln max(ψ⁺/ψ_c)_hom,
with ψ⁺ the AT1 driving energy of the elastic field under the unit macroscopic strain (ε̄₁₁ = 1), ψ_c = 3G_c/(16ℓ) the local
(phase-dependent) threshold, the average taken over a periodic disk D_r of radius r covering **both phases**, and the
normalization by the homogeneous cell. The radius is **fixed at r = 1.5ℓ = 0.0375 L** (ℓ = 0.025 L), the value chosen on
the design set in Draft v0.9. No other radius is tested confirmatorily.

Model under test: linear ridge surrogate (ridge 10⁻³, standardized inputs, intercept) **E1 + J_{1.5ℓ}**, with
E1 = (f, ε̄₁₁ᵐ, ε̄₂₂ᵐ, ε̄₁₂ᵐ, Jensen gap of the matrix, Jensen gap of the inclusions), trained once on the **180 design
inclusion cells** of the main campaign (target R_σ = ln σ_peak/σ_peak,hom). Comparators trained identically:
E1 + R_H (R_H = ln B, the weakest-link hazard) and E1 + J_ψ (pointwise onset, r = 0). E1 alone and J_{1.5ℓ} alone are
reported as exploratory.

## 2. New cells

New seed **20261102** (never used before), N = 127, κ = E_i/E_m = 10, ν = 0.3, G_c,i/G_c,m = 3, random sequential
addition of non-overlapping inclusions (`gen_inclusions`, unchanged), generated in the order below.

| Set | Family | Aspect ratio | n | f | Realizations | Cells |
|---|---|---|---|---|---|---|
| C1 (in distribution) | disks | 1 | 4, 9, 16 | 0.1, 0.2, 0.3 | 2 | 18 |
| C1 (in distribution) | ellipses | 3 | 4, 9 | 0.1, 0.2, 0.3 | 2 | 12 |
| C2 (more inclusions) | disks | 1 | 25 | 0.1, 0.2, 0.3 | 2 | 6 |
| C2 (more inclusions) | ellipses | 3 | 16 | 0.1, 0.2, 0.3 | 2 | 6 |

Total 42 cells. Phase field: `PhaseFieldRVE(mask).run()` with all default settings, identical to the main campaign and the blind sets
(AT1, hybrid split, ℓ = 0.025 L, k_res = 10⁻³, uniaxial strain control). σ_peak,hom = 4.452871.

## 3. Hypotheses and decision rules (evaluated by `tcd_frozen.py analyze`, exactly as written)

Only cells whose phase-field run separates are used; the number excluded is reported.

- **H1 (primary).** On C1, R²_oos(E1 + J_{1.5ℓ}) ≥ 0.80. Confirmed if the inequality holds, refuted otherwise.
- **H2.** On C1, E1 + J_{1.5ℓ} predicts better than both comparators: the 5% quantile of the paired cell bootstrap
  (2000 resamples of the C1 cells, seed 20261103) of R²_oos(E1 + J_{1.5ℓ}) − R²_oos(E1 + R_H) and of
  R²_oos(E1 + J_{1.5ℓ}) − R²_oos(E1 + J_ψ) are both > 0.
- **H3 (secondary, extrapolation in n).** On C2, R²_oos(E1 + J_{1.5ℓ}) ≥ 0.70.
- If fewer than 80% of the C1 cells separate, H1 and H2 are declared inconclusive.

## 4. Reporting

All three outcomes are reported in the manuscript whatever they are. Any other analysis of these cells (other radii,
other models, the equal-coverage statistic) is labelled exploratory.
