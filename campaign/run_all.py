"""Single entry point that reproduces every number, table and figure of Draft v0.9.

    python run_all.py              # analysis only, from the stored raw outputs (minutes)
    python run_all.py --solve      # also re-run all solves (about 120 CPU-hours for the phase field)
    python run_all.py --solve -j 24

Stages (in order). Solves are cached and resumable; analysis stages are deterministic
(fixed random seeds).
"""
import argparse
import subprocess
import sys
import time

SOLVE = [
    ("elastic campaign (216 RVEs + 36 at N=255)", ["campaign_run.py"]),
    ("AT1 phase-field campaign", ["pf_campaign.py", "{j}"]),
    ("homogeneous cell: residual stiffness and load step", ["v09_dissipation_check.py"]),
    ("residual-stiffness check on six cells", ["v09_kres_check.py"]),
    ("phase-field resolution check, set 1", ["pf_resolution.py", "2,13,16,30,54,59,96,160", "255"]),
    ("phase-field resolution check, set 2",
     ["pf_resolution.py", "7,28,49,57,73,107,137,157,167,177,187,207", "255", "_b"]),
    ("phase-field sensitivity variants", ["pf_variants.py", "{j}"]),
    ("elastic fields for the rotated variant", ["pf_variant_elastic.py"]),
    ("blind set: generation + elastic (protocol frozen in blind/)", ["blind_generate.py"]),
    ("blind set: phase field", ["blind_pf.py", "{j}"]),
    ("frozen J_1.5ell test: generation + elastic (protocol frozen in tcd_frozen/)", ["tcd_frozen/tcd_frozen.py", "generate"]),
    ("frozen J_1.5ell test: phase field", ["tcd_frozen/tcd_frozen.py", "pf", "{j}"]),
]
ANALYSE = [
    ("elastic ladders and figures", ["campaign_analyze.py"]),
    ("elastic counterexample at three resolutions", ["check_counterexample.py"]),
    ("phase-field analysis and figures", ["pf_analyze.py"]),
    ("four-target ladders, ligament descriptors, CIs, LOCO", ["v06_analyze.py"]),
    ("robustness variants", ["v06_variants_analyze.py"]),
    ("crack-path extraction sensitivity", ["path_sensitivity.py"]),
    ("phase-field resolution summary", ["pf_resolution_analyze.py"]),
    ("blind validation (H1-H4)", ["blind_analyze.py"]),
    ("blind pair figure", ["blind_pair_fig.py"]),
    ("Fig. 4 redraw", ["fig_omega_levels_v07.py"]),
    ("robustness controls (m-sweep, n>1 exclusion, paired diffs)", ["robustness_checks.py"]),
    ("v0.9: elastic field statistics, two-point maps, weighted cut", ["v09_fields.py"]),
    ("v0.9: ladders, conditional tests, exact bound, metric checks", ["v09_analysis.py"]),
    ("v0.9: dissipation split, J_psi at N=255, strength decomposition", ["v09_postprocess.py"]),
    ("v0.9: resolution summary (20 cells)", ["v09_resolution_all.py"]),
    ("v0.9: Mahalanobis metric and power analysis", ["v09_power.py"]),
    ("v0.9: final conditional tests (n>1, both samplers, both metrics)", ["v09_cond_final.py"]),
    ("v0.9: calibration of the residual sampler", ["v09_cond_calib_model.py"]),
    ("v0.9: strength decomposition variance and critical-distance onset", ["v09_r3_decomp_tcd.py"]),
    ("v0.9: graduated power control, sampler calibration, Ledoit-Wolf metric", ["v09_r3_power.py"]),
    ("v0.9: critical-distance radius scan and blind check", ["v09_tcd_blind.py"]),
    ("v0.9: critical-distance radius vs regularization length", ["v09_tcd_ell.py"]),
    ("v0.9: critical-distance level statistics", ["v09_add_J15.py"]),
    ("v0.9: critical-distance figure", ["v09_fig_tcd.py"]),
    ("v0.9: conditional test with the radius sweep repeated in each resample", ["v09_r4_sweep_test.py"]),
    ("v0.9: resolution sensitivity of J_r (20 cells at N=255)", ["v09_r4_res255.py"]),
    ("v0.9: frozen J_1.5ell test, decision rules H1-H3", ["tcd_frozen/tcd_frozen.py", "analyze"]),
    ("v0.9: frozen J_1.5ell test, exploratory diagnostics (not decision rules)", ["tcd_frozen/tcd_exploratory.py"]),
    ("v0.9: figures", ["v09_figures.py"]),
    ("v0.9: supplementary tables", ["v09_supp_tables.py"]),
]


def run(label, args, j):
    cmd = [sys.executable, "-u"] + [a.format(j=j) for a in args]
    print(f"\n=== {label}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"=== done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args()
    for label, args in (SOLVE if a.solve else []) + ANALYSE:
        run(label, args, a.j)


if __name__ == "__main__":
    main()
