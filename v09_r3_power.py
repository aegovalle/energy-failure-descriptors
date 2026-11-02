r"""Round-3 statistics:
(a) graduated power control for the main null results (minimum detectable effect)
(b) calibration of the local-swap sampler under synthetic conditional nulls
(c) split-moment tests with a Ledoit--Wolf shrunk Mahalanobis metric
Outputs: v09_r3_power.csv, v09_r3_swapcal.csv, v09_r3_lw.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial.distance import pdist

exec(open("v09_cond_final.py", encoding="utf-8").read().split("rows = []")[0])   # data, samplers, stats


def swap_draw_factory(Xk, eta):
    Zk = zscore(Xk); n = len(eta)
    Dd = np.sqrt(((Zk[:, None] - Zk[None]) ** 2).sum(-1)); np.fill_diagonal(Dd, np.inf)
    nbrs = [np.unique(np.r_[np.flatnonzero(Dd[i] <= 1e-12), np.argsort(Dd[i], kind="stable")[:5]]) for i in range(n)]

    def draw(rg):
        perm = np.arange(n); free = np.ones(n, bool)
        for i in rg.permutation(n):
            if not free[i]:
                continue
            c = [j for j in nbrs[i] if free[j] and j != i]
            if not c:
                continue
            j = c[rg.integers(len(c))]; free[i] = free[j] = False
            if rg.random() < 0.5:
                perm[i], perm[j] = j, i
        return eta[perm]
    return draw


def emb_factory(Xk, metric):
    if metric == "std":
        Zk = zscore(Xk)
        return lambda e: np.hstack([Zk, zscore(e)])
    if metric == "mahal":
        return lambda e: whiten(np.hstack([Xk, e]))
    return lambda e: lw_whiten(np.hstack([Xk, e]))


def lw_whiten(A):
    A = np.asarray(A, float); A = (A - A.mean(0)) / A.std(0)
    n, p = A.shape
    S = A.T @ A / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2)
    b2 = min(sum(np.sum((np.outer(x, x) - S) ** 2) for x in A) / n ** 2, d2)
    Sig = (b2 / d2) * mu * np.eye(p) + (1 - b2 / d2) * S
    w, V = np.linalg.eigh(Sig)
    return A @ V @ np.diag(w ** -0.5) @ V.T


def run_test(Xk, eta, R, sampler, metric, B, rg):
    emb = emb_factory(Xk, metric)
    obs = stats_sel(emb(eta), R)
    if sampler == "res":
        fit, res = fl_parts(Xk, eta)
        def draw(r):
            idx = np.arange(len(eta))
            for g in np.unique(STRATA):
                k = np.flatnonzero(STRATA == g); idx[k] = k[r.permutation(len(k))]
            return fit + res[idx]
    else:
        draw = swap_draw_factory(Xk, eta)
    nl = np.array([stats_sel(emb(draw(rg)), R) for _ in range(B)])
    return (1 + (nl <= obs).sum(0)) / (1 + B)


SETUPS = {"P1": ("RH20", E0, GAP), "P2": ("RH20", E0, COV), "P3": ("R_sigma", E0, GAP)}


def power_job(args):
    code, lam, noise, seed = args
    rg = np.random.default_rng(seed)
    tg, base, add = SETUPS[code]
    Xk = D[base].values; eta = D[add].values.astype(float); R0 = D[tg].values.astype(float)
    Zk = zscore(Xk)
    Phi = np.column_stack([np.ones(len(Zk)), Zk])
    g = Phi @ np.linalg.lstsq(Phi, R0, rcond=None)[0]                     # E0 (base) part of the target
    pc = zscore(eta)
    u = np.linalg.svd(pc - pc.mean(0), full_matrices=False)[0][:, 0]
    fit, res = fl_parts(Xk, u)                                            # part of the block not predictable from Psi_k
    z = (res - res.mean()) / res.std()
    sd = R0.std()
    out = []
    for rep in range(25):
        R = g + lam * sd * z + rg.normal(0, noise, len(g))
        share = (lam * sd) ** 2 / np.var(R)
        for sampler in ("res", "swap"):
            p = run_test(Xk, eta, R, sampler, "std", 200, rg)
            out.append(dict(code=code, lam=lam, noise=noise, rep=rep, sampler=sampler, share=share,
                            p_max=p[0], p_mean=p[1]))
    return out


def swapcal_job(args):
    code, seed = args
    rg = np.random.default_rng(seed)
    lab = dict((c[0], c) for c in TESTS)[code]
    _, _, tg, base, add, _ = lab
    R = D[tg].values.astype(float); Xk = D[base].values; eta = D[add].values.astype(float)
    fit, res = fl_parts(Xk, eta)
    ps = []
    for _ in range(60):
        idx = np.arange(len(eta))
        for gg in np.unique(STRATA):
            k = np.flatnonzero(STRATA == gg); idx[k] = k[rg.permutation(len(k))]
        ps.append(run_test(Xk, fit + res[idx], R, "swap", "std", 200, rg))
    ps = np.array(ps)
    return dict(code=code, rej05_max=float(np.mean(ps[:, 0] <= 0.05)), rej05_mean=float(np.mean(ps[:, 1] <= 0.05)))


def lw_job(args):
    code, seed = args
    rg = np.random.default_rng(seed)
    lab = dict((c[0], c) for c in TESTS)[code]
    _, name, tg, base, add, _ = lab
    R = D[tg].values.astype(float); Xk = D[base].values; eta = D[add].values.astype(float)
    p = run_test(Xk, eta, R, "res", "lw", 2000, rg)
    emb = emb_factory(Xk, "lw")
    return dict(code=code, test=name, p_max=p[0], p_mean=p[1])


if __name__ == "__main__":
    jobs = []
    s = 1000
    for code in ("P1", "P2"):
        for noise in (0.15, 0.5):
            for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
                s += 1; jobs.append((code, lam, noise, s))
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        s += 1; jobs.append(("P3", lam, 0.01, s))
    with Pool(min(30, len(jobs))) as pool:
        res = pool.map(power_job, jobs)
        P = pd.DataFrame([r for rr in res for r in rr]); P.to_csv("v09_r3_power.csv", index=False)
        cal = pool.map(swapcal_job, [(c, 5000 + k) for k, c in enumerate(("P1", "P2", "P3", "P4", "P5"))])
        pd.DataFrame(cal).to_csv("v09_r3_swapcal.csv", index=False)
        lw = pool.map(lw_job, [(c, 6000 + k) for k, c in enumerate(("X2", "X3", "P1", "P2"))])
        pd.DataFrame(lw).to_csv("v09_r3_lw.csv", index=False)
    pd.set_option("display.width", 200)
    S = P.groupby(["code", "noise", "lam", "sampler"]).agg(share=("share", "mean"),
                                                         pow_max=("p_max", lambda x: np.mean(x <= 0.05)),
                                                         pow_mean=("p_mean", lambda x: np.mean(x <= 0.05))).reset_index()
    print(S.round(3).to_string(index=False)); print(pd.DataFrame(cal)); print(pd.DataFrame(lw))
