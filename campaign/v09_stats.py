r"""Statistics for the v0.9 re-analysis.

* omega_sel        equal-coverage statistic with the selected pairs returned and a
                   vectorized response-permutation null (any number of permutations)
* mcshane          exact sample bound for L-Lipschitz surrogates (all pairs, q-free):
                   e*(L) = 1/2 max_ij [ |R_i - R_j| - L d_ij ]_+
* lp_check         linear-programming verification of e*(L) on a subsample
* cond_perm        conditional permutation test for an added block eta given Psi_k:
                   eta is resampled by random local swaps between near neighbours in
                   the Psi_k metric (approximate CPT, Berrett et al. 2020). The marginal
                   of eta, hence its standardization and the descriptor dimension, is
                   preserved, so the test is free of the neighbourhood-emptying effect.
* holm             Holm step-down adjustment
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import linprog


def zscore(A):
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[:, None]
    s = A.std(0)
    keep = s > 1e-12
    Z = np.zeros_like(A)
    Z[:, keep] = (A[:, keep] - A[:, keep].mean(0)) / s[keep]
    return Z[:, keep]


def pair_index(n):
    return np.triu_indices(n, 1)


def omega_sel(Z, R, q, nperm=0, rng=None, II=None, JJ=None, d=None):
    """Z: already standardized (and possibly weighted) descriptor matrix."""
    n = len(R)
    if II is None:
        II, JJ = pair_index(n)
    if d is None:
        d = pdist(Z) if Z.shape[1] else np.zeros(len(II))
    nq = int(np.ceil(q * len(d) - 1e-9))
    dcut = np.partition(d, nq - 1)[nq - 1]
    sel = d <= dcut + 1e-12
    dR = np.abs(R[II] - R[JJ])
    om = float(dR[sel].max())
    out = dict(Omega=om, npairs=int(sel.sum()), nq=nq, sel=sel)
    if nperm:
        Is, Js = II[sel], JJ[sel]
        P = np.argsort(rng.random((nperm, n)), axis=1)
        Rp = R[P]
        nulls = np.abs(Rp[:, Is] - Rp[:, Js]).max(1)
        out.update(Omega_null=float(nulls.mean()), ratio=float(om / nulls.mean()),
                   p_perm=float((1 + np.sum(nulls <= om)) / (1 + nperm)),
                   null_q05=float(np.quantile(nulls, 0.05) / nulls.mean()))
    return out


def mcshane(Z, R, Ls):
    II, JJ = pair_index(len(R))
    d = pdist(Z) if Z.shape[1] else np.zeros(len(II))
    dR = np.abs(R[II] - R[JJ])
    Ls = np.atleast_1d(Ls)
    e = np.array([0.5 * np.max(np.maximum(dR - L * d, 0.0)) for L in Ls])
    return e


def lipschitz_needed(Z, R, eps):
    """Smallest L for which e*(L) <= eps  (max over pairs of (|dR| - 2 eps)/d)."""
    II, JJ = pair_index(len(R))
    d = pdist(Z)
    dR = np.abs(R[II] - R[JJ])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dR > 2 * eps, (dR - 2 * eps) / d, 0.0)
    return float(np.max(v))


def lp_check(Z, R, L):
    """min e s.t. |f_i - R_i| <= e, |f_i - f_j| <= L d_ij  (variables f_1..f_n, e)."""
    n = len(R)
    II, JJ = pair_index(n)
    d = pdist(Z)
    rows, b = [], []
    for i in range(n):
        r = np.zeros(n + 1); r[i] = 1; r[n] = -1; rows.append(r); b.append(R[i])
        r = np.zeros(n + 1); r[i] = -1; r[n] = -1; rows.append(r); b.append(-R[i])
    for k, (i, j) in enumerate(zip(II, JJ)):
        r = np.zeros(n + 1); r[i] = 1; r[j] = -1; rows.append(r); b.append(L * d[k])
        rows.append(-r); b.append(L * d[k])
    c = np.zeros(n + 1); c[n] = 1
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(b), bounds=[(None, None)] * n + [(0, None)],
                  method="highs")
    return float(res.x[n])


def _local_swap(eta, nbrs, rng):
    n = len(eta)
    perm = np.arange(n)
    free = np.ones(n, bool)
    for i in rng.permutation(n):
        if not free[i]:
            continue
        cand = [j for j in nbrs[i] if free[j] and j != i]
        if not cand:
            continue
        j = cand[rng.integers(len(cand))]
        free[i] = free[j] = False
        if rng.random() < 0.5:
            perm[i], perm[j] = j, i
    return eta[perm]


def cond_perm(Zk, eta_raw, R, q, B, rng, k_nn=5, w=1.0):
    """Conditional test of H0: R independent of eta given Psi_k, statistic Omega(q) of
    the enlarged descriptor (Psi_k, w * eta). Returns observed Omega, mean conditional
    null, ratio, one-sided p (small p: eta reduces the worst-case insufficiency beyond
    what a conditionally exchangeable eta would)."""
    eta = zscore(eta_raw) * w
    n = len(R)
    Dk = np.sqrt(((Zk[:, None, :] - Zk[None, :, :]) ** 2).sum(-1)) if Zk.shape[1] else np.zeros((n, n))
    np.fill_diagonal(Dk, np.inf)
    # neighbours: all exact ties plus the k_nn nearest
    nbrs = []
    for i in range(n):
        order = np.argsort(Dk[i], kind="stable")
        ties = np.flatnonzero(Dk[i] <= 1e-12)
        nbrs.append(np.unique(np.r_[ties, order[:k_nn]]))
    II, JJ = pair_index(n)
    obs = omega_sel(np.hstack([Zk, eta]), R, q, II=II, JJ=JJ)["Omega"]
    nulls = np.empty(B)
    for b in range(B):
        e = _local_swap(eta, nbrs, rng)
        nulls[b] = omega_sel(np.hstack([Zk, e]), R, q, II=II, JJ=JJ)["Omega"]
    return dict(Omega=obs, Omega_cnull=float(nulls.mean()), ratio_c=float(obs / nulls.mean()),
                p_cond=float((1 + np.sum(nulls <= obs)) / (1 + B)))


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
