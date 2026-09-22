import torch

RIDGE_SCALES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def ridge_fit(X, Y, scales):
    """Ridge regression Y ~ X for each lambda in `scales` (relative to the
    mean per-feature variance of X). Solves in whichever of the primal
    (d x d) or dual (n x n) form is smaller.
    Returns a list of (scale, W, x_mean, y_mean): prediction is
    (Z - x_mean) @ W + y_mean."""
    xm = X.mean(0)
    ym = Y.mean(0)
    Xc = X - xm
    Yc = Y - ym
    n, d = Xc.shape
    lam0 = float((Xc ** 2).sum()) / d
    primal = d <= n
    G = Xc.T @ Xc if primal else Xc @ Xc.T
    rhs = Xc.T @ Yc if primal else Yc
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    fits = []
    for s in scales:
        sol = torch.linalg.solve(G + s * lam0 * eye, rhs)
        fits.append((s, sol if primal else Xc.T @ sol, xm, ym))
    return fits


def ridge_select_and_fit(X, Y):
    """Choose lambda on the most recent 20% of the (time-ordered) samples,
    then refit on all of them.

    Returns (W, x_mean, y_mean, scale, holdout_fvu), where holdout_fvu is the
    fraction of the held-out samples' variance left unexplained by the
    chosen lambda (fitted without them)."""
    n_hold = max(1, X.shape[0] // 5)
    Xh, Yh = X[-n_hold:], Y[-n_hold:]
    best_s, best_err = None, None
    for s, W, xm, ym in ridge_fit(X[:-n_hold], Y[:-n_hold], RIDGE_SCALES):
        err = float((((Xh - xm) @ W + ym - Yh) ** 2).sum())
        if best_err is None or err < best_err:
            best_s, best_err = s, err
    var = float(((Yh - Yh.mean(0)) ** 2).sum())
    holdout_fvu = best_err / var if var > 0 else float('nan')
    _, W, xm, ym = ridge_fit(X, Y, [best_s])[0]
    return W, xm, ym, best_s, holdout_fvu
