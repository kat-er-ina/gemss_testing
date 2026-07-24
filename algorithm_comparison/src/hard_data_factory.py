"""Honest / hard synthetic generator with OVERLAPPING valid solutions.

The shipped generator (``gemss.data_handling.generate_artificial_dataset``) has
three properties that make recovery near-trivial and that quietly favour GEMSS:

1. All alternative supports are linear combinations of the *first* support, so
   the entire signal lives in one ``sparsity``-dimensional subspace.
2. Signal features have ~10-100x the marginal variance of the noise features, so
   they are separable almost without the label.
3. Supports are *disjoint*, so GEMSS's headline claim -- recovering overlapping
   alternative solutions that sequential masking cannot -- is never tested.

This module fixes all three:

* **Genuine r-dimensional mechanism.** A rank-``latent_rank`` factor matrix Z
  generates the response ``y = Z @ b``. Signal features are random linear
  combinations of *all* r latent factors, so the signal is not a single
  direction.
* **Equal variance.** Signal and noise features are both unit-variance (and
  optionally standardized), removing the variance giveaway. Noise features are
  drawn independently of Z (genuinely irrelevant, same scale).
* **Controlled overlap.** Solutions share a common core of ``overlap`` features
  plus private features. Every planted support has size ``sparsity`` >=
  ``latent_rank`` and full-rank latent loadings, so each is an *exact* valid
  solution -- yet they overlap.

Ground truth = the union of planted supports (features with non-zero latent
loading); all other features are noise. Any size-``sparsity`` subset of the union
with full-rank loadings is also a valid solution -- a realistically large
Rashomon set.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Robustness helpers (T7): non-Gaussian feature noise and a nonlinear response.
# Both are strictly opt-in -- the default (``noise_dist="gauss"``,
# ``nonlinear=False``) reproduces the original code path *bit-for-bit*, drawing
# exactly the same RNG values in the same order, so existing artifacts are
# unchanged.
# ---------------------------------------------------------------------------
def _draw_feature_noise(rng, shape, noise_std, noise_dist):
    """Additive feature noise on the same (unit) scale, drawn from ``noise_dist``.

    ``gauss`` is byte-identical to the original ``rng.normal(0, noise_std, shape)``
    call. ``t`` (Student-t, dof=3) and ``laplace`` are heavy-tailed / misspecified
    alternatives, each rescaled so its marginal std equals ``noise_std`` -- so only
    the tail shape, not the noise energy, changes relative to the Gaussian default.
    """
    if noise_dist == "gauss":
        return rng.normal(0.0, noise_std, size=shape)
    if noise_dist == "laplace":
        # Laplace(0, b) has variance 2 b^2 -> b = noise_std / sqrt(2) matches std.
        return rng.laplace(0.0, noise_std / np.sqrt(2.0), size=shape)
    if noise_dist == "t":
        dof = 3.0  # heavy tails, finite variance = dof/(dof-2)
        raw = rng.standard_t(dof, size=shape)
        return raw * (noise_std / np.sqrt(dof / (dof - 2.0)))
    raise ValueError(f"noise_dist must be one of gauss/t/laplace, got {noise_dist!r}")


def _apply_nonlinear(y_latent, factors):
    """Turn a linear latent response into a non-additive one (T7 nonlinear).

    Mixes the original linear response 50/50 (in z-score units) with a pairwise
    interaction of the first two response-driver ``factors`` (or a centred square
    if only one factor is available). The interaction term is not recoverable by
    any additive/linear model, so this genuinely misspecifies the linear-Gaussian
    assumption the baselines (Lasso ensembles, GEMSS's regression head) rely on.
    """
    factors = np.asarray(factors, float)
    if factors.ndim == 1:
        factors = factors[:, None]
    if factors.shape[1] >= 2:
        inter = factors[:, 0] * factors[:, 1]
    else:
        inter = factors[:, 0] ** 2 - 1.0

    def _z(v):
        v = v - v.mean()
        s = v.std()
        return v / s if s > 0 else v

    return _z(y_latent) + _z(inter)


def _finalize_response(y_latent, binarize, binary_response_ratio):
    """Shared tail: continuous response, or a quantile-thresholded binary one."""
    if binarize:
        y_prob = 1.0 / (1.0 + np.exp(-y_latent))
        thr = np.quantile(y_prob, 1.0 - binary_response_ratio)
        return (y_prob > thr).astype(float)
    return y_latent.astype(float)


def generate_overlapping_dataset(
    n_samples: int = 100,
    n_features: int = 200,
    n_solutions: int = 3,
    sparsity: int = 5,
    latent_rank: int = 2,
    overlap: int = 2,
    noise_std: float = 0.05,
    nan_ratio: float = 0.0,
    binarize: bool = True,
    binary_response_ratio: float = 0.5,
    standardize: bool = True,
    nonlinear: bool = False,
    noise_dist: str = "gauss",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Set[int], Dict[str, List[int]]]:
    """Generate data with multiple *overlapping* equivalent sparse solutions.

    Parameters
    ----------
    n_samples, n_features : int
        Dataset shape.
    n_solutions : int
        Number of planted (overlapping) valid solutions.
    sparsity : int
        Features per planted support. Must be >= ``latent_rank``.
    latent_rank : int
        Dimension r of the latent mechanism (number of independent factors that
        drive the response). The signal subspace is r-dimensional.
    overlap : int
        Size of the feature core shared by every solution (0 <= overlap < sparsity).
        ``overlap = 0`` reproduces the disjoint regime; larger values force the
        valid solutions to share features.
    noise_std : float
        White noise added to every feature, on the same (unit) scale as the
        signal -- so it perturbs rather than dominates.
    nan_ratio : float
        Fraction of entries set to NaN (seeded, unlike the shipped generator).
    binarize : bool
        If True, threshold the sigmoid of the response to ``binary_response_ratio``.
    standardize : bool
        Z-score every feature column (kills any residual variance giveaway).
    nonlinear : bool
        T7(a). If True, the response is a non-additive function of the latent
        factors (linear term + a pairwise factor interaction), not just ``Z @ b``.
        Default False reproduces the original linear response exactly.
    noise_dist : {"gauss", "t", "laplace"}
        T7(b). Distribution of the additive feature noise (the ``noise_std``
        term). ``"gauss"`` (default) is bit-identical to the original; ``"t"``
        (Student-t, dof=3) and ``"laplace"`` are heavier-tailed / misspecified
        alternatives at the same marginal std.
    seed : int
        Seed for the (single, fully reproducible) RNG.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    true_support : Set[int]
        Union of all planted support feature indices (the ground truth).
    supports : Dict[str, List[int]]
        The individual planted supports (for diagnostics / overlap checks).
    """
    if sparsity < latent_rank:
        raise ValueError(
            f"sparsity ({sparsity}) must be >= latent_rank ({latent_rank}) so each "
            "support can reconstruct the response."
        )
    if not (0 <= overlap < sparsity):
        raise ValueError(f"overlap ({overlap}) must satisfy 0 <= overlap < sparsity ({sparsity}).")

    rng = np.random.default_rng(seed)
    r = latent_rank

    # --- latent mechanism: r independent factors drive the response ----------
    Z = rng.standard_normal((n_samples, r))
    b = rng.uniform(2.0, 10.0, size=r) * rng.choice([-1.0, 1.0], size=r)
    y_latent = Z @ b

    # --- lay out overlapping supports over a shared core + private features --
    private = sparsity - overlap
    union_size = overlap + n_solutions * private
    if union_size > n_features:
        raise ValueError(
            f"union of supports ({union_size}) exceeds n_features ({n_features}); "
            "reduce n_solutions/sparsity or increase overlap/n_features."
        )
    all_signal = rng.choice(n_features, size=union_size, replace=False)
    core = all_signal[:overlap]
    rest = all_signal[overlap:]
    supports: Dict[str, List[int]] = {}
    for k in range(n_solutions):
        priv = rest[k * private : (k + 1) * private]
        supp = np.concatenate([core, priv]).astype(int)
        supports[f"solution_{k}"] = sorted(int(i) for i in supp)

    true_support: Set[int] = {int(i) for i in all_signal}

    # --- realize features ----------------------------------------------------
    X = rng.standard_normal((n_samples, n_features))  # default: pure noise cols
    # Signal columns are random full-rank combinations of the latent factors.
    loadings: Dict[int, np.ndarray] = {}
    for j in true_support:
        c = rng.standard_normal(r)
        loadings[j] = c
        X[:, j] = Z @ c

    # white noise on the same (unit) scale, then optional standardization
    X = X + _draw_feature_noise(rng, X.shape, noise_std, noise_dist)
    if standardize:
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

    # --- response ------------------------------------------------------------
    if nonlinear:
        y_latent = _apply_nonlinear(y_latent, Z)
    y = _finalize_response(y_latent, binarize, binary_response_ratio)

    # --- seeded missingness --------------------------------------------------
    if nan_ratio > 0.0:
        mask = rng.random(X.shape) < nan_ratio
        X = X.astype(float)
        X[mask] = np.nan

    return X, y, true_support, supports


def generate_overlapping_dataset_equicorr(
    n_samples: int = 100,
    n_features: int = 200,
    n_solutions: int = 3,
    sparsity: int = 5,
    latent_rank: int = 2,  # accepted for signature parity; unused by this mechanism
    overlap: int = 2,
    noise_std: float = 0.05,
    nan_ratio: float = 0.0,
    binarize: bool = True,
    binary_response_ratio: float = 0.5,
    standardize: bool = True,
    nonlinear: bool = False,
    noise_dist: str = "gauss",
    rho: float = 0.7,
    signal_frac: float = 0.6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Set[int], Dict[str, List[int]]]:
    """T4: overlapping solutions under a **block-equicorrelation** covariance.

    ``generate_overlapping_dataset`` builds every signal feature as an independent
    random loading onto ONE shared ``latent_rank``-dimensional factor matrix ``Z``,
    so the whole signal covariance is exactly rank ``latent_rank``. This function
    keeps the *same* overlapping-support layout (a shared core of ``overlap``
    features plus per-solution private features) and the *same* return contract,
    but induces a **genuinely different covariance**: an equicorrelation block per
    feature group, blocks otherwise independent except through the shared core.

    Mechanism
    ---------
    * One shared response signal ``t`` drives the label -- it is the only thing
      ``y`` depends on.
    * Each feature group (the shared core, and each solution's private features)
      gets one common factor ``g = signal_frac * t + sqrt(1 - signal_frac**2) * h``
      with ``h`` independent per group. ``g`` therefore correlates with ``t`` at
      ``signal_frac``.
    * Every feature in a group is ``sqrt(rho) * g + sqrt(1 - rho) * eps_j``. Hence:
        - within-group correlation  = ``rho`` (equicorrelation block);
        - core<->private and cross-solution correlation = ``rho * signal_frac**2``
          (blocks are otherwise independent, coupled only via the shared signal);
        - each planted support (core + one private group) correlates with ``t`` and
          is a valid predictor of ``y`` -- so the overlapping-solution structure
          survives, just under an equicorrelation covariance instead of a
          low-rank-factor one.

    ``rho`` sets the within-block collinearity (the selection difficulty);
    ``signal_frac`` sets how strongly each block tracks the response. ``latent_rank``
    is accepted only for call-site parity with the latent generator and is unused.
    All other parameters (``overlap``/``sparsity``/``n_solutions``, ``nonlinear``,
    ``noise_dist``, ``nan_ratio``, ``binarize``, ...) behave as in the latent
    generator. Returns the identical ``(X, y, true_support, supports)`` tuple.
    """
    if sparsity < 1:
        raise ValueError(f"sparsity ({sparsity}) must be >= 1.")
    if not (0 <= overlap < sparsity):
        raise ValueError(f"overlap ({overlap}) must satisfy 0 <= overlap < sparsity ({sparsity}).")
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho ({rho}) must satisfy 0 <= rho < 1.")
    if not (0.0 <= signal_frac <= 1.0):
        raise ValueError(f"signal_frac ({signal_frac}) must satisfy 0 <= signal_frac <= 1.")

    rng = np.random.default_rng(seed)

    # --- shared response driver ---------------------------------------------
    t = rng.standard_normal(n_samples)
    y_scale = rng.uniform(2.0, 10.0) * rng.choice([-1.0, 1.0])
    y_latent = y_scale * t

    # --- lay out overlapping supports over a shared core + private features --
    private = sparsity - overlap
    union_size = overlap + n_solutions * private
    if union_size > n_features:
        raise ValueError(
            f"union of supports ({union_size}) exceeds n_features ({n_features}); "
            "reduce n_solutions/sparsity or increase overlap/n_features."
        )
    all_signal = rng.choice(n_features, size=union_size, replace=False)
    core = all_signal[:overlap]
    rest = all_signal[overlap:]

    # --- per-group common factors, each partly the shared signal -------------
    a = np.sqrt(rho)
    c = np.sqrt(1.0 - rho)
    sf = signal_frac
    hf = np.sqrt(1.0 - sf * sf)

    def _group_factor():
        h = rng.standard_normal(n_samples)
        return sf * t + hf * h  # unit-variance factor, corr(., t) = signal_frac

    X = rng.standard_normal((n_samples, n_features))  # default: pure noise cols

    if overlap > 0:
        g_core = _group_factor()
        for j in core:
            X[:, int(j)] = a * g_core + c * rng.standard_normal(n_samples)

    supports: Dict[str, List[int]] = {}
    group_factors: List[np.ndarray] = []
    for k in range(n_solutions):
        priv = rest[k * private : (k + 1) * private]
        g_k = _group_factor()
        group_factors.append(g_k)
        for j in priv:
            X[:, int(j)] = a * g_k + c * rng.standard_normal(n_samples)
        supp = np.concatenate([core, priv]).astype(int)
        supports[f"solution_{k}"] = sorted(int(i) for i in supp)

    true_support: Set[int] = {int(i) for i in all_signal}

    # white noise on the same (unit) scale, then optional standardization
    X = X + _draw_feature_noise(rng, X.shape, noise_std, noise_dist)
    if standardize:
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

    # --- response ------------------------------------------------------------
    if nonlinear:
        # interaction between two response-tracking group factors (or t^2 if only
        # one group exists) -> a non-additive response no single feature is linear in.
        drivers = np.column_stack(group_factors) if len(group_factors) >= 2 else t
        y_latent = _apply_nonlinear(y_latent, drivers)
    y = _finalize_response(y_latent, binarize, binary_response_ratio)

    # --- seeded missingness --------------------------------------------------
    if nan_ratio > 0.0:
        mask = rng.random(X.shape) < nan_ratio
        X = X.astype(float)
        X[mask] = np.nan

    return X, y, true_support, supports


def generate_distinct_solutions(
    n_samples: int = 100,
    n_features: int = 200,
    n_solutions: int = 3,
    sparsity: int = 10,
    noise_std: float = 0.05,
    binarize: bool = False,
    binary_response_ratio: float = 0.5,
    standardize: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Set[int], Dict[str, List[int]]]:
    """NON-exchangeable distinct solutions -- the fair benchmark for the count test.

    ``generate_overlapping_dataset`` shares ONE latent factor matrix ``Z`` across
    every signal feature, so the whole signal lives in a single ``latent_rank``-
    dimensional subspace and *any* full-rank subset of the union is also a valid
    solution (a combinatorial Rashomon set). That makes "do the mixture weights
    reveal the *number* of solutions?" ill-posed: the true count is not
    ``n_solutions`` but a huge combinatorial number, so near-uniform weights are
    correct, not a failure (KH's RQ5 objection).

    This generator removes the exchangeability. Each planted solution is a
    DISJOINT block of ``sparsity`` features built as a full-rank linear mixing of
      * a single shared response signal ``s`` (the only thing that drives ``y``), and
      * ``sparsity - 1`` PRIVATE nuisance factors, independent across blocks:

        ``X[:, block_k] = [s | U_k] @ M_k``,   ``M_k`` full-rank (sparsity x sparsity).

    Consequences:
      * ``s`` lies in the span of every block -> every block is an exact valid
        solution (R^2 ~ 1 reconstructing ``y``).
      * isolating ``s`` requires inverting the full ``M_k`` -> *every* feature in a
        block is necessary; no within-block feature is substitutable.
      * nuisances are independent across blocks -> a partial or cross-block subset
        cannot isolate ``s``; block j cannot substitute for block k.

    Therefore the number of valid size-``sparsity`` supports is EXACTLY
    ``n_solutions``. If GEMSS's weight perplexity / distinct-support count tracks
    ``n_solutions`` here, the weights *do* reveal the count; if it stays flat, the
    honest-negative survives on a benchmark where the count is well-defined.

    Returns the same ``(X, y, true_support, supports)`` tuple as
    ``generate_overlapping_dataset`` (solutions are disjoint, so ``true_support``
    is their disjoint union).
    """
    rng = np.random.default_rng(seed)
    union_size = n_solutions * sparsity
    if union_size > n_features:
        raise ValueError(
            f"union of disjoint supports ({union_size}) exceeds n_features "
            f"({n_features}); reduce n_solutions/sparsity or increase n_features."
        )
    if n_samples <= sparsity:
        # With n <= D a generic size-D support has full row rank and can span y,
        # so arbitrary and cross-block supports also become exact solutions and the
        # "exactly n_solutions" contract breaks. Reject the regime up front.
        raise ValueError(
            f"n_samples ({n_samples}) <= sparsity ({sparsity}): a size-{sparsity} "
            f"support can then interpolate y, so the 'exactly n_solutions' benchmark "
            f"is ill-defined. Use n_samples > sparsity."
        )

    # --- shared response signal: the ONLY driver of y ------------------------
    s = rng.standard_normal(n_samples)

    # --- lay out disjoint blocks ---------------------------------------------
    all_signal = rng.choice(n_features, size=union_size, replace=False)
    supports: Dict[str, List[int]] = {}

    X = rng.standard_normal((n_samples, n_features))  # default: pure noise cols
    for k in range(n_solutions):
        block = all_signal[k * sparsity : (k + 1) * sparsity].astype(int)
        supports[f"solution_{k}"] = sorted(int(i) for i in block)
        # private nuisances, independent across blocks
        U_k = rng.standard_normal((n_samples, sparsity - 1))
        latent_k = np.column_stack([s, U_k])              # (n, sparsity)
        # full-rank mixing: redraw until well-conditioned (almost always first try)
        while True:
            M_k = rng.standard_normal((sparsity, sparsity))
            if np.linalg.cond(M_k) < 1e3:
                break
        X[:, block] = latent_k @ M_k

    true_support: Set[int] = {int(i) for i in all_signal}

    # white noise on the same (unit) scale, then optional standardization
    X = X + rng.normal(0.0, noise_std, size=X.shape)
    if standardize:
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

    # --- response: driven solely by s ----------------------------------------
    if binarize:
        y_prob = 1.0 / (1.0 + np.exp(-s))
        thr = np.quantile(y_prob, 1.0 - binary_response_ratio)
        y = (y_prob > thr).astype(float)
    else:
        y = s.astype(float)

    return X, y, true_support, supports


def support_recovery_check(
    X: np.ndarray, y_latent: np.ndarray, supports: Dict[str, List[int]]
) -> Dict[str, float]:
    """Diagnostic: R^2 of reconstructing the (continuous) response from each support.

    A value near 1.0 confirms the planted support is an exact valid solution.
    Intended for the continuous target (pass ``y_latent``), not the binarized one.
    """
    out: Dict[str, float] = {}
    yc = y_latent - y_latent.mean()
    ss_tot = float((yc**2).sum()) or 1.0
    for name, supp in supports.items():
        Xs = np.nan_to_num(X[:, supp], nan=0.0)
        w, *_ = np.linalg.lstsq(Xs, y_latent, rcond=None)
        resid = y_latent - Xs @ w
        out[name] = 1.0 - float((resid**2).sum()) / ss_tot
    return out
