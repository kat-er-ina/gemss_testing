"""Classical sparse-selection baselines for the stress test.

The original ALFESE comparison pitted GEMSS (a sparse linear-Gaussian method)
against generic filters (MI, mRMR, FCBF, greedy, model-importance) on
linear-Gaussian data. The obvious *strong* baselines for that generative model --
the ones a practitioner would actually reach for to find multiple sparse linear
explanations -- were missing. This module supplies them:

- :class:`MaskingWrapper` -- iterative-masking (a.k.a. "peeling") with a Lasso,
  Elastic-Net or L1-logistic base learner. This is the standard sequential
  heuristic for multiple solutions: fit, take the top-``sparsity`` features as a
  solution, remove them, refit on the remainder, repeat. It is the direct
  point of comparison for GEMSS's claim that a mixture can recover *overlapping*
  solutions that masking structurally cannot.

- :class:`StabilitySelectionWrapper` -- Meinshausen-Buhlmann stability selection.
  It returns a single "stable core" of features, illustrating the collapse of the
  Rashomon set into one point estimate that motivates GEMSS.

All baselines standardize features (zero mean, unit variance) -- the realistic
default, and one that removes the variance-scale giveaway present in the raw
synthetic data. Missing values are mean-imputed (these estimators cannot ingest
NaN); this is recorded honestly rather than hidden, and is a fair disadvantage
to flag against GEMSS's native NaN handling.
"""

from typing import Any, Dict, List
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.utils import resample

from .base import ModelWrapper

# Keep sweep logs readable: the l1-logistic API is mid-deprecation across sklearn
# versions but still functional with the lockfile-pinned version.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def _prepare(X: np.ndarray) -> np.ndarray:
    """Mean-impute NaNs then standardize columns to zero-mean/unit-variance."""
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_mean, inds[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs = StandardScaler().fit_transform(X)
    # Guard against constant columns producing NaN after scaling.
    return np.nan_to_num(Xs, nan=0.0)


def _make_estimator(estimator: str, task: str, alpha: float, seed: int):
    """Build the base sparse learner for the given estimator/task."""
    if task == "classification":
        # L1-penalized logistic regression. C = 1/alpha (smaller C => sparser).
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0 / max(alpha, 1e-6),
            max_iter=2000,
            random_state=seed,
        )
    if estimator == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000, random_state=seed)
    return Lasso(alpha=alpha, max_iter=5000, random_state=seed)


def _coef_magnitudes(model, n_features: int) -> np.ndarray:
    """Absolute coefficient per feature, robust to sklearn shape conventions."""
    coef = np.asarray(model.coef_).ravel()
    mag = np.abs(coef)
    if mag.shape[0] != n_features:  # defensive
        out = np.zeros(n_features)
        out[: mag.shape[0]] = mag[:n_features]
        return out
    return mag


class MaskingWrapper(ModelWrapper):
    """Iterative-masking sequential feature selection.

    Parameters
    ----------
    estimator : {'lasso', 'elasticnet', 'logistic'}
        Base sparse learner. For classification an L1-logistic model is always
        used; ``estimator`` then only labels the run.
    n_solutions : int
        Number of sequential solutions to peel off.
    sparsity : int
        Features taken per solution (top-|coef|).
    task : {'classification', 'regression'}
    alpha : float
        Regularization strength of the base learner.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        estimator: str = "lasso",
        n_solutions: int = 6,
        sparsity: int = 5,
        task: str = "regression",
        alpha: float = 0.01,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.estimator = estimator.lower()
        self.n_solutions = n_solutions
        self.sparsity = sparsity
        self.task = task
        self.alpha = alpha
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        Xs = _prepare(X)
        y = np.asarray(y).ravel()
        n_features = Xs.shape[1]

        available = np.ones(n_features, dtype=bool)
        results: Dict[str, Dict[str, List[int]]] = {}

        for k in range(self.n_solutions):
            cols = np.where(available)[0]
            if cols.size == 0:
                break
            model = _make_estimator(self.estimator, self.task, self.alpha, self.seed)
            try:
                model.fit(Xs[:, cols], y)
            except Exception:
                break
            mag = _coef_magnitudes(model, cols.size)
            if not np.any(mag > 0):
                # No signal left; fall back to top by univariate |corr| to stay defined.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mag = np.abs(
                        np.array([np.corrcoef(Xs[:, c], y)[0, 1] for c in cols])
                    )
                mag = np.nan_to_num(mag, nan=0.0)
            top_local = np.argsort(mag)[::-1][: self.sparsity]
            support = [int(cols[i]) for i in top_local]
            results[f"solution_{k}"] = {"support": support}
            available[support] = False  # mask (peel) these features

        if not results:
            results["solution_0"] = {"support": []}
        return results


class RandomizedLassoEnsembleWrapper(ModelWrapper):
    """Strong cheap baseline: Meinshausen-Buhlmann randomised Lasso run as a
    bootstrap ensemble, then CLUSTERED into m consensus solutions.

    The devil's-advocate competitor to GEMSS: for ``n_restarts`` bootstraps, fit
    L1-logistic with random per-restart feature reweighting, take the top-D
    support; agglomerative-cluster the support cloud (Jaccard) into ``n_solutions``
    groups and return each cluster's top-D consensus features. Returns m solutions
    like GEMSS, cheaply -- the test is whether they are *all* predictive.
    """

    def __init__(self, n_solutions: int = 6, sparsity: int = 5, task: str = "classification",
                 alpha: float = 0.05, n_restarts: int = 300, wmin: float = 0.2,
                 seed: int = 42, **kwargs: Any) -> None:
        self.n_solutions = n_solutions
        self.sparsity = sparsity
        self.task = task
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.wmin = wmin
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        from collections import Counter
        from sklearn.cluster import AgglomerativeClustering

        Xs = _prepare(X)
        y = np.asarray(y).ravel()
        rng = np.random.default_rng(self.seed)
        n, p = Xs.shape
        supports = []
        for b in range(self.n_restarts):
            idx = rng.choice(n, size=n, replace=True)
            Xb = Xs[idx] * rng.uniform(self.wmin, 1.0, size=p)
            model = _make_estimator(self.estimator, self.task, self.alpha, self.seed + b)
            try:
                model.fit(Xb, y[idx])
            except Exception:
                continue
            mag = _coef_magnitudes(model, p)
            supports.append(tuple(sorted(int(i) for i in np.argsort(mag)[::-1][: self.sparsity])))
        uniq = list(set(supports))
        if not uniq:
            return {"solution_0": {"support": []}}
        if len(uniq) <= self.n_solutions:
            return {f"solution_{i}": {"support": list(s)} for i, s in enumerate(uniq)}
        M = np.zeros((len(uniq), p))
        for i, s in enumerate(uniq):
            M[i, list(s)] = 1.0
        try:
            lab = AgglomerativeClustering(n_clusters=self.n_solutions, metric="jaccard",
                                          linkage="average").fit_predict(M)
        except Exception:
            lab = AgglomerativeClustering(n_clusters=self.n_solutions).fit_predict(M)
        results = {}
        for c in range(self.n_solutions):
            members = [uniq[i] for i in range(len(uniq)) if lab[i] == c]
            if not members:
                continue
            cnt = Counter(f for s in members for f in s)
            results[f"solution_{c}"] = {"support": [f for f, _ in cnt.most_common(self.sparsity)]}
        return results or {"solution_0": {"support": list(uniq[0])}}

    estimator = "logistic"


class StabilitySelectionWrapper(ModelWrapper):
    """Meinshausen-Buhlmann stability selection -> a single stable core.

    Subsamples the data ``n_bootstrap`` times, fits an L1 model on each subsample,
    and selects the features whose selection frequency exceeds ``threshold``
    (capped at ``max_features`` by frequency). Returns one solution -- the point of
    the baseline is precisely that it collapses alternatives into a stable core.

    Parameters
    ----------
    n_solutions : int
        Accepted for interface symmetry; stability selection returns one core.
    sparsity : int
        Upper bound on the size of the returned core.
    task : {'classification', 'regression'}
    alpha : float
        L1 strength for the base learner.
    n_bootstrap : int
        Number of subsamples.
    sample_fraction : float
        Fraction of samples per subsample.
    threshold : float
        Minimum selection frequency to keep a feature.
    seed : int
    """

    def __init__(
        self,
        n_solutions: int = 1,
        sparsity: int = 5,
        task: str = "regression",
        alpha: float = 0.02,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.sparsity = sparsity
        self.task = task
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, List[int]]]:
        Xs = _prepare(X)
        y = np.asarray(y).ravel()
        n_samples, n_features = Xs.shape
        counts = np.zeros(n_features)
        n_sub = max(2, int(self.sample_fraction * n_samples))

        for b in range(self.n_bootstrap):
            idx = resample(
                np.arange(n_samples),
                replace=False,
                n_samples=n_sub,
                random_state=self.seed + b,
            )
            model = _make_estimator(self.estimator_name, self.task, self.alpha, self.seed + b)
            try:
                model.fit(Xs[idx], y[idx])
            except Exception:
                continue
            mag = _coef_magnitudes(model, n_features)
            counts += (mag > 1e-8).astype(float)

        freq = counts / max(self.n_bootstrap, 1)
        selected = np.where(freq >= self.threshold)[0]
        # Cap to the most stable `sparsity` features.
        if selected.size > self.sparsity:
            order = np.argsort(freq[selected])[::-1][: self.sparsity]
            selected = selected[order]
        support = [int(i) for i in selected]
        return {"solution_0": {"support": support}}

    estimator_name = "lasso"
