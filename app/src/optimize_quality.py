#!/usr/bin/env python3
"""Surrogate-based process parameter optimization for FDM/MEX quality models.

Each trained sklearn Pipeline (from run_quality_analysis) can be wrapped as an
objective function and optimised over the 4 FDM input variables using an
algorithm matched to the model type:

  Model          Primary            Secondary
  ─────────────────────────────────────────────
  mlp_small      PSO                DE, Bayesian, Gradient
  svr_rbf        Bayesian           PSO, DE
  elastic_net    Bayesian / DOE     Gradient
  gpr            Bayesian-GPR(EI)   PSO, DE
  knn_reg        Bayesian           PSO
  svc_rbf        Bayesian           PSO
  knn_clf        Bayesian

Usage:
    result = optimize_single(fitted_pipeline, "svr_rbf", x_df, y, "minimize", bounds)
    mresult = optimize_multi({"obj1": pipe1, "obj2": pipe2}, "svr_rbf", x_df,
                             {"obj1": "minimize", "obj2": "minimize"}, bounds)
"""
from __future__ import annotations

import time
import warnings
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Algorithm registry  (ordered: first entry = UI default)
# ---------------------------------------------------------------------------
OPTIMIZER_REGISTRY: dict[str, list[str]] = {
    "mlp_small":  ["pso", "de", "bayesian", "gradient"],
    "svr_rbf":    ["bayesian", "pso", "de"],
    "elastic_net": ["doe", "bayesian", "gradient"],
    "gpr":        ["bayesian_gpr", "pso", "de"],
    "knn_reg":    ["bayesian", "pso"],
    "svc_rbf":    ["bayesian", "pso"],
    "knn_clf":    ["bayesian"],
}

ALGO_LABELS: dict[str, str] = {
    "pso":         "PSO (Particle Swarm)",
    "de":          "DE (Differential Evolution)",
    "bayesian":    "Bayesian Optimization (GP)",
    "bayesian_gpr": "Bayesian Opt — GPR canonical (EI)",
    "doe":         "DOE / LHS (Latin Hypercube)",
    "gradient":    "Gradient (L-BFGS-B)",
}

# KNN models need CV-based evaluation (fragile on small neighbourhoods)
_KNN_MODELS = {"knn_reg", "knn_clf"}

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class OptimizationResult(TypedDict):
    optimal_inputs: dict[str, float]     # col_name → value
    predicted_value: float | None
    algo_used: str
    convergence: list[float]             # best-so-far at each iteration/eval
    n_evaluations: int
    duration_sec: float
    error: str | None


class MultiOptimizationResult(TypedDict):
    pareto_inputs: list[dict[str, float]]
    pareto_predicted: list[dict[str, float]]  # obj_name → value
    hypervolume: float | None
    n_evaluations: int
    error: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_data_bounds(x_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return {col: (min, max)} derived from training data."""
    bounds: dict[str, tuple[float, float]] = {}
    for col in x_df.columns:
        series = pd.to_numeric(x_df[col], errors="coerce").dropna()
        if len(series) > 0:
            bounds[col] = (float(series.min()), float(series.max()))
    return bounds


def _bounds_list(bounds: dict[str, tuple[float, float]],
                 col_names: list[str]) -> list[tuple[float, float]]:
    return [bounds[c] for c in col_names]


def _arr_to_df(x: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(x.reshape(1, -1), columns=col_names)


# ---------------------------------------------------------------------------
# Objective function builders
# ---------------------------------------------------------------------------

def make_objective_fn(
    pipeline: Pipeline,
    col_names: list[str],
    direction: str,
    *,
    model_name: str = "",
    x_df: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    cv_folds: int = 5,
):
    """Return (scalar_fn, vectorized_fn) for use by optimizers.

    For KNN models the scalar_fn wraps a cross-validated RMSE/accuracy to
    avoid artefacts from small neighbourhood predictions on ~50 samples.

    direction: "minimize" → objective is the raw predicted value
               "maximize" → objective is negated (optimizers always minimise)
    """
    sign = 1.0 if direction == "minimize" else -1.0
    use_cv = model_name in _KNN_MODELS and x_df is not None and y is not None

    def _scalar(x1d: np.ndarray) -> float:
        if use_cv:
            # Temporarily re-fit with candidate point appended, then CV-score
            # This is expensive but correct for KNN fragility avoidance.
            # We fall back to plain predict when CV fails.
            try:
                x_aug = pd.concat(
                    [x_df, _arr_to_df(x1d, col_names)], ignore_index=True
                )
                y_aug = pd.concat(
                    [y, pd.Series([np.nan], name=y.name)], ignore_index=True
                )
                valid = y_aug.notna()
                x_v = x_aug.loc[valid].reset_index(drop=True)
                y_v = y_aug.loc[valid].reset_index(drop=True)
                if len(x_v) < cv_folds + 1:
                    raise ValueError("Not enough rows for CV")
                scores = cross_val_score(
                    pipeline, x_v, y_v,
                    cv=min(cv_folds, len(x_v)),
                    scoring=("neg_root_mean_squared_error"
                             if hasattr(pipeline.named_steps.get("model"), "predict")
                             else "accuracy"),
                    n_jobs=1,
                )
                return float(sign * (-np.mean(scores)))
            except Exception:  # noqa: BLE001
                pass
        # Standard path
        df_row = _arr_to_df(x1d, col_names)
        try:
            pred = pipeline.predict(df_row)
            val = float(np.asarray(pred).ravel()[0])
        except Exception:  # noqa: BLE001
            return 1e9
        return sign * val

    def _vectorized(X: np.ndarray) -> np.ndarray:
        """For PSO: X shape (n_particles, n_dim) → costs (n_particles,)."""
        return np.array([_scalar(X[i]) for i in range(X.shape[0])])

    return _scalar, _vectorized


# ---------------------------------------------------------------------------
# Optimizer runners
# ---------------------------------------------------------------------------

def run_pso(
    objective_fn_vec,
    bounds: dict[str, tuple[float, float]],
    col_names: list[str],
    n_particles: int = 30,
    n_iter: int = 100,
) -> tuple[np.ndarray, float, list[float]]:
    """Particle Swarm Optimization via pyswarms."""
    try:
        import pyswarms.single as ps  # type: ignore
    except ImportError as exc:
        raise ImportError("pyswarms is required: pip install pyswarms") from exc

    lb = np.array([bounds[c][0] for c in col_names])
    ub = np.array([bounds[c][1] for c in col_names])
    limits = (lb, ub)

    convergence: list[float] = []

    def _cost(X: np.ndarray) -> np.ndarray:
        costs = objective_fn_vec(X)
        convergence.append(float(np.min(costs)))
        return costs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = ps.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(col_names),
            options=options,
            bounds=limits,
        )
        best_cost, best_pos = optimizer.optimize(_cost, iters=n_iter, verbose=False)

    # Convert to running best
    running_best: list[float] = []
    cur_best = float("inf")
    for c in convergence:
        cur_best = min(cur_best, c)
        running_best.append(cur_best)

    return best_pos, float(best_cost), running_best


def run_de(
    objective_fn_scalar,
    bounds: dict[str, tuple[float, float]],
    col_names: list[str],
    max_iter: int = 100,
    popsize: int = 15,
) -> tuple[np.ndarray, float, list[float]]:
    """Differential Evolution via scipy."""
    from scipy.optimize import differential_evolution  # type: ignore

    scipy_bounds = [bounds[c] for c in col_names]
    convergence: list[float] = []

    def _callback(xk: np.ndarray, convergence_val: float) -> bool:
        convergence.append(float(objective_fn_scalar(xk)))
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = differential_evolution(
            objective_fn_scalar,
            bounds=scipy_bounds,
            maxiter=max_iter,
            popsize=popsize,
            seed=42,
            callback=_callback,
            tol=1e-6,
        )

    # Build running best from callback values
    running_best: list[float] = []
    cur_best = float("inf")
    for c in convergence:
        cur_best = min(cur_best, c)
        running_best.append(cur_best)
    if not running_best:
        running_best = [float(result.fun)]

    return result.x, float(result.fun), running_best


def run_bayesian_opt(
    objective_fn_scalar,
    bounds: dict[str, tuple[float, float]],
    col_names: list[str],
    n_calls: int = 50,
    n_initial_points: int = 10,
) -> tuple[np.ndarray, float, list[float]]:
    """Bayesian optimization via scikit-optimize (meta-GP surrogate).
    Falls back to scipy minimize with random restarts if skopt unavailable."""
    try:
        from skopt import gp_minimize  # type: ignore
        from skopt.space import Real  # type: ignore

        space = [Real(bounds[c][0], bounds[c][1], name=c) for c in col_names]
        convergence: list[float] = []

        def _wrapped(*args):
            val = objective_fn_scalar(np.array(args[0]))
            convergence.append(val)
            return float(val)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gp_minimize(
                _wrapped,
                space,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                random_state=42,
                noise=1e-10,
            )

        running_best: list[float] = []
        cur_best = float("inf")
        for c in convergence:
            cur_best = min(cur_best, c)
            running_best.append(cur_best)

        return np.array(res.x), float(res.fun), running_best

    except ImportError:
        # Fallback: random restarts + L-BFGS-B (equivalent to gradient opt)
        return run_gradient_opt(objective_fn_scalar, bounds, col_names, n_restarts=n_calls)


def run_bayesian_opt_gpr(
    gpr_pipeline: Pipeline,
    col_names: list[str],
    direction: str,
    bounds: dict[str, tuple[float, float]],
    n_iter: int = 50,
    n_initial: int = 10,
    n_candidates: int = 200,
) -> tuple[np.ndarray, float, list[float]]:
    """Canonical GPR Bayesian Optimization using Expected Improvement.

    The GPR pipeline IS the surrogate — its predict(return_std=True) provides
    (mu, sigma) which drive the EI acquisition function. No meta-surrogate is
    used, unlike run_bayesian_opt().
    """
    from scipy.stats import norm as sp_norm  # type: ignore

    sign = 1.0 if direction == "minimize" else -1.0
    scipy_bounds = [bounds[c] for c in col_names]
    rng = np.random.default_rng(42)
    n_dims = len(col_names)

    # Extract just the GPR estimator from the pipeline for predict with std
    # The pipeline has steps: prep → model (TransformedTargetRegressor wrapping GPR)
    def _gpr_predict(X: np.ndarray):
        """Return (mu_raw, sigma_raw) using the GPR pipeline's last step."""
        # We need the GPR regressor inside the pipeline to get return_std=True.
        # Navigate: pipeline["model"].regressor_ is the fitted GPR.
        model_step = gpr_pipeline.named_steps.get("model")
        prep_step = gpr_pipeline.named_steps.get("prep")

        try:
            # Transform input features
            if prep_step is not None:
                X_t = prep_step.transform(
                    pd.DataFrame(X, columns=col_names)
                )
            else:
                X_t = X

            # Get the underlying GPR (may be wrapped in TransformedTargetRegressor)
            if hasattr(model_step, "regressor_"):
                gpr = model_step.regressor_
                mu, sigma = gpr.predict(X_t, return_std=True)
                # Inverse-transform target if needed
                if hasattr(model_step, "transformer_") and hasattr(model_step.transformer_, "inverse_transform"):
                    mu = model_step.transformer_.inverse_transform(
                        mu.reshape(-1, 1)
                    ).ravel()
            else:
                # Fallback: use pipeline predict (no std)
                mu = gpr_pipeline.predict(pd.DataFrame(X, columns=col_names))
                sigma = np.ones(len(mu)) * 0.1
        except Exception:  # noqa: BLE001
            mu = gpr_pipeline.predict(pd.DataFrame(X, columns=col_names))
            sigma = np.ones(len(mu)) * 0.1

        return np.asarray(mu).ravel(), np.asarray(sigma).ravel()

    def _ei(X_cands: np.ndarray, best_so_far: float) -> np.ndarray:
        """Expected Improvement: EI(x) = (best-mu)*Phi(Z) + sigma*phi(Z)."""
        mu, sigma = _gpr_predict(X_cands)
        obj_mu = sign * mu  # apply direction sign
        sigma = np.maximum(sigma, 1e-9)
        Z = (best_so_far - obj_mu) / sigma
        ei = (best_so_far - obj_mu) * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)
        return ei

    # Seed with random LHS samples
    lb = np.array([bounds[c][0] for c in col_names])
    ub = np.array([bounds[c][1] for c in col_names])

    X_init = rng.uniform(lb, ub, size=(n_initial, n_dims))
    y_init = np.array([
        sign * float(np.asarray(gpr_pipeline.predict(
            pd.DataFrame(X_init[[i]], columns=col_names)
        )).ravel()[0])
        for i in range(n_initial)
    ])

    best_idx = int(np.argmin(y_init))
    best_x = X_init[best_idx].copy()
    best_y = float(y_init[best_idx])

    convergence: list[float] = [best_y]

    for _ in range(n_iter - n_initial):
        # Sample candidate points and pick the one with max EI
        X_cands = rng.uniform(lb, ub, size=(n_candidates, n_dims))
        ei_vals = _ei(X_cands, best_y)
        next_idx = int(np.argmax(ei_vals))
        x_next = X_cands[next_idx]

        # Evaluate objective at selected point
        y_next = sign * float(np.asarray(gpr_pipeline.predict(
            pd.DataFrame(x_next.reshape(1, -1), columns=col_names)
        )).ravel()[0])

        if y_next < best_y:
            best_y = y_next
            best_x = x_next.copy()

        convergence.append(best_y)

    # Convert sign back for returned value
    actual_best = sign * best_y
    return best_x, float(actual_best), [sign * v for v in convergence]


def run_doe(
    objective_fn_scalar,
    bounds: dict[str, tuple[float, float]],
    col_names: list[str],
    n_lhs: int = 50,
    n_refine: int = 25,
) -> tuple[np.ndarray, float, list[float]]:
    """DOE-style optimization: LHS phase + local grid refinement.

    Phase 1: Latin Hypercube Sampling over full bounds.
    Phase 2: Factorial grid refinement around the best LHS region.
    """
    from scipy.stats.qmc import LatinHypercube  # type: ignore
    from scipy.optimize import minimize  # type: ignore

    n_dims = len(col_names)
    lb = np.array([bounds[c][0] for c in col_names])
    ub = np.array([bounds[c][1] for c in col_names])

    convergence: list[float] = []

    # Phase 1: LHS
    sampler = LatinHypercube(d=n_dims, seed=42)
    lhs_unit = sampler.random(n=n_lhs)
    lhs_scaled = lb + lhs_unit * (ub - lb)

    lhs_values: list[float] = []
    cur_best = float("inf")
    best_x = lhs_scaled[0].copy()
    for xi in lhs_scaled:
        val = float(objective_fn_scalar(xi))
        lhs_values.append(val)
        if val < cur_best:
            cur_best = val
            best_x = xi.copy()
        convergence.append(cur_best)

    # Phase 2: Local refinement around best LHS point using L-BFGS-B
    # Define a neighbourhood: 20% of the full range centred on best_x
    radius = 0.2 * (ub - lb)
    local_lb = np.maximum(lb, best_x - radius)
    local_ub = np.minimum(ub, best_x + radius)
    refine_bounds = list(zip(local_lb.tolist(), local_ub.tolist()))

    rng = np.random.default_rng(42)
    best_refine_val = cur_best
    for _ in range(n_refine):
        x0 = rng.uniform(local_lb, local_ub)
        try:
            res = minimize(
                objective_fn_scalar, x0,
                method="L-BFGS-B",
                bounds=refine_bounds,
                options={"maxiter": 50, "ftol": 1e-8},
            )
            val = float(res.fun)
        except Exception:  # noqa: BLE001
            val = float(objective_fn_scalar(x0))
            res = type("R", (), {"x": x0, "fun": val})()

        if val < best_refine_val:
            best_refine_val = val
            best_x = np.asarray(res.x).ravel().copy()
        convergence.append(min(best_refine_val, cur_best))

    return best_x, float(min(best_refine_val, cur_best)), convergence


def run_gradient_opt(
    objective_fn_scalar,
    bounds: dict[str, tuple[float, float]],
    col_names: list[str],
    n_restarts: int = 20,
) -> tuple[np.ndarray, float, list[float]]:
    """Multi-start L-BFGS-B optimization. For Elastic Net and MLP fine-tuning."""
    from scipy.optimize import minimize  # type: ignore

    lb = np.array([bounds[c][0] for c in col_names])
    ub = np.array([bounds[c][1] for c in col_names])
    scipy_bounds = list(zip(lb.tolist(), ub.tolist()))
    rng = np.random.default_rng(42)

    best_val = float("inf")
    best_x = lb + (ub - lb) * 0.5
    convergence: list[float] = []

    for _ in range(n_restarts):
        x0 = rng.uniform(lb, ub)
        try:
            res = minimize(
                objective_fn_scalar, x0,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            val = float(res.fun)
        except Exception:  # noqa: BLE001
            val = float(objective_fn_scalar(x0))
            res = type("R", (), {"x": x0, "fun": val})()

        if val < best_val:
            best_val = val
            best_x = np.asarray(res.x).ravel().copy()
        convergence.append(best_val)

    return best_x, best_val, convergence


# ---------------------------------------------------------------------------
# Unified entry point — single-objective
# ---------------------------------------------------------------------------

def optimize_single(
    fitted_pipeline: Pipeline,
    model_name: str,
    x_df: pd.DataFrame,
    y: pd.Series,
    direction: str,
    bounds: dict[str, tuple[float, float]],
    *,
    algo: str | None = None,
    n_iter: int = 100,
) -> OptimizationResult:
    """Optimize a single objective using the best-matched algorithm.

    Parameters
    ----------
    fitted_pipeline : fitted sklearn Pipeline (prep + model)
    model_name      : key from OPTIMIZER_REGISTRY
    x_df            : training features (used for KNN CV objective and bounds)
    y               : training target (used for KNN CV objective)
    direction       : "minimize" or "maximize"
    bounds          : {col_name: (lo, hi)}; use get_data_bounds(x_df) as default
    algo            : override algorithm key; None = use first from registry
    n_iter          : budget (iterations for PSO/DE/Gradient, calls for BO)
    """
    col_names = list(x_df.columns)
    available = OPTIMIZER_REGISTRY.get(model_name, ["bayesian"])
    chosen = algo if algo in available else available[0]

    t0 = time.perf_counter()
    convergence: list[float] = []
    best_x: np.ndarray | None = None
    best_val: float | None = None
    error: str | None = None

    try:
        scalar_fn, vec_fn = make_objective_fn(
            fitted_pipeline, col_names, direction,
            model_name=model_name, x_df=x_df, y=y,
        )

        if chosen == "pso":
            best_x, best_val, convergence = run_pso(
                vec_fn, bounds, col_names,
                n_particles=min(30, max(10, n_iter // 5)),
                n_iter=n_iter,
            )
        elif chosen == "de":
            best_x, best_val, convergence = run_de(
                scalar_fn, bounds, col_names, max_iter=n_iter,
            )
        elif chosen == "bayesian_gpr":
            best_x, best_val, convergence = run_bayesian_opt_gpr(
                fitted_pipeline, col_names, direction, bounds,
                n_iter=n_iter, n_initial=min(10, n_iter // 5),
            )
        elif chosen == "doe":
            n_lhs = max(20, n_iter * 2 // 3)
            n_refine = max(10, n_iter // 3)
            best_x, best_val, convergence = run_doe(
                scalar_fn, bounds, col_names,
                n_lhs=n_lhs, n_refine=n_refine,
            )
        elif chosen == "gradient":
            best_x, best_val, convergence = run_gradient_opt(
                scalar_fn, bounds, col_names, n_restarts=n_iter,
            )
        else:  # "bayesian" (default / fallback)
            best_x, best_val, convergence = run_bayesian_opt(
                scalar_fn, bounds, col_names,
                n_calls=n_iter,
                n_initial_points=min(10, n_iter // 4),
            )

    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        best_x = np.array([
            (bounds[c][0] + bounds[c][1]) / 2.0 for c in col_names
        ])
        best_val = None

    duration = time.perf_counter() - t0

    optimal_inputs = {
        c: float(best_x[i]) for i, c in enumerate(col_names)
    } if best_x is not None else {}

    # Evaluate the pipeline at the optimal point for reporting
    predicted_value: float | None = None
    if best_x is not None and error is None:
        try:
            df_opt = _arr_to_df(best_x, col_names)
            pred = fitted_pipeline.predict(df_opt)
            predicted_value = float(np.asarray(pred).ravel()[0])
        except Exception:  # noqa: BLE001
            pass

    return OptimizationResult(
        optimal_inputs=optimal_inputs,
        predicted_value=predicted_value,
        algo_used=chosen,
        convergence=convergence,
        n_evaluations=len(convergence),
        duration_sec=round(duration, 3),
        error=error,
    )


# ---------------------------------------------------------------------------
# Multi-objective via NSGA-II (pymoo)
# ---------------------------------------------------------------------------

def optimize_multi(
    pipelines_dict: dict[str, Pipeline],
    model_name: str,
    x_df: pd.DataFrame,
    directions_dict: dict[str, str],
    bounds: dict[str, tuple[float, float]],
    *,
    n_gen: int = 50,
    pop_size: int = 40,
) -> MultiOptimizationResult:
    """Multi-objective optimization using NSGA-II (pymoo).

    Discovers the Pareto front across multiple quality objectives.

    Parameters
    ----------
    pipelines_dict  : {obj_name: fitted_pipeline}
    model_name      : common model type key (used for algorithm routing)
    x_df            : training features (columns = input variable names)
    directions_dict : {obj_name: "minimize"|"maximize"}
    bounds          : {col_name: (lo, hi)}
    n_gen           : number of NSGA-II generations
    pop_size        : population size

    Returns
    -------
    MultiOptimizationResult with pareto_inputs and pareto_predicted lists.
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore
        from pymoo.core.problem import Problem  # type: ignore
        from pymoo.optimize import minimize as pymoo_minimize  # type: ignore
        from pymoo.termination import get_termination  # type: ignore
        from pymoo.indicators.hv import HV  # type: ignore
    except ImportError as exc:
        return MultiOptimizationResult(
            pareto_inputs=[],
            pareto_predicted={},
            hypervolume=None,
            n_evaluations=0,
            error=f"pymoo is required: pip install pymoo — {exc}",
        )

    col_names = list(x_df.columns)
    obj_names = list(pipelines_dict.keys())
    n_obj = len(obj_names)
    n_var = len(col_names)

    lb = np.array([bounds[c][0] for c in col_names])
    ub = np.array([bounds[c][1] for c in col_names])
    signs = np.array([
        1.0 if directions_dict.get(name, "minimize") == "minimize" else -1.0
        for name in obj_names
    ])

    class _FDMProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=0,
                xl=lb,
                xu=ub,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((len(X), n_obj))
            for j, obj_name in enumerate(obj_names):
                pipe = pipelines_dict[obj_name]
                df_X = pd.DataFrame(X, columns=col_names)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        preds = np.asarray(pipe.predict(df_X)).ravel()
                    except Exception:  # noqa: BLE001
                        preds = np.zeros(len(X))
                F[:, j] = signs[j] * preds
            out["F"] = F

    problem = _FDMProblem()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algorithm = NSGA2(pop_size=pop_size)
        termination = get_termination("n_gen", n_gen)
        res = pymoo_minimize(
            problem, algorithm, termination, seed=42, verbose=False
        )

    if res.X is None or len(res.X) == 0:
        return MultiOptimizationResult(
            pareto_inputs=[],
            pareto_predicted=[],
            hypervolume=None,
            n_evaluations=pop_size * n_gen,
            error="NSGA-II returned no Pareto solutions.",
        )

    pareto_inputs = [
        {c: float(res.X[i, j]) for j, c in enumerate(col_names)}
        for i in range(len(res.X))
    ]

    # Invert signs to get actual objective values
    F_actual = res.F * signs  # undo sign flip
    pareto_predicted = [
        {obj_names[j]: float(F_actual[i, j]) for j in range(n_obj)}
        for i in range(len(res.F))
    ]

    # Compute hypervolume with a reference point = worst + 10% margin
    hv_val: float | None = None
    try:
        ref_point = F_actual.max(axis=0) * 1.1
        # HV requires a minimization framing; re-sign if maximizing
        F_for_hv = res.F.copy()  # already in minimization framing (signs applied)
        ref_for_hv = F_for_hv.max(axis=0) * 1.1
        hv_indicator = HV(ref_point=ref_for_hv)
        hv_val = float(hv_indicator.do(F_for_hv))
    except Exception:  # noqa: BLE001
        pass

    return MultiOptimizationResult(
        pareto_inputs=pareto_inputs,
        pareto_predicted=pareto_predicted,
        hypervolume=hv_val,
        n_evaluations=pop_size * n_gen,
        error=None,
    )
