#!/usr/bin/env python3
"""Interactive Flask web app for MEX/FDM quality analysis.

Run:
    python app.py [--data PATH] [--port 5000]

The browser UI lets you choose which algorithms to run, configure their
hyperparameters, and launch the analysis without touching the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent / "src"))

import io
import itertools
import json
import math
import re as _re
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, Response, jsonify, request, stream_with_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate as _cross_validate, train_test_split as _tt_split
from sklearn.metrics import (
    mean_absolute_error as _mae, mean_squared_error as _mse, r2_score as _r2,
    accuracy_score as _acc, f1_score as _f1, balanced_accuracy_score as _bal_acc,
)

from run_quality_analysis import (
    DEFAULT_CLS_CONFIGS,
    DEFAULT_REG_CONFIGS,
    EXPERIMENT_COLUMN,
    INPUT_COLUMNS,
    MLP_SWEEP_GRID,
    PRECISION_RENAME_MAP,
    ROUGHNESS_BIAS_EXPERIMENTS,
    ROUGHNESS_TARGET_COLUMNS,
    apply_user_mapping,
    build_preprocessor,
    classification_models,
    discretize_target,
    evaluate_classification,
    evaluate_regression,
    html_table,
    make_eda_charts,
    normalize_dataframe_types,
    read_sheet_clean,
    regression_models,
    slugify,
)
from optimize_quality import (
    OPTIMIZER_REGISTRY as _OPT_REGISTRY,
    ALGO_LABELS as _ALGO_LABELS,
    get_data_bounds,
    optimize_single,
    optimize_multi,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*should be used with.*sklearn\.utils\.parallel\.Parallel.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Algorithm registry – drives both the UI and parameter validation
# ---------------------------------------------------------------------------
ALGO_REGISTRY: dict[str, dict[str, Any]] = {
    "regression": {
        "elastic_net": {
            "label": "Elastic Net",
            "params": [
                {"id": "alpha", "label": "Alpha (regularización)", "type": "sweep-float",
                 "default": "0.01",
                 "help": "0.01  ·  0.001,0.01,0.1  ·  0.001-0.1"},
                {"id": "l1_ratio", "label": "L1 Ratio (0–1)", "type": "sweep-float",
                 "default": "0.5",
                 "help": "0.5  ·  0.1,0.5,0.9"},
            ],
        },
        "svr_rbf": {
            "label": "SVR (RBF)",
            "params": [
                {"id": "C", "label": "C (regularización)", "type": "sweep-float",
                 "default": "10.0",
                 "help": "10  ·  1,10,100"},
                {"id": "epsilon", "label": "Epsilon", "type": "sweep-float",
                 "default": "0.05",
                 "help": "0.05  ·  0.01,0.05,0.1"},
                {"id": "gamma", "label": "Gamma", "type": "sweep-text",
                 "default": "scale",
                 "help": "scale  ·  scale,auto  ·  scale,auto,0.1"},
            ],
        },
        "knn_reg": {
            "label": "KNN Regressor",
            "params": [
                {"id": "n_neighbors", "label": "Nº vecinos", "type": "sweep-int",
                 "default": "5",
                 "help": "5  ·  3,5,7,9  ·  2-8"},
                {"id": "weights", "label": "Ponderación", "type": "multiselect",
                 "default": ["distance"], "options": ["uniform", "distance"]},
            ],
        },
        "gpr": {
            "label": "Proceso Gaussiano (GPR)",
            "params": [
                {"id": "alpha", "label": "Alpha (nivel de ruido)", "type": "sweep-float",
                 "default": "1e-6",
                 "help": "1e-6  ·  1e-6,1e-4,1e-2"},
            ],
        },
        "mlp_small": {
            "label": "MLP (Red Neuronal)",
            "params": [
                {"id": "hidden_layer_sizes", "label": "Capas ocultas",
                 "type": "sweep-layers", "default": "32;16",
                 "help": "32;16  ·  8-32;16  ·  4,8,16,32;4,8,16"},
                {"id": "activation", "label": "Activación",
                 "type": "multiselect", "default": ["relu"],
                 "options": ["relu", "tanh", "logistic", "identity"]},
                {"id": "alpha", "label": "Alpha (regularización L2)",
                 "type": "sweep-float", "default": "0.0001",
                 "help": "0.0001  ·  0.0001,0.001,0.01  ·  0.0001-0.01"},
                {"id": "learning_rate_init", "label": "Tasa de aprendizaje inicial",
                 "type": "sweep-float", "default": "0.001",
                 "help": "0.001  ·  0.0001,0.001,0.005"},
            ],
        },
    },
    "classification": {
        "svc_rbf": {
            "label": "SVC (RBF)",
            "params": [
                {"id": "C", "label": "C (regularización)", "type": "sweep-float",
                 "default": "10.0",
                 "help": "10  ·  1,10,100"},
                {"id": "gamma", "label": "Gamma", "type": "sweep-text",
                 "default": "scale",
                 "help": "scale  ·  scale,auto  ·  scale,auto,0.1"},
            ],
        },
        "knn_clf": {
            "label": "KNN Clasificador",
            "params": [
                {"id": "n_neighbors", "label": "Nº vecinos", "type": "sweep-int",
                 "default": "5",
                 "help": "5  ·  3,5,7,9  ·  2-8"},
                {"id": "weights", "label": "Ponderación", "type": "multiselect",
                 "default": ["distance"], "options": ["uniform", "distance"]},
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# Sweep param parsing helpers
# ---------------------------------------------------------------------------


def _powerset_cols(cols: list[str]):
    """Yield every non-empty subset of cols, ordered by size then position."""
    for r in range(1, len(cols) + 1):
        yield from itertools.combinations(cols, r)


def _iter_combos(
    output_combos_raw: list[list[str]],
    pool: list[str],
    roughness: bool,
) -> list[tuple[str, ...]]:
    """Return the combos to run for this pool (precision or roughness).

    If output_combos_raw is provided, filter to combos whose columns all
    belong to pool.  If empty/absent, fall back to the full powerset.
    """
    if not pool:
        return []
    if output_combos_raw:
        return [tuple(c) for c in output_combos_raw if c and all(col in pool for col in c)]
    return list(_powerset_cols(pool))


_INT_RANGE_RE = _re.compile(r'^(\d+)-(\d+)$')
_FLOAT_RANGE_RE = _re.compile(
    r'^([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)-([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$'
)


def _parse_sweep_tokens(raw: str, is_float: bool = False) -> list:
    """Parse a sweep spec: single value, comma-list, or a-b range.

    Integer ranges yield powers of 2 in [a, b].
    Float ranges yield powers of 10 in [a, b].
    """
    raw = raw.strip()
    if ',' in raw:
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        return [float(p) if is_float else int(float(p)) for p in parts]
    pat = _FLOAT_RANGE_RE if is_float else _INT_RANGE_RE
    m = pat.match(raw)
    if m:
        if is_float:
            a, b = float(m.group(1)), float(m.group(2))
            lo = math.floor(math.log10(a)) if a > 0 else 0
            hi = math.ceil(math.log10(b)) if b > 0 else 0
            vals = [10.0 ** e for e in range(lo, hi + 1) if a <= 10.0 ** e <= b]
            return vals or [a]
        else:
            a, b = int(m.group(1)), int(m.group(2))
            vals, v = [], 1
            while v <= b:
                if v >= a:
                    vals.append(v)
                v *= 2
            return vals or list(range(a, b + 1))
    return [float(raw) if is_float else int(float(raw))]


def _coerce_text_val(v: str) -> Any:
    """Try to convert a sweep-text token to float, else keep as str."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _parse_algo_sweep_params(
    algo_id: str, algo_type: str, cfg: dict
) -> tuple[dict | None, dict]:
    """Parse any algorithm config into (sweep_grid | None, single_config).

    Handles all param types: sweep-float, sweep-int, sweep-layers,
    sweep-text, multiselect, number, text, select.
    """
    param_defs = ALGO_REGISTRY[algo_type][algo_id]["params"]
    grid: dict[str, list] = {}
    single: dict[str, Any] = {}
    is_sweep = False

    for pdef in param_defs:
        pid = pdef["id"]
        ptype = pdef["type"]
        val = cfg.get(pid, pdef.get("default"))

        if ptype == "sweep-float":
            vals = _parse_sweep_tokens(str(val), is_float=True)
            grid[pid] = vals
            single[pid] = vals[0]
            if len(vals) > 1:
                is_sweep = True

        elif ptype == "sweep-int":
            vals = _parse_sweep_tokens(str(val), is_float=False)
            grid[pid] = [int(v) for v in vals]
            single[pid] = int(vals[0])
            if len(vals) > 1:
                is_sweep = True

        elif ptype == "sweep-layers":
            layer_specs = [t.strip() for t in str(val).split(";") if t.strip()]
            layer_options = [_parse_sweep_tokens(s, is_float=False) for s in layer_specs]
            hls = list(itertools.product(*layer_options))
            grid[pid] = hls
            single[pid] = hls[0]
            if len(hls) > 1:
                is_sweep = True

        elif ptype == "sweep-text":
            parts = [p.strip() for p in str(val).split(",") if p.strip()]
            if not parts:
                parts = [str(val)]
            coerced = [_coerce_text_val(p) for p in parts]
            grid[pid] = coerced
            single[pid] = coerced[0]
            if len(coerced) > 1:
                is_sweep = True

        elif ptype == "multiselect":
            vals = val if isinstance(val, list) else [val]
            grid[pid] = vals
            single[pid] = vals[0]
            if len(vals) > 1:
                is_sweep = True

        else:
            # number, text, select — single value, no sweep
            grid[pid] = [val]
            single[pid] = val

    return (grid if is_sweep else None), single


# ---------------------------------------------------------------------------
# Generic sweep runner and chart helpers
# ---------------------------------------------------------------------------

def _make_sweep_charts(sweep_df: pd.DataFrame) -> list[str]:
    """Build bar charts for any algorithm sweep results."""
    if sweep_df.empty:
        return []
    charts: list[str] = []
    cfg = {"responsive": True}

    _non_param = {
        "objective", "rank", "score",
        "cv_mae", "cv_rmse", "cv_r2", "cv_rmse_std",
        "holdout_mae", "holdout_rmse", "holdout_r2",
        "cv_accuracy", "cv_f1_macro", "cv_balanced_accuracy",
        "holdout_accuracy", "holdout_f1_macro", "holdout_balanced_accuracy",
    }
    param_cols = [c for c in sweep_df.columns if c not in _non_param]
    is_reg = "cv_rmse" in sweep_df.columns

    for objective in sweep_df["objective"].unique():
        subset = sweep_df[sweep_df["objective"] == objective].head(20).copy()
        subset["config"] = subset.apply(
            lambda r: " | ".join(f"{c}={r.get(c, '?')}" for c in param_cols), axis=1
        )

        if is_reg:
            metric_defs = [
                ("cv_rmse",      "CV RMSE",      "#1f77b4"),
                ("holdout_rmse", "Holdout RMSE", "#ff7f0e"),
                ("cv_mae",       "CV MAE",       "#2ca02c"),
                ("holdout_mae",  "Holdout MAE",  "#d62728"),
            ]
        else:
            metric_defs = [
                ("cv_f1_macro",          "CV F1 Macro",       "#1f77b4"),
                ("holdout_f1_macro",     "Holdout F1 Macro",  "#ff7f0e"),
                ("cv_accuracy",          "CV Accuracy",       "#2ca02c"),
                ("holdout_accuracy",     "Holdout Accuracy",  "#d62728"),
            ]

        fig = _go.Figure()
        for col, name, color in metric_defs:
            if col not in subset.columns:
                continue
            err = subset["cv_rmse_std"].tolist() if (col == "cv_rmse" and "cv_rmse_std" in subset.columns) else None
            fig.add_trace(_go.Bar(
                name=name, x=subset[col], y=subset["config"],
                orientation="h", marker_color=color,
                error_x=dict(type="data", array=err) if err else None,
                hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.4f}}<extra></extra>",
            ))
        title_sfx = "RMSE (↓)" if is_reg else "F1 Macro (↑)"
        fig.update_layout(
            title=f"Sweep — {objective} — Top 20 · {title_sfx}",
            xaxis_title=title_sfx,
            yaxis=dict(autorange="reversed"),
            barmode="group",
            height=max(420, 34 * len(subset)),
            margin=dict(l=300),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        charts.append(_pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg))

    return charts


def _run_generic_sweep_task(
    task_id: str, label: str, obj_name: str,
    algo_id: str, task_type: str,
    x_df: pd.DataFrame, y: pd.Series,
    param_grid: dict,
    class_q: int = 3,
) -> dict:
    """Grid-search any algorithm over its param_grid."""
    tab_id = ("reg" if task_type == "regression" else "cls") + "-" + algo_id
    try:
        if task_type == "classification":
            y_class = discretize_target(pd.to_numeric(y, errors="coerce"), q=class_q)
            if y_class is None:
                raise ValueError("No se puede discretizar el objetivo.")
            valid_idx = y_class.dropna().index
            x_df = x_df.loc[valid_idx].reset_index(drop=True)
            y_use = y_class.loc[valid_idx].astype(int).reset_index(drop=True)
        else:
            valid_mask = pd.to_numeric(y, errors="coerce").notna()
            x_df = x_df.loc[valid_mask].reset_index(drop=True)
            y_use = pd.to_numeric(y, errors="coerce").loc[valid_mask].reset_index(drop=True)

        if len(x_df) < 10:
            raise ValueError("Datos insuficientes para el sweep.")

        preprocessor = build_preprocessor(x_df)
        single_cfg = {k: (v[0] if isinstance(v, list) and v else v)
                      for k, v in param_grid.items()}

        if task_type == "regression":
            pipeline = list(regression_models(
                preprocessor, multi_output=False,
                enabled_configs={algo_id: single_cfg}
            ).values())[0]
            prefixed_grid = {f"model__regressor__{k}": v for k, v in param_grid.items()}
            scoring = {"mae": "neg_mean_absolute_error",
                       "rmse": "neg_root_mean_squared_error", "r2": "r2"}
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            pipeline = list(classification_models(
                preprocessor, enabled_configs={algo_id: single_cfg}
            ).values())[0]
            prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()}
            scoring = {"accuracy": "accuracy",
                       "f1_macro": "f1_macro",
                       "balanced_acc": "balanced_accuracy"}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        stratify = y_use if task_type == "classification" else None
        x_train, x_test, y_train, y_test = _tt_split(
            x_df, y_use, test_size=0.2, random_state=42, stratify=stratify
        )

        gs = GridSearchCV(
            pipeline, param_grid=prefixed_grid, cv=cv,
            scoring=scoring, refit=False, n_jobs=1,
            return_train_score=False, error_score=np.nan,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gs.fit(x_df, y_use)

        cv_res = pd.DataFrame(gs.cv_results_)
        key_prefix = "model__regressor__" if task_type == "regression" else "model__"
        param_cv_cols = [c for c in cv_res.columns if c.startswith(f"param_{key_prefix}")]
        rename_map = {c: c.replace(f"param_{key_prefix}", "") for c in param_cv_cols}

        if task_type == "regression":
            result = cv_res[
                param_cv_cols + ["mean_test_mae", "mean_test_rmse", "mean_test_r2", "std_test_rmse"]
            ].copy().rename(columns={
                **rename_map,
                "mean_test_mae": "cv_mae", "mean_test_rmse": "cv_rmse",
                "mean_test_r2": "cv_r2", "std_test_rmse": "cv_rmse_std",
            })
            result["cv_mae"] = -result["cv_mae"]
            result["cv_rmse"] = -result["cv_rmse"]
            result = result.dropna(subset=["cv_rmse"])
        else:
            result = cv_res[
                param_cv_cols + ["mean_test_accuracy", "mean_test_f1_macro", "mean_test_balanced_acc"]
            ].copy().rename(columns={
                **rename_map,
                "mean_test_accuracy": "cv_accuracy",
                "mean_test_f1_macro": "cv_f1_macro",
                "mean_test_balanced_acc": "cv_balanced_accuracy",
            })
            result = result.dropna(subset=["cv_f1_macro"])

        # Holdout metrics for each combination
        holdout_rows: list[dict] = []
        for _, row in result.iterrows():
            combo_cfg = {k: row[k] for k in rename_map.values() if k in row.index}
            try:
                prep2 = build_preprocessor(x_train)
                if task_type == "regression":
                    pipe2 = list(regression_models(
                        prep2, multi_output=False,
                        enabled_configs={algo_id: combo_cfg}
                    ).values())[0]
                else:
                    pipe2 = list(classification_models(
                        prep2, enabled_configs={algo_id: combo_cfg}
                    ).values())[0]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    pipe2.fit(x_train, y_train)
                y_pred = pipe2.predict(x_test)
                if task_type == "regression":
                    holdout_rows.append({
                        "holdout_mae":  float(_mae(y_test, y_pred)),
                        "holdout_rmse": float(math.sqrt(_mse(y_test, y_pred))),
                        "holdout_r2":   float(_r2(y_test, y_pred)),
                    })
                else:
                    holdout_rows.append({
                        "holdout_accuracy":          float(_acc(y_test, y_pred)),
                        "holdout_f1_macro":          float(_f1(y_test, y_pred, average="macro")),
                        "holdout_balanced_accuracy": float(_bal_acc(y_test, y_pred)),
                    })
            except Exception:  # noqa: BLE001
                if task_type == "regression":
                    holdout_rows.append({"holdout_mae": np.nan, "holdout_rmse": np.nan, "holdout_r2": np.nan})
                else:
                    holdout_rows.append({"holdout_accuracy": np.nan, "holdout_f1_macro": np.nan,
                                         "holdout_balanced_accuracy": np.nan})

        holdout_df = pd.DataFrame(holdout_rows, index=result.index)
        result = pd.concat([result, holdout_df], axis=1)

        # Score & rank
        if task_type == "regression":
            for col in ["cv_rmse", "holdout_rmse", "holdout_r2"]:
                result[col] = pd.to_numeric(result[col], errors="coerce")
            rr = result["cv_rmse"].max() - result["cv_rmse"].min()
            hr = result["holdout_rmse"].max() - result["holdout_rmse"].min()
            r2r = result["holdout_r2"].max() - result["holdout_r2"].min()
            nc = ((result["cv_rmse"] - result["cv_rmse"].min()) / rr) if rr > 0 else 0.0
            nh = ((result["holdout_rmse"] - result["holdout_rmse"].min()) / hr) if hr > 0 else 0.0
            nr = (1 - (result["holdout_r2"] - result["holdout_r2"].min()) / r2r) if r2r > 0 else 0.0
            result["score"] = 0.40 * nc + 0.40 * nh + 0.20 * nr
        else:
            result["cv_f1_macro"] = pd.to_numeric(result["cv_f1_macro"], errors="coerce")
            fr = result["cv_f1_macro"].max() - result["cv_f1_macro"].min()
            result["score"] = (
                (1 - (result["cv_f1_macro"] - result["cv_f1_macro"].min()) / fr)
                if fr > 0 else 0.0
            )

        result["rank"] = result["score"].rank(method="min", ascending=True).astype(int)
        result.insert(0, "objective", obj_name)
        result = result.sort_values("rank").reset_index(drop=True)

        table_html = html_table(result) if not result.empty else "<p>Sin resultados</p>"
        chart_html = _charts_html(_make_sweep_charts(result))

        return {"task_id": task_id, "label": label, "task": "algo_sweep",
                "obj": obj_name, "tab_id": tab_id,
                "table_html": table_html, "chart_html": chart_html, "error": None}

    except Exception as exc:  # noqa: BLE001
        return {"task_id": task_id, "label": label, "task": "algo_sweep",
                "obj": obj_name, "tab_id": tab_id,
                "table_html": "", "chart_html": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Per-combo sweep runner  (one CV + holdout cycle for a single param set)
# ---------------------------------------------------------------------------

def _run_single_combo_task(
    combo_id: str, tab_id: str, obj_name: str, sweep_key: str,
    algo_id: str, task_type: str,
    x_df: pd.DataFrame, y_use: pd.Series,
    x_train: pd.DataFrame, x_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    params: dict,
) -> dict:
    """Run 5-fold CV + holdout evaluation for a single parameter combination."""
    _base: dict = {
        "task": "sweep_combo",
        "task_id": combo_id,
        "combo_id": combo_id,
        "tab_id": tab_id,
        "obj": obj_name,
        "sweep_key": sweep_key,
        "algo_id": algo_id,
        "task_type": task_type,
        "params": _json_safe(params),
    }
    try:
        preprocessor = build_preprocessor(x_df)
        prep_train   = build_preprocessor(x_train)

        if task_type == "regression":
            pipelines = regression_models(preprocessor, multi_output=False,
                                          enabled_configs={algo_id: params})
            if not pipelines:
                raise ValueError(f"No pipeline built for {algo_id}")
            pipeline = next(iter(pipelines.values()))
            scoring  = {"mae": "neg_mean_absolute_error",
                        "rmse": "neg_root_mean_squared_error", "r2": "r2"}
            cv_obj   = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            pipelines = classification_models(preprocessor,
                                              enabled_configs={algo_id: params})
            if not pipelines:
                raise ValueError(f"No pipeline built for {algo_id}")
            pipeline = next(iter(pipelines.values()))
            scoring  = {"accuracy": "accuracy",
                        "f1_macro": "f1_macro",
                        "balanced_acc": "balanced_accuracy"}
            cv_obj   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv_res = _cross_validate(pipeline, x_df, y_use, cv=cv_obj,
                                     scoring=scoring, n_jobs=1,
                                     return_train_score=False)

        if task_type == "regression":
            cv_metrics: dict = {
                "cv_mae":      float(-np.mean(cv_res["test_mae"])),
                "cv_rmse":     float(-np.mean(cv_res["test_rmse"])),
                "cv_r2":       float(np.mean(cv_res["test_r2"])),
                "cv_rmse_std": float(np.std(-cv_res["test_rmse"])),
            }
        else:
            cv_metrics = {
                "cv_accuracy":          float(np.mean(cv_res["test_accuracy"])),
                "cv_f1_macro":          float(np.mean(cv_res["test_f1_macro"])),
                "cv_balanced_accuracy": float(np.mean(cv_res["test_balanced_acc"])),
            }

        # Holdout
        if task_type == "regression":
            pipe_h = next(iter(regression_models(
                prep_train, multi_output=False,
                enabled_configs={algo_id: params},
            ).values()))
        else:
            pipe_h = next(iter(classification_models(
                prep_train, enabled_configs={algo_id: params},
            ).values()))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipe_h.fit(x_train, y_train)
        y_pred = pipe_h.predict(x_test)

        if task_type == "regression":
            holdout_metrics: dict = {
                "holdout_mae":  float(_mae(y_test, y_pred)),
                "holdout_rmse": float(math.sqrt(_mse(y_test, y_pred))),
                "holdout_r2":   float(_r2(y_test, y_pred)),
            }
        else:
            holdout_metrics = {
                "holdout_accuracy":          float(_acc(y_test, y_pred)),
                "holdout_f1_macro":          float(_f1(y_test, y_pred, average="macro")),
                "holdout_balanced_accuracy": float(_bal_acc(y_test, y_pred)),
            }

        row = {**_json_safe(params), **cv_metrics, **holdout_metrics}
        return {**_base, "row": row, "error": None}

    except Exception as exc:  # noqa: BLE001
        return {**_base, "row": _json_safe(params), "error": str(exc)}


def _compute_sweep_charts_event(
    tab_id: str, obj: str, safe_key: str, task_type: str,
    rows: list[dict],
) -> dict:
    """Build ranking + chart HTML after all combos for a sweep group complete."""
    _base: dict = {"task": "algo_sweep_charts",
                   "tab_id": tab_id, "obj": obj, "safe_key": safe_key}
    if not rows:
        return {**_base, "chart_html": "", "error": None}
    try:
        result = pd.DataFrame(rows)
        if task_type == "regression":
            for col in ["cv_rmse", "holdout_rmse", "holdout_r2"]:
                result[col] = pd.to_numeric(result[col], errors="coerce")
            rr  = result["cv_rmse"].max()    - result["cv_rmse"].min()
            hr  = result["holdout_rmse"].max() - result["holdout_rmse"].min()
            r2r = result["holdout_r2"].max()  - result["holdout_r2"].min()
            nc = ((result["cv_rmse"]    - result["cv_rmse"].min())    / rr)  if rr  > 0 else 0.0
            nh = ((result["holdout_rmse"] - result["holdout_rmse"].min()) / hr)  if hr  > 0 else 0.0
            nr = (1 - (result["holdout_r2"] - result["holdout_r2"].min()) / r2r) if r2r > 0 else 0.0
            result["score"] = 0.40 * nc + 0.40 * nh + 0.20 * nr
        else:
            result["cv_f1_macro"] = pd.to_numeric(result["cv_f1_macro"], errors="coerce")
            fr = result["cv_f1_macro"].max() - result["cv_f1_macro"].min()
            result["score"] = (
                (1 - (result["cv_f1_macro"] - result["cv_f1_macro"].min()) / fr)
                if fr > 0 else 0.0
            )
        result["rank"] = result["score"].rank(method="min", ascending=True).astype(int)
        result.insert(0, "objective", obj)
        result = result.sort_values("rank").reset_index(drop=True)
        chart_html = _charts_html(_make_sweep_charts(result))
        return {**_base, "chart_html": chart_html, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {**_base, "chart_html": "", "error": str(exc)}


app = Flask(__name__)

_data_cache: tuple[pd.DataFrame, pd.DataFrame] | None = None
_raw_df_cache: pd.DataFrame | None = None
DATA_PATH = Path(__file__).parent / "data/data.xlsx"
SHEET_NAME = "Datos a utilizar en Minitab"

# Export session store: session_id → cached data
_export_store: dict[str, dict] = {}
_EXPORT_STORE_MAX = 8   # keep at most N sessions in RAM

# Column order for metric CSVs (mirrors JS REG_COLS / CLS_COLS)
_REG_METRIC_COLS = [
    "objective", "rank", "cv_mae", "cv_rmse", "train_rmse", "overfit_gap",
    "cv_r2", "cv_rmse_std", "holdout_mae", "holdout_rmse", "holdout_r2",
]
_CLS_METRIC_COLS = [
    "objective", "rank", "cv_accuracy", "cv_f1_macro", "cv_balanced_accuracy",
    "train_balanced_accuracy", "overfit_gap",
    "holdout_accuracy", "holdout_f1_macro", "holdout_balanced_accuracy",
]


def _load_raw_df() -> pd.DataFrame:
    """Load and normalise the raw DataFrame (no column-selection applied)."""
    global _raw_df_cache
    if _raw_df_cache is None:
        df = read_sheet_clean(DATA_PATH, SHEET_NAME)
        df = normalize_dataframe_types(df)
        df.columns = [str(c).strip() for c in df.columns]
        _raw_df_cache = df
    return _raw_df_cache.copy()


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _data_cache
    if _data_cache is None:
        df = read_sheet_clean(DATA_PATH, SHEET_NAME)
        df = normalize_dataframe_types(df)
        df.columns = [str(c).strip() for c in df.columns]
        df, x_df = apply_user_mapping(df)
        _data_cache = (df, x_df)
    df, x_df = _data_cache
    return df.copy(), x_df.copy()


def _prepare_data_dynamic(
    input_cols: list[str], output_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (full_df, x_df, y_df) from user-selected columns."""
    raw = _load_raw_df()
    all_needed = input_cols + output_cols
    missing = [c for c in all_needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas: {missing}")
    df = raw.copy()
    for c in all_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    x_df = df[input_cols].copy()
    return df, x_df, df[output_cols].copy()


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MEX/FDM Quality Analyzer</title>
<link rel="stylesheet" href="https://cdn.datatables.net/2.0.8/css/dataTables.dataTables.min.css"/>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/2.0.8/js/dataTables.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Tahoma,sans-serif;background:#f8fafc;color:#1f2937;display:flex;height:100vh;overflow:hidden}
.layout{display:flex;width:100%;height:100vh}
/* ---- Sidebar ---- */
.sidebar{width:270px;min-width:250px;background:#1e293b;color:#e2e8f0;display:flex;flex-direction:column;overflow-y:auto;padding:14px 12px;gap:10px}
.app-title{font-size:.95rem;font-weight:700;color:#f1f5f9;line-height:1.3}
.data-info{font-size:.72rem;color:#94a3b8;padding:5px 8px;background:#0f172a;border-radius:5px}
.section-head{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:4px}
.algo-card{background:#334155;border-radius:6px;padding:6px 8px;margin-bottom:4px;transition:background .15s}
.algo-card.enabled{background:#1d4ed8}
.algo-row{display:flex;align-items:center;gap:6px}
.algo-row input[type=checkbox]{width:13px;height:13px;cursor:pointer;flex-shrink:0;accent-color:#38bdf8}
.algo-label{flex:1;font-size:.82rem;cursor:pointer;user-select:none}
.cfg-btn{background:rgba(255,255,255,.15);border:none;color:#e2e8f0;border-radius:4px;padding:1px 7px;cursor:pointer;font-size:.74rem;line-height:1.6}
.cfg-btn:disabled{opacity:.35;cursor:not-allowed}
.cfg-btn:not(:disabled):hover{background:rgba(255,255,255,.3)}
.param-summary{font-size:.67rem;color:#93c5fd;margin-top:2px;padding-left:19px;word-break:break-all;min-height:.85rem}
.opt-row{display:flex;align-items:center;gap:6px;font-size:.82rem;margin-bottom:4px;cursor:pointer}
.opt-row input[type=checkbox]{accent-color:#38bdf8}
.opt-row input[type=number]{width:52px;padding:2px 4px;border-radius:4px;border:1px solid #475569;background:#334155;color:#e2e8f0;font-size:.79rem}
.run-btn{width:100%;padding:9px;background:#0ea5e9;border:none;border-radius:6px;color:#fff;font-size:.9rem;font-weight:600;cursor:pointer;margin-top:auto}
.run-btn:hover{background:#0284c7}
.run-btn:disabled{background:#475569;cursor:not-allowed}
/* ---- Main content ---- */
.content{flex:1;display:flex;flex-direction:column;overflow:hidden}
#placeholder{display:flex;align-items:center;justify-content:center;flex:1;color:#94a3b8;font-size:.95rem;text-align:center}
#spinner{display:none;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:12px;color:#64748b}
.ring{width:40px;height:40px;border:4px solid #e5e7eb;border-top-color:#0ea5e9;border-radius:50%;animation:spin .85s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* ---- Tabs ---- */
.tab-bar{display:flex;gap:2px;padding:8px 14px 0;background:#fff;border-bottom:2px solid #e5e7eb;flex-shrink:0;overflow-x:auto;scrollbar-width:thin}
.tab-btn{padding:6px 14px;border:none;background:none;font-size:.82rem;font-weight:500;color:#6b7280;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;white-space:nowrap;border-radius:4px 4px 0 0;transition:color .12s,border-color .12s}
.tab-btn:hover{color:#1f2937;background:#f8fafc}
.tab-btn.active{color:#0ea5e9;border-bottom-color:#0ea5e9;background:#f0f9ff}
.tab-panels{flex:1;overflow-y:auto;padding:12px 16px}
.tab-panel{display:none}
.tab-panel.active{display:block}
/* ---- Content inside tabs ---- */
.panel-section{margin-bottom:14px}
.panel-section h3{font-size:.88rem;color:#0f172a;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #e5e7eb}
.tbl{border-collapse:collapse;width:100%;font-size:.79rem;margin-bottom:8px}
.tbl th,.tbl td{border:1px solid #e5e7eb;padding:4px 8px;text-align:left}
.tbl th{background:#f1f5f9;font-weight:600}
tr:nth-child(even) td{background:#f8fafc}
.chart-wrap{margin:6px 0}
tfoot input{width:100%;box-sizing:border-box;font-size:.7rem;padding:2px 3px;margin-top:1px;border:1px solid #d1d5db;border-radius:3px}
.dataTables_wrapper{font-size:.79rem;margin-bottom:12px}
.err{color:#dc2626;padding:10px;background:#fef2f2;border-radius:5px;border-left:3px solid #dc2626}
/* ---- Modal ---- */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.48);display:flex;align-items:center;justify-content:center;z-index:900}
.overlay.hidden{display:none}
.modal{background:#fff;border-radius:10px;width:420px;max-width:95vw;max-height:80vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.28)}
.modal-hdr{display:flex;align-items:center;justify-content:space-between;padding:13px 16px;border-bottom:1px solid #e5e7eb}
.modal-hdr h3{font-size:.9rem;color:#0f172a}
.close-x{background:none;border:none;font-size:1rem;cursor:pointer;color:#6b7280;padding:2px 5px;border-radius:4px}
.close-x:hover{background:#f3f4f6}
.modal-body{padding:13px 16px;display:flex;flex-direction:column;gap:9px}
.f-field{display:flex;flex-direction:column;gap:3px}
.f-field label{font-size:.8rem;color:#374151;font-weight:500}
.f-field input,.f-field select{padding:5px 8px;border:1px solid #d1d5db;border-radius:5px;font-size:.83rem;color:#1f2937}
.f-field input:focus,.f-field select:focus{outline:none;border-color:#0ea5e9;box-shadow:0 0 0 2px rgba(14,165,233,.2)}
.modal-ftr{display:flex;justify-content:flex-end;gap:6px;padding:10px 16px;border-top:1px solid #e5e7eb}
.btn{padding:5px 13px;border-radius:5px;border:none;font-size:.81rem;cursor:pointer;font-weight:500}
.btn-primary{background:#0ea5e9;color:#fff}.btn-primary:hover{background:#0284c7}
.btn-sec{background:#f1f5f9;color:#475569;border:1px solid #e2e8f0}.btn-sec:hover{background:#e2e8f0}
.btn-ghost{background:none;color:#6b7280}.btn-ghost:hover{background:#f9fafb}
.ms-group{display:flex;flex-wrap:wrap;gap:6px 14px;padding:4px 0}
.ms-opt{display:flex;align-items:center;gap:5px;font-size:.82rem;color:#374151;cursor:pointer}
.ms-opt input[type=checkbox]{accent-color:#0ea5e9;width:14px;height:14px}
.f-hint{font-size:.71rem;color:#6b7280;font-style:italic;margin-top:2px}
/* ---- Progress bar ---- */
.progress-wrap{padding:6px 14px;background:#fff;border-bottom:1px solid #e5e7eb;flex-shrink:0;position:relative}
.progress-track{position:relative;height:22px;background:#e5e7eb;border-radius:11px;cursor:default;overflow:visible}
.progress-fill{height:100%;background:linear-gradient(90deg,#0ea5e9,#38bdf8);border-radius:11px;transition:width .35s ease;min-width:0}
.progress-label{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font-size:.71rem;font-weight:600;color:#1f2937;white-space:nowrap;pointer-events:none}
.progress-popup{display:none;position:absolute;top:calc(100% + 6px);left:0;right:0;max-height:280px;overflow-y:auto;background:#fff;border:1px solid #e5e7eb;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.14);z-index:800;padding:5px 0}
.progress-wrap:hover .progress-popup{display:block}
.ptask{display:flex;align-items:center;gap:8px;padding:3px 12px;font-size:.73rem;color:#374151;transition:background .1s}
.ptask:hover{background:#f8fafc}
.ptask.done .picon{color:#22c55e}
.ptask.pending .picon{color:#f59e0b}
.ptask.error .picon{color:#ef4444}
.picon{flex-shrink:0;font-size:.82rem}
.pname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* ---- Collapsible objective groups ---- */
details.obj-group{margin:4px 0;border:1px solid #e5e7eb;border-radius:6px;overflow:hidden}
details.obj-group>summary{padding:5px 10px;cursor:pointer;font-size:.79rem;font-weight:600;color:#0f172a;background:#f8fafc;user-select:none;list-style:none;display:flex;align-items:center;gap:6px}
details.obj-group>summary::before{content:"▶";font-size:.65rem;transition:transform .15s;flex-shrink:0}
details.obj-group[open]>summary::before{transform:rotate(90deg)}
details.obj-group>summary:hover{background:#f1f5f9}
details.obj-group>.obj-body{padding:4px 0}
/* ---- Column role table (inside variables modal) ---- */
.var-modal{max-width:560px;width:96vw}
.col-table-wrap{max-height:420px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:6px}
.col-table{width:100%;border-collapse:collapse;font-size:.82rem}
.col-table thead th{position:sticky;top:0;background:#f1f5f9;color:#374151;font-weight:600;padding:6px 10px;text-align:center;white-space:nowrap;border-bottom:2px solid #e5e7eb}
.col-table thead th:first-child{text-align:left;width:55%}
.col-table tbody tr:hover{background:#f8fafc}
.col-table tbody td{padding:5px 10px;vertical-align:middle;border-bottom:1px solid #f1f5f9}
.col-table tbody td:first-child{color:#1f2937;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:220px}
.col-table tbody td:not(:first-child){text-align:center}
.col-table input[type=radio]{accent-color:#0284c7;width:15px;height:15px;cursor:pointer}
.col-table tr.role-input td:first-child{color:#0369a1;font-weight:500}
.col-table tr.role-output td:first-child{color:#065f46;font-weight:500}
/* summary pill in sidebar */
.var-btn-row{display:flex;align-items:center;gap:6px}
.var-open-btn{flex:1;background:#334155;border:none;color:#e2e8f0;padding:5px 9px;border-radius:5px;font-size:.75rem;cursor:pointer;text-align:left;transition:background .15s}
.var-open-btn:hover{background:#475569}
.var-badge{font-size:.68rem;color:#94a3b8;margin-top:2px}
.col-sel-none{font-size:.7rem;color:#64748b;padding:3px 4px;font-style:italic}
/* ---- Output combinations section ---- */
.combo-section{margin-top:12px;border-top:1px solid #e5e7eb;padding-top:10px}
.combo-section-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.combo-section-title{font-size:.78rem;font-weight:600;color:#374151}
.combo-ctrl-btns{display:flex;gap:5px}
.combo-ctrl-btn{font-size:.7rem;color:#0284c7;background:none;border:none;cursor:pointer;padding:0 2px;text-decoration:underline}
.combo-list{max-height:200px;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
.combo-item{display:flex;align-items:center;gap:7px;padding:3px 6px;border-radius:4px;background:#f8fafc;font-size:.78rem;color:#1f2937;cursor:pointer}
.combo-item:hover{background:#f0f9ff}
.combo-item input[type=checkbox]{accent-color:#0284c7;width:14px;height:14px;cursor:pointer;flex-shrink:0}
.combo-item .combo-size{font-size:.67rem;color:#94a3b8;white-space:nowrap;margin-left:auto}
.combo-empty{font-size:.73rem;color:#94a3b8;font-style:italic;padding:4px 0}
/* ---- Metric header tooltip ---- */
th[data-tip]{position:relative;cursor:help;border-bottom:1px dashed #94a3b8}
th[data-tip]::after{content:attr(data-tip);position:absolute;left:50%;top:100%;transform:translateX(-50%);margin-top:4px;background:#1e293b;color:#f1f5f9;font-size:.72rem;font-weight:400;padding:6px 9px;border-radius:5px;white-space:pre-line;width:240px;z-index:9999;pointer-events:none;opacity:0;transition:opacity .15s;line-height:1.45;box-shadow:0 3px 12px rgba(0,0,0,.4)}
th[data-tip]:hover::after{opacity:1}
/* ---- Algo info button ---- */
.info-btn{background:none;border:1px solid #475569;color:#94a3b8;border-radius:4px;padding:1px 6px;font-size:.72rem;cursor:pointer;line-height:1.4;transition:background .12s,color .12s}
.info-btn:hover{background:#334155;color:#e2e8f0}
/* ---- Info modal ---- */
.info-modal{max-width:660px;width:96vw;max-height:90vh;display:flex;flex-direction:column}
.info-modal .modal-body{overflow-y:auto;padding:14px 18px;line-height:1.65;font-size:.88rem}
.info-modal h4{margin:13px 0 4px;color:#0f172a;font-size:.9rem;border-bottom:1px solid #e5e7eb;padding-bottom:3px}
.info-modal h4:first-child{margin-top:4px}
.info-modal ul{padding-left:18px;margin:4px 0 8px}
.info-modal li{margin-bottom:3px}
.info-modal .param-table{width:100%;border-collapse:collapse;font-size:.82rem;margin:6px 0 10px}
.info-modal .param-table th,.info-modal .param-table td{border:1px solid #e5e7eb;padding:5px 8px;text-align:left;vertical-align:top}
.info-modal .param-table th{background:#f1f5f9;font-weight:600}
.info-modal .tag{display:inline-block;padding:2px 7px;border-radius:10px;font-size:.75rem;margin:2px 2px}
.info-modal .tag-pro{background:#dcfce7;color:#166534}
.info-modal .tag-con{background:#fee2e2;color:#991b1b}
/* ---- Live sweep combo table ---- */
.combo-spinner-inline{display:inline-block;width:12px;height:12px;border:2px solid #e5e7eb;border-top-color:#0ea5e9;border-radius:50%;animation:spin .85s linear infinite;vertical-align:middle}
.sweep-cell-pending{min-width:58px;text-align:right;color:#94a3b8;padding:3px 7px}
.sweep-cell-done{min-width:58px;text-align:right;padding:3px 7px}
.sweep-cell-error{min-width:58px;text-align:center;color:#dc2626;font-weight:600;padding:3px 7px}
.sweep-row-error>td{background:#fef2f2!important}
.sweep-live-tbl{font-size:.77rem}
.sweep-live-tbl td,.sweep-live-tbl th{padding:3px 7px}
.sweep-opt-btn{background:#4f46e5;color:#fff;border:none;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:.82rem;transition:background .13s}
.sweep-opt-btn:hover{background:#7c3aed}
/* ---- Sweep sort + filter ---- */
.sweep-filter-bar{display:flex;align-items:center;gap:8px;padding:6px 0 4px}
.sweep-filter-input{flex:1;padding:4px 8px;border:1px solid #d1d5db;border-radius:5px;font-size:.78rem;outline:none}
.sweep-filter-input:focus{border-color:#0284c7;box-shadow:0 0 0 2px rgba(2,132,199,.15)}
.sweep-filter-count{font-size:.72rem;color:#64748b;white-space:nowrap}
.sweep-live-tbl thead th.sortable-th{cursor:pointer;user-select:none;white-space:nowrap}
.sweep-live-tbl thead th.sortable-th:hover{background:#e0f2fe}
.sweep-live-tbl thead th.sort-asc::after{content:" ▲";font-size:.65em;color:#0284c7}
.sweep-live-tbl thead th.sort-desc::after{content:" ▼";font-size:.65em;color:#0284c7}
.sweep-row-hidden{display:none}
.sweep-config-badge{display:inline-block;background:#4f46e5;color:#fff;border-radius:4px;font-size:.65rem;padding:1px 6px;margin-left:6px;font-weight:600;cursor:default;vertical-align:middle}
/* ---- Export buttons ---- */
.export-row{display:flex;gap:5px;margin-top:5px}
.export-btn{flex:1;padding:6px 4px;background:#1e3a5f;border:1px solid #334155;color:#93c5fd;border-radius:5px;font-size:.74rem;cursor:pointer;text-align:center;transition:background .13s}
.export-btn:hover:not(:disabled){background:#1d4ed8;color:#fff;border-color:#3b82f6}
.export-btn:disabled{opacity:.38;cursor:not-allowed}
/* ---- Export results modal ---- */
.export-modal{max-width:500px;width:96vw}
.export-modal .modal-body{padding:12px 16px;gap:7px}
.exp-section{font-size:.78rem;font-weight:600;color:#374151;margin:8px 0 4px;border-top:1px solid #e5e7eb;padding-top:6px}
.exp-section:first-child{border-top:none;margin-top:0;padding-top:0}
.exp-check-row{display:flex;align-items:center;gap:6px;font-size:.82rem;color:#374151;padding:2px 0}
.exp-check-row input[type=checkbox]{accent-color:#0ea5e9;width:14px;height:14px;flex-shrink:0}
.exp-sel-btns{display:flex;gap:5px;margin-top:5px}
/* ---- Optimization modal ---- */
.opt-modal{max-width:920px;width:96vw;max-height:92vh;display:flex;flex-direction:column;overflow:hidden}
.opt-modal .modal-body{overflow:hidden;padding:0;display:flex;flex-direction:row;flex:1;min-height:0}
.opt-config-col{width:300px;min-width:200px;flex-shrink:0;border-right:1px solid #e5e7eb;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.opt-result-col{flex:1;overflow-y:auto;padding:14px 16px;background:#f8fafc;display:flex;flex-direction:column}
@media(max-width:620px){.opt-modal .modal-body{flex-direction:column}.opt-config-col{width:auto;border-right:none;border-bottom:1px solid #e5e7eb}}
.opt-section{font-size:.76rem;font-weight:600;color:#374151;margin:10px 0 4px;border-top:1px solid #e5e7eb;padding-top:8px;text-transform:uppercase;letter-spacing:.03em}
.opt-section.opt-first{border-top:none;margin-top:0;padding-top:0}
.algo-radio-group{display:flex;flex-wrap:wrap;gap:5px 14px;padding:3px 0}
.algo-radio-opt{display:flex;align-items:center;gap:5px;font-size:.82rem;cursor:pointer;color:#374151}
.algo-radio-opt input[type=radio]{accent-color:#0ea5e9;width:14px;height:14px}
.algo-badge{font-size:.67rem;padding:1px 6px;border-radius:10px;background:#dbeafe;color:#1e40af;margin-left:4px}
.dir-group{display:flex;gap:14px}
.dir-opt{display:flex;align-items:center;gap:5px;font-size:.83rem;cursor:pointer;color:#374151}
.dir-opt input[type=radio]{accent-color:#0ea5e9;width:14px;height:14px}
.bounds-table{width:100%;border-collapse:collapse;font-size:.78rem;margin-top:3px}
.bounds-table th{text-align:left;padding:3px 5px;background:#f1f5f9;border:1px solid #e5e7eb;font-weight:600;font-size:.74rem}
.bounds-table td{padding:2px 4px;border:1px solid #e5e7eb;vertical-align:middle}
.bounds-table input[type=number]{width:76px;padding:2px 4px;border:1px solid #d1d5db;border-radius:4px;font-size:.76rem}
.opt-result-placeholder{flex:1;display:flex;align-items:center;justify-content:center;color:#9ca3af;text-align:center;padding:20px}
.opt-result-box{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:12px 14px;display:none}
.opt-result-box.error{background:#fef2f2;border-color:#fca5a5}
.opt-result-title{font-size:.85rem;font-weight:700;color:#166534;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.opt-result-title.error{color:#991b1b}
.opt-result-section{font-size:.72rem;font-weight:600;color:#374151;margin:10px 0 4px;text-transform:uppercase;letter-spacing:.03em}
.opt-params-table{width:100%;border-collapse:collapse;font-size:.8rem;margin-bottom:4px}
.opt-params-table th,.opt-params-table td{border:1px solid #bbf7d0;padding:4px 8px;text-align:left}
.opt-params-table th{background:#dcfce7;font-weight:600}
.opt-all-preds-table{width:100%;border-collapse:collapse;font-size:.8rem}
.opt-all-preds-table th{text-align:left;padding:4px 8px;background:#ede9fe;border:1px solid #ddd6fe;font-weight:600;font-size:.76rem}
.opt-all-preds-table td{padding:4px 8px;border:1px solid #ddd6fe}
.opt-all-preds-table tr.preds-opt td{background:#f5f3ff;font-weight:600}
.opt-all-preds-table tr.preds-opt td:first-child{border-left:3px solid #7c3aed}
.opt-meta{font-size:.71rem;color:#6b7280;margin-top:8px;border-top:1px solid #d1fae5;padding-top:6px}
.opt-spinner{display:inline-block;width:14px;height:14px;border:2px solid #e5e7eb;border-top-color:#0ea5e9;border-radius:50%;animation:spin .85s linear infinite;vertical-align:middle;margin-right:6px}
/* ---- Per-row optimize button ---- */
.row-opt-btn{padding:1px 7px;background:#7c3aed;border:none;color:#fff;border-radius:4px;font-size:.78rem;cursor:pointer;transition:background .13s;white-space:nowrap}
.row-opt-btn:hover{background:#6d28d9}
.tbl th.opt-col,.tbl td.opt-col{width:36px;text-align:center;padding:2px 4px;border-left:1px solid #e5e7eb}
/* ---- Winner-row highlight ---- */
.winner-row td{background:#f0fdf4 !important}
.winner-row td:nth-child(2){font-weight:700;color:#15803d}
/* ---- Multi-objective modal ---- */
.multi-obj-modal{max-width:720px;width:96vw;max-height:92vh;display:flex;flex-direction:column;overflow:hidden}
.multi-obj-modal .modal-body{overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;flex:1}
.multi-obj-row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.multi-obj-row .f-field{min-width:180px;flex:1}
.pareto-chart{margin-top:6px}
/* ---- Multi-obj trigger button ---- */
.multi-obj-btn{padding:4px 11px;background:#0ea5e9;border:none;color:#fff;border-radius:5px;font-size:.75rem;font-weight:500;cursor:pointer;transition:background .13s;display:none}
.multi-obj-btn:hover{background:#0284c7}
</style>
</head>
<body>
<div class="layout">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="app-title">MEX/FDM Quality Analyzer</div>
    <div class="data-info" id="data-info">Cargando datos…</div>

    <div>
      <div class="section-head">Variables</div>
      <div class="var-btn-row">
        <button class="var-open-btn" onclick="openVarModal()">&#9998; Seleccionar variables…</button>
      </div>
      <div class="var-badge" id="var-badge">Cargando…</div>
    </div>

    <div>
      <div class="section-head">Algoritmos — Regresión</div>
      <div id="reg-algos"></div>
    </div>

    <div>
      <div class="section-head">Algoritmos — Clasificación</div>
      <div id="cls-algos"></div>
    </div>

    <div>
      <div class="section-head">Opciones</div>
      <label class="opt-row">
        Clases para clasificación:&nbsp;
        <input type="number" id="class-q" value="3" min="2" max="10">
      </label>
    </div>

    <button class="run-btn" id="run-btn" onclick="runAnalysis()">&#9654; Ejecutar Análisis</button>
    <div class="export-row">
      <button class="export-btn" onclick="exportConfig()" title="Descargar configuración como JSON">&#11123; Config</button>
      <button class="export-btn" id="export-results-btn" disabled onclick="openExportModal()" title="Exportar resultados como ZIP">&#11123; Resultados</button>
    </div>
  </aside>

  <!-- Content -->
  <main class="content">
    <div id="placeholder">
      Seleccione uno o más algoritmos y haga clic en <br><strong>Ejecutar Análisis</strong> para comenzar.
    </div>
    <div id="spinner"><div class="ring"></div><span>Ejecutando análisis…</span></div>
    <div id="results" style="display:none;flex:1;display:none;flex-direction:column;overflow:hidden"></div>
  </main>
</div>

<!-- Variables modal -->
<div class="overlay hidden" id="var-overlay" onclick="varOverlayClick(event)">
  <div class="modal var-modal">
    <div class="modal-hdr">
      <h3>Seleccionar variables</h3>
      <button class="close-x" onclick="closeVarModal()">&#x2715;</button>
    </div>
    <div class="modal-body" style="padding:12px 14px">
      <div class="col-table-wrap">
        <table class="col-table">
          <thead><tr><th>Columna</th><th>Entrada</th><th>Salida</th><th>Ignorar</th></tr></thead>
          <tbody id="col-role-body"><tr><td colspan="4" class="col-sel-none">Cargando…</td></tr></tbody>
        </table>
      </div>
      <div class="combo-section" id="combo-section">
        <div class="combo-section-hdr">
          <span class="combo-section-title">Combinaciones de salida a analizar</span>
          <div class="combo-ctrl-btns">
            <button class="combo-ctrl-btn" onclick="_combosSelectAll(true)">Todas</button>
            <button class="combo-ctrl-btn" onclick="_combosSelectAll(false)">Ninguna</button>
          </div>
        </div>
        <div class="combo-list" id="combo-list"><span class="combo-empty">Seleccione al menos una variable de salida.</span></div>
      </div>
    </div>
    <div class="modal-ftr">
      <button class="btn btn-primary" onclick="saveVarModal()">Guardar</button>
      <button class="btn btn-ghost" onclick="closeVarModal()">Cancelar</button>
    </div>
  </div>
</div>

<!-- Algo info modal -->
<div class="overlay hidden" id="info-overlay" onclick="infoOverlayClick(event)">
  <div class="modal info-modal">
    <div class="modal-hdr">
      <h3 id="info-modal-title">Información del algoritmo</h3>
      <button class="close-x" onclick="closeInfoModal()">&#x2715;</button>
    </div>
    <div class="modal-body" id="info-modal-body"></div>
    <div class="modal-ftr">
      <button class="btn btn-ghost" onclick="closeInfoModal()">Cerrar</button>
    </div>
  </div>
</div>

<!-- Algo config modal -->
<div class="overlay hidden" id="overlay" onclick="overlayClick(event)">
  <div class="modal">
    <div class="modal-hdr">
      <h3 id="modal-title">Configurar algoritmo</h3>
      <button class="close-x" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
    <div class="modal-ftr">
      <button class="btn btn-sec" onclick="resetConfig()">Restablecer</button>
      <button class="btn btn-primary" onclick="saveConfig()">Guardar</button>
      <button class="btn btn-ghost" onclick="closeModal()">Cancelar</button>
    </div>
  </div>
</div>

<!-- Export results modal -->
<div class="overlay hidden" id="export-overlay" onclick="exportOverlayClick(event)">
  <div class="modal export-modal">
    <div class="modal-hdr">
      <h3>Exportar resultados</h3>
      <button class="close-x" onclick="closeExportModal()">&#x2715;</button>
    </div>
    <div class="modal-body" id="export-modal-body"></div>
    <div class="modal-ftr">
      <button class="btn btn-primary" id="do-export-btn" onclick="doExport()">&#11123; Generar ZIP</button>
      <button class="btn btn-ghost" onclick="closeExportModal()">Cancelar</button>
    </div>
  </div>
</div>

<!-- Optimization modal -->
<div class="overlay hidden" id="opt-overlay" onclick="optOverlayClick(event)">
  <div class="modal opt-modal">
    <div class="modal-hdr">
      <h3 id="opt-modal-title">Optimizar par&#225;metros</h3>
      <button class="close-x" onclick="closeOptModal()">&#x2715;</button>
    </div>
    <div class="modal-body" id="opt-modal-body">
      <!-- LEFT: configuration -->
      <div class="opt-config-col">
        <div class="opt-section opt-first">Algoritmo de optimizaci&#243;n</div>
        <div class="algo-radio-group" id="opt-algo-group"></div>
        <div class="opt-section">Direcci&#243;n</div>
        <div class="dir-group">
          <label class="dir-opt"><input type="radio" name="opt-dir" value="minimize" checked> Minimizar</label>
          <label class="dir-opt"><input type="radio" name="opt-dir" value="maximize"> Maximizar</label>
        </div>
        <div class="opt-section">Iteraciones / Evaluaciones</div>
        <div class="f-field" style="max-width:130px">
          <input type="number" id="opt-n-iter" value="50" min="10" max="500" step="10">
        </div>
        <div class="opt-section">L&#237;mites de variables de entrada</div>
        <div style="overflow-x:auto">
          <table class="bounds-table">
            <thead><tr><th>Variable</th><th>M&#237;n</th><th>M&#225;x</th></tr></thead>
            <tbody id="opt-bounds-body"></tbody>
          </table>
        </div>
      </div>
      <!-- RIGHT: result -->
      <div class="opt-result-col">
        <div class="opt-result-placeholder" id="opt-result-placeholder">
          <div>
            <div style="font-size:2.8rem;margin-bottom:10px;opacity:.25">&#9889;</div>
            <div style="font-size:.83rem">Presione <strong>Optimizar</strong> para<br>encontrar los par&#225;metros &#243;ptimos</div>
          </div>
        </div>
        <div class="opt-result-box" id="opt-result-box">
          <div class="opt-result-title" id="opt-result-title">&#10003; Par&#225;metros &#243;ptimos encontrados</div>
          <div class="opt-result-section">Par&#225;metros &#243;ptimos de entrada</div>
          <table class="opt-params-table" id="opt-params-table">
            <thead><tr><th>Variable</th><th>Valor &#243;ptimo</th></tr></thead>
            <tbody id="opt-params-body"></tbody>
          </table>
          <div class="opt-result-section" id="opt-allpreds-label" style="margin-top:12px">Predicciones del modelo con par&#225;metros &#243;ptimos</div>
          <table class="opt-all-preds-table" id="opt-all-preds-table">
            <thead><tr><th>Objetivo</th><th>Predicci&#243;n</th></tr></thead>
            <tbody id="opt-all-preds-body"></tbody>
          </table>
          <div class="opt-meta" id="opt-meta"></div>
          <div id="opt-convergence-chart" style="margin-top:8px"></div>
        </div>
        <div class="opt-result-box error" id="opt-error-box" style="display:none">
          <div class="opt-result-title error" id="opt-error-title">Error</div>
          <div id="opt-error-msg" style="font-size:.82rem;color:#991b1b"></div>
        </div>
      </div>
    </div>
    <div class="modal-ftr">
      <button class="btn btn-primary" id="opt-run-btn" onclick="runOptimization()">&#9654; Optimizar</button>
      <button class="btn multi-obj-btn" id="opt-multi-btn" onclick="openMultiObjModal()">&#9783; Multi-objetivo</button>
      <button class="btn btn-ghost" onclick="closeOptModal()">Cerrar</button>
    </div>
  </div>
</div>

<!-- Multi-objective optimization modal -->
<div class="overlay hidden" id="multi-obj-overlay" onclick="multiObjOverlayClick(event)">
  <div class="modal multi-obj-modal">
    <div class="modal-hdr">
      <h3>Optimizaci&#243;n multi-objetivo (NSGA-II)</h3>
      <button class="close-x" onclick="closeMultiObjModal()">&#x2715;</button>
    </div>
    <div class="modal-body">
      <div class="multi-obj-row">
        <div class="f-field">
          <label class="f-label">Objetivo 1</label>
          <select id="multi-obj-sel-1" class="f-input"></select>
          <div class="dir-group" style="margin-top:5px">
            <label class="dir-opt"><input type="radio" name="mdir-1" value="minimize" checked> Min</label>
            <label class="dir-opt"><input type="radio" name="mdir-1" value="maximize"> Max</label>
          </div>
        </div>
        <div class="f-field">
          <label class="f-label">Objetivo 2</label>
          <select id="multi-obj-sel-2" class="f-input"></select>
          <div class="dir-group" style="margin-top:5px">
            <label class="dir-opt"><input type="radio" name="mdir-2" value="minimize" checked> Min</label>
            <label class="dir-opt"><input type="radio" name="mdir-2" value="maximize"> Max</label>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;min-width:120px">
          <div class="f-field">
            <label class="f-label">Generaciones</label>
            <input type="number" id="multi-n-gen" class="f-input" value="50" min="10" max="500" step="10">
          </div>
          <div class="f-field">
            <label class="f-label">Poblaci&#243;n</label>
            <input type="number" id="multi-pop-size" class="f-input" value="40" min="10" max="200" step="10">
          </div>
        </div>
      </div>
      <div id="pareto-chart-container" class="pareto-chart"></div>
    </div>
    <div class="modal-ftr">
      <button class="btn btn-primary" id="multi-run-btn" onclick="runMultiObjOpt()">&#9654; Ejecutar NSGA-II</button>
      <button class="btn btn-ghost" onclick="closeMultiObjModal()">Cerrar</button>
    </div>
  </div>
</div>

<script>
const REGISTRY = __REGISTRY__;
const OPT_REGISTRY = __OPT_REGISTRY__;
const OPT_ALGO_LABELS = __OPT_ALGO_LABELS__;
const algoConfigs = {};
let curAlgo = null, curType = null;

// ---------- Metric tooltips ----------
const COL_TOOLTIPS = {
  objective:               'Objetivo de predicción\n(columna o combinación de columnas de salida)',
  rank:                    'Ranking del modelo (por objetivo)\n1 = mejor. Basado en RMSE-CV, brecha de sobreajuste y R² holdout.',
  cv_mae:                  'MAE — Validación cruzada\nError absoluto medio promedio en CV.\nMenor es mejor.',
  cv_rmse:                 'RMSE — Validación cruzada\nRaíz del error cuadrático medio en CV.\nPenaliza errores grandes. Menor es mejor.',
  train_rmse:              'RMSE — Entrenamiento (CV)\nRMSE medio en los pliegues de entrenamiento.\nComparar con RMSE-CV para detectar sobreajuste.',
  overfit_gap:             'Brecha de sobreajuste\nRMSE-CV − RMSE-entrenamiento.\nValores altos indican que el modelo memoriza en vez de generalizar.\nIdeal: cerca de 0.',
  cv_r2:                   'R² — Validación cruzada\nCoeficiente de determinación en CV.\n1 = predicción perfecta; 0 = modelo nulo.',
  cv_rmse_std:             'RMSE Std — Validación cruzada\nDesviación estándar del RMSE entre folds.\nMide estabilidad del modelo.',
  holdout_mae:             'MAE — Conjunto de prueba\nError absoluto medio en datos nunca vistos.',
  holdout_rmse:            'RMSE — Conjunto de prueba\nError cuadrático medio en datos nunca vistos.',
  holdout_r2:              'R² — Conjunto de prueba\nCoeficiente de determinación en datos nunca vistos.',
  cv_accuracy:             'Exactitud — Validación cruzada\nFracción de muestras clasificadas correctamente.',
  cv_f1_macro:             'F1 Macro — Validación cruzada\nPromedio no ponderado del F1 por clase.\nRobusta ante desbalance de clases.',
  cv_balanced_accuracy:    'Exactitud balanceada — CV\nPromedio del recall por clase. Útil en clases desbalanceadas.',
  train_balanced_accuracy: 'Exactitud balanceada — Entrenamiento (CV)\nValor en los pliegues de entrenamiento.\nComparar con CV para detectar sobreajuste.',
  holdout_accuracy:        'Exactitud — Conjunto de prueba',
  holdout_f1_macro:        'F1 Macro — Conjunto de prueba',
  holdout_balanced_accuracy: 'Exactitud balanceada — Conjunto de prueba',
};

// ---------- Algorithm info ----------
const ALGO_INFO = {
  elastic_net: {
    name: 'Elastic Net',
    theory: `¿Qué es una regresión lineal?
Imagina que quieres predecir la rugosidad de una pieza a partir de la temperatura de impresión. La regresión lineal traza la "línea recta que mejor se ajusta" a los datos: multiplica cada variable de entrada por un número (llamado coeficiente o peso) y los suma para obtener la predicción.

¿Qué problema resuelve Elastic Net?
Cuando hay muchas variables de entrada, algunos coeficientes pueden crecer demasiado, haciendo que el modelo se "memorice" los datos de entrenamiento pero falle con datos nuevos. Esto se llama sobreajuste. Elastic Net evita esto añadiendo una penalización: si un coeficiente se hace muy grande, la penalización aumenta el error y el modelo aprende a mantenerlos pequeños.

¿Cómo combina Lasso y Ridge?
Elastic Net mezcla dos tipos de penalización:
• Lasso (L1): empuja algunos coeficientes exactamente a cero, eliminando variables irrelevantes automáticamente (selección de variables).
• Ridge (L2): encoge todos los coeficientes de forma proporcional sin llegar a cero, útil cuando varias variables están correlacionadas entre sí.
El parámetro l1_ratio controla cuánto de cada uno se usa.`,
    params: [
      ['alpha', 'Controla cuánto se penaliza el tamaño de los coeficientes. Alpha = 0 es regresión lineal pura (sin penalización). Valores altos hacen el modelo más simple pero pueden perder precisión.'],
      ['l1_ratio', 'Proporción de penalización Lasso vs Ridge. 0 = solo Ridge (no elimina variables); 1 = solo Lasso (puede eliminar muchas variables); 0.5 = mezcla igual.'],
    ],
    pros: ['Robusto con predictores correlacionados','Selección automática de variables','Rápido y escalable','Fácil de interpretar: cada coeficiente indica el efecto de cada variable'],
    cons: ['Solo captura relaciones lineales — si la rugosidad depende del cuadrado de la temperatura, no lo detecta','Sensible a outliers si no se normaliza'],
    usage: 'Punto de partida sólido. Funciona bien cuando hay muchas variables y se sospecha que solo algunas son relevantes.',
  },
  svr_rbf: {
    name: 'SVR con kernel RBF',
    theory: `¿Qué es la idea principal de SVR?
Imagina que tus datos son puntos en un plano. SVR (Support Vector Regression) intenta trazar una "franja" o tubo alrededor de una línea de predicción. Los puntos que caen dentro del tubo no se penalizan — solo se penalizan los que están afuera. Esto hace al modelo más robusto ante valores atípicos (outliers).

¿Qué pasa cuando la relación no es una línea recta?
Aquí entra el truco del kernel. Un kernel es una función matemática que transforma los datos a un espacio donde sí es posible trazar una línea recta, aunque en el espacio original la relación sea curva o compleja. Es como "doblar el espacio" para que los datos queden alineados.

El kernel RBF (Radial Basis Function):
El kernel RBF mide qué tan "cerca" están dos puntos usando su distancia. Dos puntos muy cercanos tienen una influencia muy alta entre sí; puntos lejanos casi no se influyen. El resultado es que el modelo puede aprender formas muy complejas (curvas, superficies onduladas, etc.) sin que el usuario tenga que especificarlas manualmente.`,
    params: [
      ['C', 'Controla cuánto se penaliza cada punto que cae fuera del tubo. C alto = el modelo se esfuerza mucho por no cometer errores de entrenamiento (puede sobreajustar). C bajo = el modelo es más flexible y acepta más errores (más generalizable).'],
      ['epsilon (ε)', 'Define el ancho del tubo de tolerancia. Puntos dentro del tubo no afectan el entrenamiento. Valor pequeño = el modelo es muy exigente consigo mismo; valor grande = ignora pequeñas desviaciones.'],
      ['gamma', 'Controla el radio de influencia de cada punto de entrenamiento. "scale" es un valor automático recomendado. Gamma alto = cada punto solo influye en sus vecinos muy cercanos (modelo muy local, puede sobreajustar).'],
    ],
    pros: ['Potente para datos no lineales','Robusto ante outliers gracias al tubo ε','Funciona bien con pocos datos'],
    cons: ['Muy lento cuando hay miles de datos','Difícil de interpretar — no se sabe qué variable importa más','Sensible a la escala: hay que normalizar los datos'],
    usage: 'Adecuado para conjuntos pequeños o medianos con relaciones no lineales. Requiere normalización previa.',
  },
  knn_reg: {
    name: 'K-Vecinos más Cercanos (Regresión)',
    theory: `¿Cuál es la idea?
KNN (K-Nearest Neighbors) es quizás el algoritmo más intuitivo de todos. Para predecir el valor de un punto nuevo, simplemente busca los K puntos más parecidos en los datos de entrenamiento y promedia sus valores reales.

Ejemplo concreto:
Quieres predecir la rugosidad para una pieza impresa a 210 °C, 0.2 mm de capa, 60 mm/s y 20% de relleno. KNN busca los 5 experimentos más similares (los "vecinos más cercanos") en los datos históricos y hace el promedio de sus rugosidades.

¿Qué significa "más cercano"?
Se usa la distancia euclidiana: la misma que mides con una regla en el espacio, pero en múltiples dimensiones. Si las variables tienen escalas muy diferentes (temperatura en cientos, relleno en porcentaje), la distancia estaría sesgada — por eso es importante normalizar los datos.

Ponderación por distancia:
Con weights="distance", los vecinos más cercanos tienen más peso en el promedio. Si un vecino está muy cerca, su valor importa más que el de uno lejano.

Aprendizaje "perezoso":
A diferencia de otros modelos, KNN no aprende una función matemática durante el entrenamiento — simplemente guarda todos los datos. El trabajo real ocurre al momento de predecir.`,
    params: [
      ['n_neighbors (K)', 'Cuántos vecinos se consultan. K=1 usa solo el punto más cercano (muy sensible al ruido). K grande promedia más puntos (predicciones más suaves pero puede perder detalle local). Se recomienda probar valores impares como 3, 5, 7.'],
      ['weights', '"uniform": todos los K vecinos contribuyen igual al promedio. "distance": vecinos más cercanos contribuyen más.'],
    ],
    pros: ['Sin suposiciones sobre la forma de la relación entre variables','Sencillo de entender e interpretar','Se adapta automáticamente a cualquier forma de los datos'],
    cons: ['Lento al predecir: busca entre todos los datos cada vez','Maldición de la dimensionalidad: con muchas variables, todas las distancias se parecen y el concepto de "cercanía" pierde sentido','Sensible a variables irrelevantes y a la escala'],
    usage: 'Útil como referencia y cuando la relación entre variables es localmente compleja. Funciona bien con pocos datos y pocas variables.',
  },
  gpr: {
    name: 'Gaussian Process Regression (GPR)',
    theory: `¿Qué es un proceso gaussiano? (Empezando desde cero)
Una distribución gaussiana (o "campana de Gauss") describe la probabilidad de que una variable tome distintos valores. Por ejemplo, las alturas de personas siguen una distribución gaussiana: la mayoría están cerca del promedio y pocas son muy altas o muy bajas.

Un proceso gaussiano extiende esta idea: en lugar de describir la probabilidad de un número, describe la probabilidad de una función completa. Es decir, GPR no solo predice "la rugosidad será 1.5 μm" sino "la rugosidad probablemente está entre 1.3 y 1.7 μm" — dando una medida de incertidumbre.

¿Cómo funciona en la práctica?
GPR aprende dos cosas: la predicción más probable y cuán seguro está de ella. Cerca de los datos de entrenamiento, la incertidumbre es baja. Lejos de los datos, la incertidumbre crece. Esto es muy valioso en experimentos científicos donde los datos son pocos y costosos.

¿Qué es el kernel?
El kernel es una función que mide qué tan "similares" son dos puntos de entrada. Si dos experimentos tienen parámetros similares, sus salidas deberían ser parecidas — el kernel formaliza esa intuición. El kernel RBF asume que la función varía suavemente. El término WhiteNoise representa el ruido inherente en las mediciones.`,
    params: [
      ['alpha', 'Nivel de ruido en las mediciones. Un valor pequeño asume que las mediciones son muy precisas; un valor grande asume que hay bastante ruido experimental. Si el modelo sobreajusta, aumentar alpha ayuda.'],
      ['normalize_y', 'Si está activado (True), centra la variable de salida en cero antes de ajustar. Recomendado cuando los valores de salida son muy grandes o muy pequeños.'],
    ],
    pros: ['Único en dar predicciones con intervalos de confianza','Funciona excepcionalmente bien con pocos datos','Muy flexible — puede aprender casi cualquier forma'],
    cons: ['Se vuelve muy lento cuando hay más de ~1000 datos (escala cúbica)','Requiere elegir bien el kernel','Difícil de escalar a problemas grandes'],
    usage: 'Excelente para experimentos de diseño con pocos datos (como el presente caso). Ideal cuando se necesita saber qué tan confiable es la predicción.',
  },
  mlp_small: {
    name: 'Red Neuronal MLP (MLPRegressor)',
    theory: `¿Qué es una neurona artificial?
Inspirada en el cerebro humano, una neurona artificial recibe varias entradas (por ejemplo, temperatura, velocidad), las multiplica por pesos, las suma y aplica una función de activación para decidir qué señal enviar hacia adelante. La función de activación introduce la no-linealidad: sin ella, apilar muchas neuronas sería equivalente a una sola regresión lineal.

¿Qué es una red de capas? (MLP = Multilayer Perceptron)
Las neuronas se organizan en capas:
• Capa de entrada: recibe las variables de proceso (temperatura, velocidad, etc.)
• Capas ocultas: procesan la información de forma progresivamente más abstracta
• Capa de salida: produce la predicción final (rugosidad, precisión, etc.)
Cada conexión entre neuronas tiene un peso que se ajusta durante el entrenamiento.

¿Cómo aprende? (Backpropagation)
El modelo hace una predicción, compara con el valor real y calcula el error. Luego "propaga el error hacia atrás" por toda la red para ajustar cada peso un pequeño paso en la dirección que reduce el error. Esto se repite miles de veces. El tamaño de ese "pequeño paso" es la tasa de aprendizaje.

¿Qué es el sobreajuste y cómo lo evita alpha?
Si la red tiene demasiadas neuronas o se entrena demasiado, puede memorizar los datos en lugar de aprender patrones generales. El parámetro alpha añade una penalización matemática que impide que los pesos crezcan demasiado, forzando al modelo a ser más simple y generalizable.`,
    params: [
      ['hidden_layer_sizes', 'Define cuántas capas ocultas hay y cuántas neuronas tiene cada una. Ejemplo: (64, 32) significa 2 capas ocultas, la primera con 64 neuronas y la segunda con 32. Más neuronas = más capacidad de aprender patrones complejos, pero más riesgo de sobreajuste.'],
      ['activation', 'La función que transforma la señal en cada neurona. "relu" (Rectified Linear Unit): pasa los valores positivos sin cambios y convierte los negativos en cero — simple, eficiente y muy popular. "tanh": produce valores entre -1 y 1, útil cuando los datos están centrados alrededor de cero.'],
      ['alpha', 'Fuerza de la regularización L2 (penalización de pesos grandes). Valor mayor = red más simple = menos sobreajuste, pero puede perder precisión. Valor menor = la red puede aprender patrones más complejos.'],
      ['learning_rate_init', 'Qué tan grandes son los pasos al ajustar los pesos durante el entrenamiento. Valor pequeño = ajuste lento pero estable. Valor grande = ajuste rápido pero puede "pasarse" del mínimo y oscilar sin converger.'],
    ],
    pros: ['Captura relaciones no lineales muy complejas','Flexible: se puede adaptar la arquitectura al problema','Soporta múltiples variables de salida'],
    cons: ['Caja negra: difícil saber qué variables importan más','Requiere más datos que otros modelos para generalizar bien','Muy sensible a la escala de los datos y a los hiperparámetros','Puede no converger si los hiperparámetros no son adecuados'],
    usage: 'Adecuado cuando se sospecha una relación altamente no lineal. Usar el sweep de hiperparámetros para encontrar la mejor configuración.',
  },
  svc_rbf: {
    name: 'SVC con kernel RBF (Clasificación)',
    theory: `De predicción de números a predicción de categorías
En clasificación ya no predecimos un número continuo (como "la rugosidad es 1.5 μm") sino una categoría (como "calidad alta", "calidad media", "calidad baja"). Para esto, primero se convierte la variable continua en clases usando cuantiles: los datos se dividen en grupos de igual tamaño.

¿Qué hace SVC?
SVC (Support Vector Classifier) busca la frontera que mejor separa las clases. Si las clases fueran puntos de dos colores en un plano, SVC busca la línea (o curva) que las divide dejando el mayor espacio posible entre la línea y los puntos más cercanos. Ese espacio se llama margen. Un margen grande significa que la frontera es más robusta.

¿Por qué el kernel RBF?
Si las clases no son linealmente separables (no se puede trazar una línea recta entre ellas), el kernel RBF transforma los datos a un espacio de mayor dimensión donde sí es posible encontrar una frontera lineal. El resultado es una frontera curva en el espacio original.

¿Qué es class_weight="balanced"?
Si hay muchos más experimentos de "calidad media" que de "calidad alta", el modelo podría ignorar la clase minoritaria. Con class_weight="balanced", se asigna automáticamente más peso a las clases con menos ejemplos, corrigiendo este sesgo.`,
    params: [
      ['C', 'Penalización por clasificar mal un punto. C alto = el modelo intenta no equivocarse en ningún punto de entrenamiento (puede sobreajustar). C bajo = acepta algunos errores en entrenamiento para tener una frontera más simple y general.'],
      ['gamma', 'Radio de influencia de cada punto del conjunto de entrenamiento al definir la frontera. "scale" es el valor automático recomendado. Gamma alto = fronteras muy locales y detalladas (riesgo de sobreajuste). Gamma bajo = fronteras más suaves y globales.'],
    ],
    pros: ['Robusto con clases desbalanceadas gracias a class_weight','Efectivo en espacios de alta dimensión','Buen margen de generalización con pocos datos'],
    cons: ['Lento con muchos datos','No produce probabilidades directamente','Sensible a la escala de las variables'],
    usage: 'Clasificación en conjuntos pequeños o medianos. Combinado con discretización por cuantiles, identifica rangos de calidad de impresión.',
  },
  knn_clf: {
    name: 'K-Vecinos más Cercanos (Clasificación)',
    theory: `Misma idea que KNN para regresión, adaptada a categorías
La lógica es idéntica a KNN-Regresión: busca los K experimentos más similares al nuevo punto. La diferencia es cómo se combina la respuesta: en lugar de promediar números, se hace una votación. La clase que más vecinos tengan gana.

Ejemplo:
Se tienen 5 vecinos: 3 son "calidad alta" y 2 son "calidad media". La predicción es "calidad alta" porque tiene más votos.

Con pesos por distancia:
Los vecinos más cercanos votan con más fuerza. Si el vecino más cercano es "calidad alta" pero los otros 4 están lejos y son "calidad media", el resultado dependerá de cuánto más cerca está el primero.

¿Por qué es útil como punto de referencia?
KNN no hace ninguna suposición sobre la forma de los datos. Si un modelo más sofisticado no supera a KNN, es una señal de que los datos quizás no tienen patrones claros o que el modelo complejo tiene problemas.`,
    params: [
      ['n_neighbors (K)', 'Cuántos vecinos votan. K=1 usa solo el punto más cercano — muy sensible al ruido. K mayor = votaciones más estables pero puede perder fronteras locales detalladas.'],
      ['weights', '"uniform": todos los vecinos tienen el mismo voto. "distance": los vecinos más cercanos votan con más peso.'],
    ],
    pros: ['Simple e intuitivo — fácil de explicar','Sin suposiciones sobre la distribución de los datos','Se adapta a fronteras de decisión complejas'],
    cons: ['Lento al predecir: compara con todos los datos cada vez','Sensible a variables irrelevantes y a diferencias de escala','No produce probabilidades calibradas'],
    usage: 'Referencia robusta para clasificación cuando no se conoce la estructura de los datos.',
  },
};

// ---------- Init ----------
document.addEventListener('DOMContentLoaded', () => {
  renderGroup('regression', document.getElementById('reg-algos'));
  renderGroup('classification', document.getElementById('cls-algos'));
  fetchDataInfo();
});

function renderGroup(type, container) {
  for (const [id, algo] of Object.entries(REGISTRY[type])) {
    const card = document.createElement('div');
    card.className = 'algo-card';
    card.id = 'card-' + id;
    const hasInfo = !!ALGO_INFO[id];
    card.innerHTML =
      '<div class="algo-row">' +
        '<input type="checkbox" id="chk-' + id + '" data-algo="' + id + '" data-type="' + type + '" onchange="onToggle(this)">' +
        '<label class="algo-label" for="chk-' + id + '">' + algo.label + '</label>' +
        (hasInfo ? '<button class="info-btn" onclick="openInfoModal(\'' + id + '\')" title="Información del algoritmo">ⓘ</button>' : '') +
        '<button class="cfg-btn" id="cfgbtn-' + id + '" disabled onclick="openModal(\'' + id + '\',\'' + type + '\')">⚙ Config</button>' +
      '</div>' +
      '<div class="param-summary" id="sum-' + id + '">predeterminado</div>';
    container.appendChild(card);
  }
}

function onToggle(chk) {
  const id = chk.dataset.algo;
  document.getElementById('card-' + id).classList.toggle('enabled', chk.checked);
  document.getElementById('cfgbtn-' + id).disabled = !chk.checked;
  if (!chk.checked) {
    delete algoConfigs[id];
    document.getElementById('sum-' + id).textContent = 'predeterminado';
  }
}

// ---------- Info modal ----------
function openInfoModal(algoId) {
  const info = ALGO_INFO[algoId];
  if (!info) return;
  document.getElementById('info-modal-title').textContent = info.name;
  const body = document.getElementById('info-modal-body');

  const paramRows = (info.params || []).map(([p, desc]) =>
    `<tr><td><strong>${p}</strong></td><td>${desc}</td></tr>`
  ).join('');

  const pros = (info.pros || []).map(p => `<span class="tag tag-pro">✓ ${p}</span>`).join(' ');
  const cons = (info.cons || []).map(c => `<span class="tag tag-con">✗ ${c}</span>`).join(' ');

  body.innerHTML =
    `<h4>Fundamento teórico</h4><p>${info.theory.replace(/\n/g,'<br>')}</p>` +
    (paramRows ? `<h4>Parámetros</h4><table class="param-table"><thead><tr><th>Parámetro</th><th>Descripción y efecto</th></tr></thead><tbody>${paramRows}</tbody></table>` : '') +
    (pros ? `<h4>Ventajas</h4><div style="margin:4px 0 8px">${pros}</div>` : '') +
    (cons ? `<h4>Desventajas</h4><div style="margin:4px 0 8px">${cons}</div>` : '') +
    (info.usage ? `<h4>Cuándo usarlo</h4><p>${info.usage}</p>` : '');

  document.getElementById('info-overlay').classList.remove('hidden');
}
function closeInfoModal() {
  document.getElementById('info-overlay').classList.add('hidden');
}
function infoOverlayClick(e) {
  if (e.target === document.getElementById('info-overlay')) closeInfoModal();
}

// ---------- Algo config modal ----------
function openModal(algoId, algoType) {
  curAlgo = algoId; curType = algoType;
  const algo = REGISTRY[algoType][algoId];
  document.getElementById('modal-title').textContent = 'Configurar: ' + algo.label;
  const saved = algoConfigs[algoId] || {};
  const body = document.getElementById('modal-body');
  body.innerHTML = '';
  for (const p of algo.params) {
    const val = (saved[p.id] !== undefined) ? saved[p.id] : p.default;
    const div = document.createElement('div');
    div.className = 'f-field';
    if (p.type === 'select') {
      const opts = p.options.map(o => '<option value="' + o + '"' + (o == val ? ' selected' : '') + '>' + o + '</option>').join('');
      div.innerHTML = '<label for="pm-' + p.id + '">' + p.label + '</label><select id="pm-' + p.id + '">' + opts + '</select>';
    } else if (p.type === 'multiselect') {
      const curVals = Array.isArray(val) ? val : [val];
      const boxes = p.options.map(o =>
        '<label class="ms-opt"><input type="checkbox" name="pm-' + p.id + '" value="' + o + '"' + (curVals.includes(o) ? ' checked' : '') + '> ' + o + '</label>'
      ).join('');
      div.innerHTML = '<label>' + p.label + '</label><div class="ms-group">' + boxes + '</div>';
    } else if (p.type === 'sweep-layers' || p.type === 'sweep-float' || p.type === 'sweep-int' || p.type === 'sweep-text') {
      div.innerHTML = '<label for="pm-' + p.id + '">' + p.label + '</label>' +
        '<input type="text" id="pm-' + p.id + '" value="' + val + '">' +
        '<span class="f-hint">' + (p.help || '') + '</span>';
    } else {
      const attrs = [
        'type="' + p.type + '"',
        'id="pm-' + p.id + '"',
        'value="' + val + '"',
        'step="' + (p.step || 'any') + '"',
        p.min !== undefined ? 'min="' + p.min + '"' : '',
        p.max !== undefined ? 'max="' + p.max + '"' : '',
      ].filter(Boolean).join(' ');
      div.innerHTML = '<label for="pm-' + p.id + '">' + p.label + '</label><input ' + attrs + '>';
    }
    body.appendChild(div);
  }
  document.getElementById('overlay').classList.remove('hidden');
}

function closeModal() { document.getElementById('overlay').classList.add('hidden'); }
function overlayClick(e) { if (e.target === document.getElementById('overlay')) closeModal(); }

function saveConfig() {
  const algo = REGISTRY[curType][curAlgo];
  const params = {};
  for (const p of algo.params) {
    if (p.type === 'multiselect') {
      const checks = document.querySelectorAll('input[name="pm-' + p.id + '"]:checked');
      params[p.id] = checks.length
        ? [...checks].map(c => c.value)
        : (Array.isArray(p.default) ? p.default : [p.default]);
    } else {
      const el = document.getElementById('pm-' + p.id);
      params[p.id] = (p.type === 'number' && p.integer) ? parseInt(el.value, 10)
                   : (p.type === 'number')               ? parseFloat(el.value)
                   :                                        el.value;
    }
  }
  algoConfigs[curAlgo] = params;
  updateSummary(curAlgo);
  closeModal();
}

function resetConfig() {
  delete algoConfigs[curAlgo];
  openModal(curAlgo, curType);
}

function updateSummary(id) {
  const p = algoConfigs[id];
  document.getElementById('sum-' + id).textContent =
    p ? Object.entries(p).map(([k,v]) => k + '=' + v).join(' · ') : 'predeterminado';
}

// ---------- Tabs ----------
function showTab(id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + id));
}

// Re-execute <script> tags injected via innerHTML (browsers skip them by default)
function execScripts(container) {
  container.querySelectorAll('script').forEach(old => {
    const s = document.createElement('script');
    [...old.attributes].forEach(a => s.setAttribute(a.name, a.value));
    s.textContent = old.textContent;
    old.parentNode.replaceChild(s, old);
  });
}

// ---------- Streaming state ----------
let _totalTasks = 0, _doneTasks = 0, _taskStatus = {}, _taskList = [];
let _sessionId = null, _availableTabs = [];

const REG_COLS = ['objective','rank','cv_mae','cv_rmse','train_rmse','overfit_gap','cv_r2','cv_rmse_std','holdout_mae','holdout_rmse','holdout_r2'];
const CLS_COLS = ['objective','rank','cv_accuracy','cv_f1_macro','cv_balanced_accuracy','train_balanced_accuracy','overfit_gap','holdout_accuracy','holdout_f1_macro','holdout_balanced_accuracy'];

// ---------- Run ----------
async function runAnalysis() {
  const reg = {}, cls = {};
  document.querySelectorAll('[data-type="regression"]:checked').forEach(c => { reg[c.dataset.algo] = algoConfigs[c.dataset.algo] || null; });
  document.querySelectorAll('[data-type="classification"]:checked').forEach(c => { cls[c.dataset.algo] = algoConfigs[c.dataset.algo] || null; });

  const payload = {
    regression: reg,
    classification: cls,
    class_q: parseInt(document.getElementById('class-q').value, 10) || 3,
    input_cols:    [...document.querySelectorAll('#col-role-body input[value="input"]:checked')].map(r => r.dataset.col),
    output_cols:   [...document.querySelectorAll('#col-role-body input[value="output"]:checked')].map(r => r.dataset.col),
    output_combos: _getSelectedCombos(),
  };

  if (!payload.input_cols.length || !payload.output_cols.length) {
    alert('Seleccione al menos una variable de entrada y una de salida.');
    document.getElementById('run-btn').disabled = false;
    return;
  }

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('spinner').style.display = 'flex';
  const resultsEl = document.getElementById('results');
  resultsEl.style.display = 'none';
  resultsEl.innerHTML = '';
  document.getElementById('run-btn').disabled = true;
  document.getElementById('export-results-btn').disabled = true;
  _sessionId = null; _availableTabs = [];
  _totalTasks = 0; _doneTasks = 0; _taskStatus = {}; _taskList = [];

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      resultsEl.innerHTML = '<div class="err">Error: ' + (d.error || resp.statusText) + '</div>';
      resultsEl.style.display = 'block';
      return;
    }

    document.getElementById('spinner').style.display = 'none';
    resultsEl.style.display = 'flex';
    resultsEl.style.flexDirection = 'column';
    resultsEl.style.overflow = 'hidden';

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        _handleSSEFrame(frame, resultsEl);
      }
    }
  } catch (e) {
    resultsEl.innerHTML = '<div class="err">Error de conexión: ' + e.message + '</div>';
    resultsEl.style.display = 'block';
  } finally {
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
  }
}

function _handleSSEFrame(frame, resultsEl) {
  let eventType = 'message', dataStr = '';
  for (const line of frame.split('\n')) {
    if (line.startsWith('event: ')) eventType = line.slice(7).trim();
    else if (line.startsWith('data: ')) dataStr += line.slice(6);
  }
  if (!dataStr) return;
  let data;
  try { data = JSON.parse(dataStr); } catch { return; }

  if (eventType === 'init')   _handleInit(data, resultsEl);
  else if (eventType === 'result') _handleResult(data);
  else if (eventType === 'done')   _handleDone();
}

function _handleInit(data, resultsEl) {
  _sessionId = data.session_id || null;
  _totalTasks = data.total;
  _taskList   = data.task_list || [];
  _taskList.forEach(t => { _taskStatus[t.id] = 'pending'; });

  // Build available-tabs list for export modal
  const mlbls = data.model_labels || {};
  _availableTabs = [];
  (data.reg_models || []).forEach(m => {
    _availableTabs.push({tab_id: 'reg-' + m, label: (mlbls[m] || m) + ' (regresión)'});
  });
  (data.cls_models || []).forEach(m => {
    _availableTabs.push({tab_id: 'cls-' + m, label: (mlbls[m] || m) + ' (clasificación)'});
  });

  // Build skeleton HTML
  resultsEl.innerHTML = _buildSkeleton(data);

  // Inject EDA
  const edaPanel = document.getElementById('panel-eda');
  if (edaPanel && data.eda_html) { edaPanel.innerHTML = data.eda_html; execScripts(edaPanel); }

  // Cache bounds sent with the init event
  if (data.bounds_by_tab) Object.assign(_boundsCache, data.bounds_by_tab);

  // Pre-populate sweep tables with pending rows
  _initSweepTables(data.sweep_combo_list || []);

  // Activate first tab
  const firstBtn = resultsEl.querySelector('.tab-btn');
  if (firstBtn) showTab(firstBtn.dataset.tab);

  _updateProgress();
}

function _buildSkeleton(data) {
  const mlbls = data.model_labels || {};

  // Progress bar
  let html = '<div class="progress-wrap">' +
    '<div class="progress-track">' +
      '<div class="progress-fill" id="prog-fill" style="width:0%"></div>' +
      '<span class="progress-label" id="prog-label">0 / ' + data.total + ' análisis</span>' +
    '</div>' +
    '<div class="progress-popup" id="prog-popup"></div>' +
  '</div>';

  let tabBar = '<button class="tab-btn" data-tab="eda" onclick="showTab(\'eda\')">Análisis Exploratorio</button>';
  let panels = '<div class="tab-panel" id="panel-eda"></div>';

  (data.reg_models || []).forEach(m => {
    const tid = 'reg-' + m;
    const lbl = (mlbls[m] || m) + ' (reg)';
    tabBar += '<button class="tab-btn" data-tab="' + tid + '" onclick="showTab(\'' + tid + '\')">' + lbl + '</button>';
    panels += _modelPanel(tid, 'regression', (data.sweep_reg_models || []).includes(m));
  });
  (data.cls_models || []).forEach(m => {
    const tid = 'cls-' + m;
    const lbl = (mlbls[m] || m) + ' (cls)';
    tabBar += '<button class="tab-btn" data-tab="' + tid + '" onclick="showTab(\'' + tid + '\')">' + lbl + '</button>';
    panels += _modelPanel(tid, 'classification', (data.sweep_cls_models || []).includes(m));
  });

  html += '<div class="tab-bar">' + tabBar + '</div>';
  html += '<div class="tab-panels">' + panels + '</div>';
  return html;
}

function _modelPanel(tabId, taskType, hasSweep) {
  const cols = taskType === 'regression' ? REG_COLS : CLS_COLS;
  const ths = cols.map(c => {
    const tip = COL_TOOLTIPS[c];
    return tip ? `<th data-tip="${tip}">${c} &#9432;</th>` : `<th>${c}</th>`;
  }).join('');
  const tbId = 'tbody-' + tabId;
  const chId = 'charts-' + tabId;
  // Extract model name from tabId (format: "reg-{model_name}" or "cls-{model_name}")
  const modelName = tabId.replace(/^(reg|cls)-/, '');
  let html = '<div class="tab-panel" id="panel-' + tabId + '">';
  html += '<div class="panel-section"><h3>M\u00e9tricas</h3>' +
    '<div class="panel-section"><table class="tbl dt-table" id="tbl-' + tabId + '">' +
    '<thead><tr>' + ths + '<th class="opt-col">&#9889;</th></tr></thead>' +
    '<tbody id="' + tbId + '"></tbody>' +
    '<tfoot><tr>' + ths + '<th class="opt-col">&#9889;</th></tr></tfoot>' +
    '</table></div>';
  html += '<div class="panel-section"><h3>Gr\u00e1ficas</h3><div id="' + chId + '"></div></div>';
  if (hasSweep) {
    html += '<div class="panel-section" id="sweep-panel-' + tabId + '" style="display:none">' +
      '<h3>Sweep de hiperpar\u00e1metros</h3>' +
      '<div id="sweep-content-' + tabId + '"></div></div>';
  }
  html += '</div>';  // close panel-section (Métricas)
  html += '</div>';  // close tab-panel
  return html;
}  // tabId → last objective string

function _handleResult(data) {
  // algo_sweep_charts is emitted automatically (not a counted task)
  if (data.task !== 'algo_sweep_charts') {
    _doneTasks++;
    _taskStatus[data.task_id] = data.error ? 'error' : 'done';
    _updateProgress();
    // For sweep_combo errors we still want to update the row, so don't return early
    if (data.error && data.task !== 'sweep_combo') return;
  }

  const prefix = data.task === 'regression' ? 'reg' : data.task === 'classification' ? 'cls' : null;
  const cols   = data.task === 'regression' ? REG_COLS : CLS_COLS;

  if (prefix) {
    for (const [modelName, mData] of Object.entries(data.models || {})) {
      const tabId = prefix + '-' + modelName;

      // Append metrics row
      const tbody = document.getElementById('tbody-' + tabId);
      if (tbody) {
        const tr = document.createElement('tr');
        const isWinner = mData.metrics && mData.metrics.rank === 1;
        if (isWinner) tr.className = 'winner-row';
        cols.forEach(col => {
          const td = document.createElement('td');
          if (col === 'objective') {
            td.textContent = data.obj;
          } else if (col === 'rank' && isWinner) {
            td.textContent = '🏆 1';
          } else {
            const v = mData.metrics ? mData.metrics[col] : undefined;
            td.textContent = v != null ? (typeof v === 'number' ? v.toFixed(4) : v) : '—';
          }
          tr.appendChild(td);
        });
        // Per-row optimize button
        const optTd = document.createElement('td');
        optTd.className = 'opt-col';
        const safeObj = data.obj.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        optTd.innerHTML = '<button class="row-opt-btn" onclick="openOptModal(\'' +
          tabId + '\', \'' + modelName + '\', \'' + safeObj + '\')" title="Optimizar con este objetivo">&#9889;</button>';
        tr.appendChild(optTd);
        tbody.appendChild(tr);
      }

      // Append collapsible chart group
      const chartsEl = document.getElementById('charts-' + tabId);
      if (chartsEl && mData.chart_html) {
        const det = document.createElement('details');
        det.className = 'obj-group';
        det.innerHTML = '<summary>' + data.obj + '</summary>' +
          '<div class="obj-body"><div class="chart-wrap">' + mData.chart_html + '</div></div>';
        chartsEl.appendChild(det);
        execScripts(det);
      }

      // Register for multi-objective panel
      _registerObjSpec(tabId, modelName, data.obj);
    }
  } else if (data.task === 'sweep_combo') {
    // Update the pre-populated pending row for this combo
    const row = document.querySelector('[data-combo-id="' + data.combo_id + '"]');
    if (row) {
      if (data.error) {
        row.classList.add('sweep-row-error');
        row.querySelectorAll('td[data-metric]').forEach(cell => {
          cell.className = 'sweep-cell-error';
          cell.innerHTML = '&#x2717;';
        });
      } else {
        row.querySelectorAll('td[data-metric]').forEach(cell => {
          const metric = cell.dataset.metric;
          const v = data.row ? data.row[metric] : undefined;
          cell.className = 'sweep-cell-done';
          cell.innerHTML = (v == null) ? '&mdash;' : (typeof v === 'number' ? v.toFixed(4) : v);
        });
      }
    }
  } else if (data.task === 'algo_sweep_charts') {
    // Charts arrive once all combos for a group complete
    if (data.chart_html) {
      const container = document.querySelector('[data-sweep-charts="' + data.safe_key + '"]');
      if (container) { container.innerHTML = data.chart_html; execScripts(container); }
    }
  } else if (data.task === 'algo_sweep') {
    // Legacy handler (kept for safety; no longer emitted by the server)
    const sweepPanel   = document.getElementById('sweep-panel-' + data.tab_id);
    const sweepContent = document.getElementById('sweep-content-' + data.tab_id);
    if (sweepPanel)  sweepPanel.style.display = 'block';
    if (sweepContent && (data.table_html || data.chart_html)) {
      const det = document.createElement('details');
      det.className = 'obj-group';
      det.innerHTML = '<summary>' + data.obj + '</summary>' +
        '<div class="obj-body">' + (data.table_html || '') + (data.chart_html || '') + '</div>';
      sweepContent.appendChild(det);
      execScripts(det);
    }
  }
}

function _handleDone() {
  initDataTables();
  const eb = document.getElementById('export-results-btn');
  if (eb) eb.disabled = false;
}

// ---------- Sweep table pre-population ----------
function _initSweepTables(comboList) {
  if (!comboList || !comboList.length) return;

  // Store combo params for sweep-row optimization
  for (const c of comboList) {
    const mn = c.tab_id.replace(/^(reg|cls)-/, '');
    _sweepComboParams[c.combo_id] = {
      tab_id: c.tab_id, model_name: mn,
      obj: c.obj, task_type: c.task_type,
      params: c.params,
    };
  }

  // Group by tab_id → obj
  const byTab = {};
  for (const c of comboList) {
    if (!byTab[c.tab_id]) byTab[c.tab_id] = {};
    if (!byTab[c.tab_id][c.obj]) byTab[c.tab_id][c.obj] = [];
    byTab[c.tab_id][c.obj].push(c);
  }

  for (const [tabId, objGroups] of Object.entries(byTab)) {
    const panel   = document.getElementById('sweep-panel-' + tabId);
    const content = document.getElementById('sweep-content-' + tabId);
    if (!panel || !content) continue;
    panel.style.display = 'block';

    for (const [obj, combos] of Object.entries(objGroups)) {
      const isReg      = combos[0].task_type === 'regression';
      const paramNames = Object.keys(combos[0].params || {});
      const metricCols = isReg
        ? ['cv_mae','cv_rmse','cv_r2','cv_rmse_std','holdout_mae','holdout_rmse','holdout_r2']
        : ['cv_accuracy','cv_f1_macro','cv_balanced_accuracy',
           'holdout_accuracy','holdout_f1_macro','holdout_balanced_accuracy'];
      const safeKey  = combos[0].safe_key;
      const allCols  = [...paramNames, ...metricCols];

      const theadCells = allCols.map(c => {
        const tip = COL_TOOLTIPS[c];
        return tip ? `<th data-tip="${tip}">${c} &#9432;</th>` : `<th>${c}</th>`;
      }).join('') + '<th>Acción</th>';

      const tbodyRows = combos.map(c => {
        const paramCells = paramNames.map(p => {
          const v = c.params ? c.params[p] : '';
          const disp = Array.isArray(v) ? v.join(',') : (v == null ? '&mdash;' : v);
          return `<td>${disp}</td>`;
        }).join('');
        const metricCells = metricCols.map(m =>
          `<td class="sweep-cell-pending" data-metric="${m}"><span class="combo-spinner-inline"></span></td>`
        ).join('');
        const optBtn = `<td><button class="sweep-opt-btn" onclick="openOptFromSweep('${c.combo_id}')" title="Optimizar con estos parámetros">&#9889;</button></td>`;
        return `<tr data-combo-id="${c.combo_id}">${paramCells}${metricCells}${optBtn}</tr>`;
      }).join('');

      const det = document.createElement('details');
      det.className = 'obj-group';
      det.setAttribute('data-sweep-obj', obj);
      det.innerHTML =
        `<summary>${obj} &mdash; ${combos.length} combinaciones</summary>` +
        `<div class="obj-body">` +
          `<div class="sweep-filter-bar">` +
            `<input class="sweep-filter-input" type="search" placeholder="Filtrar filas\u2026" aria-label="Filtrar">` +
            `<span class="sweep-filter-count"></span>` +
          `</div>` +
          `<div style="overflow-x:auto"><table class="tbl sweep-live-tbl">` +
            `<thead><tr>${theadCells}</tr></thead>` +
            `<tbody>${tbodyRows}</tbody>` +
          `</table></div>` +
          `<div class="chart-wrap" data-sweep-charts="${safeKey}"></div>` +
        `</div>`;
      content.appendChild(det);
      _makeSweepInteractive(det);
    }
  }
}
function _makeSweepInteractive(det) {
  const table      = det.querySelector('table.sweep-live-tbl');
  const filterInput = det.querySelector('.sweep-filter-input');
  const countEl    = det.querySelector('.sweep-filter-count');
  if (!table) return;

  // --- Sort ---
  let _sortCol = -1, _sortAsc = true;
  table.querySelectorAll('thead th').forEach((th, idx) => {
    if (th.textContent.trim() === 'Acción') return;
    th.classList.add('sortable-th');
    th.addEventListener('click', () => {
      if (_sortCol === idx) { _sortAsc = !_sortAsc; }
      else { _sortCol = idx; _sortAsc = true; }
      table.querySelectorAll('thead th').forEach(h => h.classList.remove('sort-asc','sort-desc'));
      th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
      _sortSweepTable(table, idx, _sortAsc);
      _updateSweepCount(table, countEl);
    });
  });

  // --- Filter ---
  if (filterInput) {
    filterInput.addEventListener('input', () => {
      _filterSweepTable(table, filterInput.value);
      _updateSweepCount(table, countEl);
    });
  }

  _updateSweepCount(table, countEl);
}

function _sortSweepTable(table, colIdx, asc) {
  const tbody = table.querySelector('tbody');
  const rows = [...tbody.rows];
  rows.sort((a, b) => {
    const ta = a.cells[colIdx] ? a.cells[colIdx].textContent.trim() : '';
    const tb = b.cells[colIdx] ? b.cells[colIdx].textContent.trim() : '';
    const na = parseFloat(ta), nb = parseFloat(tb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? ta.localeCompare(tb) : tb.localeCompare(ta);
  });
  rows.forEach(r => tbody.appendChild(r));
}

function _filterSweepTable(table, query) {
  const q = query.trim().toLowerCase();
  table.querySelectorAll('tbody tr').forEach(tr => {
    if (!q) { tr.classList.remove('sweep-row-hidden'); return; }
    const text = [...tr.cells].map(td => td.textContent).join(' ').toLowerCase();
    tr.classList.toggle('sweep-row-hidden', !text.includes(q));
  });
}

function _updateSweepCount(table, countEl) {
  if (!countEl) return;
  const total   = table.querySelectorAll('tbody tr').length;
  const visible = table.querySelectorAll('tbody tr:not(.sweep-row-hidden)').length;
  countEl.textContent = visible < total ? `${visible} de ${total}` : `${total} filas`;
}

function _updateProgress() {
  const fill  = document.getElementById('prog-fill');
  const label = document.getElementById('prog-label');
  const popup = document.getElementById('prog-popup');
  if (!fill) return;
  const pct = _totalTasks > 0 ? (_doneTasks / _totalTasks * 100) : 0;
  fill.style.width = pct + '%';
  label.textContent = _doneTasks + ' / ' + _totalTasks + ' análisis';
  if (popup) {
    popup.innerHTML = _taskList.map(t => {
      const st = _taskStatus[t.id] || 'pending';
      const icon = st === 'done' ? '✓' : st === 'error' ? '✗' : '⏳';
      return '<div class="ptask ' + st + '"><span class="picon">' + icon + '</span>' +
             '<span class="pname">' + t.label + '</span></div>';
    }).join('');
  }
}

function initDataTables() {
  if (!window.jQuery || !$.fn.DataTable) return;
  $('.dt-table').each(function() {
    if (!$.fn.DataTable.isDataTable(this)) {
      $(this).DataTable({
        pageLength: 25,
        language: {
          search: 'Buscar:', lengthMenu: 'Mostrar _MENU_ registros',
          info: 'Mostrando _START_ a _END_ de _TOTAL_ registros',
          infoEmpty: 'Sin resultados',
          paginate: { first:'Primera', last:'Última', next:'Siguiente', previous:'Anterior' },
        },
        initComplete: function() {
          this.api().columns().every(function() {
            const col = this;
            $('<input type="text" placeholder="' + $(col.header()).text() + '" />')
              .appendTo($(col.footer()).empty())
              .on('keyup change clear', function() { if (col.search() !== this.value) col.search(this.value).draw(); });
          });
        },
      });
    }
  });
}

// ---------- Export ----------
function exportConfig() {
  const reg = {}, cls = {};
  document.querySelectorAll('[data-type="regression"]:checked').forEach(c => {
    reg[c.dataset.algo] = algoConfigs[c.dataset.algo] || null;
  });
  document.querySelectorAll('[data-type="classification"]:checked').forEach(c => {
    cls[c.dataset.algo] = algoConfigs[c.dataset.algo] || null;
  });
  const config = {
    exported_at: new Date().toISOString(),
    variables: {
      input:  [...document.querySelectorAll('#col-role-body input[value="input"]:checked')].map(r => r.dataset.col),
      output: [...document.querySelectorAll('#col-role-body input[value="output"]:checked')].map(r => r.dataset.col),
    },
    algorithms: { regression: reg, classification: cls },
    options: { class_q: parseInt(document.getElementById('class-q').value, 10) || 3 },
  };
  const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'analysis_config.json';
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
}

function openExportModal() {
  if (!_sessionId) { alert('Ejecute un análisis antes de exportar resultados.'); return; }
  const body = document.getElementById('export-modal-body');
  body.innerHTML =
    '<p class="exp-section">Incluir en el ZIP</p>' +
    '<label class="exp-check-row"><input type="checkbox" id="exp-config" checked> Configuración (config.json)</label>' +
    '<label class="exp-check-row"><input type="checkbox" id="exp-eda" checked> Análisis exploratorio (EDA)</label>' +
    '<p class="exp-section">Análisis disponibles</p>' +
    _availableTabs.map(t =>
      '<label class="exp-check-row"><input type="checkbox" class="exp-tab-chk" value="' + t.tab_id + '" checked> ' + t.label + '</label>'
    ).join('') +
    '<div class="exp-sel-btns">' +
    '<button class="btn btn-sec" onclick="expSelectAll(true)">Todos</button>' +
    '<button class="btn btn-sec" onclick="expSelectAll(false)">Ninguno</button>' +
    '</div>';
  document.getElementById('export-overlay').classList.remove('hidden');
}

function closeExportModal() { document.getElementById('export-overlay').classList.add('hidden'); }
function exportOverlayClick(e) { if (e.target === document.getElementById('export-overlay')) closeExportModal(); }
function expSelectAll(v) { document.querySelectorAll('.exp-tab-chk').forEach(c => c.checked = v); }

async function doExport() {
  const selected = [...document.querySelectorAll('.exp-tab-chk:checked')].map(c => c.value);
  const btn = document.getElementById('do-export-btn');
  btn.disabled = true; btn.textContent = 'Generando…';
  try {
    const resp = await fetch('/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: _sessionId,
        selected,
        include_config: document.getElementById('exp-config').checked,
        include_eda: document.getElementById('exp-eda').checked,
      }),
    });
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      alert('Error al exportar: ' + (d.error || resp.statusText));
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'analysis_export.zip';
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
    closeExportModal();
  } finally {
    btn.disabled = false; btn.textContent = '⬇ Generar ZIP';
  }
}

async function fetchDataInfo() {
  try {
    const r = await fetch('/data_info');
    const d = await r.json();
    document.getElementById('data-info').textContent = d.n_obs + ' observaciones · ' + d.n_cols + ' variables';
    _populateColSelectors(d.columns || []);
  } catch {
    document.getElementById('data-info').textContent = 'Datos no disponibles';
  }
}

function _populateColSelectors(cols) {
  const tbody = document.getElementById('col-role-body');
  tbody.innerHTML = '';

  const numeric = cols.filter(c => c.is_numeric);
  if (!numeric.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="col-sel-none">Sin columnas numéricas</td></tr>';
    _updateVarBadge();
    _rebuildCombos();
    return;
  }
  numeric.forEach((c, i) => {
    const uid = 'cr' + i;
    const tr = document.createElement('tr');
    tr.className = 'role-none';
    tr.innerHTML =
      `<td title="${c.name}">${c.name}</td>` +
      `<td><input type="radio" name="${uid}" value="input" data-col="${c.name}" onchange="onRoleChange(this)"></td>` +
      `<td><input type="radio" name="${uid}" value="output" data-col="${c.name}" onchange="onRoleChange(this)"></td>` +
      `<td><input type="radio" name="${uid}" value="none" data-col="${c.name}" checked onchange="onRoleChange(this)"></td>`;
    tbody.appendChild(tr);
  });
  _updateVarBadge();
  _rebuildCombos();
}

function onRoleChange(radio) {
  const tr = radio.closest('tr');
  tr.className = 'role-' + radio.value;
  _rebuildCombos();
}

function _rebuildCombos() {
  const outputs = [...document.querySelectorAll('#col-role-body input[value="output"]:checked')]
    .map(r => r.dataset.col);
  const listEl = document.getElementById('combo-list');
  if (!outputs.length) {
    listEl.innerHTML = '<span class="combo-empty">Seleccione al menos una variable de salida.</span>';
    return;
  }
  // Generate all non-empty subsets ordered by size
  const combos = [];
  for (let r = 1; r <= outputs.length; r++) {
    _combinations(outputs, r).forEach(c => combos.push(c));
  }
  // Preserve existing checked state by key
  const prevState = {};
  listEl.querySelectorAll('input[type=checkbox][data-combo-key]').forEach(cb => {
    prevState[cb.dataset.comboKey] = cb.checked;
  });
  listEl.innerHTML = '';
  combos.forEach(combo => {
    const key = combo.join('\x00');
    const checked = key in prevState ? prevState[key] : true;
    const label = document.createElement('label');
    label.className = 'combo-item';
    const sizeLabel = combo.length === 1 ? 'individual' :
      combo.length === 2 ? 'par' : combo.length + ' variables';
    label.innerHTML =
      `<input type="checkbox" data-combo-key="${key}" ${checked ? 'checked' : ''}>` +
      `<span>${combo.join(' + ')}</span>` +
      `<span class="combo-size">${sizeLabel}</span>`;
    listEl.appendChild(label);
  });
}

function _combinations(arr, r) {
  if (r === 1) return arr.map(x => [x]);
  const result = [];
  arr.forEach((v, i) => {
    _combinations(arr.slice(i + 1), r - 1).forEach(rest => result.push([v, ...rest]));
  });
  return result;
}

function _combosSelectAll(checked) {
  document.querySelectorAll('#combo-list input[type=checkbox]')
    .forEach(cb => { cb.checked = checked; });
}

function _getSelectedCombos() {
  const result = [];
  document.querySelectorAll('#combo-list input[type=checkbox]:checked').forEach(cb => {
    result.push(cb.dataset.comboKey.split('\x00'));
  });
  return result;
}

function _updateVarBadge() {
  const ins  = [...document.querySelectorAll('#col-role-body input[value="input"]:checked')].length;
  const outs = [...document.querySelectorAll('#col-role-body input[value="output"]:checked')].length;
  const badge = document.getElementById('var-badge');
  if (ins === 0 && outs === 0) { badge.textContent = 'Sin variables seleccionadas'; return; }
  const parts = [];
  if (ins)  parts.push(ins  + ' entrada' + (ins  > 1 ? 's' : ''));
  if (outs) parts.push(outs + ' salida'  + (outs > 1 ? 's' : ''));
  badge.textContent = parts.join(' · ');
}

// ---------- Variables modal ----------
function openVarModal() {
  document.getElementById('var-overlay').classList.remove('hidden');
}
function closeVarModal() {
  document.getElementById('var-overlay').classList.add('hidden');
}
function varOverlayClick(e) {
  if (e.target === document.getElementById('var-overlay')) closeVarModal();
}
function saveVarModal() {
  _updateVarBadge();
  closeVarModal();
}

// =====================================================================
// Optimization UI
// =====================================================================
let _curOptTabId = null;
let _curOptModelName = null;
let _curOptObjective = null;
let _curOptBoundsData = null;
let _curSweepParams = null;   // params dict from a selected sweep row (null = use default config)

// Map combo_id → {tab_id, model_name, obj, task_type, params}
const _sweepComboParams = {};
// Bounds cache: tab_id → {col: [lo, hi]} (populated from init payload)
const _boundsCache = {};

async function openOptModal(tabId, modelName, objective, sweepParams) {
  _curOptTabId = tabId;
  _curOptModelName = modelName;
  _curOptObjective = objective;
  _curSweepParams = sweepParams || null;

  const sweepLabel = _curSweepParams
    ? ' <span class="sweep-config-badge" title="' +
        Object.entries(_curSweepParams).map(([k,v]) => k+'='+v).join(', ') +
        '">&#128295; Config del sweep</span>'
    : '';
  document.getElementById('opt-modal-title').innerHTML =
    'Optimizar: ' + (modelName || tabId) + ' &mdash; ' + objective + sweepLabel;

  // Show/hide multi-obj button based on available specs
  const multiBtn = document.getElementById('opt-multi-btn');
  if (multiBtn) multiBtn.style.display = _availableObjSpecs.length >= 2 ? 'inline-block' : 'none';

  // Reset result column to placeholder
  document.getElementById('opt-result-placeholder').style.display = 'flex';
  document.getElementById('opt-result-box').style.display = 'none';
  document.getElementById('opt-error-box').style.display = 'none';
  document.getElementById('opt-convergence-chart').innerHTML = '';

  // Populate algorithm radio buttons from OPT_REGISTRY
  const algos = OPT_REGISTRY[modelName] || ['bayesian'];
  const grp = document.getElementById('opt-algo-group');
  grp.innerHTML = '';
  algos.forEach((a, i) => {
    const lbl = OPT_ALGO_LABELS[a] || a;
    const badge = i === 0 ? '<span class="algo-badge">Recomendado</span>' : '';
    grp.innerHTML +=
      '<label class="algo-radio-opt">' +
        '<input type="radio" name="opt-algo" value="' + a + '"' + (i === 0 ? ' checked' : '') + '> ' +
        lbl + badge +
      '</label>';
  });

  // Set default iterations based on algorithm
  const firstAlgo = algos[0] || 'bayesian';
  document.getElementById('opt-n-iter').value =
    (firstAlgo === 'pso' || firstAlgo === 'de') ? 100 : 50;

  document.getElementById('opt-result-box').style.display = 'none';
  document.getElementById('opt-overlay').classList.remove('hidden');

  const boundsBody = document.getElementById('opt-bounds-body');

  function _fillBounds(boundsData) {
    if (!boundsData || !Object.keys(boundsData).length) {
      boundsBody.innerHTML = '<tr><td colspan="3" style="color:#94a3b8;font-size:.75rem">No se encontraron límites para este modelo.</td></tr>';
      return;
    }
    _curOptBoundsData = boundsData;
    boundsBody.innerHTML = '';
    for (const [col, [lo, hi]] of Object.entries(boundsData)) {
      boundsBody.innerHTML +=
        '<tr>' +
          '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + col + '">' + col + '</td>' +
          '<td><input type="number" id="blo-' + CSS.escape(col) + '" value="' + lo.toFixed(4) + '" step="any"></td>' +
          '<td><input type="number" id="bhi-' + CSS.escape(col) + '" value="' + hi.toFixed(4) + '" step="any"></td>' +
        '</tr>';
    }
  }

  // Use pre-cached bounds (sent with the init event) — instant, no server roundtrip
  if (_boundsCache[tabId]) {
    _fillBounds(_boundsCache[tabId]);
  } else {
    // Fallback: fetch from server (needed if modal opened before init or cache miss)
    boundsBody.innerHTML = '<tr><td colspan="3" style="color:#94a3b8;font-size:.75rem">Cargando límites…</td></tr>';
    try {
      const r = await fetch('/bounds?session_id=' + encodeURIComponent(_sessionId) +
                            '&tab_id=' + encodeURIComponent(tabId));
      const data = await r.json();
      if (data.error) {
        boundsBody.innerHTML = '<tr><td colspan="3" class="err">' + data.error + '</td></tr>';
      } else {
        _boundsCache[tabId] = data;
        _fillBounds(data);
      }
    } catch (e) {
      boundsBody.innerHTML = '<tr><td colspan="3" class="err">Error al obtener límites: ' + e.message + '</td></tr>';
    }
  }
}

function closeOptModal() {
  document.getElementById('opt-overlay').classList.add('hidden');
}
function optOverlayClick(e) {
  if (e.target === document.getElementById('opt-overlay')) closeOptModal();
}

// Called when user clicks ⚡ on a sweep table row
function openOptFromSweep(comboId) {
  const spec = _sweepComboParams[comboId];
  if (!spec) return;
  openOptModal(spec.tab_id, spec.model_name, spec.obj, spec.params);
}

async function runOptimization() {
  const algo = document.querySelector('input[name="opt-algo"]:checked')?.value || 'bayesian';
  const direction = document.querySelector('input[name="opt-dir"]:checked')?.value || 'minimize';
  const nIter = parseInt(document.getElementById('opt-n-iter').value, 10) || 50;

  // Collect bounds
  const bounds = {};
  if (_curOptBoundsData) {
    for (const col of Object.keys(_curOptBoundsData)) {
      const escaped = CSS.escape(col);
      const loEl = document.getElementById('blo-' + escaped);
      const hiEl = document.getElementById('bhi-' + escaped);
      if (loEl && hiEl) {
        bounds[col] = [parseFloat(loEl.value), parseFloat(hiEl.value)];
      }
    }
  }

  const resultBox = document.getElementById('opt-result-box');
  const errorBox  = document.getElementById('opt-error-box');
  const btn = document.getElementById('opt-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="opt-spinner"></span> Optimizando…';
  document.getElementById('opt-result-placeholder').style.display = 'none';
  resultBox.style.display = 'none';
  errorBox.style.display  = 'none';

  try {
    const resp = await fetch('/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: _sessionId,
        tab_id: _curOptTabId,
        model_name: _curOptModelName,
        objective: _curOptObjective,
        direction,
        algo,
        n_iter: nIter,
        bounds,
        sweep_params: _curSweepParams || undefined,
      }),
    });
    const data = await resp.json();

    if (data.error) {
      errorBox.style.display = 'block';
      document.getElementById('opt-error-title').textContent = 'Error al optimizar';
      document.getElementById('opt-error-msg').textContent = data.error;
    } else {
      resultBox.style.display = 'block';

      // ── Optimal inputs table ──
      const tbody = document.getElementById('opt-params-body');
      tbody.innerHTML = '';
      for (const [k, v] of Object.entries(data.optimal_inputs || {})) {
        tbody.innerHTML +=
          '<tr><td title="' + k + '">' + k + '</td>' +
          '<td><strong>' + Number(v).toFixed(4) + '</strong></td></tr>';
      }

      // ── All predictions table ──
      const allPreds = data.all_predictions || {};
      const primaryComponents = new Set(data.primary_components || []);
      // When the target is a combined objective, highlight its components instead of
      // the aggregated mean row (which is less meaningful to the user).
      const isCombo = primaryComponents.size > 0;
      const predBody = document.getElementById('opt-all-preds-body');
      predBody.innerHTML = '';
      const predEntries = Object.entries(allPreds);
      if (predEntries.length > 0) {
        document.getElementById('opt-allpreds-label').style.display = '';
        document.getElementById('opt-all-preds-table').style.display = '';
        // Sort: primary components first, then the combined mean row (if combo), then rest
        const sorted = [...predEntries].sort((a, b) => {
          const aComp = primaryComponents.has(a[0]) ? 2 : (a[0] === _curOptObjective ? (isCombo ? -1 : 2) : 0);
          const bComp = primaryComponents.has(b[0]) ? 2 : (b[0] === _curOptObjective ? (isCombo ? -1 : 2) : 0);
          return bComp - aComp;
        });
        for (const [obj, val] of sorted) {
          const isComponent = primaryComponents.has(obj);
          const isOpt = isCombo ? isComponent : obj === _curOptObjective;
          const isCombinedMean = isCombo && obj === _curOptObjective;
          // Make label readable: replace _ with space
          const shortLabel = obj.replace(/_/g, ' ');
          const dispVal = val == null ? '—'
            : typeof val === 'number' ? val.toFixed(4)
            : val;
          const dirMark = isOpt
            ? (direction === 'minimize' ? ' &#8595;min' : ' &#8593;max')
            : '';
          const meanNote = isCombinedMean ? ' <span style="font-weight:400;opacity:.6;font-size:.72rem">(media)</span>' : '';
          predBody.innerHTML +=
            '<tr class="' + (isOpt ? 'preds-opt' : '') + '">' +
            '<td title="' + obj + '" style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' +
            shortLabel + dirMark + meanNote + '</td>' +
            '<td><strong>' + dispVal + '</strong></td></tr>';
        }
      } else {
        document.getElementById('opt-allpreds-label').style.display = 'none';
        document.getElementById('opt-all-preds-table').style.display = 'none';
      }

      // ── Meta info ──
      const algLabel = OPT_ALGO_LABELS[data.algo_used] || data.algo_used;
      document.getElementById('opt-meta').innerHTML =
        '<strong>Algoritmo:</strong> ' + algLabel +
        ' &nbsp;·&nbsp; <strong>Evaluaciones:</strong> ' + (data.n_evaluations || '—') +
        ' &nbsp;·&nbsp; <strong>Tiempo:</strong> ' +
        (data.duration_sec != null ? data.duration_sec.toFixed(1) + 's' : '—');

      // ── Convergence chart ──
      const convergence = data.convergence || [];
      const cvEl = document.getElementById('opt-convergence-chart');
      if (convergence.length > 1) {
        const fig = {
          data: [{
            x: convergence.map((_, i) => i + 1),
            y: convergence,
            type: 'scatter', mode: 'lines+markers',
            name: 'Mejor valor', line: { color: '#7c3aed', width: 2 },
            marker: { size: 4 },
            hovertemplate: 'Iter %{x}: %{y:.6f}<extra></extra>',
          }],
          layout: {
            title: { text: 'Curva de convergencia', font: { size: 12 } },
            xaxis: { title: 'Iteración', titlefont: { size: 11 } },
            yaxis: { title: direction === 'minimize' ? 'Valor mínimo' : 'Valor máximo', titlefont: { size: 11 } },
            height: 200, margin: { t: 30, b: 40, l: 56, r: 10 },
            responsive: true,
          },
        };
        cvEl.innerHTML = '';
        Plotly.newPlot(cvEl, fig.data, fig.layout, { responsive: true });
      } else {
        cvEl.innerHTML = '';
      }
    }
  } catch (e) {
    errorBox.style.display = 'block';
    document.getElementById('opt-error-title').textContent = 'Error de conexión';
    document.getElementById('opt-error-msg').textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Optimizar';
  }
}

// =====================================================================
// Multi-objective optimization panel
// =====================================================================
let _availableObjSpecs = [];   // [{tab_id, model_name, objective, label}]

function _registerObjSpec(tabId, modelName, objective) {
  const key = tabId + '||' + modelName + '||' + objective;
  if (!_availableObjSpecs.find(s => s.key === key)) {
    _availableObjSpecs.push({
      key, tab_id: tabId, model_name: modelName, objective,
      label: modelName + ' — ' + objective,
    });
    _updateMultiObjSelects();
  }
}

function _updateMultiObjSelects() {
  ['multi-obj-sel-1', 'multi-obj-sel-2'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = _availableObjSpecs.map(s =>
      '<option value="' + s.key + '"' + (s.key === cur ? ' selected' : '') + '>' + s.label + '</option>'
    ).join('');
  });
  // Enable multi-obj button in opt-modal footer when >=2 objectives are available
  const btn = document.getElementById('opt-multi-btn');
  if (btn && _availableObjSpecs.length >= 2) btn.style.display = 'inline-block';
}

function openMultiObjModal() {
  _updateMultiObjSelects();
  document.getElementById('multi-obj-overlay').classList.remove('hidden');
}
function closeMultiObjModal() {
  document.getElementById('multi-obj-overlay').classList.add('hidden');
}
function multiObjOverlayClick(e) {
  if (e.target === document.getElementById('multi-obj-overlay')) closeMultiObjModal();
}

async function runMultiObjOpt() {
  const sel1 = document.getElementById('multi-obj-sel-1')?.value;
  const sel2 = document.getElementById('multi-obj-sel-2')?.value;
  const dir1 = document.querySelector('input[name="mdir-1"]:checked')?.value || 'minimize';
  const dir2 = document.querySelector('input[name="mdir-2"]:checked')?.value || 'minimize';
  const nGen = parseInt(document.getElementById('multi-n-gen').value, 10) || 50;
  const popSize = parseInt(document.getElementById('multi-pop-size').value, 10) || 40;

  const spec1 = _availableObjSpecs.find(s => s.key === sel1);
  const spec2 = _availableObjSpecs.find(s => s.key === sel2);
  if (!spec1 || !spec2) { alert('Seleccione dos objetivos diferentes.'); return; }

  const btn = document.getElementById('multi-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="opt-spinner"></span> Ejecutando NSGA-II…';
  document.getElementById('pareto-chart-container').innerHTML = '<span style="color:#6b7280;font-size:.8rem">Computando frente de Pareto…</span>';

  try {
    const resp = await fetch('/optimize-multi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: _sessionId,
        objectives: [
          { tab_id: spec1.tab_id, model_name: spec1.model_name, objective: spec1.objective, direction: dir1 },
          { tab_id: spec2.tab_id, model_name: spec2.model_name, objective: spec2.objective, direction: dir2 },
        ],
        n_gen: nGen,
        pop_size: popSize,
      }),
    });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('pareto-chart-container').innerHTML =
        '<div class="err">' + data.error + '</div>';
      return;
    }

    // Build Pareto scatter plot
    const px = (data.pareto_predicted || []).map(p => {
      const vals = Object.values(p);
      return vals[0];
    });
    const py = (data.pareto_predicted || []).map(p => {
      const vals = Object.values(p);
      return vals[1] != null ? vals[1] : vals[0];
    });
    const objKeys = data.pareto_predicted?.length ? Object.keys(data.pareto_predicted[0]) : ['Obj 1', 'Obj 2'];
    const hoverTexts = (data.pareto_inputs || []).map((inp, i) =>
      Object.entries(inp).map(([k, v]) => k + ': ' + Number(v).toFixed(4)).join('<br>') +
      '<br>' + (objKeys[0] || 'Obj 1') + ': ' + (px[i] != null ? Number(px[i]).toFixed(6) : '—') +
      '<br>' + (objKeys[1] || 'Obj 2') + ': ' + (py[i] != null ? Number(py[i]).toFixed(6) : '—')
    );

    const fig = {
      data: [{
        x: px, y: py, mode: 'markers+lines',
        marker: { color: '#7c3aed', size: 8, opacity: 0.85 },
        text: hoverTexts, hoverinfo: 'text',
        name: 'Soluciones Pareto',
        line: { color: '#c4b5fd', width: 1, dash: 'dot' },
      }],
      layout: {
        title: { text: 'Frente de Pareto — NSGA-II', font: { size: 13 } },
        xaxis: { title: objKeys[0] || 'Objetivo 1' },
        yaxis: { title: objKeys[1] || 'Objetivo 2' },
        height: 380,
        margin: { t: 40, b: 50, l: 70, r: 20 },
        annotations: data.hypervolume != null ? [{
          text: 'Hipervolumen: ' + data.hypervolume.toFixed(4),
          xref: 'paper', yref: 'paper', x: 0.01, y: 0.97,
          xanchor: 'left', yanchor: 'top', showarrow: false,
          font: { size: 11, color: '#374151' },
          bgcolor: 'rgba(255,255,255,.75)',
        }] : [],
      },
    };
    document.getElementById('pareto-chart-container').innerHTML = '<div id="pareto-plot"></div>';
    Plotly.newPlot('pareto-plot', fig.data, fig.layout, { responsive: true });
  } catch (e) {
    document.getElementById('pareto-chart-container').innerHTML =
      '<div class="err">Error: ' + e.message + '</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Ejecutar NSGA-II';
  }
}

</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Per-model chart helpers (tab view)
# ---------------------------------------------------------------------------
import plotly.graph_objects as _go
import plotly.io as _pio
from plotly.subplots import make_subplots as _make_subplots
from sklearn.metrics import confusion_matrix as _cm


def _charts_html(charts: list[str]) -> str:
    return "".join(f'<div class="chart-wrap">{c}</div>' for c in charts)


def _wrap_chart_html(fragment: str, title: str = "Gráfica") -> str:
    """Wrap a Plotly HTML fragment into a standalone downloadable HTML page."""
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!DOCTYPE html>\n<html lang=\"es\">\n<head>"
        "<meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
        f"<title>{safe_title}</title>"
        "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>"
        "<style>body{margin:16px;font-family:'Segoe UI',Tahoma,sans-serif;color:#1f2937}"
        "h2{color:#0f172a;font-size:1rem;margin-bottom:10px}</style>\n</head>\n<body>"
        f"<h2>{safe_title}</h2>\n{fragment}\n</body>\n</html>"
    )


# ---------------------------------------------------------------------------
# JSON-safety + per-objective chart generators (used by task workers)
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/inf/numpy scalars to make obj JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def _strip_private(d: dict) -> dict:
    """Return a shallow copy of *d* with all underscore-prefixed keys removed.

    Private keys (``_fitted``, ``_x_df``, ``_y``, ``_class_q``) hold
    sklearn Pipelines and DataFrames which must not be JSON-serialised.
    ``_cache()`` reads them *before* this function strips them.
    """
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _reg_obj_chart_html(art: dict, obj_name: str, model_name: str) -> str:
    """Actual-vs-predicted + residual scatter (+ optional MLP loss curve) for one objective × model."""
    cfg = {"responsive": True}
    y_true = np.asarray(art["y_test"]).reshape(-1)
    y_pred = np.asarray(art["y_pred"]).reshape(-1)
    residuals = y_true - y_pred
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    fig = _make_subplots(rows=1, cols=2, subplot_titles=("Real vs Predicho", "Residuales"))
    fig.add_trace(_go.Scatter(x=y_true.tolist(), y=y_pred.tolist(), mode="markers",
                              marker=dict(color="#1f77b4", opacity=0.7),
                              hovertemplate="Real: %{x:.4f}<br>Pred: %{y:.4f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(_go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                              line=dict(color="red", dash="dash"), showlegend=False),
                  row=1, col=1)
    fig.add_trace(_go.Scatter(x=y_pred.tolist(), y=residuals.tolist(), mode="markers",
                              marker=dict(color="#ff7f0e", opacity=0.7),
                              hovertemplate="Pred: %{x:.4f}<br>Res: %{y:.4f}<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(_go.Scatter(x=[float(y_pred.min()), float(y_pred.max())], y=[0.0, 0.0],
                              mode="lines", line=dict(color="red", dash="dash"), showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(title_text="Real", row=1, col=1)
    fig.update_yaxes(title_text="Predicho", row=1, col=1)
    fig.update_xaxes(title_text="Predicho", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    fig.update_layout(height=300, showlegend=False, margin=dict(t=36, b=24))
    html = _pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg)

    # Loss curve (MLP only)
    loss_curve: list | None = art.get("loss_curve")
    if loss_curve and len(loss_curve) > 1:
        fig_lc = _go.Figure(_go.Scatter(
            x=list(range(1, len(loss_curve) + 1)),
            y=loss_curve,
            mode="lines",
            line=dict(color="#7c3aed", width=2),
            hovertemplate="Época %{x}<br>Pérdida: %{y:.5f}<extra></extra>",
        ))
        fig_lc.update_layout(
            title="Curva de pérdida (entrenamiento)",
            xaxis_title="Época",
            yaxis_title="Pérdida (MSE)",
            height=240,
            margin=dict(t=36, b=24),
        )
        html += _pio.to_html(fig_lc, full_html=False, include_plotlyjs=False, config=cfg)

    return html


def _cls_obj_chart_html(art: dict, obj_name: str, model_name: str) -> str:
    """Confusion matrix for one objective × model."""
    cfg = {"responsive": True}
    labels = art["labels"]
    cm = _cm(art["y_test"], art["y_pred"], labels=labels)
    fig = _go.Figure(_go.Heatmap(
        z=cm.tolist(), x=[f"Pred {lb}" for lb in labels], y=[f"Real {lb}" for lb in labels],
        colorscale="Blues", text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
    ))
    fig.update_layout(xaxis_title="Predicho", yaxis_title="Real",
                      height=300, margin=dict(t=28, b=24))
    return _pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg)


# ---------------------------------------------------------------------------
# Per-objective task workers  (run inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _run_reg_task(
    task_id: str, label: str, obj_name: str,
    x_df: pd.DataFrame, y: "pd.Series | pd.DataFrame",
    reg_configs: "dict | None",
) -> dict:
    try:
        tbl, arts = evaluate_regression(x_df, y, obj_name, reg_configs=reg_configs)
        models: dict[str, Any] = {}
        if not tbl.empty:
            for _, row in tbl.iterrows():
                mn = str(row["model"])
                metrics = _json_safe({k: v for k, v in row.items()
                                      if k not in ("objective", "task", "model")})
                art = arts.get(mn, {})
                chart_html = _reg_obj_chart_html(art, obj_name, mn) if art else ""
                models[mn] = {"metrics": metrics, "chart_html": chart_html}
        return {"task_id": task_id, "label": label, "task": "regression",
                "obj": obj_name, "models": models, "error": None,
                "_x_df": x_df,
                "_y": y if not isinstance(y, pd.DataFrame) else y.iloc[:, 0],
                "_reg_configs": reg_configs,
                "_multi_output": isinstance(y, pd.DataFrame)}
    except Exception as exc:  # noqa: BLE001
        return {"task_id": task_id, "label": label, "task": "regression",
                "obj": obj_name, "models": {}, "error": str(exc),
                "_x_df": x_df, "_y": None, "_reg_configs": reg_configs,
                "_multi_output": isinstance(y, pd.DataFrame)}


def _run_cls_task(
    task_id: str, label: str, obj_name: str,
    x_df: pd.DataFrame, y: pd.Series,
    class_q: int, cls_configs: "dict | None",
) -> dict:
    try:
        tbl, arts = evaluate_classification(x_df, y, obj_name, class_q, cls_configs=cls_configs)
        models: dict[str, Any] = {}
        if not tbl.empty:
            for _, row in tbl.iterrows():
                mn = str(row["model"])
                metrics = _json_safe({k: v for k, v in row.items()
                                      if k not in ("objective", "task", "model")})
                art = arts.get(mn, {})
                chart_html = _cls_obj_chart_html(art, obj_name, mn) if art else ""
                models[mn] = {"metrics": metrics, "chart_html": chart_html}
        return {"task_id": task_id, "label": label, "task": "classification",
                "obj": obj_name, "models": models, "error": None,
                "_x_df": x_df, "_y": y,
                "_class_q": class_q, "_cls_configs": cls_configs}
    except Exception as exc:  # noqa: BLE001
        return {"task_id": task_id, "label": label, "task": "classification",
                "obj": obj_name, "models": {}, "error": str(exc),
                "_x_df": x_df, "_y": y, "_class_q": class_q, "_cls_configs": cls_configs}


def _run_sweep_task(
    task_id: str, label: str, obj_name: str,
    x_df: pd.DataFrame, y: pd.Series,
    sweep_grid: dict,
) -> dict:
    try:
        sweep_df = run_mlp_sweep(x_df, y, obj_name, sweep_grid)
        table_html = html_table(sweep_df) if not sweep_df.empty else "<p>Sin resultados</p>"
        chart_html = _charts_html(make_mlp_sweep_charts(sweep_df))
        return {"task_id": task_id, "label": label, "task": "sweep",
                "obj": obj_name, "table_html": table_html, "chart_html": chart_html, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"task_id": task_id, "label": label, "task": "sweep",
                "obj": obj_name, "table_html": "", "chart_html": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> Response:
    registry_json = json.dumps(ALGO_REGISTRY)
    opt_registry_json = json.dumps(_OPT_REGISTRY)
    opt_labels_json = json.dumps(_ALGO_LABELS)
    html = (_HTML
            .replace("__REGISTRY__", registry_json)
            .replace("__OPT_REGISTRY__", opt_registry_json)
            .replace("__OPT_ALGO_LABELS__", opt_labels_json))
    return Response(html, mimetype="text/html")


@app.route("/data_info")
def data_info() -> Response:
    try:
        raw = _load_raw_df()
        numeric_cols = raw.select_dtypes(include=["number"]).columns.tolist()
        all_cols = [
            {"name": c, "is_numeric": c in numeric_cols}
            for c in raw.columns
            if str(c).strip()
        ]
        return jsonify({"n_obs": len(raw), "n_cols": len(raw.columns), "columns": all_cols})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/export", methods=["POST"])
def export_results() -> Response:
    """Build and return a ZIP archive of selected analysis results."""
    body_ex = request.get_json(force=True) or {}
    session_id_ex: str = body_ex.get("session_id", "")
    selected: set[str] = set(body_ex.get("selected", []))
    include_config: bool = bool(body_ex.get("include_config", True))
    include_eda: bool    = bool(body_ex.get("include_eda", True))

    store = _export_store.get(session_id_ex)
    if not store:
        return jsonify({"error": "Sesión no encontrada o expirada. Ejecute el análisis de nuevo."}), 404

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"analysis_{ts}"
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # --- config.json ---
        if include_config:
            cfg_data = store.get("config") or {}
            zf.writestr(
                f"{prefix}/config.json",
                json.dumps(cfg_data, indent=2, ensure_ascii=False),
            )

        # --- EDA charts (single combined HTML) ---
        if include_eda and store.get("eda_html"):
            zf.writestr(
                f"{prefix}/charts/eda.html",
                _wrap_chart_html(store["eda_html"], "EDA — Análisis Exploratorio"),
            )

        # --- Metrics tables (one CSV per selected model tab) ---
        for tab_id, tdata in store.get("metric_tables", {}).items():
            if tab_id not in selected:
                continue
            rows = tdata.get("rows", [])
            if not rows:
                continue
            cols = (_REG_METRIC_COLS if tdata["type"] == "regression"
                    else _CLS_METRIC_COLS)
            df_m = pd.DataFrame(rows)
            # Ensure column order and fill any missing cols with NaN
            ordered_cols = [c for c in cols if c in df_m.columns]
            extra_cols   = [c for c in df_m.columns if c not in cols]
            df_m = df_m[ordered_cols + extra_cols]
            csv_buf = io.StringIO()
            df_m.to_csv(csv_buf, index=False)
            zf.writestr(f"{prefix}/metrics/{tab_id}.csv", csv_buf.getvalue())

        # --- Per-objective charts for selected tabs ---
        for chart_key, html in store.get("charts", {}).items():
            tab_id = chart_key.split("__")[0]
            if tab_id not in selected:
                continue
            safe_name = _re.sub(r"[^a-zA-Z0-9_\-]", "_", chart_key)
            readable  = chart_key.replace("__", " — ").replace("_", " ")
            zf.writestr(
                f"{prefix}/charts/{safe_name}.html",
                _wrap_chart_html(html, readable),
            )

        # --- Sweep tables (one CSV per objective × sweep) ---
        for sk, sdata in store.get("sweep_tables", {}).items():
            if sdata.get("tab_id", "") not in selected:
                continue
            rows = sdata.get("rows", [])
            if not rows:
                continue
            df_s = pd.DataFrame(rows)
            csv_buf = io.StringIO()
            df_s.to_csv(csv_buf, index=False)
            safe_sk = _re.sub(r"[^a-zA-Z0-9_\-]", "_", sk)
            zf.writestr(f"{prefix}/sweeps/{safe_sk}.csv", csv_buf.getvalue())

        # --- Sweep charts ---
        for sk, html in store.get("sweep_charts", {}).items():
            if not html:
                continue
            tab_id = sk.split("__")[0]
            if tab_id not in selected:
                continue
            safe_sk = _re.sub(r"[^a-zA-Z0-9_\-]", "_", sk)
            readable = sk.replace("__", " — ").replace("_", " ")
            zf.writestr(
                f"{prefix}/charts/sweep_{safe_sk}.html",
                _wrap_chart_html(html, f"Sweep — {readable}"),
            )

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{prefix}.zip"',
            "Content-Type": "application/zip",
        },
    )


@app.route("/run", methods=["POST"])
def run() -> Response:  # noqa: C901
    body = request.get_json(force=True) or {}
    reg_selection: dict[str, dict | None] = body.get("regression", {})
    cls_selection: dict[str, dict | None] = body.get("classification", {})
    class_q: int = int(body.get("class_q", 3))
    input_cols: list[str]  = body.get("input_cols",  []) or []
    output_cols: list[str] = body.get("output_cols", []) or []
    # output_combos: explicit list of combinations to run, e.g. [["Ra_X"], ["Ra_X","Ra_Y"]]
    # If absent or empty fall back to full powerset of output_cols
    output_combos_raw: list[list[str]] = body.get("output_combos", []) or []

    reg_configs: dict[str, dict] | None = (
        None if "regression" not in body
        else {k: (v or {}) for k, v in reg_selection.items()}
    )
    cls_configs: dict[str, dict] | None = (
        None if "classification" not in body
        else {k: (v or {}) for k, v in cls_selection.items()}
    )

    # Parse sweep grids for every selected algorithm
    # algo_id → ("regression"|"classification", sweep_grid)
    algo_sweep_grids: dict[str, tuple[str, dict]] = {}

    if reg_configs:
        for algo_id in list(reg_configs.keys()):
            cfg = reg_configs[algo_id] or {}
            sweep_grid, single_cfg = _parse_algo_sweep_params(algo_id, "regression", cfg)
            reg_configs[algo_id] = single_cfg
            if sweep_grid:
                algo_sweep_grids[algo_id] = ("regression", sweep_grid)

    if cls_configs:
        for algo_id in list(cls_configs.keys()):
            cfg = cls_configs[algo_id] or {}
            sweep_grid, single_cfg = _parse_algo_sweep_params(algo_id, "classification", cfg)
            cls_configs[algo_id] = single_cfg
            if sweep_grid:
                algo_sweep_grids[algo_id] = ("classification", sweep_grid)

    try:
        if input_cols and output_cols:
            df, x_df, _y_df = _prepare_data_dynamic(input_cols, output_cols)
        else:
            df, x_df = get_data()
            input_cols  = list(INPUT_COLUMNS)
            output_cols = list(PRECISION_RENAME_MAP.values()) + list(ROUGHNESS_TARGET_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Error al cargar datos: {exc}"}), 500

    # Classify output columns into roughness (those with bias experiments to exclude)
    # and precision groups based on whether they appear in the legacy maps.
    # With dynamic selection we use a simple heuristic: roughness cols contain Ra/Rq/Rz.
    _roughness_kw = ("ra_", "rq_", "rz_", "ra,", "rq,", "rz,")
    precision_cols = [c for c in output_cols
                      if not any(c.lower().startswith(k) for k in _roughness_kw)]
    roughness_cols = [c for c in output_cols
                      if any(c.lower().startswith(k) for k in _roughness_kw)]
    # Fallback: if everything ended up in one bucket put it all in precision_cols
    if not precision_cols and roughness_cols:
        precision_cols, roughness_cols = roughness_cols, []

    bias_mask = (~df[EXPERIMENT_COLUMN].isin(ROUGHNESS_BIAS_EXPERIMENTS)
                 if EXPERIMENT_COLUMN in df.columns
                 else pd.Series(True, index=df.index))

    algo_labels = {k: v["label"] for group in ALGO_REGISTRY.values() for k, v in group.items()}

    if reg_configs is None:
        reg_models = list(DEFAULT_REG_CONFIGS.keys())
    elif reg_configs:
        reg_models = list(reg_configs.keys())
    else:
        reg_models = []

    if cls_configs is None:
        cls_models = list(DEFAULT_CLS_CONFIGS.keys())
    elif cls_configs:
        cls_models = list(cls_configs.keys())
    else:
        cls_models = []

    sweep_reg_models = [aid for aid, (tt, _) in algo_sweep_grids.items() if tt == "regression"]
    sweep_cls_models = [aid for aid, (tt, _) in algo_sweep_grids.items() if tt == "classification"]

    # Build flat task list (one entry per objective × task-type)
    tasks: list[dict] = []

    def _add_reg(cols_list: list[str], x: pd.DataFrame, y: "pd.Series | pd.DataFrame",
                 prefix: str) -> None:
        obj = prefix + "_" + "+".join(slugify(c) for c in cols_list)
        short = "+".join(cols_list)
        tasks.append({"kind": "reg", "task_id": f"reg::{obj}",
                      "label": f"reg: {short}", "obj": obj,
                      "x": x, "y": y, "configs": reg_configs, "extra": {}})

    def _add_cls(cols_list: list[str], x: pd.DataFrame, y: pd.Series,
                 prefix: str) -> None:
        obj = prefix + "_cls_" + "+".join(slugify(c) for c in cols_list)
        short = "+".join(cols_list)
        tasks.append({"kind": "cls", "task_id": f"cls::{obj}",
                      "label": f"cls: {short}", "obj": obj,
                      "x": x, "y": y, "configs": cls_configs, "extra": {"class_q": class_q}})

    def _add_algo_sweep(cols_list: list[str], x: pd.DataFrame, y: pd.Series,
                        prefix: str, algo_id: str, task_type: str, sweep_grid: dict) -> None:
        obj = prefix + "_sweep_" + algo_id + "_" + "+".join(slugify(c) for c in cols_list)
        short = "+".join(cols_list)
        tasks.append({"kind": "algo_sweep",
                      "task_id": f"sweep::{algo_id}::{obj}",
                      "label": f"sweep {algo_id}: {short}", "obj": obj,
                      "x": x, "y": y, "configs": None,
                      "extra": {"algo_id": algo_id, "task_type": task_type,
                                "sweep_grid": sweep_grid, "class_q": class_q}})

    if reg_configs is None or reg_configs:
        for combo in _iter_combos(output_combos_raw, precision_cols, roughness=False):
            cl = list(combo)
            if all(c in precision_cols for c in cl):
                y = (pd.to_numeric(df[cl[0]], errors="coerce") if len(cl) == 1
                     else df[cl].apply(pd.to_numeric, errors="coerce"))
                _add_reg(cl, x_df, y, "precision")
        for combo in _iter_combos(output_combos_raw, roughness_cols, roughness=True):
            cl = list(combo)
            if all(c in roughness_cols for c in cl):
                x_r = x_df.loc[bias_mask].reset_index(drop=True)
                y_r = (pd.to_numeric(df.loc[bias_mask, cl[0]], errors="coerce").reset_index(drop=True)
                       if len(cl) == 1
                       else df.loc[bias_mask, cl].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))
                _add_reg(cl, x_r, y_r, "roughness")

    if cls_configs is None or cls_configs:
        for combo in _iter_combos(output_combos_raw, precision_cols, roughness=False):
            cl = list(combo)
            if all(c in precision_cols for c in cl):
                y_c = (df[cl[0]] if len(cl) == 1
                       else df[cl].apply(pd.to_numeric, errors="coerce").mean(axis=1))
                _add_cls(cl, x_df, y_c, "precision")
        for combo in _iter_combos(output_combos_raw, roughness_cols, roughness=True):
            cl = list(combo)
            if all(c in roughness_cols for c in cl):
                x_r = x_df.loc[bias_mask].reset_index(drop=True)
                y_r = (df.loc[bias_mask, cl[0]].reset_index(drop=True) if len(cl) == 1
                       else df.loc[bias_mask, cl].apply(pd.to_numeric, errors="coerce")
                       .mean(axis=1).reset_index(drop=True))
                _add_cls(cl, x_r, y_r, "roughness")

    # Sweep tasks — one per (algo_with_sweep × output_column)
    for algo_id, (task_type, sweep_grid) in algo_sweep_grids.items():
        for combo in _iter_combos(output_combos_raw, precision_cols, roughness=False):
            cl = list(combo)
            if all(c in precision_cols for c in cl):
                if task_type == "regression":
                    y_s = (pd.to_numeric(df[cl[0]], errors="coerce") if len(cl) == 1
                           else df[cl].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1))
                    _add_algo_sweep(cl, x_df, y_s, "precision", algo_id, task_type, sweep_grid)
                else:
                    y_s = (df[cl[0]] if len(cl) == 1
                           else df[cl].apply(pd.to_numeric, errors="coerce").mean(axis=1))
                    _add_algo_sweep(cl, x_df, y_s, "precision", algo_id, task_type, sweep_grid)
        for combo in _iter_combos(output_combos_raw, roughness_cols, roughness=True):
            cl = list(combo)
            if all(c in roughness_cols for c in cl):
                x_r = x_df.loc[bias_mask].reset_index(drop=True)
                if task_type == "regression":
                    y_r = (pd.to_numeric(df.loc[bias_mask, cl[0]], errors="coerce").reset_index(drop=True)
                           if len(cl) == 1
                           else df.loc[bias_mask, cl].apply(pd.to_numeric, errors="coerce")
                           .mean(axis=1).reset_index(drop=True))
                else:
                    y_r = (df.loc[bias_mask, cl[0]].reset_index(drop=True) if len(cl) == 1
                           else df.loc[bias_mask, cl].apply(pd.to_numeric, errors="coerce")
                           .mean(axis=1).reset_index(drop=True))
                _add_algo_sweep(cl, x_r, y_r, "roughness", algo_id, task_type, sweep_grid)

    session_id = str(uuid.uuid4())

    def generate():
        # --- Export session store ---
        store: dict = {
            "config": _json_safe(body),
            "eda_html": "",
            "metric_tables": {},   # tab_id → {type, rows}
            "sweep_tables": {},    # sweep_key → {tab_id, obj, rows}
            "sweep_charts": {},    # sweep_key → html
            "charts": {},          # "tab_id__slug" → html
            "x_df_store": {},      # tab_id → {x_df, y_by_obj, input_cols, task, reg_configs, cls_configs, class_q}
        }
        # Prune oldest session if store is getting large
        while len(_export_store) >= _EXPORT_STORE_MAX:
            _export_store.pop(next(iter(_export_store)), None)
        _export_store[session_id] = store

        def _cache(r: dict) -> None:
            """Accumulate a result event into the export store."""
            task = r.get("task")
            if task in ("regression", "classification") and not r.get("error"):
                prefix = "reg" if task == "regression" else "cls"
                for mn, mdata in r.get("models", {}).items():
                    tid = f"{prefix}-{mn}"
                    if tid not in store["metric_tables"]:
                        store["metric_tables"][tid] = {"type": task, "rows": []}
                    row: dict = {"objective": r.get("obj", "")}
                    row.update(mdata.get("metrics") or {})
                    store["metric_tables"][tid]["rows"].append(row)
                    if mdata.get("chart_html"):
                        ck = f"{tid}__{slugify(r.get('obj', ''))}"
                        store["charts"][ck] = mdata["chart_html"]

                # Store x_df, y, and configs per tab for on-demand optimization
                x_df_r = r.get("_x_df")
                y_r = r.get("_y")
                if x_df_r is not None:
                    for mn in (r.get("models") or {}):
                        tid = f"{prefix}-{mn}"
                        if tid not in store["x_df_store"]:
                            store["x_df_store"][tid] = {
                                "x_df": x_df_r,
                                "input_cols": list(x_df_r.columns),
                                "y_by_obj": {},
                                "task": task,
                                "class_q": r.get("_class_q", 3),
                                "reg_configs": r.get("_reg_configs"),
                                "cls_configs": r.get("_cls_configs"),
                                "multi_output": r.get("_multi_output", False),
                            }
                        else:
                            # Update fields that may have been pre-populated with None by sweep setup
                            entry = store["x_df_store"][tid]
                            entry["x_df"] = x_df_r
                            entry["input_cols"] = list(x_df_r.columns)
                            entry["task"] = task
                            entry["class_q"] = r.get("_class_q", 3)
                            entry["reg_configs"] = r.get("_reg_configs")
                            entry["cls_configs"] = r.get("_cls_configs")
                            entry["multi_output"] = r.get("_multi_output", False)
                        if y_r is not None:
                            obj = r.get("obj", "")
                            store["x_df_store"][tid]["y_by_obj"][obj] = y_r

            elif task == "sweep_combo" and not r.get("error") and r.get("row"):
                sk = r.get("sweep_key", "")
                if sk:
                    if sk not in store["sweep_tables"]:
                        store["sweep_tables"][sk] = {
                            "tab_id": r.get("tab_id", ""),
                            "obj": r.get("obj", ""),
                            "algo_id": r.get("algo_id", ""),
                            "task_type": r.get("task_type", "regression"),
                            "param_names": list((r.get("params") or {}).keys()),
                            "rows": [],
                        }
                    store["sweep_tables"][sk]["rows"].append(r["row"])
            elif task == "algo_sweep_charts" and r.get("chart_html"):
                sk = f"{r.get('tab_id', '')}__{r.get('obj', '')}"
                store["sweep_charts"][sk] = r["chart_html"]

        # Separate regular tasks from sweep tasks; expand sweep tasks into per-combo tasks
        regular_tasks       = [t for t in tasks if t["kind"] != "algo_sweep"]
        algo_sweep_tasks_raw = [t for t in tasks if t["kind"] == "algo_sweep"]

        sweep_combo_tasks: list[dict] = []
        sweep_groups: dict[str, dict] = {}   # sweep_key → tracking state
        sweep_combo_list: list[dict]  = []   # sent in init so frontend pre-populates table

        for t in algo_sweep_tasks_raw:
            algo_id    = t["extra"]["algo_id"]
            task_type  = t["extra"]["task_type"]
            sweep_grid = t["extra"]["sweep_grid"]
            class_q    = t["extra"].get("class_q", 3)
            obj        = t["obj"]
            tab_id     = ("reg" if task_type == "regression" else "cls") + "-" + algo_id
            sweep_key  = f"{tab_id}__{obj}"
            safe_key   = _re.sub(r"[^a-zA-Z0-9_-]", "_", sweep_key)

            # Prepare valid data + train/test split once per group
            try:
                x_raw = t["x"]
                y_raw = t["y"]
                if task_type == "classification":
                    y_class = discretize_target(
                        pd.to_numeric(y_raw, errors="coerce"), q=class_q
                    )
                    if y_class is None:
                        continue
                    valid_idx = y_class.dropna().index
                    x_use    = x_raw.loc[valid_idx].reset_index(drop=True)
                    y_use    = y_class.loc[valid_idx].astype(int).reset_index(drop=True)
                    stratify = y_use
                else:
                    valid_mask = pd.to_numeric(y_raw, errors="coerce").notna()
                    x_use      = x_raw.loc[valid_mask].reset_index(drop=True)
                    y_use      = (pd.to_numeric(y_raw, errors="coerce")
                                  .loc[valid_mask].reset_index(drop=True))
                    stratify   = None
                if len(x_use) < 10:
                    continue
                x_train, x_test, y_train, y_test = _tt_split(
                    x_use, y_use, test_size=0.2, random_state=42, stratify=stratify
                )
            except Exception:  # noqa: BLE001
                continue

            param_names = list(sweep_grid.keys())
            combos      = list(itertools.product(*[sweep_grid[k] for k in param_names]))

            if sweep_key not in sweep_groups:
                sweep_groups[sweep_key] = {
                    "tab_id": tab_id, "obj": obj, "safe_key": safe_key,
                    "task_type": task_type, "param_names": param_names,
                    "total": 0, "done": 0, "rows": [],
                }

            for i, combo in enumerate(combos):
                params   = dict(zip(param_names, combo))
                combo_id = f"combo__{safe_key}__{i}"
                sweep_groups[sweep_key]["total"] += 1

                sweep_combo_list.append({
                    "combo_id": combo_id, "tab_id": tab_id,
                    "obj": obj, "safe_key": safe_key,
                    "task_type": task_type,
                    "params": _json_safe(params),
                })
                sweep_combo_tasks.append({
                    "kind": "sweep_combo",
                    "task_id": combo_id,
                    "label": f"sweep {algo_id}:{obj} #{i + 1}/{len(combos)}",
                    "combo_id": combo_id, "tab_id": tab_id,
                    "obj": obj, "sweep_key": sweep_key,
                    "algo_id": algo_id, "task_type": task_type,
                    "params": params,
                    "x_use": x_use, "y_use": y_use,
                    "x_train": x_train, "x_test": x_test,
                    "y_train": y_train, "y_test": y_test,
                })

        all_tasks = regular_tasks + sweep_combo_tasks

        # Pre-populate x_df_store with sweep task data so /optimize can find y for sweep objs
        for t in algo_sweep_tasks_raw:
            a_id  = t["extra"]["algo_id"]
            t_type = t["extra"]["task_type"]
            cq    = t["extra"].get("class_q", 3)
            t_tab = ("reg" if t_type == "regression" else "cls") + "-" + a_id
            if t_tab not in store["x_df_store"]:
                store["x_df_store"][t_tab] = {
                    "x_df": t["x"], "input_cols": list(t["x"].columns),
                    "y_by_obj": {}, "task": t_type,
                    "class_q": cq,
                    "reg_configs": None, "cls_configs": None,
                    "multi_output": False,
                }
            store["x_df_store"][t_tab]["y_by_obj"][t["obj"]] = t["y"]

        # Pre-compute bounds per tab so the frontend doesn't need a separate /bounds fetch
        bounds_by_tab: dict[str, dict] = {}

        def _tab_bounds(x: pd.DataFrame) -> dict:
            return {c: list(v) for c, v in get_data_bounds(x).items()}

        # Regular tasks: derive tab_ids from configs
        for t in regular_tasks:
            prefix = "reg" if t["kind"] == "reg" else "cls"
            t_configs = t.get("configs")
            if prefix == "reg":
                model_names = list(t_configs.keys() if t_configs is not None else DEFAULT_REG_CONFIGS.keys())
            else:
                model_names = list(t_configs.keys() if t_configs is not None else DEFAULT_CLS_CONFIGS.keys())
            for mn in model_names:
                tid = f"{prefix}-{mn}"
                if tid not in bounds_by_tab:
                    bounds_by_tab[tid] = _tab_bounds(t["x"])

        # Sweep tasks: already have the correct x per tab
        for t in algo_sweep_tasks_raw:
            a_id = t["extra"]["algo_id"]
            t_type = t["extra"]["task_type"]
            t_tab = ("reg" if t_type == "regression" else "cls") + "-" + a_id
            if t_tab not in bounds_by_tab:
                bounds_by_tab[t_tab] = _tab_bounds(t["x"])

        # EDA is fast — do it before opening the executor
        used_cols = [c for c in input_cols + precision_cols + roughness_cols
                     if c in df.columns]
        eda_html = _charts_html(make_eda_charts(df[used_cols], precision_cols + roughness_cols))

        init_payload = {
            "session_id": session_id,
            "total": len(all_tasks),
            "task_list": [{"id": t["task_id"], "label": t["label"], "task": t["kind"]}
                          for t in all_tasks],
            "reg_models": reg_models,
            "cls_models": cls_models,
            "sweep_reg_models": sweep_reg_models,
            "sweep_cls_models": sweep_cls_models,
            "model_labels": algo_labels,
            "eda_html": eda_html,
            "sweep_combo_list": sweep_combo_list,
            "bounds_by_tab": bounds_by_tab,
        }
        yield f"event: init\ndata: {json.dumps(init_payload)}\n\n"
        store["eda_html"] = eda_html

        def _run(t: dict) -> dict:
            if t["kind"] == "reg":
                return _run_reg_task(t["task_id"], t["label"], t["obj"],
                                     t["x"], t["y"], t["configs"])
            if t["kind"] == "cls":
                return _run_cls_task(t["task_id"], t["label"], t["obj"],
                                     t["x"], t["y"], t["extra"]["class_q"], t["configs"])
            if t["kind"] == "sweep_combo":
                return _run_single_combo_task(
                    t["combo_id"], t["tab_id"], t["obj"], t["sweep_key"],
                    t["algo_id"], t["task_type"],
                    t["x_use"], t["y_use"],
                    t["x_train"], t["x_test"], t["y_train"], t["y_test"],
                    t["params"],
                )
            return {"task_id": t["task_id"], "label": t.get("label", "?"),
                    "task": "error", "obj": t.get("obj", "?"),
                    "models": {}, "error": "Unknown task kind"}

        with ThreadPoolExecutor(max_workers=min(8, len(all_tasks) or 1)) as ex:
            futures = {ex.submit(_run, t): t for t in all_tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"task_id": "?", "label": "?", "task": "error",
                              "obj": "?", "models": {}, "error": str(exc)}
                _cache(result)
                yield f"event: result\ndata: {json.dumps(_strip_private(result))}\n\n"

                # After every combo completes, check if its group is fully done
                if result.get("task") == "sweep_combo":
                    sk = result.get("sweep_key", "")
                    if sk in sweep_groups:
                        sg = sweep_groups[sk]
                        if result.get("error") is None and result.get("row"):
                            sg["rows"].append(result["row"])
                        sg["done"] += 1
                        if sg["done"] >= sg["total"]:
                            charts_evt = _compute_sweep_charts_event(
                                sg["tab_id"], sg["obj"], sg["safe_key"],
                                sg["task_type"], sg["rows"],
                            )
                            yield f"event: result\ndata: {json.dumps(charts_evt)}\n\n"
                            _cache(charts_evt)

        yield "event: done\ndata: {}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Optimization endpoints
# ---------------------------------------------------------------------------

@app.route("/algo-registry")
def algo_registry_endpoint() -> Response:
    """Return OPTIMIZER_REGISTRY and algorithm labels as JSON."""
    return jsonify({
        "registry": _OPT_REGISTRY,
        "labels": _ALGO_LABELS,
    })


@app.route("/bounds")
def bounds_endpoint() -> Response:
    """Return data-derived bounds {col: [min, max]} for a session's tab."""
    session_id = request.args.get("session_id", "")
    tab_id = request.args.get("tab_id", "")
    store = _export_store.get(session_id)
    if not store:
        return jsonify({"error": "Sesión no encontrada o expirada."}), 404
    xdf_entry = store.get("x_df_store", {}).get(tab_id)
    if xdf_entry is None:
        # Fall back to any available x_df
        all_entries = list(store.get("x_df_store", {}).values())
        if not all_entries:
            return jsonify({"error": "No hay datos de entrada disponibles para este tab."}), 404
        xdf_entry = all_entries[0]
    x_df = xdf_entry["x_df"]
    raw_bounds = get_data_bounds(x_df)
    return jsonify({c: list(v) for c, v in raw_bounds.items()})


@app.route("/optimize", methods=["POST"])
def optimize_endpoint() -> Response:
    """Run single-objective optimization on a fitted model.

    Request body (JSON):
        session_id  : str
        tab_id      : str   (e.g. "reg-svr_rbf")
        model_name  : str   (e.g. "svr_rbf")
        objective   : str   (obj_name string from results)
        direction   : "minimize" | "maximize"
        algo        : str | null  (override; null = use registry default)
        n_iter      : int   (default 50)
        bounds      : {col: [lo, hi]}  (optional; falls back to data bounds)
    """
    body = request.get_json(force=True) or {}
    session_id: str = body.get("session_id", "")
    tab_id: str = body.get("tab_id", "")
    model_name: str = body.get("model_name", "")
    objective: str = body.get("objective", "")
    direction: str = body.get("direction", "minimize")
    algo: str | None = body.get("algo") or None
    n_iter: int = int(body.get("n_iter", 50))
    # Optional: specific hyperparameter config chosen from sweep table
    sweep_params: dict | None = body.get("sweep_params") or None

    store = _export_store.get(session_id)
    if not store:
        return jsonify({"error": "Sesión no encontrada o expirada. Ejecute el análisis de nuevo."}), 404

    # Retrieve stored data and configs
    xdf_entry = store.get("x_df_store", {}).get(tab_id)
    if xdf_entry is None:
        return jsonify({"error": "Datos de entrada no encontrados para este tab."}), 404

    x_df: pd.DataFrame = xdf_entry["x_df"]
    y_by_obj: dict = xdf_entry.get("y_by_obj", {})
    y_raw: pd.Series = y_by_obj.get(objective, pd.Series(dtype=float))
    if y_raw.empty:
        return jsonify({"error": f"No hay datos del objetivo '{objective}' para este tab."}), 404

    task_type: str = xdf_entry.get("task", "regression")
    class_q: int = xdf_entry.get("class_q", 3)

    # Build bounds
    raw_bounds = get_data_bounds(x_df)
    user_bounds_raw: dict = body.get("bounds") or {}
    bounds: dict[str, tuple[float, float]] = {}
    for col in x_df.columns:
        if col in user_bounds_raw and len(user_bounds_raw[col]) == 2:
            bounds[col] = (float(user_bounds_raw[col][0]), float(user_bounds_raw[col][1]))
        elif col in raw_bounds:
            bounds[col] = raw_bounds[col]

    # Determine model config: sweep_params override → stored configs → None (use defaults)
    if sweep_params:
        cfg_model: dict | None = {model_name: sweep_params}
    elif task_type == "classification":
        stored_cls = xdf_entry.get("cls_configs")
        cfg_model = {model_name: stored_cls.get(model_name, {})} if stored_cls else None
    else:
        stored_reg = xdf_entry.get("reg_configs")
        cfg_model = {model_name: stored_reg.get(model_name, {})} if stored_reg else None

    # Fit the model on-demand on the full valid dataset
    try:
        if task_type == "classification":
            y_num = pd.to_numeric(y_raw, errors="coerce")
            y_target = discretize_target(y_num, q=class_q)
            valid_mask = y_target.notna()
            x_full = x_df.loc[valid_mask].reset_index(drop=True)
            y_full = y_target.loc[valid_mask].astype(int).reset_index(drop=True)
            preprocessor_full = build_preprocessor(x_full)
            pipes = classification_models(preprocessor_full, enabled_configs=cfg_model)
        else:
            y_target = pd.to_numeric(y_raw, errors="coerce")
            valid_mask = y_target.notna() & x_df.notna().all(axis=1)
            x_full = x_df.loc[valid_mask].reset_index(drop=True)
            y_full = y_target.loc[valid_mask].reset_index(drop=True)
            # multi_output requires a 2-D y; a single objective always produces 1-D
            multi_output = False
            preprocessor_full = build_preprocessor(x_full)
            pipes = regression_models(preprocessor_full, multi_output=multi_output,
                                      enabled_configs=cfg_model)

        if model_name not in pipes:
            return jsonify({"error": f"Modelo '{model_name}' no disponible para este tab."}), 404

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipes[model_name].fit(x_full, y_full)
        fitted = pipes[model_name]
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Error al entrenar el modelo para optimización: {exc}"}), 500

    try:
        result = optimize_single(
            fitted, model_name, x_df, y_raw,
            direction, bounds,
            algo=algo,
            n_iter=n_iter,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    result_dict = dict(result)

    # Predict all objectives with the optimal inputs so the UI can show a full output table
    optimal_inputs = result_dict.get("optimal_inputs") or {}
    all_predictions: dict = {}
    if optimal_inputs and not result_dict.get("error"):
        # The optimized objective prediction is already known
        all_predictions[objective] = result_dict.get("predicted_value")
        opt_x = pd.DataFrame([optimal_inputs])
        for obj_name, y_series in y_by_obj.items():
            if obj_name == objective:
                continue
            try:
                if task_type == "classification":
                    y_n = pd.to_numeric(y_series, errors="coerce")
                    y_t = discretize_target(y_n, q=class_q)
                    vm = y_t.notna()
                    xf = x_df.loc[vm].reset_index(drop=True)
                    yf = y_t.loc[vm].astype(int).reset_index(drop=True)
                    pp = build_preprocessor(xf)
                    ps = classification_models(pp, enabled_configs=cfg_model)
                else:
                    y_n = pd.to_numeric(y_series, errors="coerce")
                    vm = y_n.notna() & x_df.notna().all(axis=1)
                    xf = x_df.loc[vm].reset_index(drop=True)
                    yf = y_n.loc[vm].reset_index(drop=True)
                    pp = build_preprocessor(xf)
                    ps = regression_models(pp, multi_output=False, enabled_configs=cfg_model)
                if model_name not in ps:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    ps[model_name].fit(xf, yf)
                pred_raw = ps[model_name].predict(opt_x)[0]
                all_predictions[obj_name] = (
                    float(pred_raw) if isinstance(pred_raw, (int, float, np.floating, np.integer))
                    else str(pred_raw)
                )
            except Exception:  # noqa: BLE001
                pass
    result_dict["all_predictions"] = _json_safe(all_predictions)

    # When the optimized objective is a combination (joined with "+"), identify which
    # individual-output objectives in all_predictions are its components so the UI
    # can highlight them instead of showing just the aggregated mean prediction.
    primary_components: list[str] = []
    if "+" in objective and task_type == "regression":
        parts = objective.split("+")
        for part in parts:
            # parts[0] is the full key of the first component; later parts are only the slug suffix
            if part in y_by_obj and "+" not in part:
                primary_components.append(part)
            else:
                match = next(
                    (k for k in y_by_obj if "+" not in k and k != objective and k.endswith(part)),
                    None,
                )
                if match:
                    primary_components.append(match)
    result_dict["primary_components"] = primary_components

    return jsonify(result_dict)


@app.route("/optimize-multi", methods=["POST"])
def optimize_multi_endpoint() -> Response:
    """Run multi-objective NSGA-II optimization over two objectives.

    Request body (JSON):
        session_id  : str
        objectives  : [{tab_id, model_name, objective, direction}]
        bounds      : {col: [lo, hi]}  (optional)
        n_gen       : int (default 50)
        pop_size    : int (default 40)
    """
    body = request.get_json(force=True) or {}
    session_id: str = body.get("session_id", "")
    obj_specs: list[dict] = body.get("objectives", [])
    n_gen: int = int(body.get("n_gen", 50))
    pop_size: int = int(body.get("pop_size", 40))

    store = _export_store.get(session_id)
    if not store:
        return jsonify({"error": "Sesión no encontrada o expirada."}), 404

    if len(obj_specs) < 2:
        return jsonify({"error": "Se requieren al menos 2 objetivos para optimización multi-objetivo."}), 400

    pipelines_dict: dict[str, Any] = {}
    directions_dict: dict[str, str] = {}
    x_df_ref: pd.DataFrame | None = None
    model_name_ref: str = ""

    for spec in obj_specs:
        tab_id = spec.get("tab_id", "")
        mn = spec.get("model_name", "")
        obj = spec.get("objective", "")
        direction = spec.get("direction", "minimize")

        xdf_entry = store.get("x_df_store", {}).get(tab_id)
        if xdf_entry is None:
            return jsonify({"error": f"Datos de entrada no encontrados para {tab_id}."}), 404

        x_df_e: pd.DataFrame = xdf_entry["x_df"]
        y_raw: pd.Series = xdf_entry.get("y_by_obj", {}).get(obj, pd.Series(dtype=float))
        if y_raw.empty:
            return jsonify({"error": f"No hay datos del objetivo '{obj}' para {tab_id}."}), 404

        task_type: str = xdf_entry.get("task", "regression")
        class_q: int = xdf_entry.get("class_q", 3)

        try:
            if task_type == "classification":
                y_num = pd.to_numeric(y_raw, errors="coerce")
                y_target = discretize_target(y_num, q=class_q)
                valid_mask = y_target.notna()
                x_full = x_df_e.loc[valid_mask].reset_index(drop=True)
                y_full = y_target.loc[valid_mask].astype(int).reset_index(drop=True)
                configs = xdf_entry.get("cls_configs")
                cfg_model = {mn: configs.get(mn, {})} if configs else None
                preprocessor_full = build_preprocessor(x_full)
                pipes = classification_models(preprocessor_full, enabled_configs=cfg_model)
            else:
                y_target = pd.to_numeric(y_raw, errors="coerce")
                valid_mask = y_target.notna() & x_df_e.notna().all(axis=1)
                x_full = x_df_e.loc[valid_mask].reset_index(drop=True)
                y_full = y_target.loc[valid_mask].reset_index(drop=True)
                configs = xdf_entry.get("reg_configs")
                cfg_model = {mn: configs.get(mn, {})} if configs else None
                multi_output = xdf_entry.get("multi_output", False)
                preprocessor_full = build_preprocessor(x_full)
                pipes = regression_models(preprocessor_full, multi_output=multi_output,
                                          enabled_configs=cfg_model)

            if mn not in pipes:
                return jsonify({"error": f"Modelo '{mn}' no disponible para {tab_id}."}), 404

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                pipes[mn].fit(x_full, y_full)
            pipe = pipes[mn]
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Error al entrenar {mn} para optimización: {exc}"}), 500

        key = f"{obj}__{mn}"
        pipelines_dict[key] = pipe
        directions_dict[key] = direction
        if x_df_ref is None:
            x_df_ref = x_df_e
            model_name_ref = mn

    if x_df_ref is None:
        return jsonify({"error": "Datos de entrada no disponibles."}), 404

    raw_bounds = get_data_bounds(x_df_ref)
    user_bounds_raw: dict = body.get("bounds") or {}
    bounds: dict[str, tuple[float, float]] = {}
    for col in x_df_ref.columns:
        if col in user_bounds_raw and len(user_bounds_raw[col]) == 2:
            bounds[col] = (float(user_bounds_raw[col][0]), float(user_bounds_raw[col][1]))
        elif col in raw_bounds:
            bounds[col] = raw_bounds[col]

    try:
        result = optimize_multi(
            pipelines_dict, model_name_ref, x_df_ref,
            directions_dict, bounds,
            n_gen=n_gen, pop_size=pop_size,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(dict(result))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MEX/FDM Quality Analyzer web app.")
    parser.add_argument("--data", type=Path, default=None,
                        help="Override path to source Excel file.")
    parser.add_argument("--sheet", type=str, default=SHEET_NAME,
                        help="Sheet name to read.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.data:
        DATA_PATH = args.data.resolve()
    SHEET_NAME = args.sheet
    app.run(host=args.host, port=args.port, debug=True, use_reloader=True)
