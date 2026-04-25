#!/usr/bin/env python3
"""Builds a reproducible ML analysis and HTML report for MEX/FDM quality targets.

Targets supported:
- Dimensional precision (single-output regression + optional classification)
- Surface roughness (single-output regression + optional classification)
- Joint multi-output regression when both targets are present
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.linear_model import ElasticNet


warnings.filterwarnings("ignore", category=ConvergenceWarning)


INPUT_COLUMNS = [
    "Temperatura de la boquilla (°C)",
    "Altura de la capa (mm)",
    "Velocidad de impresión (mm/s)",
    "Porcentaje de relleno (%)",
]

ROUGHNESS_TARGET_COLUMNS = [
    "Ra_0,8Promedio (μm)",
    "Ra_2,5Promedio (μm)",
    "Rq_0,8Promedio (μm)",
    "Rq_2,5Promedio (μm)",
    "Rz_0,8Promedio (μm)",
    "Rz_2,5Promedio (μm)",
]

PRECISION_RENAME_MAP = {
    "VernierLargo-Desviación dimensional (mm)": "Desviación dimensional largo-Vernier",
    "VernierAncho-Desviación dimensional (mm)": "Desviación dimensional ancho-Vernier",
    "VernierAltura-Desviación dimensional (mm)": "Desviación dimensional altura-Vernier"
}

EXPERIMENT_COLUMN = "Experimento no."
ROUGHNESS_BIAS_EXPERIMENTS = {28, 38, 40}

MLP_SWEEP_GRID: dict[str, list] = {
    "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
    "alpha": [1e-5, 1e-4, 1e-3],
    "learning_rate_init": [1e-4, 1e-3, 5e-3],
    "activation": ["relu", "tanh"],
}

DEFAULT_REG_CONFIGS: dict[str, dict[str, Any]] = {
    "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
    "svr_rbf": {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
    "knn_reg": {"n_neighbors": 5, "weights": "distance"},
    "gpr": {"alpha": 1e-6, "normalize_y": False},
    "mlp_small": {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
    },
}

DEFAULT_CLS_CONFIGS: dict[str, dict[str, Any]] = {
    "svc_rbf": {"C": 10.0, "gamma": "scale"},
    "knn_clf": {"n_neighbors": 5, "weights": "distance"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quality analysis and generate an HTML report."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../../data/Réplica Shirmohammadi 2021 Datos.xlsx"),
        help="Path to the source Excel file.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Datos a utilizar en Minitab",
        help="Sheet name to analyze.",
    )
    parser.add_argument(
        "--precision-target",
        type=str,
        default=None,
        help="Column name for dimensional precision target. If omitted, inferred from column names.",
    )
    parser.add_argument(
        "--roughness-target",
        type=str,
        default=None,
        help="Column name for surface roughness target. If omitted, inferred from column names.",
    )
    parser.add_argument(
        "--class-q",
        type=int,
        default=3,
        help="Number of quantile classes for optional classification path.",
    )
    parser.add_argument(
        "--disable-classification",
        action="store_true",
        help="Skip classification experiments.",
    )
    parser.add_argument(
        "--disable-mlp-sweep",
        action="store_true",
        help="Skip MLPRegressor hyperparameter sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./report_output"),
        help="Directory where report and figures are generated.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower())
    return text.strip("_") or "plot"


def infer_targets(columns: list[str]) -> tuple[str | None, str | None]:
    precision_tokens = ["precision", "dimensional", "deviation", "error_dim", "msde"]
    roughness_tokens = ["rugos", "rough", "ra", "rz", "surface"]

    precision_col = None
    roughness_col = None

    for col in columns:
        col_l = col.lower()
        if precision_col is None and any(tok in col_l for tok in precision_tokens):
            if not any(tok in col_l for tok in roughness_tokens):
                precision_col = col
        if roughness_col is None and any(tok in col_l for tok in roughness_tokens):
            roughness_col = col

    return precision_col, roughness_col


def read_sheet_clean(data_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(data_path, sheet_name=sheet_name, header=None)

    header_row = None
    for idx in range(min(len(raw), 25)):
        row_values = [str(v).lower() for v in raw.iloc[idx].tolist()]
        joined = " | ".join(row_values)
        if "experimento" in joined and (
            "temperatura" in joined or "velocidad" in joined or "ra " in joined
        ):
            header_row = idx
            break

    if header_row is None:
        # Fallback to default parser if a clear metadata/header split is not detected.
        df = pd.read_excel(data_path, sheet_name=sheet_name)
    else:
        headers = raw.iloc[header_row].tolist()
        headers = [str(h).strip() if pd.notna(h) else "" for h in headers]

        # Remove the common leading empty column from the raw worksheet layout.
        if headers and headers[0] == "":
            raw = raw.iloc[:, 1:]
            headers = headers[1:]

        df = raw.iloc[header_row + 1 :].copy()
        df.columns = headers

    # Remove unnamed/empty columns and rows that are fully empty.
    df = df.loc[:, [c for c in df.columns if str(c).strip() and "unnamed" not in str(c).lower()]]
    df = df.dropna(how="all").reset_index(drop=True)

    # Coerce date-like columns where applicable and leave other values untouched.
    for col in df.columns:
        if "fecha" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def ensure_targets(
    df: pd.DataFrame,
    precision_target: str | None,
    roughness_target: str | None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    out = df.copy()

    if not precision_target:
        dim_dev_cols = [
            c
            for c in out.columns
            if "desviaci" in c.lower() and "dimensional" in c.lower()
        ]
        if len(dim_dev_cols) == 1:
            precision_target = dim_dev_cols[0]
        elif len(dim_dev_cols) > 1:
            out["precision_dimensional_index_mm"] = (
                out[dim_dev_cols].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1)
            )
            precision_target = "precision_dimensional_index_mm"

    if not roughness_target:
        preferred = [
            c
            for c in out.columns
            if "ra_" in c.lower() and "promedio" in c.lower()
        ]
        if preferred:
            if len(preferred) == 1:
                roughness_target = preferred[0]
            else:
                out["roughness_index_um"] = (
                    out[preferred].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                )
                roughness_target = "roughness_index_um"
        else:
            broad = [
                c
                for c in out.columns
                if any(tok in c.lower() for tok in ["ra", "rq", "rz", "rugos", "rough"])
                and "promedio" in c.lower()
            ]
            if len(broad) == 1:
                roughness_target = broad[0]
            elif len(broad) > 1:
                out["roughness_index_um"] = (
                    out[broad].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                )
                roughness_target = "roughness_index_um"

    return out, precision_target, roughness_target


def normalize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in {"M", "m"}:
            continue

        series = out[col]
        if series.dtype == "O":
            cleaned = series.replace(r"^\s*$", np.nan, regex=True)
            as_text = cleaned.astype(str).str.replace(",", ".", regex=False).str.strip()
            as_text = as_text.replace({"nan": np.nan, "none": np.nan, "": np.nan})
            numeric_try = pd.to_numeric(as_text, errors="coerce")

            # Convert to numeric when most values are numeric-like.
            if numeric_try.notna().mean() >= 0.80:
                out[col] = numeric_try
            else:
                out[col] = cleaned.astype("string")

    return out


def apply_user_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = df.copy()

    required_cols = (
        INPUT_COLUMNS
        + ROUGHNESS_TARGET_COLUMNS
        + list(PRECISION_RENAME_MAP.keys())
        + [EXPERIMENT_COLUMN]
    )
    missing = [c for c in required_cols if c not in mapped.columns]
    if missing:
        raise ValueError(f"Required mapped columns were not found: {missing}")

    for src, dst in PRECISION_RENAME_MAP.items():
        mapped[dst] = pd.to_numeric(mapped[src], errors="coerce")

    for c in ROUGHNESS_TARGET_COLUMNS + INPUT_COLUMNS:
        mapped[c] = pd.to_numeric(mapped[c], errors="coerce")

    mapped[EXPERIMENT_COLUMN] = pd.to_numeric(mapped[EXPERIMENT_COLUMN], errors="coerce")
    x_features = mapped[INPUT_COLUMNS].copy()

    return mapped, x_features


def build_preprocessor(x_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_df.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def _extract_loss_curve(pipeline: Pipeline) -> list[float] | None:
    """Return MLPRegressor.loss_curve_ from a fitted Pipeline, or None."""
    try:
        est = pipeline.named_steps.get("model")
        # Unwrap TransformedTargetRegressor
        if hasattr(est, "regressor_"):
            est = est.regressor_
        if hasattr(est, "loss_curve_"):
            return [float(v) for v in est.loss_curve_]
    except Exception:  # noqa: BLE001
        pass
    return None


def _add_winner_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite winner_score and rank per objective group.

    Criteria (lower score = better model):
      40 % – normalised CV RMSE          (generalisation error)
      40 % – normalised |overfit_gap|    (train-CV gap; lower = less overfit)
      20 % – normalised (1 – holdout R²) (holdout quality; lower = higher R²)
    """
    df = df.copy()
    if df.empty:
        return df

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi - lo > 1e-12 else pd.Series(0.0, index=s.index)

    norm_cv  = _norm(df["cv_rmse"])
    norm_gap = _norm(df["overfit_gap"].abs())
    norm_r2  = _norm(1.0 - df["holdout_r2"])

    df["winner_score"] = 0.40 * norm_cv + 0.40 * norm_gap + 0.20 * norm_r2
    df["rank"] = df["winner_score"].rank(method="min", ascending=True).astype(int)
    return df


def _add_winner_score_cls(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite winner_score and rank for classification results.

    Criteria (lower score = better model):
      40 % – normalised (1 – CV F1-macro)        (generalisation)
      40 % – normalised |overfit_gap|             (train-CV gap)
      20 % – normalised (1 – holdout F1-macro)   (holdout quality)
    """
    df = df.copy()
    if df.empty:
        return df

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi - lo > 1e-12 else pd.Series(0.0, index=s.index)

    norm_cv  = _norm(1.0 - df["cv_f1_macro"])
    norm_gap = _norm(df["overfit_gap"].abs())
    norm_ho  = _norm(1.0 - df["holdout_f1_macro"])

    df["winner_score"] = 0.40 * norm_cv + 0.40 * norm_gap + 0.20 * norm_ho
    df["rank"] = df["winner_score"].rank(method="min", ascending=True).astype(int)
    return df


def regression_models(
    preprocessor: ColumnTransformer,
    multi_output: bool,
    enabled_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Pipeline]:
    """Build regression pipelines.

    Parameters
    ----------
    enabled_configs:
        Mapping of model name → parameter overrides.  Keys present in the dict
        determine which models are built; ``None`` values fall back to
        ``DEFAULT_REG_CONFIGS``.  Pass ``None`` to build all models with
        their default parameters.
    """
    configs: dict[str, dict[str, Any]] = (
        {name: {} for name in DEFAULT_REG_CONFIGS}
        if enabled_configs is None
        else enabled_configs
    )

    base_models: dict[str, Any] = {}
    for name, user_params in configs.items():
        p = {**DEFAULT_REG_CONFIGS.get(name, {}), **(user_params or {})}
        if name == "elastic_net":
            base_models[name] = ElasticNet(
                alpha=float(p["alpha"]),
                l1_ratio=float(p["l1_ratio"]),
                random_state=42,
            )
        elif name == "svr_rbf":
            gamma = p["gamma"]
            try:
                gamma = float(gamma)
            except (ValueError, TypeError):
                pass
            base_models[name] = SVR(
                C=float(p["C"]),
                epsilon=float(p["epsilon"]),
                gamma=gamma,
            )
        elif name == "knn_reg":
            base_models[name] = KNeighborsRegressor(
                n_neighbors=int(p["n_neighbors"]),
                weights=str(p["weights"]),
            )
        elif name == "gpr":
            base_models[name] = GaussianProcessRegressor(
                kernel=1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3),
                alpha=float(p["alpha"]),
                normalize_y=bool(p.get("normalize_y", False)),
                random_state=42,
            )
        elif name == "mlp_small":
            hls = p["hidden_layer_sizes"]
            if isinstance(hls, str):
                hls = tuple(int(x.strip()) for x in hls.split(",") if x.strip())
            base_models[name] = MLPRegressor(
                hidden_layer_sizes=hls,
                activation=str(p["activation"]),
                alpha=float(p["alpha"]),
                learning_rate_init=float(p["learning_rate_init"]),
                max_iter=1000,
                random_state=42,
            )

    models: dict[str, Pipeline] = {}
    for name, model in base_models.items():
        scaled_estimator = TransformedTargetRegressor(
            regressor=model,
            transformer=StandardScaler(),
        )
        final_estimator = MultiOutputRegressor(scaled_estimator) if multi_output else scaled_estimator
        models[name] = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", final_estimator),
            ]
        )
    return models


def classification_models(
    preprocessor: ColumnTransformer,
    enabled_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Pipeline]:
    configs: dict[str, dict[str, Any]] = (
        {name: {} for name in DEFAULT_CLS_CONFIGS}
        if enabled_configs is None
        else enabled_configs
    )

    models: dict[str, Pipeline] = {}
    for name, user_params in configs.items():
        p = {**DEFAULT_CLS_CONFIGS.get(name, {}), **(user_params or {})}
        if name == "svc_rbf":
            gamma = p["gamma"]
            try:
                gamma = float(gamma)
            except (ValueError, TypeError):
                pass
            clf = SVC(
                C=float(p["C"]),
                gamma=gamma,
                class_weight="balanced",
                probability=True,
                random_state=42,
            )
        elif name == "knn_clf":
            clf = KNeighborsClassifier(
                n_neighbors=int(p["n_neighbors"]),
                weights=str(p["weights"]),
            )
        else:
            continue
        models[name] = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    return models


def evaluate_regression(
    x_df: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    objective_name: str,
    reg_configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(y, pd.DataFrame):
        valid_mask = y.notna().all(axis=1)
    else:
        valid_mask = y.notna()

    x_df = x_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(x_df) < 10:
        return pd.DataFrame(), {}

    preprocessor = build_preprocessor(x_df)
    multi_output = isinstance(y, pd.DataFrame)
    models = regression_models(preprocessor, multi_output=multi_output, enabled_configs=reg_configs)
    if not models:
        return pd.DataFrame(), {}

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    rows: list[dict[str, Any]] = []
    holdout_artifacts: dict[str, Any] = {}

    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        cv = cross_validate(
            model,
            x_df,
            y,
            cv=rkf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        cv_rmse_val    = -float(np.mean(cv["test_rmse"]))
        train_rmse_val = -float(np.mean(cv["train_rmse"]))

        row = {
            "objective":    objective_name,
            "task":         "regression",
            "model":        name,
            "cv_mae":       -float(np.mean(cv["test_mae"])),
            "cv_rmse":      cv_rmse_val,
            "cv_r2":        float(np.mean(cv["test_r2"])),
            "cv_rmse_std":  float(np.std(-cv["test_rmse"])),
            "train_mae":    -float(np.mean(cv["train_mae"])),
            "train_rmse":   train_rmse_val,
            "train_r2":     float(np.mean(cv["train_r2"])),
            "overfit_gap":  round(cv_rmse_val - train_rmse_val, 6),
            "holdout_mae":  float(mean_absolute_error(y_test, y_pred)),
            "holdout_rmse": float(rmse(y_test, y_pred)),
            "holdout_r2":   float(r2_score(y_test, y_pred)),
        }
        rows.append(row)

        holdout_artifacts[name] = {
            "y_test":     y_test,
            "y_pred":     y_pred,
            "loss_curve": _extract_loss_curve(model),
        }

    result_df = pd.DataFrame(rows)
    result_df = _add_winner_score(result_df)
    result_df = result_df.sort_values(["objective", "rank"]).reset_index(drop=True)
    return result_df, holdout_artifacts


def discretize_target(y: pd.Series, q: int) -> pd.Series | None:
    y_non_null = y.dropna()
    if y_non_null.nunique() < 2:
        return None
    try:
        classes = pd.qcut(y_non_null, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None

    full = pd.Series(index=y.index, dtype="float")
    full.loc[y_non_null.index] = classes.astype(float)
    return full.astype("Int64")


def evaluate_classification(
    x_df: pd.DataFrame,
    y_raw: pd.Series,
    objective_name: str,
    class_q: int,
    cls_configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_raw = pd.to_numeric(y_raw, errors="coerce")
    y_class = discretize_target(y_raw, q=class_q)
    if y_class is None:
        return pd.DataFrame(), {}

    valid_idx = y_class.dropna().index
    x = x_df.loc[valid_idx]
    y = y_class.loc[valid_idx].astype(int)

    if y.nunique() < 2:
        return pd.DataFrame(), {}

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        return pd.DataFrame(), {}

    preprocessor = build_preprocessor(x)
    models = classification_models(preprocessor, enabled_configs=cls_configs)
    if not models:
        return pd.DataFrame(), {}
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_acc": "balanced_accuracy",
    }

    rows: list[dict[str, Any]] = []
    holdout_artifacts: dict[str, Any] = {}

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        return pd.DataFrame(), {}

    for name, model in models.items():
        cv = cross_validate(
            model,
            x,
            y,
            cv=rskf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        cv_f1_val    = float(np.mean(cv["test_f1_macro"]))
        train_f1_val = float(np.mean(cv["train_f1_macro"]))

        row = {
            "objective":               objective_name,
            "task":                    "classification",
            "model":                   name,
            "cv_accuracy":             float(np.mean(cv["test_accuracy"])),
            "cv_f1_macro":             cv_f1_val,
            "cv_balanced_accuracy":    float(np.mean(cv["test_balanced_acc"])),
            "train_accuracy":          float(np.mean(cv["train_accuracy"])),
            "train_f1_macro":          train_f1_val,
            "train_balanced_accuracy": float(np.mean(cv["train_balanced_acc"])),
            "overfit_gap":             round(train_f1_val - cv_f1_val, 6),
            "holdout_accuracy":        float(accuracy_score(y_test, y_pred)),
            "holdout_f1_macro":        float(f1_score(y_test, y_pred, average="macro")),
            "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        }
        rows.append(row)

        holdout_artifacts[name] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "labels": sorted(y.unique().tolist()),
        }

    result_df = pd.DataFrame(rows)
    result_df = _add_winner_score_cls(result_df)
    result_df = result_df.sort_values(["objective", "rank"]).reset_index(drop=True)
    return result_df, holdout_artifacts


def run_mlp_sweep(
    x_df: pd.DataFrame,
    y: pd.Series,
    objective_name: str,
    param_grid: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Sweep MLPRegressor hyperparameters and return a results DataFrame with full metrics."""
    if param_grid is None:
        param_grid = MLP_SWEEP_GRID

    valid_mask = y.notna()
    x_df = x_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(x_df) < 10:
        return pd.DataFrame()

    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(x_df)
    base_mlp = MLPRegressor(max_iter=1000, random_state=42)
    scaled_mlp = TransformedTargetRegressor(regressor=base_mlp, transformer=StandardScaler())
    pipeline = Pipeline([("prep", preprocessor), ("model", scaled_mlp)])

    prefixed_grid = {f"model__regressor__{k}": v for k, v in param_grid.items()}

    scoring = {
        "mae":  "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2":   "r2",
    }

    gs = GridSearchCV(
        pipeline,
        param_grid=prefixed_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring,
        refit=False,
        n_jobs=1,
        return_train_score=False,
        error_score=np.nan,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gs.fit(x_df, y)

    cv_res = pd.DataFrame(gs.cv_results_)
    param_cols = [c for c in cv_res.columns if c.startswith("param_model__regressor__")]
    rename_map = {c: c.replace("param_model__regressor__", "") for c in param_cols}

    result = cv_res[
        param_cols + [
            "mean_test_mae", "mean_test_rmse", "mean_test_r2",
            "std_test_rmse",
        ]
    ].copy()
    result = result.rename(columns={
        **rename_map,
        "mean_test_mae":  "cv_mae",
        "mean_test_rmse": "cv_rmse",
        "mean_test_r2":   "cv_r2",
        "std_test_rmse":  "cv_rmse_std",
    })
    result["cv_mae"]  = -result["cv_mae"]
    result["cv_rmse"] = -result["cv_rmse"]
    result = result.dropna(subset=["cv_rmse"])

    # Compute holdout metrics for every candidate configuration
    holdout_rows: list[dict] = []
    for i, row in result.iterrows():
        params = {k: row[k] for k in rename_map.values() if k in row.index}
        try:
            est = Pipeline([("prep", build_preprocessor(x_train)),
                            ("model", TransformedTargetRegressor(
                                regressor=MLPRegressor(
                                    max_iter=1000, random_state=42, **params
                                ),
                                transformer=StandardScaler(),
                            ))])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                est.fit(x_train, y_train)
            y_pred = est.predict(x_test)
            holdout_rows.append({
                "holdout_mae":  float(mean_absolute_error(y_test, y_pred)),
                "holdout_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "holdout_r2":   float(r2_score(y_test, y_pred)),
            })
        except Exception:  # noqa: BLE001
            holdout_rows.append({"holdout_mae": np.nan, "holdout_rmse": np.nan, "holdout_r2": np.nan})

    holdout_df = pd.DataFrame(holdout_rows, index=result.index)
    result = pd.concat([result, holdout_df], axis=1)

    # Composite rank: normalised cv_rmse (40%) + holdout_rmse (40%) + (1-r2) holdout (20%)
    for col in ["cv_rmse", "holdout_rmse", "holdout_r2", "holdout_mae"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    rmse_range = result["cv_rmse"].max() - result["cv_rmse"].min()
    ho_range   = result["holdout_rmse"].max() - result["holdout_rmse"].min()
    r2_range   = result["holdout_r2"].max() - result["holdout_r2"].min()

    norm_cv   = ((result["cv_rmse"]   - result["cv_rmse"].min())   / rmse_range) if rmse_range > 0 else 0.0
    norm_ho   = ((result["holdout_rmse"] - result["holdout_rmse"].min()) / ho_range)   if ho_range  > 0 else 0.0
    norm_r2   = (1 - (result["holdout_r2"] - result["holdout_r2"].min()) / r2_range)   if r2_range  > 0 else 0.0

    result["score"] = 0.40 * norm_cv + 0.40 * norm_ho + 0.20 * norm_r2
    result["rank"]  = result["score"].rank(method="min", ascending=True).astype(int)

    result.insert(0, "objective", objective_name)
    result = result.sort_values("rank").reset_index(drop=True)
    # Reorder columns for readability
    param_names = list(rename_map.values())
    ordered = (["objective", "rank"] + param_names +
               ["cv_mae", "cv_rmse", "cv_r2", "cv_rmse_std",
                "holdout_mae", "holdout_rmse", "holdout_r2", "score"])
    result = result[[c for c in ordered if c in result.columns]]
    return result


def make_eda_charts(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    charts: list[str] = []
    cfg = {"responsive": True}

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        text = [[f"{v:.2f}" for v in row] for row in corr.values]
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="Viridis",
                zmin=-1,
                zmax=1,
                text=text,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(title="Mapa de correlación", height=520)
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg))

    for col in target_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Distribución", "Boxplot"),
        )
        fig.add_trace(
            go.Histogram(x=data, name="Frecuencia", marker_color="#1f77b4", opacity=0.8),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(x=data, name=col, marker_color="#ff7f0e", boxpoints="all", jitter=0.4),
            row=1,
            col=2,
        )
        fig.update_layout(title=col, height=380, showlegend=False)
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg))

    return charts


def make_regression_charts(
    reg_results: pd.DataFrame,
    reg_artifacts: dict[str, dict[str, Any]],
) -> list[str]:
    charts: list[str] = []
    cfg = {"responsive": True}
    if reg_results.empty:
        return charts

    for objective in reg_results["objective"].unique():
        subset = reg_results[reg_results["objective"] == objective].copy()

        fig = go.Figure(
            go.Bar(
                x=subset["model"],
                y=subset["cv_rmse"],
                marker_color="#2ca02c",
                error_y=dict(type="data", array=subset["cv_rmse_std"].tolist()),
                hovertemplate="<b>%{x}</b><br>CV RMSE: %{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"CV RMSE por modelo — {objective}",
            xaxis_title="Modelo",
            yaxis_title="CV RMSE",
            height=380,
        )
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg))

        best_model = subset.sort_values("cv_rmse").iloc[0]["model"]
        artifact = reg_artifacts[objective][best_model]
        y_true = np.asarray(artifact["y_test"]).reshape(-1)
        y_pred = np.asarray(artifact["y_pred"]).reshape(-1)
        residuals = y_true - y_pred

        min_v = float(np.nanmin([y_true.min(), y_pred.min()]))
        max_v = float(np.nanmax([y_true.max(), y_pred.max()]))

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Real vs Predicho — {objective} ({best_model})",
                f"Residuales — {objective} ({best_model})",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Obs",
                hovertemplate="Real: %{x:.4f}<br>Pred: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residual",
                hovertemplate="Pred: %{x:.4f}<br>Res: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[float(y_pred.min()), float(y_pred.max())],
                y=[0.0, 0.0],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Real", row=1, col=1)
        fig.update_yaxes(title_text="Predicho", row=1, col=1)
        fig.update_xaxes(title_text="Predicho", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)
        fig.update_layout(height=420, showlegend=False)
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg))

    return charts


def make_classification_charts(
    cls_results: pd.DataFrame,
    cls_artifacts: dict[str, dict[str, Any]],
) -> list[str]:
    charts: list[str] = []
    cfg = {"responsive": True}
    if cls_results.empty:
        return charts

    for objective in cls_results["objective"].unique():
        subset = cls_results[cls_results["objective"] == objective].copy()
        best_model = subset.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
        artifact = cls_artifacts[objective][best_model]

        labels = artifact["labels"]
        cm = confusion_matrix(
            artifact["y_test"],
            artifact["y_pred"],
            labels=labels,
        )

        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=[f"Pred {lb}" for lb in labels],
                y=[f"Real {lb}" for lb in labels],
                colorscale="Blues",
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title=f"Matriz de confusión — {objective} ({best_model})",
            xaxis_title="Predicho",
            yaxis_title="Real",
            height=420,
        )
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False, config=cfg))

    return charts


def make_mlp_sweep_charts(sweep_df: pd.DataFrame) -> list[str]:
    """Create horizontal bar charts for MLP hyperparameter sweep results."""
    charts: list[str] = []
    cfg = {"responsive": True}
    if sweep_df.empty:
        return charts

    for objective in sweep_df["objective"].unique():
        subset = sweep_df[sweep_df["objective"] == objective].head(20).copy()
        subset["config"] = subset.apply(
            lambda r: (
                f"hl={r.get('hidden_layer_sizes', '?')} "
                f"α={r.get('alpha', '?')} "
                f"lr={r.get('learning_rate_init', '?')} "
                f"act={r.get('activation', '?')}"
            ),
            axis=1,
        )

        # Multi-metric grouped bar: cv_rmse, holdout_rmse, cv_mae, holdout_mae
        metric_cols = [
            ("cv_rmse",      "CV RMSE",      "#1f77b4"),
            ("holdout_rmse", "Holdout RMSE", "#ff7f0e"),
            ("cv_mae",       "CV MAE",       "#2ca02c"),
            ("holdout_mae",  "Holdout MAE",  "#d62728"),
        ]
        fig_bar = go.Figure()
        for col, name, color in metric_cols:
            if col in subset.columns:
                err = subset["cv_rmse_std"].tolist() if col == "cv_rmse" else None
                fig_bar.add_trace(go.Bar(
                    name=name,
                    x=subset[col],
                    y=subset["config"],
                    orientation="h",
                    marker_color=color,
                    error_x=dict(type="data", array=err) if err else None,
                    hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.4f}}<extra></extra>",
                ))
        fig_bar.update_layout(
            title=f"MLP Sweep — {objective} (Top 20 por ranking)",
            xaxis_title="Error (menor es mejor)",
            yaxis=dict(autorange="reversed"),
            barmode="group",
            height=max(420, 34 * len(subset)),
            margin=dict(l=300),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        charts.append(pio.to_html(fig_bar, full_html=False, include_plotlyjs=False, config=cfg))

        # R² chart
        if "cv_r2" in subset.columns or "holdout_r2" in subset.columns:
            fig_r2 = go.Figure()
            for col, name, color in [("cv_r2", "CV R²", "#9467bd"),
                                      ("holdout_r2", "Holdout R²", "#8c564b")]:
                if col in subset.columns:
                    fig_r2.add_trace(go.Bar(
                        name=name, x=subset[col], y=subset["config"],
                        orientation="h", marker_color=color,
                        hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.4f}}<extra></extra>",
                    ))
            fig_r2.update_layout(
                title=f"MLP Sweep R² — {objective} (Top 20)",
                xaxis_title="R² (mayor es mejor)",
                yaxis=dict(autorange="reversed"),
                barmode="group",
                height=max(420, 34 * len(subset)),
                margin=dict(l=300),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            )
            charts.append(pio.to_html(fig_r2, full_html=False, include_plotlyjs=False, config=cfg))

    return charts


def html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No results available for this section.</p>"

    # If a "rank" column exists, build the table manually so winner rows get
    # the CSS class and rank-1 cells get the trophy badge.
    if "rank" in df.columns:
        def _fmt(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v) if v is not None else ""

        headers = "".join(f"<th>{c}</th>" for c in df.columns)
        rows_html = []
        for _, row in df.iterrows():
            is_winner = row["rank"] == 1
            cls = ' class="winner-row"' if is_winner else ""
            cells = []
            for col in df.columns:
                v = row[col]
                if col == "rank" and is_winner:
                    cells.append(f"<td>&#127942; 1</td>")
                else:
                    cells.append(f"<td>{_fmt(v)}</td>")
            rows_html.append(f"<tr{cls}>{''.join(cells)}</tr>")

        return (
            '<table class="tbl dt-table" border="0">'
            f"<thead><tr>{headers}</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
        )

    return df.to_html(index=False, classes="tbl dt-table", border=0, float_format=lambda x: f"{x:.4f}")


def _variable_table_html(
    input_cols: list[str],
    precision_cols: list[str],
    roughness_cols: list[str],
) -> str:
    rows = []
    for col in input_cols:
        rows.append(("Entrada", col))
    for col in precision_cols:
        rows.append(("Salida — Precisión dimensional", col))
    for col in roughness_cols:
        rows.append(("Salida — Rugosidad superficial", col))

    body = "".join(
        f"<tr><td>{tipo}</td><td>{var}</td></tr>"
        for tipo, var in rows
    )
    return (
        '<table class="tbl"><thead><tr>'
        "<th>Tipo de variable</th><th>Nombre de la variable</th>"
        "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )


def render_report(
    out_path: Path,
    df: pd.DataFrame,
    precision_target: str | None,
    roughness_target: str | None,
    input_cols: list[str],
    precision_cols: list[str],
    roughness_cols: list[str],
    reg_results: pd.DataFrame,
    cls_results: pd.DataFrame,
    eda_charts: list[str],
    reg_charts: list[str],
    cls_charts: list[str],
    mlp_sweep_results: pd.DataFrame,
    mlp_sweep_charts: list[str],
) -> None:
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[num_cols].describe().T.reset_index() if num_cols else pd.DataFrame()

    var_table = _variable_table_html(input_cols, precision_cols, roughness_cols)

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MEX/FDM Quality Report</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/2.0.8/css/dataTables.dataTables.min.css" />
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/2.0.8/js/dataTables.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3 {{ color: #0f172a; }}
    .meta {{ background: #f8fafc; padding: 12px 16px; border-left: 4px solid #0ea5e9; margin-bottom: 20px; }}
    .tbl {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px; font-size: 0.92rem; }}
    .tbl th, .tbl td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    .tbl th {{ background: #f1f5f9; }}
    .kpi {{ display: inline-block; margin-right: 18px; margin-bottom: 8px; padding: 8px 10px; background: #f8fafc; border-radius: 6px; }}
    tr:nth-child(even) td {{ background: #f8fafc; }}
    .chart-wrap {{ margin: 16px 0; }}
    tfoot input {{ width: 100%; box-sizing: border-box; font-size: 0.78rem; padding: 2px 4px; margin-top: 2px;
                   border: 1px solid #d1d5db; border-radius: 3px; }}
    .dataTables_wrapper {{ font-size: 0.9rem; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>Reporte de Análisis de Calidad MEX/FDM</h1>
  <div class="meta">
    <div class="kpi"><strong>Observaciones:</strong> {len(df)}</div>
    <div class="kpi"><strong>Variables usadas:</strong> {df.shape[1]}</div>
    <div class="kpi"><strong>Variables de entrada:</strong> {len(input_cols)}</div>
    <div class="kpi"><strong>Variables de salida (precisión):</strong> {len(precision_cols)}</div>
    <div class="kpi"><strong>Variables de salida (rugosidad):</strong> {len(roughness_cols)}</div>
  </div>

  <h2>1. Variables de Entrada y Salida</h2>
  <p>Las variables de entrada son los parámetros de proceso controlados durante la impresión; las variables de salida son los indicadores de calidad a predecir.</p>
  {var_table}

  <h2>2. Datos y Calidad</h2>
  <h3>Valores Faltantes</h3>
  {html_table(missing)}
  <h3>Estadísticas Descriptivas</h3>
  {html_table(summary)}

  <h2>3. Análisis Exploratorio Visual</h2>
  {''.join(f'<div class="chart-wrap">{c}</div>' for c in eda_charts)}

  <h2>4. Resultados de Regresión</h2>
  {html_table(reg_results)}
  {''.join(f'<div class="chart-wrap">{c}</div>' for c in reg_charts)}

  <h2>5. Barrido de Hiperparámetros MLP</h2>
  <p>Se prueban distintas combinaciones de capas ocultas, α (regularización L2), tasa de aprendizaje y función de activación mediante búsqueda exhaustiva con validación cruzada de 5 pliegues. Resultados ordenados por CV RMSE ascendente.</p>
  {html_table(mlp_sweep_results)}
  {''.join(f'<div class="chart-wrap">{c}</div>' for c in mlp_sweep_charts)}

  <h2>6. Resultados de Clasificación (Opcional)</h2>
  {html_table(cls_results)}
  {''.join(f'<div class="chart-wrap">{c}</div>' for c in cls_charts)}

  <h2>7. Notas Técnicas y Control de Calidad</h2>
  <ul>
    <li>La validación cruzada usa folds repetidos para reducir la varianza en conjuntos de datos pequeños.</li>
    <li>Las métricas de holdout complementan los resultados de validación cruzada.</li>
    <li>La comparación incluye objetivos individuales y salida múltiple (precisión dimensional).</li>
    <li>Para regresión: MAE, RMSE y R²; para clasificación: F1-macro y exactitud balanceada.</li>
    <li>Se excluyen los experimentos 28, 38 y 40 del análisis de rugosidad por sesgo documentado.</li>
  </ul>

  <script>
  $(document).ready(function () {{
    $('.dt-table').each(function () {{
      var $tbl = $(this);
      var $tfoot = $('<tfoot><tr></tr></tfoot>');
      $tbl.find('thead th').each(function () {{
        $tfoot.find('tr').append('<th></th>');
      }});
      $tbl.append($tfoot);
      $tbl.DataTable({{
        pageLength: 25,
        language: {{
          search: 'Buscar:',
          lengthMenu: 'Mostrar _MENU_ registros',
          info: 'Mostrando _START_ a _END_ de _TOTAL_ registros',
          infoEmpty: 'Sin resultados',
          paginate: {{ first: 'Primera', last: 'Última', next: 'Siguiente', previous: 'Anterior' }}
        }},
        initComplete: function () {{
          this.api().columns().every(function () {{
            var col = this;
            var title = $(col.header()).text();
            $('<input type="text" placeholder="' + title + '" />')
              .appendTo($(col.footer()).empty())
              .on('keyup change clear', function () {{
                if (col.search() !== this.value) {{
                  col.search(this.value).draw();
                }}
              }});
          }});
        }}
      }});
    }});
  }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    asset_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df = read_sheet_clean(args.data, args.sheet)
    df = normalize_dataframe_types(df)
    df.columns = [str(c).strip() for c in df.columns]
    df, x_df = apply_user_mapping(df)

    precision_target = ", ".join(PRECISION_RENAME_MAP.values())
    roughness_target = ", ".join(ROUGHNESS_TARGET_COLUMNS)

    reg_tables: list[pd.DataFrame] = []
    reg_artifacts_by_objective: dict[str, dict[str, Any]] = {}

    for col_name in PRECISION_RENAME_MAP.values():
        y_precision = pd.to_numeric(df[col_name], errors="coerce")
        objective = f"precision_{slugify(col_name)}"
        tbl, artifacts = evaluate_regression(x_df, y_precision, objective)
        reg_tables.append(tbl)
        reg_artifacts_by_objective[objective] = artifacts

    bias_mask = ~df[EXPERIMENT_COLUMN].isin(ROUGHNESS_BIAS_EXPERIMENTS)
    for col_name in ROUGHNESS_TARGET_COLUMNS:
        x_rough = x_df.loc[bias_mask].reset_index(drop=True)
        y_roughness = pd.to_numeric(df.loc[bias_mask, col_name], errors="coerce").reset_index(drop=True)
        objective = f"roughness_{slugify(col_name)}_no_bias"
        tbl, artifacts = evaluate_regression(x_rough, y_roughness, objective)
        reg_tables.append(tbl)
        reg_artifacts_by_objective[objective] = artifacts

    y_multi = df[list(PRECISION_RENAME_MAP.values())].apply(pd.to_numeric, errors="coerce")
    tbl, artifacts = evaluate_regression(x_df, y_multi, "precision_multi_output")
    reg_tables.append(tbl)
    reg_artifacts_by_objective["precision_multi_output"] = artifacts

    reg_results = pd.concat(reg_tables, ignore_index=True) if reg_tables else pd.DataFrame()

    cls_tables: list[pd.DataFrame] = []
    cls_artifacts_by_objective: dict[str, dict[str, Any]] = {}

    if not args.disable_classification:
        for col_name in PRECISION_RENAME_MAP.values():
            objective = f"precision_classes_{slugify(col_name)}"
            tbl, artifacts = evaluate_classification(
                x_df, df[col_name], objective, args.class_q
            )
            if not tbl.empty:
                cls_tables.append(tbl)
                cls_artifacts_by_objective[objective] = artifacts

        for col_name in ROUGHNESS_TARGET_COLUMNS:
            x_rough = x_df.loc[bias_mask].reset_index(drop=True)
            y_rough = df.loc[bias_mask, col_name].reset_index(drop=True)
            objective = f"roughness_classes_{slugify(col_name)}_no_bias"
            tbl, artifacts = evaluate_classification(
                x_rough, y_rough, objective, args.class_q
            )
            if not tbl.empty:
                cls_tables.append(tbl)
                cls_artifacts_by_objective[objective] = artifacts

    cls_results = pd.concat(cls_tables, ignore_index=True) if cls_tables else pd.DataFrame()

    mlp_sweep_tables: list[pd.DataFrame] = []
    if not args.disable_mlp_sweep:
        y_precision_comp = (
            df[list(PRECISION_RENAME_MAP.values())]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .mean(axis=1)
        )
        tbl = run_mlp_sweep(x_df, y_precision_comp, "precision_composite")
        mlp_sweep_tables.append(tbl)

        y_roughness_comp = (
            df.loc[bias_mask, ROUGHNESS_TARGET_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .reset_index(drop=True)
        )
        x_rough_sweep = x_df.loc[bias_mask].reset_index(drop=True)
        tbl = run_mlp_sweep(x_rough_sweep, y_roughness_comp, "roughness_composite_no_bias")
        mlp_sweep_tables.append(tbl)

    mlp_sweep_results = pd.concat(mlp_sweep_tables, ignore_index=True) if mlp_sweep_tables else pd.DataFrame()

    precision_cols = list(PRECISION_RENAME_MAP.values())
    roughness_cols = ROUGHNESS_TARGET_COLUMNS
    used_cols = [c for c in INPUT_COLUMNS + precision_cols + roughness_cols if c in df.columns]
    df_report = df[used_cols].copy()

    eda_targets = precision_cols + roughness_cols
    eda_charts = make_eda_charts(df_report, eda_targets)
    reg_charts = make_regression_charts(reg_results, reg_artifacts_by_objective)
    cls_charts = make_classification_charts(cls_results, cls_artifacts_by_objective)
    mlp_sweep_charts = make_mlp_sweep_charts(mlp_sweep_results)

    report_path = output_dir / "quality_report.html"
    render_report(
        out_path=report_path,
        df=df_report,
        precision_target=precision_target,
        roughness_target=roughness_target,
        input_cols=INPUT_COLUMNS,
        precision_cols=precision_cols,
        roughness_cols=roughness_cols,
        reg_results=reg_results,
        cls_results=cls_results,
        eda_charts=eda_charts,
        reg_charts=reg_charts,
        cls_charts=cls_charts,
        mlp_sweep_results=mlp_sweep_results,
        mlp_sweep_charts=mlp_sweep_charts,
    )

    reg_csv = output_dir / "regression_results.csv"
    cls_csv = output_dir / "classification_results.csv"
    sweep_csv = output_dir / "mlp_sweep_results.csv"
    reg_results.to_csv(reg_csv, index=False)
    cls_results.to_csv(cls_csv, index=False)
    mlp_sweep_results.to_csv(sweep_csv, index=False)

    print(f"Report generated: {report_path}")
    print(f"Regression results: {reg_csv}")
    if not cls_results.empty:
        print(f"Classification results: {cls_csv}")
    else:
        print("Classification results skipped or unavailable.")
    if not mlp_sweep_results.empty:
        print(f"MLP sweep results: {sweep_csv}")
    else:
        print("MLP sweep skipped or unavailable.")


if __name__ == "__main__":
    main()
