#!/usr/bin/env python3
"""CLI script for standalone optimization of trained FDM quality models.

Usage examples:
    python run_optimization.py --model svr_rbf --target Ra_0,8Promedio --direction minimize
    python run_optimization.py --model elastic_net --target Ra_2,5Promedio --algo doe
    python run_optimization.py --model mlp_small --target Rz_0,8Promedio --n-iter 100 --output-csv results.csv

Run `python run_optimization.py --help` for full options.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so this script can be run from any directory
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from run_quality_analysis import (
    INPUT_COLUMNS,
    ROUGHNESS_TARGET_COLUMNS,
    PRECISION_RENAME_MAP,
    build_preprocessor,
    discretize_target,
    read_sheet_clean,
    regression_models,
    classification_models,
)
from optimize_quality import (
    OPTIMIZER_REGISTRY,
    ALGO_LABELS,
    get_data_bounds,
    optimize_single,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_DATA = THIS_DIR.parent.parent / "Réplica Shirmohammadi con RSM correcto.mpx"
DEFAULT_EXCEL = list(THIS_DIR.parent.parent.glob("*.xlsx"))
KNOWN_REGRESSION_TARGETS = list(ROUGHNESS_TARGET_COLUMNS) + list(PRECISION_RENAME_MAP.values())
KNOWN_CLASSIFICATION_TARGETS = KNOWN_REGRESSION_TARGETS  # classification also uses same raw cols


def _find_data_file() -> Path:
    """Try to locate the default Excel data file."""
    root = THIS_DIR.parent.parent
    candidates = list(root.glob("*.xlsx")) + list(root.glob("data/*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "Could not find a data file. Use --data to specify the path."
    )


def _load_df(data_path: Path, sheet: str) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return read_sheet_clean(data_path, sheet)
    elif suffix == ".csv":
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data file format: {suffix}")


def _build_x_y(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    class_q: int,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Extract x_df and y from the raw dataframe. Returns (x_df, y, resolved_target)."""
    # Map precision columns if needed
    df = df.copy()
    df.rename(columns=PRECISION_RENAME_MAP, inplace=True)

    # Find target column (partial match)
    all_targets = [c for c in df.columns if c not in INPUT_COLUMNS]
    match = [c for c in all_targets if target_col.lower() in c.lower()]
    if not match:
        raise ValueError(
            f"Target '{target_col}' not found. Available columns:\n  " +
            "\n  ".join(all_targets)
        )
    resolved = match[0]
    print(f"[info] Using target column: '{resolved}'")

    # Feature matrix
    x_cols = [c for c in INPUT_COLUMNS if c in df.columns]
    if not x_cols:
        raise ValueError(f"None of INPUT_COLUMNS found in data. Columns: {list(df.columns)}")
    x_df = df[x_cols].copy()

    # Target
    y_raw = pd.to_numeric(df[resolved], errors="coerce")
    if task == "classification":
        y = discretize_target(y_raw, q=class_q)
        y = y.astype(int)
    else:
        y = y_raw

    # Drop NaN rows
    valid = x_df.notna().all(axis=1) & y.notna()
    x_df = x_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    print(f"[info] Dataset: {len(x_df)} rows × {len(x_df.columns)} features")
    return x_df, y, resolved


def _fit_model(
    model_name: str,
    task: str,
    x_df: pd.DataFrame,
    y: pd.Series,
    class_q: int,
):
    """Fit and return a sklearn Pipeline for the given model."""
    preprocessor = build_preprocessor(x_df)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if task == "regression":
            pipes = regression_models(preprocessor, multi_output=False)
            if model_name not in pipes:
                available = sorted(pipes.keys())
                raise ValueError(
                    f"Regression model '{model_name}' not found. Available: {available}"
                )
            pipe = pipes[model_name]
            pipe.fit(x_df, y)
        else:
            pipes = classification_models(preprocessor)
            if model_name not in pipes:
                available = sorted(pipes.keys())
                raise ValueError(
                    f"Classification model '{model_name}' not found. Available: {available}"
                )
            pipe = pipes[model_name]
            pipe.fit(x_df, y)
    return pipe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize FDM quality model parameters using metaheuristic algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data", type=Path, default=None,
                   help="Path to Excel (.xlsx) or CSV data file. "
                        "Auto-detected if omitted.")
    p.add_argument("--sheet", default="Datos a utilizar en Minitab",
                   help="Sheet name in Excel file (default: 'Datos a utilizar en Minitab').")
    p.add_argument("--model", required=True,
                   help="Model name, e.g. svr_rbf, elastic_net, mlp_small, gpr, knn_reg, "
                        "svc_rbf, knn_clf.")
    p.add_argument("--target", required=True,
                   help="Target column name or partial match, e.g. 'Ra_0,8' or 'Rz_2,5'.")
    p.add_argument("--task", choices=["regression", "classification"], default="regression",
                   help="Task type (default: regression).")
    p.add_argument("--class-q", type=int, default=3,
                   help="Number of quality classes for classification (default: 3).")
    p.add_argument("--direction", choices=["minimize", "maximize"], default="minimize",
                   help="Optimization direction (default: minimize).")
    p.add_argument("--algo", default=None,
                   help="Override optimization algorithm. See OPTIMIZER_REGISTRY for options.")
    p.add_argument("--n-iter", type=int, default=50,
                   help="Number of iterations / evaluations (default: 50).")
    p.add_argument("--bounds", type=json.loads, default=None,
                   help='JSON string of bounds: \'{"col": [lo, hi], ...}\'. '
                        "Defaults to data min/max per column.")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Save optimal inputs to a CSV file.")
    p.add_argument("--list-algos", action="store_true",
                   help="List available algorithms per model and exit.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_algos:
        print("\nAvailable optimization algorithms per model:")
        for mn, algos in OPTIMIZER_REGISTRY.items():
            print(f"  {mn:<16}: " + ", ".join(f"{a} ({ALGO_LABELS.get(a, a)})" for a in algos))
        return

    # Locate data file
    data_path: Path = args.data or _find_data_file()
    print(f"[info] Data file: {data_path}")

    # Load and prepare data
    df = _load_df(data_path, args.sheet)
    x_df, y, resolved_target = _build_x_y(df, args.target, args.task, args.class_q)

    # Fit model
    print(f"[info] Fitting {args.model} ({args.task}) on full dataset…")
    fitted = _fit_model(args.model, args.task, x_df, y, args.class_q)
    print("[info] Model fitted.")

    # Prepare bounds
    data_bounds = get_data_bounds(x_df)
    if args.bounds:
        bounds = {
            c: (float(args.bounds[c][0]), float(args.bounds[c][1]))
            if c in args.bounds else data_bounds[c]
            for c in x_df.columns
        }
    else:
        bounds = data_bounds

    print(f"\n[info] Bounds used:")
    for col, (lo, hi) in bounds.items():
        print(f"  {col}: [{lo:.4g}, {hi:.4g}]")

    # Determine algorithm
    algo = args.algo
    if algo:
        print(f"\n[info] Algorithm: {algo} ({ALGO_LABELS.get(algo, algo)})")
    else:
        registry_algos = OPTIMIZER_REGISTRY.get(args.model, [])
        default_algo = registry_algos[0] if registry_algos else "gradient"
        print(f"\n[info] Algorithm: {default_algo} ({ALGO_LABELS.get(default_algo, default_algo)}) "
              f"[default for {args.model}]")

    print(f"[info] Direction: {args.direction}")
    print(f"[info] Iterations: {args.n_iter}\n")
    print("Running optimization…")

    result = optimize_single(
        fitted, args.model, x_df, y,
        args.direction, bounds,
        algo=algo,
        n_iter=args.n_iter,
    )

    # Print results
    if result.get("error"):
        print(f"\n[ERROR] Optimization failed: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS FOUND")
    print("=" * 60)
    for col, val in (result.get("optimal_inputs") or {}).items():
        print(f"  {col}: {val:.6f}")

    pred = result.get("predicted_value")
    print(f"\nPredicted {'minimum' if args.direction == 'minimize' else 'maximum'}: "
          f"{pred:.6f}" if pred is not None else "\nPredicted value: N/A")

    algo_used = result.get("algo_used", "?")
    print(f"\nAlgorithm used : {algo_used} ({ALGO_LABELS.get(algo_used, algo_used)})")
    print(f"Evaluations    : {result.get('n_evaluations', '—')}")
    print(f"Duration       : {result.get('duration_sec', 0):.2f}s")
    print("=" * 60)

    # Save CSV if requested
    if args.output_csv:
        rows = [{"column": k, "optimal_value": v}
                for k, v in (result.get("optimal_inputs") or {}).items()]
        rows.append({"column": f"predicted_{resolved_target}", "optimal_value": pred})
        out_df = pd.DataFrame(rows)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
