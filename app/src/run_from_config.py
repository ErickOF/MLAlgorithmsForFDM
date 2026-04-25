#!/usr/bin/env python3
"""CLI script for running quality analysis from a JSON configuration file.

The JSON config is in the format exported by the web application's
"Export Config" button:

    {
      "exported_at": "2026-04-24T...",
      "variables": {
        "input":  ["Temperatura de la boquilla (°C)", ...],
        "output": ["Ra_0,8Promedio (μm)", ...]
      },
      "algorithms": {
        "regression":     {"mlp_small": {"hidden_layer_sizes": [32, 16], ...}, "svr_rbf": null},
        "classification": {"svc_rbf": null}
      },
      "options": {
        "class_q": 3
      }
    }

An empty algorithms object (``{}``) selects all default models for that task type.
Use ``--skip-regression`` or ``--skip-classification`` to disable a task entirely.

Usage examples:
    python run_from_config.py config.json
    python run_from_config.py config.json --data ../../data/Replica.xlsx
    python run_from_config.py config.json --output-dir ./results --no-report
    python run_from_config.py config.json --skip-classification
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_quality_analysis import (
    EXPERIMENT_COLUMN,
    ROUGHNESS_BIAS_EXPERIMENTS,
    evaluate_classification,
    evaluate_regression,
    make_classification_charts,
    make_eda_charts,
    make_regression_charts,
    normalize_dataframe_types,
    read_sheet_clean,
    render_report,
    slugify,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA = THIS_DIR.parent.parent / "data" / "Réplica Shirmohammadi 2021 Datos.xlsx"
DEFAULT_SHEET = "Datos a utilizar en Minitab"

# Roughness column prefixes — same heuristic used by the web app
_ROUGHNESS_KW = ("ra_", "rq_", "rz_", "ra,", "rq,", "rz,")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_file(hint: Path | None) -> Path:
    if hint is not None:
        if not hint.exists():
            raise FileNotFoundError(f"Data file not found: {hint}")
        return hint
    for candidate in (
        [DEFAULT_DATA]
        + list(THIS_DIR.parent.parent.glob("*.xlsx"))
        + list(THIS_DIR.parent.parent.glob("data/*.xlsx"))
        + list(THIS_DIR.parent.parent.glob("data/*.csv"))
    ):
        p = Path(candidate)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not auto-detect a data file. Use --data to specify the path."
    )


def _load_df(data_path: Path, sheet: str) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = read_sheet_clean(data_path, sheet)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data file format: {suffix}")
    df = normalize_dataframe_types(df)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _is_roughness(col: str) -> bool:
    return any(col.lower().startswith(k) for k in _ROUGHNESS_KW)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run quality analysis from an exported JSON configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "config",
        type=Path,
        help="Path to the JSON config file exported from the web app.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the Excel (.xlsx) or CSV data file. Auto-detected if omitted.",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=DEFAULT_SHEET,
        help=f"Sheet name in the Excel file (default: '{DEFAULT_SHEET}').",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./report_output"),
        help="Directory where CSV results and the HTML report are saved (default: ./report_output).",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation; only save CSV results.",
    )
    p.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip all regression tasks.",
    )
    p.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip all classification tasks.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = _parse_args()

    # ---- Load config -------------------------------------------------------
    config_path = args.config.resolve()
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        cfg: dict = json.load(fh)

    variables: dict = cfg.get("variables", {})
    input_cols: list[str] = variables.get("input", [])
    output_cols: list[str] = variables.get("output", [])

    if not input_cols:
        sys.exit("Config 'variables.input' is empty — specify at least one input column.")
    if not output_cols:
        sys.exit("Config 'variables.output' is empty — specify at least one output column.")

    algorithms: dict = cfg.get("algorithms", {})
    reg_selection: dict[str, dict | None] = algorithms.get("regression", {})
    cls_selection: dict[str, dict | None] = algorithms.get("classification", {})
    options: dict = cfg.get("options", {})
    class_q: int = int(options.get("class_q", 3))

    # Empty dict → None means "run all default models" (mirrors web app behaviour)
    reg_configs: dict[str, dict] | None = (
        None if not reg_selection
        else {k: (v or {}) for k, v in reg_selection.items()}
    )
    cls_configs: dict[str, dict] | None = (
        None if not cls_selection
        else {k: (v or {}) for k, v in cls_selection.items()}
    )

    # ---- Load data ---------------------------------------------------------
    data_path = _find_data_file(args.data)
    print(f"[info] Data:   {data_path}")
    print(f"[info] Config: {config_path}")

    df = _load_df(data_path, args.sheet)

    missing = [c for c in input_cols + output_cols if c not in df.columns]
    if missing:
        print(
            f"[error] Columns not found in data: {missing}\n"
            f"        Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Coerce all selected columns to numeric
    for col in input_cols + output_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x_df = df[input_cols].copy()

    # Bias mask for roughness columns (biased experiments excluded)
    bias_mask: pd.Series = (
        ~df[EXPERIMENT_COLUMN].isin(ROUGHNESS_BIAS_EXPERIMENTS)
        if EXPERIMENT_COLUMN in df.columns
        else pd.Series(True, index=df.index)
    )

    # Classify output columns by domain
    roughness_cols = [c for c in output_cols if _is_roughness(c)]
    precision_cols = [c for c in output_cols if not _is_roughness(c)]
    # Fall back: if nothing matched as roughness treat everything as precision
    if not precision_cols and roughness_cols:
        precision_cols, roughness_cols = roughness_cols, []

    # ---- Prepare output directory ------------------------------------------
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)

    # ---- Run analysis ------------------------------------------------------
    reg_tables: list[pd.DataFrame] = []
    reg_artifacts: dict[str, dict[str, Any]] = {}
    cls_tables: list[pd.DataFrame] = []
    cls_artifacts: dict[str, dict[str, Any]] = {}

    def _run_reg(x: pd.DataFrame, y: "pd.Series | pd.DataFrame", objective: str) -> None:
        print(f"  [regression]     {objective}")
        tbl, arts = evaluate_regression(x, y, objective, reg_configs)
        if not tbl.empty:
            reg_tables.append(tbl)
            reg_artifacts[objective] = arts

    def _run_cls(x: pd.DataFrame, y_raw: pd.Series, objective: str) -> None:
        print(f"  [classification] {objective}")
        tbl, arts = evaluate_classification(x, y_raw, objective, class_q, cls_configs)
        if not tbl.empty:
            cls_tables.append(tbl)
            cls_artifacts[objective] = arts

    # -- Regression ----------------------------------------------------------
    if not args.skip_regression:
        print("[info] Running regression tasks…")
        for col in precision_cols:
            y = pd.to_numeric(df[col], errors="coerce")
            _run_reg(x_df, y, f"precision_{slugify(col)}")
        if len(precision_cols) > 1:
            y_multi = df[precision_cols].apply(pd.to_numeric, errors="coerce")
            _run_reg(x_df, y_multi, "precision_multi_output")

        for col in roughness_cols:
            xr = x_df.loc[bias_mask].reset_index(drop=True)
            yr = pd.to_numeric(df.loc[bias_mask, col], errors="coerce").reset_index(drop=True)
            _run_reg(xr, yr, f"roughness_{slugify(col)}")
        if len(roughness_cols) > 1:
            xr = x_df.loc[bias_mask].reset_index(drop=True)
            y_multi = (
                df.loc[bias_mask, roughness_cols]
                .apply(pd.to_numeric, errors="coerce")
                .reset_index(drop=True)
            )
            _run_reg(xr, y_multi, "roughness_multi_output")
    else:
        print("[info] Regression skipped (--skip-regression).")

    # -- Classification ------------------------------------------------------
    if not args.skip_classification:
        print("[info] Running classification tasks…")
        for col in precision_cols:
            _run_cls(x_df, df[col], f"precision_cls_{slugify(col)}")
        for col in roughness_cols:
            xr = x_df.loc[bias_mask].reset_index(drop=True)
            yr = df.loc[bias_mask, col].reset_index(drop=True)
            _run_cls(xr, yr, f"roughness_cls_{slugify(col)}")
    else:
        print("[info] Classification skipped (--skip-classification).")

    # ---- Consolidate and save CSV results ----------------------------------
    reg_results = pd.concat(reg_tables, ignore_index=True) if reg_tables else pd.DataFrame()
    cls_results = pd.concat(cls_tables, ignore_index=True) if cls_tables else pd.DataFrame()

    reg_csv = output_dir / "regression_results.csv"
    cls_csv = output_dir / "classification_results.csv"
    reg_results.to_csv(reg_csv, index=False)
    cls_results.to_csv(cls_csv, index=False)
    print(f"[output] Regression results:     {reg_csv}")
    if not cls_results.empty:
        print(f"[output] Classification results: {cls_csv}")

    # ---- Generate HTML report ----------------------------------------------
    if not args.no_report:
        print("[info] Generating HTML report…")
        all_output_cols = precision_cols + roughness_cols
        used_cols = [c for c in input_cols + all_output_cols if c in df.columns]
        df_report = df[used_cols].copy()

        eda_charts = make_eda_charts(df_report, all_output_cols)
        reg_charts = make_regression_charts(reg_results, reg_artifacts)
        cls_charts = make_classification_charts(cls_results, cls_artifacts)

        report_path = output_dir / "quality_report.html"
        render_report(
            out_path=report_path,
            df=df_report,
            precision_target=", ".join(precision_cols) if precision_cols else None,
            roughness_target=", ".join(roughness_cols) if roughness_cols else None,
            input_cols=input_cols,
            precision_cols=precision_cols,
            roughness_cols=roughness_cols,
            reg_results=reg_results,
            cls_results=cls_results,
            eda_charts=eda_charts,
            reg_charts=reg_charts,
            cls_charts=cls_charts,
            mlp_sweep_results=pd.DataFrame(),
            mlp_sweep_charts=[],
        )
        print(f"[output] HTML report:            {report_path}")
    else:
        print("[info] HTML report skipped (--no-report).")

    print("[done]")


if __name__ == "__main__":
    main()
