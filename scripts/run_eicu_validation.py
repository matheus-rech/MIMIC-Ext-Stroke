#!/usr/bin/env python3
"""CLI entry-point for eICU-CRD 2.0 external validation of the stroke cohort.

Usage
-----
    cd /Users/matheusrech/MIMIC-Ext-Stroke
    source .venv/bin/activate

    python scripts/run_eicu_validation.py \
        --eicu-path /path/to/eicu/csv.gz/ \
        --output-path ./outputs/eicu_validation/

    # Optionally compare with existing MIMIC cohort:
    python scripts/run_eicu_validation.py \
        --eicu-path /path/to/eicu/csv.gz/ \
        --mimic-cohort ./outputs/cohort/stroke_cohort.parquet \
        --output-path ./outputs/eicu_validation/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Allow running as ``python scripts/run_eicu_validation.py`` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.eicu_validation import (
    extract_eicu_stroke_cohort,
    extract_eicu_stroke_timeseries,
    compare_cohort_demographics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eicu_validation")


def _build_config(args: argparse.Namespace) -> dict:
    """Build a config dict compatible with extract_eicu_stroke_cohort."""
    # Start from project config.yaml if present, else sensible defaults
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {},
            "cohort": {"min_icu_los_hours": 6, "max_icu_los_days": 30},
            "timeseries": {"max_hours": 72},
        }

    # Override with CLI arguments
    config["data"]["eicu_path"] = str(args.eicu_path)
    config["data"]["eicu_output_path"] = str(args.output_path)

    if args.min_icu_los_hours is not None:
        config["cohort"]["min_icu_los_hours"] = args.min_icu_los_hours
    if args.max_icu_los_days is not None:
        config["cohort"]["max_icu_los_days"] = args.max_icu_los_days
    if args.max_hours is not None:
        config.setdefault("timeseries", {})["max_hours"] = args.max_hours

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract stroke cohort from eICU-CRD 2.0 for external validation.",
    )
    parser.add_argument(
        "--eicu-path",
        type=Path,
        required=True,
        help="Directory containing eICU CSV.GZ files (patient.csv.gz, diagnosis.csv.gz, ...)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/eicu_validation"),
        help="Output directory for eICU cohort and timeseries parquets.",
    )
    parser.add_argument(
        "--mimic-cohort",
        type=Path,
        default=None,
        help="Path to MIMIC stroke_cohort.parquet for demographic comparison.",
    )
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Skip time-series extraction (cohort only).",
    )
    parser.add_argument("--min-icu-los-hours", type=int, default=None)
    parser.add_argument("--max-icu-los-days", type=int, default=None)
    parser.add_argument("--max-hours", type=int, default=None)

    args = parser.parse_args()
    config = _build_config(args)

    # --- Validate eICU path ---
    eicu_dir = Path(args.eicu_path)
    required_files = ["patient.csv.gz", "diagnosis.csv.gz"]
    for fname in required_files:
        if not (eicu_dir / fname).exists():
            logger.error("Missing required eICU file: %s/%s", eicu_dir, fname)
            sys.exit(1)

    logger.info("=" * 72)
    logger.info("eICU-CRD 2.0 Stroke External Validation")
    logger.info("=" * 72)

    # ---- Step 1: Cohort extraction ----
    eicu_cohort = extract_eicu_stroke_cohort(config)
    logger.info("eICU cohort: %d patients", len(eicu_cohort))

    if eicu_cohort.empty:
        logger.error("No stroke patients found in eICU data. Aborting.")
        sys.exit(1)

    # ---- Step 2: Time-series extraction (optional) ----
    if not args.skip_timeseries:
        ts = extract_eicu_stroke_timeseries(config, eicu_cohort)
        logger.info(
            "eICU timeseries: %d rows, %d stays, %d hours max",
            len(ts),
            ts["stay_id"].nunique() if not ts.empty else 0,
            int(ts["hour"].max()) if not ts.empty else 0,
        )

    # ---- Step 3: Compare with MIMIC (if provided) ----
    if args.mimic_cohort is not None:
        mimic_path = Path(args.mimic_cohort)
        if not mimic_path.exists():
            logger.warning("MIMIC cohort file not found: %s -- skipping comparison.", mimic_path)
        else:
            logger.info("Loading MIMIC cohort from %s", mimic_path)
            mimic_cohort = pd.read_parquet(mimic_path)
            comparison = compare_cohort_demographics(mimic_cohort, eicu_cohort)

            # Save comparison table
            out_path = Path(args.output_path)
            comparison.to_csv(out_path / "cohort_comparison.csv", index=False)
            logger.info("Comparison table saved -> %s", out_path / "cohort_comparison.csv")

    logger.info("=" * 72)
    logger.info("eICU external validation complete.")
    logger.info("Outputs -> %s", args.output_path)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
