# %% [markdown]
# # FX Pairs: VS02 Conditional IC — Volatility as State
#
# Pre-registered unit: Vertical Slice 02 / H1.
# Tests whether `vol_gk_21d` conditions the cross-sectional IC of `ret_21d`
# (exploratory backup: `mom_skip_recent`) against `fwd_ret_1d`.
#
# Setup invariants and sealed holdout match Stage 05. This script does not
# replace `05_evaluation.py`; it scores only the scoped VS02 design.
#
# Outputs under `evaluation/`:
# - `vs02_conditional_ic_timeseries.parquet`
# - `vs02_summary.parquet`

# %%
"""VS02: vol_gk_21d as state gate for short-horizon signal IC (validation only)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import yaml
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from scipy.stats import spearmanr

from utils.artifact_specs import resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir

CASE_STUDY_ID = "fx_pairs"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
evaluation_config = load_evaluation_config(CASE_STUDY_ID)

JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"
LABEL_COL = setup["labels"]["primary"]
STATE_COL = "vol_gk_21d"
PRIMARY_SIGNAL = "ret_21d"
BACKUP_SIGNAL = "mom_skip_recent"
SIGNALS = [PRIMARY_SIGNAL, BACKUP_SIGNAL]
HOLDOUT_START = date.fromisoformat(str(evaluation_config["holdout_start"]))

IC_THRESHOLD = 0.005
STABILITY_THRESHOLD = 0.60
MIN_FOLD_DAYS = 5
MIN_SESSIONS = 20
MIN_PERIODS_FULL = 5
MIN_PERIODS_TERCILE = 5  # ~20/3 pairs; drop thin daily slices

LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, LABEL_COL, setup)
LABEL_HORIZON = int(str(LABEL_BUFFER).rstrip("Dd"))

financial = pl.read_parquet(CASE_DIR / "features" / "financial.parquet")
labels = pl.read_parquet(CASE_DIR / "labels" / f"{LABEL_COL}.parquet")
needed = JOIN_COLS + [STATE_COL, *SIGNALS]
missing = [c for c in needed if c not in financial.columns]
if missing:
    raise ValueError(f"financial.parquet missing required columns: {missing}")

panel_all = (
    financial.select(needed)
    .join(labels.select(JOIN_COLS + [LABEL_COL]), on=JOIN_COLS, how="inner")
    .sort([DATE_COL, "symbol"])
)

UNIQUE_DATES = labels.select(DATE_COL).unique().sort(DATE_COL)
splits = generate_cv_splits(
    UNIQUE_DATES,
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    outcome_horizon=LABEL_BUFFER,
)


def daily_spearman_ic(
    frame: pl.DataFrame,
    feature: str,
    return_col: str,
    *,
    min_periods: int,
    extra_keys: list[str] | None = None,
) -> pl.DataFrame:
    """One Spearman IC per session (optional partition keys already on rows)."""
    rows: list[dict] = []
    keys = [DATE_COL] + (extra_keys or [])
    for group in frame.partition_by(keys, maintain_order=True):
        valid = group.select([feature, return_col]).drop_nulls()
        if len(valid) < min_periods:
            continue
        ic, _ = spearmanr(valid[feature].to_numpy(), valid[return_col].to_numpy())
        if not np.isfinite(ic):
            continue
        row = {
            DATE_COL: group[DATE_COL][0],
            "fold": int(group["fold"][0]),
            "ic": float(ic),
            "n_obs": len(valid),
        }
        for key in extra_keys or []:
            row[key] = group[key][0]
        rows.append(row)
    return pl.DataFrame(rows).sort(DATE_COL) if rows else pl.DataFrame()


def fold_sign_consistency(ic_frame: pl.DataFrame, mean_ic: float) -> dict:
    fold_means: list[float] = []
    for fold in sorted(ic_frame["fold"].unique().to_list()):
        values = ic_frame.filter(pl.col("fold") == fold).sort(DATE_COL)["ic"].to_numpy()
        if len(values) >= MIN_FOLD_DAYS:
            fold_means.append(float(np.mean(values)))
    if not fold_means:
        return {
            "n_folds": 0,
            "sign_consistency": float("nan"),
            "worst_fold_ic": float("nan"),
            "best_fold_ic": float("nan"),
            "fold_ics": [],
            "max_fold_share": float("nan"),
        }
    direction = 1.0 if mean_ic >= 0 else -1.0
    signed = [value * direction for value in fold_means]
    positive = sum(value > 0 for value in signed)
    abs_mass = np.abs(fold_means)
    max_fold_share = float(abs_mass.max() / abs_mass.sum()) if abs_mass.sum() > 0 else 1.0
    return {
        "n_folds": len(fold_means),
        "sign_consistency": positive / len(fold_means),
        "worst_fold_ic": float(min(fold_means)),
        "best_fold_ic": float(max(fold_means)),
        "fold_ics": fold_means,
        "max_fold_share": max_fold_share,
    }


def summarize_series(name: str, series: pl.DataFrame, role: str) -> dict | None:
    if series.is_empty() or len(series) < MIN_SESSIONS:
        return None
    hac = compute_ic_hac_stats(series, ic_col="ic", label_horizon=LABEL_HORIZON)
    mean_ic = float(hac["mean_ic"])
    stability = fold_sign_consistency(series, mean_ic)
    return {
        "series_id": name,
        "role": role,
        "n_sessions": int(len(series)),
        "mean_ic": mean_ic,
        "hac_pvalue": float(hac["p_value"]),
        "hac_tstat": float(hac["t_stat"]),
        "sign_consistency": float(stability["sign_consistency"]),
        "n_folds": int(stability["n_folds"]),
        "worst_fold_ic": float(stability["worst_fold_ic"]),
        "best_fold_ic": float(stability["best_fold_ic"]),
        "max_fold_share": float(stability["max_fold_share"]),
        "clears_ic_floor": bool(abs(mean_ic) >= IC_THRESHOLD),
        "clears_stability": bool(stability["sign_consistency"] >= STABILITY_THRESHOLD),
        "path_concentrated": bool(stability["max_fold_share"] >= 0.50),
    }


val_frames: list[pl.DataFrame] = []
for split in splits:
    fold = int(split["fold"])
    train_start = pd.Timestamp(split["train_start"]).date()
    train_end = pd.Timestamp(split["train_end"]).date()
    val_start = pd.Timestamp(split["val_start"]).date()
    val_end = pd.Timestamp(split["val_end"]).date()

    train = panel_all.filter(pl.col(DATE_COL).is_between(train_start, train_end, closed="both"))
    val = panel_all.filter(pl.col(DATE_COL).is_between(val_start, val_end, closed="both"))
    if val.is_empty():
        raise ValueError(f"Fold {fold}: empty validation window {val_start}..{val_end}")

    train_vol = train[STATE_COL].drop_nulls().to_numpy()
    if len(train_vol) < 100:
        raise ValueError(f"Fold {fold}: insufficient train vol observations ({len(train_vol)})")
    q33, q67 = np.quantile(train_vol, [1 / 3, 2 / 3])

    val = val.with_columns(
        [
            pl.lit(fold, dtype=pl.Int64).alias("fold"),
            pl.lit(float(q33)).alias("vol_q33_train"),
            pl.lit(float(q67)).alias("vol_q67_train"),
            pl.when(pl.col(STATE_COL).is_null())
            .then(None)
            .when(pl.col(STATE_COL) <= q33)
            .then(pl.lit(0, dtype=pl.Int8))
            .when(pl.col(STATE_COL) <= q67)
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(2, dtype=pl.Int8))
            .alias("vol_tercile"),
        ]
    ).with_columns((pl.col("vol_tercile").cast(pl.Float64) - 1.0).alias("vol_state"))
    for signal in SIGNALS:
        val = val.with_columns(
            (pl.col(signal) * pl.col("vol_state")).alias(f"{signal}__x_vol_state")
        )
    val_frames.append(val)

eval_panel = pl.concat(val_frames).sort([DATE_COL, "symbol"])
if eval_panel[DATE_COL].max() >= HOLDOUT_START:
    raise AssertionError("VS02 evaluation panel reaches the sealed holdout")
if eval_panel.select(JOIN_COLS).n_unique() != len(eval_panel):
    raise ValueError("VS02 panel has duplicate timestamp-symbol rows")

print(
    f"VS02 panel: {len(eval_panel):,} rows, "
    f"{eval_panel[DATE_COL].n_unique():,} sessions, "
    f"{eval_panel['fold'].n_unique()} folds, "
    f"{eval_panel[DATE_COL].min()}..{eval_panel[DATE_COL].max()} "
    f"(holdout from {HOLDOUT_START} sealed)"
)

TERCILE_NAMES = {0: "low", 1: "mid", 2: "high"}
series_frames: list[pl.DataFrame] = []
summaries: list[dict] = []

for signal in SIGNALS:
    base = daily_spearman_ic(
        eval_panel, signal, LABEL_COL, min_periods=MIN_PERIODS_FULL
    ).with_columns(
        [
            pl.lit(f"{signal}__unconditional").alias("series_id"),
            pl.lit("unconditional").alias("role"),
            pl.lit(signal).alias("signal"),
            pl.lit(None, dtype=pl.Utf8).alias("vol_regime"),
        ]
    )
    series_frames.append(base)
    summary = summarize_series(f"{signal}__unconditional", base, "unconditional")
    if summary:
        summary["signal"] = signal
        summary["vol_regime"] = None
        summaries.append(summary)

    gated = eval_panel.filter(pl.col("vol_tercile").is_not_null())
    for tercile, regime in TERCILE_NAMES.items():
        subset = gated.filter(pl.col("vol_tercile") == tercile)
        series_id = f"{signal}__vol_{regime}"
        series = daily_spearman_ic(
            subset,
            signal,
            LABEL_COL,
            min_periods=MIN_PERIODS_TERCILE,
            extra_keys=["vol_tercile"],
        ).with_columns(
            [
                pl.lit(series_id).alias("series_id"),
                pl.lit("conditional_tercile").alias("role"),
                pl.lit(signal).alias("signal"),
                pl.lit(regime).alias("vol_regime"),
            ]
        )
        if "vol_tercile" in series.columns:
            series = series.drop("vol_tercile")
        series_frames.append(series)
        summary = summarize_series(series_id, series, "conditional_tercile")
        if summary:
            summary["signal"] = signal
            summary["vol_regime"] = regime
            summaries.append(summary)

    interaction = f"{signal}__x_vol_state"
    inter = daily_spearman_ic(
        eval_panel, interaction, LABEL_COL, min_periods=MIN_PERIODS_FULL
    ).with_columns(
        [
            pl.lit(f"{signal}__interaction").alias("series_id"),
            pl.lit("interaction").alias("role"),
            pl.lit(signal).alias("signal"),
            pl.lit("interaction").alias("vol_regime"),
        ]
    )
    series_frames.append(inter)
    summary = summarize_series(f"{signal}__interaction", inter, "interaction")
    if summary:
        summary["signal"] = signal
        summary["vol_regime"] = "interaction"
        summaries.append(summary)

ic_timeseries = pl.concat([frame for frame in series_frames if not frame.is_empty()])
summary_frame = pl.DataFrame(summaries).sort(["signal", "role", "vol_regime"])

ic_path = EVAL_DIR / "vs02_conditional_ic_timeseries.parquet"
summary_path = EVAL_DIR / "vs02_summary.parquet"
ic_timeseries.write_parquet(ic_path)
summary_frame.write_parquet(summary_path)

print(f"Wrote {ic_path.relative_to(CASE_DIR)} ({len(ic_timeseries):,} rows)")
print(f"Wrote {summary_path.relative_to(CASE_DIR)} ({len(summary_frame)} series)")
print()
with pl.Config(tbl_rows=50, tbl_width_chars=200):
    print(
        summary_frame.select(
            [
                "series_id",
                "n_sessions",
                "mean_ic",
                "hac_pvalue",
                "sign_consistency",
                "max_fold_share",
                "clears_ic_floor",
                "clears_stability",
                "path_concentrated",
            ]
        )
    )

primary = summary_frame.filter(pl.col("signal") == PRIMARY_SIGNAL)
baseline = primary.filter(pl.col("role") == "unconditional")
baseline_abs = abs(float(baseline["mean_ic"][0])) if len(baseline) else 0.0
candidates = primary.filter(pl.col("role").is_in(["conditional_tercile", "interaction"]))

hits = []
for row in candidates.iter_rows(named=True):
    material = abs(row["mean_ic"]) > baseline_abs + 1e-12 and abs(row["mean_ic"]) >= IC_THRESHOLD
    stable = row["sign_consistency"] >= STABILITY_THRESHOLD
    not_path = not row["path_concentrated"]
    if material and (stable or row["hac_pvalue"] < 0.05) and not_path:
        hits.append(row["series_id"])

print()
print("=== VS02 confirmatory decision (primary signal only) ===")
print(f"Unconditional |IC| baseline ({PRIMARY_SIGNAL}): {baseline_abs:.6f}")
if hits:
    print(f"H1 CONFIRM candidates: {hits}")
    print("Decision: provisional CONFIRM — record in Obsidian; holdout still sealed.")
else:
    print("H1 CONFIRM candidates: none")
    print(
        "Decision: NULL on H1 under pre-registered criteria "
        "(no material, stable, non-path-concentrated conditional/interaction lift)."
    )
    print("Next research unit (separate pre-registration): H2 cross-sectional ranks.")
