from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.config import SignalColumnConfig, StorageConfig
from src.io_utils import collect_parquet_files
from src.logging_utils import configure_logging


logger = logging.getLogger(__name__)
SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_label(value: object) -> str:
    """Return a filesystem-friendly label for generated plot filenames."""
    label = SAFE_LABEL_RE.sub("_", str(value).strip())
    return label.strip("_") or "unknown"


def _non_overwriting_path(path: Path) -> Path:
    """Return path, or a numbered variant, without replacing an existing file."""
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Could not find an unused output filename near: {path}")


def default_output_path(
    *,
    data_dir: Path,
    input_path: Path,
    value_col: str,
) -> Path:
    """Build the default non-overwriting HTML path for a raw signal plot."""
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"raw_signal_{_safe_label(value_col)}_"
        f"{_safe_label(input_path.name)}_{run_label}.html"
    )
    return _non_overwriting_path(data_dir / "validation_plots" / filename)


def load_signal_from_parquet(
    input_path: Path,
    *,
    timestamp_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Load one timestamp/value pair from a Parquet file or directory."""
    files = collect_parquet_files(input_path)
    logger.info("Found %s Parquet files under %s", len(files), input_path)

    parts: list[pd.DataFrame] = []
    for file in files:
        part = pd.read_parquet(file, columns=[timestamp_col, value_col])
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, value_col])
    return df.sort_values(timestamp_col).reset_index(drop=True)


def plot_signal(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    gap_threshold_minutes: float,
    max_points: int | None,
) -> None:
    """Write an interactive HTML telemetry plot and mark large timestamp gaps."""
    if df.empty:
        raise ValueError("No valid rows available to plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    work["delta_seconds"] = work[timestamp_col].diff().dt.total_seconds()
    gap_threshold_seconds = gap_threshold_minutes * 60
    gaps = work[work["delta_seconds"] > gap_threshold_seconds]

    plot_data = work
    if max_points is not None and max_points > 0 and len(work) > max_points:
        step = max(1, len(work) // max_points)
        plot_data = work.iloc[::step].copy()
        logger.info(
            "Downsampled plot from %s to %s points for browser performance.",
            f"{len(work):,}",
            f"{len(plot_data):,}",
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=plot_data[timestamp_col],
            y=plot_data[value_col],
            mode="lines+markers",
            name=value_col,
            line={"width": 1},
            marker={"size": 3},
            customdata=plot_data["delta_seconds"],
            hovertemplate=(
                "time=%{x}<br>"
                f"{value_col}=%{{y}}<br>"
                "delta_seconds=%{customdata}<extra></extra>"
            ),
        )
    )

    shapes = []
    for _, row in gaps.iterrows():
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": row[timestamp_col],
                "x1": row[timestamp_col],
                "y0": 0,
                "y1": 1,
                "line": {"color": "red", "width": 1, "dash": "dot"},
                "opacity": 0.35,
            }
        )

    summary = (
        f"rows={len(work):,} | "
        f"first={work[timestamp_col].min()} | "
        f"last={work[timestamp_col].max()} | "
        f"gaps>{gap_threshold_minutes:g}min={len(gaps):,}"
    )

    fig.update_layout(
        title=f"{title}<br><sup>{summary}</sup>",
        xaxis_title="Time",
        yaxis_title=ylabel,
        hovermode="x unified",
        shapes=shapes,
        template="plotly_white",
    )
    fig.update_xaxes(rangeslider_visible=True)

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    logger.info("Wrote interactive plot to %s", output_path)
    if not gaps.empty:
        largest_gap = gaps["delta_seconds"].max()
        logger.info("Largest timestamp gap: %.0f seconds", largest_gap)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one raw Parquet telemetry signal over time.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw Parquet file or directory to plot.")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path.")
    parser.add_argument("--env-file", type=Path, default=None, help="Path to a .env file.")
    parser.add_argument("--value-col", default=None, help="Signal column to plot. Defaults to configured SOC column.")
    parser.add_argument("--timestamp-col", default=None, help="Timestamp column. Defaults to configured timestamp column.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--ylabel", default=None, help="Y-axis label.")
    parser.add_argument(
        "--gap-threshold-minutes",
        type=float,
        default=60.0,
        help="Mark timestamp gaps larger than this threshold.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Maximum plotted points for browser performance. Use 0 to plot all rows.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        # Load storage config so .env path behavior stays consistent with the rest
        # of the project, even though this script only needs signal columns.
        storage_config = StorageConfig.from_env(args.env_file)
        columns = SignalColumnConfig.from_env(args.env_file)
        timestamp_col = args.timestamp_col or columns.timestamp_col
        value_col = args.value_col or columns.soc_col
        output_path = (
            _non_overwriting_path(args.output)
            if args.output
            else default_output_path(
                data_dir=storage_config.data_dir,
                input_path=args.input,
                value_col=value_col,
            )
        )

        df = load_signal_from_parquet(
            args.input,
            timestamp_col=timestamp_col,
            value_col=value_col,
        )
        plot_signal(
            df,
            timestamp_col=timestamp_col,
            value_col=value_col,
            output_path=output_path,
            title=args.title or f"{value_col} over time",
            ylabel=args.ylabel or value_col,
            gap_threshold_minutes=args.gap_threshold_minutes,
            max_points=None if args.max_points == 0 else args.max_points,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Signal plotting failed.")
        else:
            logger.error("Signal plotting failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
