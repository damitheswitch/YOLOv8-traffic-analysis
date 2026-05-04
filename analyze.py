"""
Post-processing analysis for traffic pipeline results.

Run after the main pipeline finishes. Reads results.csv (pandas + matplotlib only)
and writes figures plus a text summary under output/. Traffic volume uses
adaptive time bins (10 s through 1 min) from the crossing timestamp span.

Usage:
  python analyze.py
  python analyze.py --csv path/to/results.csv --out-dir path/to/output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cardinal directions for OD matrix (rows = entry, columns = exit)
DIRECTION_ORDER = ("North", "South", "East", "West")
TIME_ANCHOR = pd.Timestamp("2000-01-01 00:00:00")

# Volume timeline figure (replaces legacy vehicles_per_minute.png name).
TRAFFIC_VOLUME_TIMELINE_FILENAME = "traffic_volume_timeline.png"
LEGACY_VOLUME_FILENAME = "vehicles_per_minute.png"

# Seconds from first to last crossing timestamp in the CSV; used only to pick bin width.
_MIN_SPAN_FOR_BIN_RULE = 1.0

# Adaptive histogram bins: [span in seconds) -> pandas offset string
# <40 -> 10s, <80 -> 15s, <120 -> 20s, <240 -> 30s, else 1min


@dataclass(frozen=True)
class VolumeBinConfig:
    """Time bucketing for crossing-count timeline and peak summary."""

    pandas_freq: str
    window_seconds: float
    human_short: str  # e.g. "15 s" for console / summary


def event_timestamp_span_seconds(df: pd.DataFrame) -> float:
    """Max(timestamp) - min(timestamp) from CSV; 0 if missing or single instant."""
    if df.empty or "timestamp" not in df.columns:
        return 0.0
    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
    if len(ts) < 2:
        return 0.0
    return float(ts.max() - ts.min())


def select_volume_bin_config(event_span_seconds: float) -> VolumeBinConfig:
    """
    Pick resample frequency from crossing-time span (seconds).

    Tiers: <40 -> 10s; <80 -> 15s; <120 -> 20s; <240 -> 30s; else 1min.
    """
    span = max(float(event_span_seconds), _MIN_SPAN_FOR_BIN_RULE)
    if span < 40:
        return VolumeBinConfig("10s", 10.0, "10 s")
    if span < 80:
        return VolumeBinConfig("15s", 15.0, "15 s")
    if span < 120:
        return VolumeBinConfig("20s", 20.0, "20 s")
    if span < 240:
        return VolumeBinConfig("30s", 30.0, "30 s")
    return VolumeBinConfig("1min", 60.0, "1 min")


def crossing_counts_volume_timeline(
    df: pd.DataFrame,
    *,
    bin_cfg: VolumeBinConfig | None = None,
) -> tuple[pd.Series, VolumeBinConfig]:
    """
    Resample completed-crossing rows to uniform time bins.

    Requires column ``event_time``. Returns (counts per bin left edge, config used).
    """
    if df.empty or "event_time" not in df.columns:
        raise ValueError("crossing_counts_volume_timeline requires non-empty df with event_time")
    span = event_timestamp_span_seconds(df)
    cfg = bin_cfg or select_volume_bin_config(span)
    counts = (
        df.set_index("event_time")
        .sort_index()
        .resample(cfg.pandas_freq, label="left", closed="left")
        .size()
        .rename("vehicles")
    )
    return counts, cfg


def _elapsed_from_anchor(index: pd.DatetimeIndex) -> np.ndarray:
    return (index - TIME_ANCHOR).total_seconds().astype(np.float64)


def _format_peak_interval(start_sec: float, end_sec: float) -> str:
    """Human-readable interval for summary (prefer minutes when timeline is long)."""
    if end_sec <= 300:
        return f"{start_sec:.0f}-{end_sec:.0f} s"
    return f"{start_sec / 60:.1f}-{end_sec / 60:.1f} min"


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df = df.dropna(how="all")
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing 'timestamp' column: {csv_path}")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    # Datetime for time-based resampling (timestamp = seconds from video start)
    df["event_time"] = TIME_ANCHOR + pd.to_timedelta(df["timestamp"], unit="s")
    if "estimated_speed" in df.columns:
        df["estimated_speed"] = pd.to_numeric(df["estimated_speed"], errors="coerce")
    return df


def plot_speed_histogram(df: pd.DataFrame, out_path: Path) -> None:
    speeds = df["estimated_speed"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(speeds, bins="auto", color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Number of vehicles")
    ax.set_title("Speed distribution of vehicles passing through the intersection")
    if len(speeds) > 0:
        m, med, s = float(speeds.mean()), float(speeds.median()), float(speeds.std(ddof=0))
        ax.axvline(m, color="darkred", linestyle="--", linewidth=1.5, label=f"Mean: {m:.1f} km/h")
        ax.axvline(med, color="darkgreen", linestyle=":", linewidth=1.5, label=f"Median: {med:.1f} km/h")
        ax.legend(loc="upper right")
        ax.text(
            0.98,
            0.95,
            f"σ = {s:.1f} km/h",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_traffic_volume_timeline(df: pd.DataFrame, out_path: Path) -> VolumeBinConfig | None:
    """
    Bar chart of crossing counts per adaptive time window.

    Returns the bin config used (for logging), or None if there was nothing to plot.
    """
    if df.empty or "event_time" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return None

    counts, cfg = crossing_counts_volume_timeline(df)
    elapsed_sec = _elapsed_from_anchor(counts.index)
    t_end = float(elapsed_sec[-1]) + cfg.window_seconds if len(elapsed_sec) else 0.0
    use_minutes_axis = t_end > 240.0
    scale = 60.0 if use_minutes_axis else 1.0
    x_left = elapsed_sec / scale
    bar_w = (cfg.window_seconds * 0.9) / scale

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x_left,
        counts.values,
        width=bar_w,
        align="edge",
        color="teal",
        edgecolor="darkslategray",
    )
    unit = "minutes" if use_minutes_axis else "seconds"
    ax.set_xlabel(f"Time from start of observation ({unit})")
    ax.set_ylabel(f"Vehicles completing a crossing (per {cfg.human_short} window)")
    ax.set_title("Traffic volume over the observation period")

    n = len(x_left)
    step = max(1, n // 24) if n > 24 else 1
    tick_idx = np.arange(0, n, step)
    ax.set_xticks(x_left[tick_idx])
    if use_minutes_axis:
        labels = [f"{x_left[i]:.1f}" for i in tick_idx]
    else:
        labels = [f"{int(round(x_left[i] * scale))}" for i in tick_idx]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cfg


def plot_entry_exit_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    for col in ("entry_side", "exit_side"):
        if col not in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.text(0.5, 0.5, f"Missing {col}", ha="center", va="center", transform=ax.transAxes)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return

    sub = df.dropna(subset=["entry_side", "exit_side"])
    ct = pd.crosstab(sub["entry_side"], sub["exit_side"])
    ct = ct.reindex(index=list(DIRECTION_ORDER), fill_value=0)
    ct = ct.reindex(columns=list(DIRECTION_ORDER), fill_value=0)
    mat = ct.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vehicle count")

    ax.set_xticks(np.arange(len(DIRECTION_ORDER)))
    ax.set_yticks(np.arange(len(DIRECTION_ORDER)))
    ax.set_xticklabels(DIRECTION_ORDER)
    ax.set_yticklabels(DIRECTION_ORDER)
    ax.set_xlabel("Exit direction")
    ax.set_ylabel("Entry direction")
    ax.set_title("Origin-destination matrix (entry to exit)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _detect_class_column(df: pd.DataFrame) -> str | None:
    for name in ("vehicle_class", "class_name", "class", "yolo_class"):
        if name in df.columns:
            return name
    return None


def plot_speed_by_class(df: pd.DataFrame, out_path: Path) -> bool:
    col = _detect_class_column(df)
    if col is None or "estimated_speed" not in df.columns:
        return False

    sub = df[[col, "estimated_speed"]].dropna()
    if sub.empty:
        return False

    grouped = sub.groupby(col, observed=True)["estimated_speed"]
    means = grouped.mean().sort_values(ascending=False)
    stds = grouped.std(ddof=0).reindex(means.index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(means))
    ax.bar(x, means.values, yerr=stds.values, capsize=4, color="slateblue", edgecolor="navy", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in means.index], rotation=20, ha="right")
    ax.set_ylabel("Estimated speed (km/h)")
    ax.set_xlabel("Vehicle class")
    ax.set_title("Average speed by vehicle type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def compute_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Traffic analysis - results summary")
    lines.append("=" * 50)

    n = len(df)
    lines.append(f"Total vehicles (completed crossings): {n}")

    if n == 0:
        lines.append("(No rows to summarize.)")
        return "\n".join(lines) + "\n"

    if "estimated_speed" in df.columns:
        sp = df["estimated_speed"].dropna()
        if len(sp) > 0:
            lines.append(f"Average speed: {sp.mean():.2f} km/h")
            lines.append(f"Median speed: {sp.median():.2f} km/h")
            lines.append(f"Speed std. dev.: {sp.std(ddof=0):.2f} km/h")

    if "entry_side" in df.columns:
        ec = df["entry_side"].dropna().astype(str)
        if not ec.empty:
            busiest = ec.value_counts().idxmax()
            lines.append(f"Busiest entry direction: {busiest}")

    if "entry_side" in df.columns and "exit_side" in df.columns:
        route = df["entry_side"].astype(str) + " -> " + df["exit_side"].astype(str)
        route_counts = route.value_counts()
        if not route_counts.empty:
            top = route_counts.index[0]
            lines.append(
                f"Most common route (entry to exit): {top} ({int(route_counts.iloc[0])} vehicles)"
            )

    if "event_time" in df.columns:
        span = event_timestamp_span_seconds(df)
        counts, cfg = crossing_counts_volume_timeline(df)
        lines.append(
            f"Crossing timestamp span: {span:.1f} s; volume timeline uses {cfg.human_short} bins."
        )
        if len(counts) > 0 and counts.sum() > 0:
            peak_left = counts.idxmax()
            start_sec = float((peak_left - TIME_ANCHOR).total_seconds())
            end_sec = start_sec + cfg.window_seconds
            interval = _format_peak_interval(start_sec, end_sec)
            lines.append(
                f"Peak traffic ({cfg.human_short} window): {interval} from start "
                f"({int(counts.max())} vehicles)"
            )

    lines.append("=" * 50)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze results.csv and export thesis-ready figures.")
    parser.add_argument("--csv", type=Path, default=Path("results.csv"), help="Input CSV path")
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Folder for PNGs and summary.txt")
    args = parser.parse_args()

    csv_path: Path = args.csv
    out_dir: Path = args.out_dir

    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path.resolve()}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    if df.empty:
        print(f"Warning: no data rows in {csv_path}")
        summary = compute_summary(df)
        print(summary)
        (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
        return

    # Figures (one function per figure; easy to comment out if needed)
    plot_speed_histogram(df, out_dir / "speed_distribution.png")
    vol_cfg = plot_traffic_volume_timeline(df, out_dir / TRAFFIC_VOLUME_TIMELINE_FILENAME)
    if vol_cfg is not None:
        span = event_timestamp_span_seconds(df)
        print(f"Volume timeline: {vol_cfg.human_short} bins (crossing timestamp span {span:.1f} s).")

    legacy = out_dir / LEGACY_VOLUME_FILENAME
    if legacy.exists():
        legacy.unlink()

    plot_entry_exit_heatmap(df, out_dir / "entry_exit_flow_heatmap.png")

    wrote_class = plot_speed_by_class(df, out_dir / "speed_by_class.png")
    if not wrote_class:
        p = out_dir / "speed_by_class.png"
        if p.exists():
            p.unlink()

    summary = compute_summary(df)
    print(summary)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"Figures and summary written to: {out_dir.resolve()}")
    if not wrote_class:
        print("Note: no vehicle class column in CSV; skipped speed-by-class figure.")


if __name__ == "__main__":
    main()
