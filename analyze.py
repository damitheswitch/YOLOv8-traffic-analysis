"""
Post-processing analysis for traffic pipeline results.

Run after the main pipeline finishes. Reads results.csv (pandas + matplotlib only)
and writes figures plus a text summary under output/.

Usage:
  python analyze.py
  python analyze.py --csv path/to/results.csv --out-dir path/to/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cardinal directions for OD matrix (rows = entry, columns = exit)
DIRECTION_ORDER = ("North", "South", "East", "West")
TIME_ANCHOR = pd.Timestamp("2000-01-01 00:00:00")


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


def plot_vehicles_per_minute(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "event_time" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    per_min = df.set_index("event_time").resample("1min").size()
    per_min = per_min.rename("vehicles")

    fig, ax = plt.subplots(figsize=(9, 5))
    elapsed_min = (per_min.index - TIME_ANCHOR).total_seconds() / 60.0
    x = np.asarray(elapsed_min, dtype=float)
    ax.bar(x, per_min.values, width=0.9, align="center", color="teal", edgecolor="darkslategray")
    ax.set_xlabel("Time from start of observation (minutes)")
    ax.set_ylabel("Vehicles completing a crossing (per 1-minute window)")
    ax.set_title("Traffic volume over the observation period")
    step = max(1, len(x) // 24) if len(x) > 24 else 1
    tick_idx = np.arange(0, len(x), step)
    ax.set_xticks(x[tick_idx])
    ax.set_xticklabels([f"{int(x[i])}" for i in tick_idx], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
        route = (
            df["entry_side"].astype(str)
            + " -> "
            + df["exit_side"].astype(str)
        )
        route_counts = route.value_counts()
        if not route_counts.empty:
            top = route_counts.index[0]
            lines.append(
                f"Most common route (entry to exit): {top} ({int(route_counts.iloc[0])} vehicles)"
            )

    if "event_time" in df.columns:
        per_min = df.set_index("event_time").resample("1min").size()
        if len(per_min) > 0 and per_min.sum() > 0:
            peak_idx = per_min.idxmax()
            peak_min = (peak_idx - TIME_ANCHOR).total_seconds() / 60.0
            lines.append(
                f"Peak traffic minute: {int(peak_min)}-{int(peak_min) + 1} min from start "
                f"({int(per_min.max())} vehicles)"
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
    plot_vehicles_per_minute(df, out_dir / "vehicles_per_minute.png")
    plot_entry_exit_heatmap(df, out_dir / "entry_exit_flow_heatmap.png")

    wrote_class = plot_speed_by_class(df, out_dir / "speed_by_class.png")
    if not wrote_class:
        # Remove stale file if present from a previous run with class data
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
