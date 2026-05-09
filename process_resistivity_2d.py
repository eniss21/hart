#!/usr/bin/env python3
"""
2D apparent-resistivity wall-detection workflow.

Implements the procedure described in
``2d_resistivity_wall_detection_processing_workflow.md`` verbatim, then
exposes a ``run_workflow`` entry point compatible with the existing Flask app.

The processing logic itself (log10 transform, P2-P98 clipping, 0.5 m grid,
3x3 median, sigma=2.5 m Gaussian background, robust MAD Z-score, Z>2.0
threshold, 3x3 morphology, minimum 12-cell components, PCA endpoint
extraction) is unchanged from the markdown specification. Parameters can
be overridden by the caller, but the defaults match the spec exactly.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from scipy.interpolate import griddata
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    gaussian_filter,
    label,
    median_filter,
)
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Parameters2D:
    """Parameter set for the 2D workflow.

    Defaults are taken from the markdown specification (Section 3).
    """

    grid_resolution_m: float = 0.5
    clip_low_pct: float = 2.0
    clip_high_pct: float = 98.0
    median_size_cells: int = 3
    gaussian_sigma_m: float = 2.5
    max_extrapolation_distance_m: float = 2.6
    robust_z_threshold: float = 2.0
    min_plan_cells: int = 12


REQUIRED_COLUMNS = ("Piketa", "Profili", "East", "North")


def _read_excel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("Input file must be .xls or .xlsx.")
    try:
        return pd.read_excel(path)
    except ImportError as exc:
        package = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
        raise ImportError(
            f"Reading {path.suffix} files requires the '{package}' package."
        ) from exc


def _detect_resistivity_column(df: pd.DataFrame) -> str:
    """The spec uses ``Rd``; accept a few common aliases as well."""

    candidates = ["Rd", "rho_ohm_m", "Rho (Ohm.m)", "Rho", "Resistivity"]
    columns = list(df.columns)
    lower_lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        match = lower_lookup.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(
        "Could not find an apparent-resistivity column "
        f"(looked for {candidates}). Available: {columns}"
    )


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input workbook is missing required column(s) for the 2D workflow: "
            f"{missing}. Expected columns: {list(REQUIRED_COLUMNS)} + Rd."
        )


def run_workflow(
    input_file: Path,
    output_folder: Path,
    params: Parameters2D | None = None,
) -> dict[str, Any]:
    """Execute the full 2D workflow and return paths to the produced files.

    The processing logic mirrors the markdown one-to-one; only the I/O layer
    is parameterised so the same routine can be called from the CLI or the
    Flask UI.
    """

    if params is None:
        params = Parameters2D()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    GRID_RESOLUTION_M = params.grid_resolution_m
    CLIP_LOW_PCT = params.clip_low_pct
    CLIP_HIGH_PCT = params.clip_high_pct
    MEDIAN_SIZE_CELLS = params.median_size_cells
    GAUSSIAN_SIGMA_M = params.gaussian_sigma_m
    MAX_EXTRAPOLATION_DISTANCE_M = params.max_extrapolation_distance_m
    ROBUST_Z_THRESHOLD = params.robust_z_threshold
    MIN_PLAN_CELLS = params.min_plan_cells
    MORPH_STRUCTURE = np.ones((3, 3), dtype=bool)

    # --- Load and clean -----------------------------------------------------
    raw = _read_excel(Path(input_file))
    raw.columns = [str(c).strip() for c in raw.columns]

    rho_col = _detect_resistivity_column(raw)
    if rho_col != "rho_ohm_m":
        raw = raw.rename(columns={rho_col: "rho_ohm_m"})

    _ensure_required_columns(raw)

    df = raw[["Piketa", "Profili", "East", "North", "rho_ohm_m"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(df.columns)).copy()
    df = df[df["rho_ohm_m"] > 0].copy()
    if df.empty:
        raise ValueError("No valid rows remain after removing non-positive or invalid values.")

    # Local survey coordinates for filtering; final map is exported in Pulkovo coordinates.
    x = df["Piketa"].to_numpy(float)
    y = df["Profili"].to_numpy(float)
    rho = df["rho_ohm_m"].to_numpy(float)
    logrho = np.log10(rho)
    p2, p98 = np.percentile(logrho, [CLIP_LOW_PCT, CLIP_HIGH_PCT])
    logclip = np.clip(logrho, p2, p98)

    xi = np.arange(np.floor(x.min()), np.ceil(x.max()) + GRID_RESOLUTION_M / 2, GRID_RESOLUTION_M)
    yi = np.arange(np.floor(y.min()), np.ceil(y.max()) + GRID_RESOLUTION_M / 2, GRID_RESOLUTION_M)
    X, Y = np.meshgrid(xi, yi)
    points = np.column_stack([x, y])
    lin = griddata(points, logclip, (X, Y), method="linear")
    near = griddata(points, logclip, (X, Y), method="nearest")
    G = np.where(np.isnan(lin), near, lin)
    # mask extrapolated cells
    tree = cKDTree(points)
    dist, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
    D = dist.reshape(X.shape)
    valid_mask = D <= MAX_EXTRAPOLATION_DISTANCE_M
    filled = np.where(valid_mask, G, near)

    # --- Processing chain ---------------------------------------------------
    median_log = median_filter(filled, size=MEDIAN_SIZE_CELLS, mode="nearest")
    sigma_cells = GAUSSIAN_SIGMA_M / GRID_RESOLUTION_M
    background_log = gaussian_filter(median_log, sigma=sigma_cells, mode="nearest")
    residual_log = median_log - background_log
    valid_residual = residual_log[valid_mask]
    residual_median = np.nanmedian(valid_residual)
    mad = median_abs_deviation(valid_residual, scale="normal", nan_policy="omit")
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(valid_residual)
    robust_z = (residual_log - residual_median) / mad
    positive_mask_raw = (robust_z > ROBUST_Z_THRESHOLD) & valid_mask
    positive_mask_clean = binary_opening(
        binary_closing(positive_mask_raw, structure=MORPH_STRUCTURE),
        structure=MORPH_STRUCTURE,
    )
    lbl, n_lbl = label(positive_mask_clean)

    # Affine transformation from local survey axes to Pulkovo coordinates.
    A = np.column_stack([np.ones(len(df)), x, y])
    coefE = np.linalg.lstsq(A, df["East"].to_numpy(float), rcond=None)[0]
    coefN = np.linalg.lstsq(A, df["North"].to_numpy(float), rcond=None)[0]

    def transform(xv, yv):
        return (
            coefE[0] + coefE[1] * xv + coefE[2] * yv,
            coefN[0] + coefN[1] * xv + coefN[2] * yv,
        )

    E, N = transform(X, Y)

    # --- Component extraction and PCA-based endpoint estimation ------------
    components = []
    kept_label_ids = []
    for cid in range(1, n_lbl + 1):
        inds = np.argwhere(lbl == cid)
        ncells = len(inds)
        if ncells < MIN_PLAN_CELLS:
            continue
        yy = Y[inds[:, 0], inds[:, 1]]
        xx = X[inds[:, 0], inds[:, 1]]
        ee, nn = transform(xx, yy)
        coords = np.column_stack([ee, nn])
        pca = PCA(n_components=2).fit(coords)
        proj = pca.transform(coords)[:, 0]
        ep1 = coords[np.argmin(proj)]
        ep2 = coords[np.argmax(proj)]
        length_m = float(np.linalg.norm(ep2 - ep1))
        edge = bool(
            (np.nanmin(xx) <= x.min() + MAX_EXTRAPOLATION_DISTANCE_M)
            or (np.nanmax(xx) >= x.max() - MAX_EXTRAPOLATION_DISTANCE_M)
            or (np.nanmin(yy) <= y.min() + MAX_EXTRAPOLATION_DISTANCE_M)
            or (np.nanmax(yy) >= y.max() - MAX_EXTRAPOLATION_DISTANCE_M)
        )
        zvals = robust_z[lbl == cid]
        rvals = residual_log[lbl == cid]
        components.append(
            {
                "component_id": cid,
                "cells": int(ncells),
                "area_m2": ncells * GRID_RESOLUTION_M ** 2,
                "start_E": float(ep1[0]),
                "start_N": float(ep1[1]),
                "end_E": float(ep2[0]),
                "end_N": float(ep2[1]),
                "length_m": length_m,
                "local_x_min": float(xx.min()),
                "local_x_max": float(xx.max()),
                "local_y_min": float(yy.min()),
                "local_y_max": float(yy.max()),
                "z_max": float(np.nanmax(zvals)),
                "z_mean": float(np.nanmean(zvals)),
                "residual_log10_max": float(np.nanmax(rvals)),
                "resistivity_ratio_max": float(10 ** np.nanmax(rvals)),
                "edge_caution": edge,
                "interpretation": (
                    "edge-related resistive candidate"
                    if edge
                    else "resistive wall/foundation candidate"
                ),
            }
        )
        kept_label_ids.append(cid)
    components_df = (
        pd.DataFrame(components)
        .sort_values(["z_max", "area_m2"], ascending=False)
        .reset_index(drop=True)
    )
    if not components_df.empty:
        components_df["wall_label"] = [f"Wall {i + 1}" for i in range(len(components_df))]
        # Confidence classification
        conf = []
        for _, row in components_df.iterrows():
            if row["edge_caution"]:
                c = "High*" if row["z_max"] >= 3.0 else "Medium*"
            elif row["z_max"] >= 3.0 and row["length_m"] >= 3.0:
                c = "High"
            else:
                c = "Medium"
            conf.append(c)
        components_df["confidence"] = conf
    else:
        components_df["wall_label"] = []
        components_df["confidence"] = []

    summary_csv = output_folder / "wall_trace_summary_same_3d_workflow.csv"
    components_df.to_csv(summary_csv, index=False)

    # --- Export processed grid table ---------------------------------------
    out_grid = pd.DataFrame(
        {
            "local_x_piketa_m": X.ravel(),
            "local_y_profili_m": Y.ravel(),
            "Easting_Pulkovo1942_Zone4_m": E.ravel(),
            "Northing_Pulkovo1942_Zone4_m": N.ravel(),
            "nearest_data_distance_m": D.ravel(),
            "inside_extrapolation_mask": valid_mask.ravel(),
            "log10_rho_interpolated_clipped": G.ravel(),
            "median_filtered_log10_rho": median_log.ravel(),
            "background_log10_rho": background_log.ravel(),
            "residual_log10_rho": residual_log.ravel(),
            "robust_z": robust_z.ravel(),
            "positive_mask_raw_z_gt_2": positive_mask_raw.ravel(),
            "positive_mask_clean": positive_mask_clean.ravel(),
            "component_id": lbl.ravel(),
        }
    )
    grid_csv = output_folder / "processed_grid_same_3d_workflow.csv"
    out_grid.to_csv(grid_csv, index=False)

    # --- Helper functions for plots ----------------------------------------
    def setup_axis(ax):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which="major", color="0.15", alpha=0.18, linewidth=0.5)
        ax.grid(which="minor", color="0.15", alpha=0.08, linewidth=0.35)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")

    # Prepare masked arrays for plotting
    plot_log = np.where(valid_mask, G, np.nan)
    plot_median = np.where(valid_mask, median_log, np.nan)
    plot_resid = np.where(valid_mask, residual_log, np.nan)
    plot_z = np.where(valid_mask, robust_z, np.nan)

    # --- Workflow panels ---------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 12), dpi=240, constrained_layout=True)
    panel_data = [
        ("A. Original apparent resistivity, log10 clipped P2-P98", plot_log, "viridis", None, "log10 apparent resistivity"),
        ("B. 3x3 median-filtered map", plot_median, "viridis", None, "log10 apparent resistivity"),
        (
            "C. Residual anomaly map",
            plot_resid,
            "RdYlBu_r",
            TwoSlopeNorm(
                vcenter=0,
                vmin=np.nanpercentile(plot_resid, 2),
                vmax=np.nanpercentile(plot_resid, 98),
            ),
            "Residual log10 apparent resistivity",
        ),
        (
            "D. Robust Z-score anomaly map",
            plot_z,
            "RdYlBu_r",
            TwoSlopeNorm(
                vcenter=0, vmin=-3.0, vmax=max(3.0, np.nanpercentile(plot_z, 99))
            ),
            "Robust Z-score",
        ),
    ]
    for ax, (title, data, cmap, norm, cblabel) in zip(axs.ravel(), panel_data):
        im = ax.pcolormesh(E, N, data, shading="auto", cmap=cmap, norm=norm)
        setup_axis(ax)
        ax.set_title(title, fontsize=11.5, weight="bold")
        if "Residual" in title or "Z-score" in title:
            cs = ax.contour(
                E,
                N,
                plot_z,
                levels=[1.5, 2.0],
                colors="k",
                linewidths=[0.8, 1.0],
                linestyles=["--", "-"],
            )
            ax.clabel(cs, fmt={1.5: "Z=1.5", 2.0: "Z=2.0"}, fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cblabel, fontsize=8.5)
    fig.suptitle(
        "Ancient Wall Detection Workflow - Same Procedure as 3D Case\nPulkovo 1942 / Albania Zone 4",
        fontsize=15,
        weight="bold",
    )
    workflow_png = output_folder / "01_workflow_maps_same_3d_procedure.png"
    fig.savefig(workflow_png, bbox_inches="tight")
    plt.close(fig)

    # --- Final interpreted map with top candidates labelled ----------------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=260)
    norm = TwoSlopeNorm(
        vcenter=0,
        vmin=np.nanpercentile(plot_resid, 2),
        vmax=np.nanpercentile(plot_resid, 98),
    )
    im = ax.pcolormesh(E, N, plot_resid, shading="auto", cmap="RdYlBu_r", norm=norm)
    setup_axis(ax)
    cs = ax.contour(
        E,
        N,
        plot_z,
        levels=[1.5, 2.0],
        colors="black",
        linewidths=[1.0, 1.25],
        linestyles=["--", "-"],
    )
    ax.clabel(cs, fmt={1.5: "Z>=1.5", 2.0: "Z>=2.0"}, fontsize=8)
    # mask contours
    ax.contour(
        E,
        N,
        positive_mask_clean.astype(float),
        levels=[0.5],
        colors="magenta",
        linewidths=0.9,
    )
    # Label all components lightly; strongest 6 with wall labels
    for idx, row in components_df.iterrows():
        xs = [row["start_E"], row["end_E"]]
        ys = [row["start_N"], row["end_N"]]
        color = (
            "deepskyblue"
            if row["edge_caution"]
            else ("cyan" if row["confidence"] == "High" else "lime")
        )
        lw = 3.0 if idx < 6 else 1.5
        alpha = 0.95 if idx < 6 else 0.65
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round")
        mx, my = np.mean(xs), np.mean(ys)
        if idx < 6:
            labeltxt = f"{row['wall_label']} ({row['confidence']})"
            ax.text(
                mx + 0.6,
                my + 0.6,
                labeltxt,
                color=color,
                fontsize=8.5,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec=color, alpha=0.72),
            )
    # scale bar and north arrow
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    bar_len = 10
    x0 = xmin + 3
    y0 = ymin + 2
    ax.plot([x0, x0 + bar_len], [y0, y0], color="k", lw=4)
    ax.plot(
        [x0 + bar_len / 2, x0 + bar_len / 2],
        [y0 - 0.25, y0 + 0.25],
        color="white",
        lw=4,
        solid_capstyle="butt",
    )
    ax.text(x0 + bar_len / 2, y0 + 1.0, "10 m", ha="center", fontsize=8.5)
    ax.annotate(
        "N",
        xy=(xmax - 6.5, ymin + 12),
        xytext=(xmax - 6.5, ymin + 4),
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k"),
        fontsize=11,
        weight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(
        "Residual log10 apparent resistivity\npositive = resistive relative to background",
        fontsize=9,
    )
    ax.set_title(
        "Final Wall-Candidate Map - 2D Adaptation of the 3D Workflow\nlog10 transform, P2-P98 clipping, sigma=2.5 m background, robust Z>2.0",
        fontsize=12.5,
        weight="bold",
    )
    final_png = output_folder / "02_final_wall_candidates_same_3d_procedure.png"
    fig.savefig(final_png, bbox_inches="tight")
    plt.close(fig)

    # --- Markdown report ---------------------------------------------------
    report = f"""# 2D Apparent Resistivity Wall Detection - Same Processing Logic as 3D Case

Input file: `{Path(input_file).name}`
Coordinate system: **Pulkovo 1942 / Albania Zone 4**
Rows used after removing non-positive resistivity: **{len(df)}**

## Processing parameters

| Parameter | Value |
|---|---:|
| Working variable | log10(apparent resistivity) |
| Percentile clipping | P2 = {p2:.4f}, P98 = {p98:.4f} |
| Grid resolution | {GRID_RESOLUTION_M:.2f} m |
| Maximum extrapolation distance | {MAX_EXTRAPOLATION_DISTANCE_M:.2f} m |
| Median filter | {MEDIAN_SIZE_CELLS}x{MEDIAN_SIZE_CELLS} cells |
| Gaussian background sigma | {GAUSSIAN_SIGMA_M:.2f} m = {sigma_cells:.1f} grid cells |
| Residual | median-filtered log10(rho) - Gaussian background |
| Robust normalization | (residual - median) / MAD |
| Positive anomaly threshold | robust Z > {ROBUST_Z_THRESHOLD:.1f} |
| Morphology | 3x3 closing followed by 3x3 opening |
| Minimum component size | {MIN_PLAN_CELLS} plan cells = {MIN_PLAN_CELLS * GRID_RESOLUTION_M ** 2:.2f} m^2 |

## Results

- Raw positive cells before morphology: **{int(positive_mask_raw.sum())}**
- Clean positive cells after morphology: **{int(positive_mask_clean.sum())}**
- Kept wall/anomaly components: **{len(components_df)}**

The CSV file `wall_trace_summary_same_3d_workflow.csv` contains all extracted components with endpoint coordinates, length, area, maximum robust Z, and edge-caution flag.

## Interpretation rule

- **High**: strong positive residual, robust Z generally >= 3, and coherent linear/compact geometry.
- **High\\***: same as High, but affected by edge uncertainty.
- **Medium**: coherent anomaly but weaker, shorter, or less persistent in plan view.

Because this is a single 2D plan map, the original 3D criterion of persistence across multiple depth slices cannot be applied. It is replaced here by 2D plan continuity, morphological cleaning, and minimum plan-area filtering.
"""
    report_path = output_folder / "processing_report_same_3d_workflow.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # --- Parameters dump (handy for reproducibility) -----------------------
    pd.DataFrame([asdict(params)]).to_csv(output_folder / "parameters_2d.csv", index=False)

    return {
        "output_folder": output_folder,
        "workflow_map": workflow_png,
        "final_map": final_png,
        "summary_csv": summary_csv,
        "grid_csv": grid_csv,
        "report": report_path,
        "wall_count": len(components_df),
        "csv": summary_csv,  # alias to match 3D workflow signature
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="2D apparent-resistivity ancient-wall detection workflow."
    )
    parser.add_argument("input_file", help="Input .xls/.xlsx workbook")
    parser.add_argument(
        "-o",
        "--output-folder",
        default="processed_2d_same_as_3d_workflow",
        help="Output folder (default: %(default)s)",
    )
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--clip-low", type=float, default=2.0)
    parser.add_argument("--clip-high", type=float, default=98.0)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--gaussian-sigma-m", type=float, default=2.5)
    parser.add_argument("--max-extrapolation-distance", type=float, default=2.6)
    parser.add_argument("--robust-z-threshold", type=float, default=2.0)
    parser.add_argument("--min-plan-cells", type=int, default=12)
    args = parser.parse_args()

    params = Parameters2D(
        grid_resolution_m=args.grid_resolution,
        clip_low_pct=args.clip_low,
        clip_high_pct=args.clip_high,
        median_size_cells=args.median_size,
        gaussian_sigma_m=args.gaussian_sigma_m,
        max_extrapolation_distance_m=args.max_extrapolation_distance,
        robust_z_threshold=args.robust_z_threshold,
        min_plan_cells=args.min_plan_cells,
    )

    outputs = run_workflow(Path(args.input_file), Path(args.output_folder), params)
    print("DONE")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
