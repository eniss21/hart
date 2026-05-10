#!/usr/bin/env python3
"""
Process apparent-resistivity depth slices and extract possible wall traces.

The implementation follows resistivity_processing_wall_detection_workflow.md:
log10 transform, P2-P98 clipping, per-depth gridding, footprint masking,
median filtering, Gaussian background removal, robust MAD thresholding,
3D morphological cleaning, connected-component filtering, plan projection,
and report exports.
"""

from __future__ import annotations

import argparse
import math
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
from scipy.interpolate import griddata
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    gaussian_filter,
    label,
    median_filter,
)
from scipy.spatial import QhullError, cKDTree


@dataclass(frozen=True)
class ProcessingParameters:
    grid_resolution: float = 0.55
    clip_low: float = 2.0
    clip_high: float = 98.0
    median_size: int = 3
    gaussian_sigma_m: float = 2.5
    robust_z_threshold: float = 2.0
    max_extrapolation_distance: float = 2.6
    min_component_voxels: int = 15
    min_component_slices: int = 2
    min_plan_cells: int = 12


DEFAULT_INPUT_FILE = "Lezha_3D.xls"
DEFAULT_OUTPUT_FOLDER = "resistivity_output"


def _normalize_name(name: Any) -> str:
    return str(name).strip()


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """
    Automatically detect coordinate, depth, and resistivity columns.

    The exact column names from both documented database examples are checked
    first, followed by case-insensitive variants.
    """

    if len(df.columns) == 7:
        original_columns = list(df.columns)
        df.columns = [
            "Profili",
            "Piketa",
            "Depth",
            "Rd",
            "SP",
            "N",
            "E",
        ]
        print(
            "Using positional 3D schema: "
            "Profili, Piketa/PK, Depth, Rd, SP, N, E. "
            f"Original headers were: {original_columns}"
        )
        return "E", "N", "Depth", "Rd"

    df.columns = [_normalize_name(c) for c in df.columns]
    columns = list(df.columns)
    lower_lookup = {c.lower(): c for c in columns}

    def first_present(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        for candidate in candidates:
            found = lower_lookup.get(candidate.lower())
            if found is not None:
                return found
        return None

    possible_x = ["X", "E", "East", "Easting", "Local_X"]
    possible_y = ["Y", "N", "North", "Northing", "Local_Y"]
    possible_depth = ["Thellësia", "Thellesia", "Depth", "Z", "Depth_m"]
    possible_rho = [
        "Rho (Ohm.m)",
        "Rd",
        "Resistivity",
        "Apparent_Resistivity",
        "Rho",
    ]

    x_col = first_present(possible_x)
    y_col = first_present(possible_y)
    depth_col = first_present(possible_depth)
    rho_col = first_present(possible_rho)

    missing = []
    if x_col is None:
        missing.append("X/E coordinate")
    if y_col is None:
        missing.append("Y/N coordinate")
    if depth_col is None:
        missing.append("depth")
    if rho_col is None:
        missing.append("resistivity")
    if missing:
        raise ValueError(
            "Could not detect required column(s): "
            + ", ".join(missing)
            + f". Available columns: {columns}"
        )

    return x_col, y_col, depth_col, rho_col


def _read_excel(input_file: Path) -> pd.DataFrame:
    if input_file.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("Input file must be .xls or .xlsx.")
    try:
        return pd.read_excel(input_file)
    except ImportError as exc:
        if input_file.suffix.lower() == ".xls":
            package = "xlrd"
        else:
            package = "openpyxl"
        raise ImportError(
            f"Reading {input_file.suffix} files requires the '{package}' package."
        ) from exc


def load_and_prepare_data(
    input_file: Path, params: ProcessingParameters
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read the database, remove invalid resistivity values, convert coordinates
    to local coordinates, compute log10 resistivity, and apply robust clipping.
    """

    df = _read_excel(input_file)
    df.columns = [_normalize_name(c) for c in df.columns]

    x_col, y_col, depth_col, rho_col = detect_columns(df)

    for col in [x_col, y_col, depth_col, rho_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[np.isfinite(df[rho_col])].copy()
    df = df[df[rho_col] > 0].copy()
    df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col]) & np.isfinite(df[depth_col])].copy()

    if df.empty:
        raise ValueError("No valid rows remain after removing non-positive or invalid values.")

    x0 = float(df[x_col].min())
    y0 = float(df[y_col].min())

    df["Local_X_m"] = df[x_col] - x0
    df["Local_Y_m"] = df[y_col] - y0

    df["Log10_Rho"] = np.log10(df[rho_col])

    p_low, p_high = np.percentile(df["Log10_Rho"], [params.clip_low, params.clip_high])

    df["Log10_Rho_Clipped"] = df["Log10_Rho"].clip(p_low, p_high)
    df["Clip_Flag"] = "kept"
    df.loc[df["Log10_Rho"] < p_low, "Clip_Flag"] = "low_clipped"
    df.loc[df["Log10_Rho"] > p_high, "Clip_Flag"] = "high_clipped"

    metadata = {
        "x_col": x_col,
        "y_col": y_col,
        "depth_col": depth_col,
        "rho_col": rho_col,
        "x0": x0,
        "y0": y0,
        "clip_low_value": float(p_low),
        "clip_high_value": float(p_high),
    }

    return df, metadata


def create_regular_grid(
    df: pd.DataFrame, params: ProcessingParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular 2D grid and valid-area mask based on nearest measured point.
    """

    x_min = float(df["Local_X_m"].min())
    x_max = float(df["Local_X_m"].max())
    y_min = float(df["Local_Y_m"].min())
    y_max = float(df["Local_Y_m"].max())
    res = params.grid_resolution

    x_grid = np.arange(
        math.floor(x_min / res) * res,
        math.ceil(x_max / res) * res + res,
        res,
    )
    y_grid = np.arange(
        math.floor(y_min / res) * res,
        math.ceil(y_max / res) * res + res,
        res,
    )

    xx, yy = np.meshgrid(x_grid, y_grid)

    real_points = df[["Local_X_m", "Local_Y_m"]].drop_duplicates().to_numpy()
    if len(real_points) == 0:
        raise ValueError("No coordinate pairs are available for grid creation.")

    tree = cKDTree(real_points)
    distance_to_nearest, _ = tree.query(np.c_[xx.ravel(), yy.ravel()], k=1)
    valid_mask = (
        distance_to_nearest.reshape(xx.shape) <= params.max_extrapolation_distance
    )

    return x_grid, y_grid, xx, yy, valid_mask


def _interpolate_slice(
    points: np.ndarray, values: np.ndarray, xx: np.ndarray, yy: np.ndarray
) -> np.ndarray:
    grid_nearest = griddata(points, values, (xx, yy), method="nearest")
    if len(np.unique(points, axis=0)) < 3:
        return grid_nearest

    try:
        grid_linear = griddata(points, values, (xx, yy), method="linear")
    except QhullError:
        return grid_nearest

    return np.where(np.isfinite(grid_linear), grid_linear, grid_nearest)


def process_depth_slices(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    valid_mask: np.ndarray,
    params: ProcessingParameters,
) -> dict[str, Any]:
    """
    Process each depth slice:
    - interpolate clipped log-resistivity;
    - apply median filter;
    - estimate Gaussian background;
    - calculate residual anomaly;
    - calculate robust Z-score;
    - threshold positive anomalies.
    """

    del x_grid, y_grid

    depth_col = metadata["depth_col"]
    depths = sorted(df[depth_col].dropna().unique())
    if not depths:
        raise ValueError("No depth slices were found.")

    filtered_slices = []
    background_slices = []
    residual_slices = []
    robust_z_slices = []
    binary_slices = []
    slice_summary = []

    gaussian_sigma_cells = params.gaussian_sigma_m / params.grid_resolution

    for depth in depths:
        slice_df = df[df[depth_col] == depth].copy()
        points = slice_df[["Local_X_m", "Local_Y_m"]].to_numpy()
        values = slice_df["Log10_Rho_Clipped"].to_numpy()

        grid_log_rho = _interpolate_slice(points, values, xx, yy)
        grid_log_rho = np.where(valid_mask, grid_log_rho, np.nan)

        fill_value = float(np.nanmedian(values))
        grid_filled = np.where(np.isfinite(grid_log_rho), grid_log_rho, fill_value)

        grid_median = median_filter(
            grid_filled, size=params.median_size, mode="nearest"
        )

        background = gaussian_filter(
            grid_median,
            sigma=gaussian_sigma_cells,
            mode="nearest",
            truncate=3.0,
        )

        residual = grid_median - background
        residual = np.where(valid_mask, residual, np.nan)
        residual_values = residual[np.isfinite(residual)]

        residual_median = float(np.nanmedian(residual_values))
        mad = float(np.nanmedian(np.abs(residual_values - residual_median)))
        if mad <= 1e-9:
            mad = float(np.nanstd(residual_values))
        if mad <= 1e-9:
            mad = 1.0

        robust_z = (residual - residual_median) / mad
        binary_anomaly = np.isfinite(robust_z) & (robust_z > params.robust_z_threshold)

        filtered_slices.append(grid_median)
        background_slices.append(background)
        residual_slices.append(residual)
        robust_z_slices.append(robust_z)
        binary_slices.append(binary_anomaly)
        slice_summary.append(
            {
                "Depth_m": depth,
                "Num_Data_Points": len(slice_df),
                "Residual_Median": residual_median,
                "Residual_MAD": mad,
                "Threshold_Z": params.robust_z_threshold,
                "Num_Anomaly_Cells": int(binary_anomaly.sum()),
            }
        )

    return {
        "depths": depths,
        "filtered_slices": np.stack(filtered_slices, axis=0),
        "background_slices": np.stack(background_slices, axis=0),
        "residual_slices": np.stack(residual_slices, axis=0),
        "robust_z_slices": np.stack(robust_z_slices, axis=0),
        "binary_slices": np.stack(binary_slices, axis=0),
        "slice_summary": pd.DataFrame(slice_summary),
    }


def clean_and_label_3d_targets(
    results: dict[str, Any], params: ProcessingParameters
) -> tuple[np.ndarray, np.ndarray]:
    """
    Morphologically clean the binary anomaly volume and identify connected 3D targets.
    """

    binary_volume = results["binary_slices"]

    closed_volume = binary_closing(
        binary_volume,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1,
    )

    opened_volume = binary_opening(
        closed_volume,
        structure=np.ones((1, 2, 2), dtype=bool),
        iterations=1,
    )

    labels_3d, num_components = label(
        opened_volume,
        structure=np.ones((3, 3, 3), dtype=int),
    )

    component_sizes = []
    for component_id in range(1, num_components + 1):
        indices = np.argwhere(labels_3d == component_id)
        num_voxels = len(indices)
        num_slices = len(np.unique(indices[:, 0]))
        if (
            num_voxels >= params.min_component_voxels
            and num_slices >= params.min_component_slices
        ):
            component_sizes.append((component_id, num_voxels))

    component_sizes = sorted(component_sizes, key=lambda item: item[1], reverse=True)

    clean_labels = np.zeros_like(labels_3d)
    for new_id, (old_id, _size) in enumerate(component_sizes, start=1):
        clean_labels[labels_3d == old_id] = new_id

    clean_volume = clean_labels > 0
    return clean_volume, clean_labels


def project_targets_to_plan(
    clean_labels: np.ndarray,
    results: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    metadata: dict[str, Any],
    params: ProcessingParameters,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Project cleaned 3D targets into plan view and extract wall traces.
    """

    depths = results["depths"]
    persistence = np.sum(clean_labels > 0, axis=0)
    plan_mask = persistence >= params.min_component_slices

    plan_labels, num_plan_components = label(
        plan_mask, structure=np.ones((3, 3), dtype=int)
    )

    wall_rows = []
    final_plan_labels = np.zeros_like(plan_labels)

    x0 = metadata["x0"]
    y0 = metadata["y0"]
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]
    wall_id = 1
    wall_columns = [
        "Wall_ID",
        "Center_" + x_col,
        "Center_" + y_col,
        x_col + "_Min",
        x_col + "_Max",
        y_col + "_Min",
        y_col + "_Max",
        "Area_m2",
        "Approx_Length_m",
        "Approx_Width_m",
        "Approx_Orientation_deg",
        "Max_Persistence_Slices",
        "Mean_Persistence_Slices",
        "Depth_Top_m",
        "Depth_Bottom_m",
    ]

    for plan_component_id in range(1, num_plan_components + 1):
        indices = np.argwhere(plan_labels == plan_component_id)
        if len(indices) < params.min_plan_cells:
            continue

        y_indices = indices[:, 0]
        x_indices = indices[:, 1]

        local_x = x_grid[x_indices]
        local_y = y_grid[y_indices]

        global_x = x0 + local_x
        global_y = y0 + local_y

        center_x = float(np.mean(global_x))
        center_y = float(np.mean(global_y))

        coords_local = np.column_stack([local_x, local_y])
        center_local = coords_local.mean(axis=0)
        coords_centered = coords_local - center_local

        if len(coords_local) >= 2:
            covariance = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]

            main_vector = eigenvectors[:, order[0]]
            secondary_vector = eigenvectors[:, order[1]]

            projection_main = coords_centered @ main_vector
            projection_secondary = coords_centered @ secondary_vector

            length = float(projection_main.max() - projection_main.min())
            width = float(projection_secondary.max() - projection_secondary.min())
            orientation_deg = float(np.degrees(np.arctan2(main_vector[1], main_vector[0])))
        else:
            length = np.nan
            width = np.nan
            orientation_deg = np.nan

        depths_present = []
        for y_index, x_index in indices:
            depth_indices = np.where(clean_labels[:, y_index, x_index] > 0)[0]
            depths_present.extend(depths[depth_index] for depth_index in depth_indices)

        if depths_present:
            depth_top = max(depths_present)
            depth_bottom = min(depths_present)
        else:
            depth_top = np.nan
            depth_bottom = np.nan

        final_plan_labels[plan_labels == plan_component_id] = wall_id

        wall_rows.append(
            {
                "Wall_ID": f"Wall {wall_id}",
                "Center_" + x_col: center_x,
                "Center_" + y_col: center_y,
                x_col + "_Min": float(np.min(global_x)),
                x_col + "_Max": float(np.max(global_x)),
                y_col + "_Min": float(np.min(global_y)),
                y_col + "_Max": float(np.max(global_y)),
                "Area_m2": float(len(indices) * params.grid_resolution**2),
                "Approx_Length_m": length,
                "Approx_Width_m": width,
                "Approx_Orientation_deg": orientation_deg,
                "Max_Persistence_Slices": int(
                    np.max(persistence[plan_labels == plan_component_id])
                ),
                "Mean_Persistence_Slices": float(
                    np.mean(persistence[plan_labels == plan_component_id])
                ),
                "Depth_Top_m": depth_top,
                "Depth_Bottom_m": depth_bottom,
            }
        )
        wall_id += 1

    wall_table = pd.DataFrame(wall_rows, columns=wall_columns)
    return persistence, final_plan_labels, wall_table


def _global_grid(
    x_grid: np.ndarray, y_grid: np.ndarray, metadata: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    return metadata["x0"] + x_grid, metadata["y0"] + y_grid


def _label_walls(
    ax: plt.Axes,
    plan_labels: np.ndarray,
    wall_table: pd.DataFrame,
    x_global: np.ndarray,
    y_global: np.ndarray,
    metadata: dict[str, Any],
    linewidth: float,
) -> None:
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]

    for _, row in wall_table.iterrows():
        wall_number = int(str(row["Wall_ID"]).split()[-1])
        mask = plan_labels == wall_number

        if np.any(mask):
            ax.contour(
                x_global,
                y_global,
                mask.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=linewidth,
            )

        center_x = row["Center_" + x_col]
        center_y = row["Center_" + y_col]
        ax.text(
            center_x,
            center_y,
            row["Wall_ID"],
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25"),
        )


def plot_anomaly_map(
    persistence: np.ndarray,
    plan_labels: np.ndarray,
    wall_table: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Plot anomaly persistence background plus wall trace outlines.
    """

    x_global, y_global = _global_grid(x_grid, y_grid, metadata)
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    anomaly_background = np.where(plan_labels > 0, persistence, np.nan)
    if not np.isfinite(anomaly_background).any():
        anomaly_background = np.zeros_like(persistence, dtype=float)

    image = ax.imshow(
        anomaly_background,
        origin="lower",
        extent=[x_global.min(), x_global.max(), y_global.min(), y_global.max()],
        cmap="YlOrRd",
        aspect="equal",
    )

    _label_walls(ax, plan_labels, wall_table, x_global, y_global, metadata, linewidth=1.5)

    ax.set_title("Possible Ancient Wall Traces from Resistivity Data")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Persistence, number of depth slices")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_clean_coordinate_map(
    plan_labels: np.ndarray,
    wall_table: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Plot a clean report-style map: coordinate grid plus wall outlines only.
    """

    x_global, y_global = _global_grid(x_grid, y_grid, metadata)
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_facecolor("white")

    _label_walls(ax, plan_labels, wall_table, x_global, y_global, metadata, linewidth=1.8)

    ax.set_xlim(x_global.min(), x_global.max())
    ax.set_ylim(y_global.min(), y_global.max())
    ax.set_title("Possible Ancient Wall Traces on Coordinate Grid")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, which="major", color="0.75", linestyle="-", linewidth=0.8)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="0.9", linestyle=":", linewidth=0.5)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_results_to_excel(
    df: pd.DataFrame,
    results: dict[str, Any],
    wall_table: pd.DataFrame,
    metadata: dict[str, Any],
    params: ProcessingParameters,
    output_file: Path,
) -> None:
    """
    Export processed points, slice summary, parameters, metadata, and wall traces.
    """

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="processed_points", index=False)
            results["slice_summary"].to_excel(writer, sheet_name="slice_summary", index=False)
            wall_table.to_excel(writer, sheet_name="wall_traces", index=False)
            pd.DataFrame([metadata]).to_excel(writer, sheet_name="metadata", index=False)
            pd.DataFrame([asdict(params)]).to_excel(writer, sheet_name="parameters", index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing .xlsx output requires the 'openpyxl' package."
        ) from exc


def run_workflow(
    input_file: Path, output_folder: Path, params: ProcessingParameters
) -> dict[str, Any]:
    output_folder.mkdir(parents=True, exist_ok=True)

    print("Reading database...")
    df, metadata = load_and_prepare_data(input_file, params)

    print("Detected columns:")
    print("X/E coordinate:", metadata["x_col"])
    print("Y/N coordinate:", metadata["y_col"])
    print("Depth:", metadata["depth_col"])
    print("Resistivity:", metadata["rho_col"])

    print("Creating regular grid...")
    x_grid, y_grid, xx, yy, valid_mask = create_regular_grid(df, params)

    print("Processing depth slices...")
    results = process_depth_slices(
        df, metadata, x_grid, y_grid, xx, yy, valid_mask, params
    )

    print("Cleaning and labelling 3D targets...")
    _clean_volume, clean_labels = clean_and_label_3d_targets(results, params)

    print("Projecting 3D targets to plan view...")
    persistence, plan_labels, wall_table = project_targets_to_plan(
        clean_labels, results, x_grid, y_grid, metadata, params
    )

    print("Possible wall traces:", len(wall_table))

    anomaly_map_path = output_folder / "wall_traces_with_anomalies.png"
    clean_map_path = output_folder / "wall_traces_coordinate_grid_only.png"
    excel_path = output_folder / "resistivity_wall_trace_results.xlsx"
    csv_path = output_folder / "wall_trace_summary.csv"

    print("Creating anomaly map...")
    plot_anomaly_map(
        persistence,
        plan_labels,
        wall_table,
        x_grid,
        y_grid,
        metadata,
        anomaly_map_path,
    )

    print("Creating clean coordinate map...")
    plot_clean_coordinate_map(
        plan_labels,
        wall_table,
        x_grid,
        y_grid,
        metadata,
        clean_map_path,
    )

    print("Exporting tables...")
    export_results_to_excel(df, results, wall_table, metadata, params, excel_path)
    wall_table.to_csv(csv_path, index=False)

    print("Processing completed.")
    print("Output folder:", output_folder)
    print("Anomaly map:", anomaly_map_path)
    print("Clean coordinate map:", clean_map_path)
    print("Excel:", excel_path)
    print("CSV:", csv_path)

    return {
        "output_folder": output_folder,
        "anomaly_map": anomaly_map_path,
        "clean_map": clean_map_path,
        "excel": excel_path,
        "csv": csv_path,
        "wall_count": len(wall_table),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect possible ancient wall traces from resistivity data."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help=f"Input .xls/.xlsx file. Default: {DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder. Default: {DEFAULT_OUTPUT_FOLDER}",
    )
    parser.add_argument("--grid-resolution", type=float, default=0.55)
    parser.add_argument("--clip-low", type=float, default=2.0)
    parser.add_argument("--clip-high", type=float, default=98.0)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--gaussian-sigma-m", type=float, default=2.5)
    parser.add_argument("--robust-z-threshold", type=float, default=2.0)
    parser.add_argument("--max-extrapolation-distance", type=float, default=2.6)
    parser.add_argument("--min-component-voxels", type=int, default=15)
    parser.add_argument("--min-component-slices", type=int, default=2)
    parser.add_argument("--min-plan-cells", type=int, default=12)
    return parser


def params_from_args(args: argparse.Namespace) -> ProcessingParameters:
    return ProcessingParameters(
        grid_resolution=args.grid_resolution,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        median_size=args.median_size,
        gaussian_sigma_m=args.gaussian_sigma_m,
        robust_z_threshold=args.robust_z_threshold,
        max_extrapolation_distance=args.max_extrapolation_distance,
        min_component_voxels=args.min_component_voxels,
        min_component_slices=args.min_component_slices,
        min_plan_cells=args.min_plan_cells,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_folder = Path(args.output_folder)
    params = params_from_args(args)

    run_workflow(input_file, output_folder, params)


if __name__ == "__main__":
    main()
