# Resistivity Data Processing Workflow for Archaeological Wall Detection

## Purpose

This Markdown file summarizes the complete workflow used to process apparent resistivity map-slice data and extract possible ancient wall traces.

It is written for coding agents and developers who need to understand both the geophysical logic and the implementation logic.

The workflow is designed for multi-depth apparent resistivity datasets, where each measurement contains:

- coordinate X/E;
- coordinate Y/N;
- depth or pseudo-depth;
- apparent resistivity.

The final objective is to identify **localized, coherent, high-resistivity residual anomalies** that may correspond to buried walls, foundations, compact stone structures, or archaeological remains.

---

# 1. Input data

The program should accept `.xls` and `.xlsx` files.

## Database 1 example

| Column | Meaning |
|---|---|
| `Profili` | profile number |
| `Piketa` | station along profile |
| `Thellësia` | depth slice |
| `Rho (Ohm.m)` | apparent resistivity |
| `SP (mV)` | self-potential, optional |
| `X` | coordinate X |
| `Y` | coordinate Y |

## Database 2 example

| Column | Meaning |
|---|---|
| `Profili` | profile number |
| `PK` | station |
| `Depth` | depth slice |
| `Rd` | apparent resistivity |
| `SP` | self-potential, optional |
| `E` | easting |
| `N` | northing |

The code should auto-detect the coordinate, depth, and resistivity columns when possible.

---

# 2. Processing concept

Ancient walls are expected to appear as:

- localized positive resistivity anomalies;
- elongated or rectangular patterns in plan view;
- coherent features across several depth slices;
- spatially connected objects, not isolated single-pixel spikes.

The goal is not to map high absolute resistivity only. The goal is to enhance **local resistive contrast relative to background**.

The key residual anomaly is:

```text
A(x, y, z) = Filtered_Log10_Rho(x, y, z) - Background_Log10_Rho(x, y, z)
```

Positive residual anomalies are interpreted as possible high-resistivity archaeological targets.

---

# 3. Full workflow

```text
raw apparent resistivity
        ↓
remove non-positive values
        ↓
convert to local coordinates
        ↓
log10 apparent resistivity
        ↓
P2–P98 robust clipping
        ↓
split by depth slice
        ↓
interpolate each depth slice to regular grid
        ↓
mask extrapolated cells outside measured footprint
        ↓
3×3 median filter
        ↓
Gaussian background estimation
        ↓
residual anomaly = filtered log-resistivity − background
        ↓
robust MAD Z-score
        ↓
positive anomaly threshold: Z > 2.0
        ↓
3D morphological cleaning
        ↓
3D connected-component filtering
        ↓
plan-view projection / persistence map
        ↓
2D connected wall traces
        ↓
coordinate-grid interpretation map
```

---

# 4. Mathematical formulation

## 4.1 Log transform

Apparent resistivity is transformed as:

```text
R_log = log10(rho_a)
```

where:

- `rho_a` is apparent resistivity;
- `R_log` is log-resistivity.

This reduces the dominance of very high resistivity values and stabilizes the distribution.

## 4.2 Percentile clipping

```text
P2 <= R_log <= P98
```

Clipped value:

```text
if R_log < P2:   R_log_clip = P2
if R_log > P98:  R_log_clip = P98
otherwise:       R_log_clip = R_log
```

This limits outliers without deleting the measurement.

## 4.3 Background removal

For each depth slice:

```text
Background = GaussianSmooth(MedianFiltered_Log10_Rho)
```

Then:

```text
Residual = MedianFiltered_Log10_Rho - Background
```

## 4.4 Robust normalization

```text
Z_robust = (Residual - median(Residual)) / MAD(Residual)
```

where:

```text
MAD = median(abs(Residual - median(Residual)))
```

## 4.5 Positive anomaly mask

```text
Mask = Z_robust > 2.0
```

Only positive residual anomalies are retained.

---

# 5. Default parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `GRID_RESOLUTION` | `0.5` m | regular grid spacing |
| `CLIP_LOW` | `2` | lower percentile |
| `CLIP_HIGH` | `98` | upper percentile |
| `MEDIAN_SIZE` | `3` | 3×3 median filter |
| `GAUSSIAN_SIGMA_M` | `2.5` m | background smoothing sigma |
| `ROBUST_Z_THRESHOLD` | `2.0` | positive anomaly threshold |
| `MAX_EXTRAPOLATION_DISTANCE` | `2.6` m | maximum distance from real data |
| `MIN_COMPONENT_VOXELS` | `15` | minimum 3D target size |
| `MIN_COMPONENT_SLICES` | `2` | target must occur in at least 2 depth slices |
| `MIN_PLAN_CELLS` | `12` | minimum plan-view size for wall trace |

These are recommended starting values. They may be adjusted depending on survey spacing, expected wall width, and noise level.

---

# 6. Interpretation rules

A possible wall trace should satisfy several criteria:

1. It is a positive local residual anomaly.
2. It is coherent in plan view.
3. It is not an isolated single-cell spike.
4. It persists across at least two depth slices.
5. It has elongated, linear, rectangular, or structurally meaningful geometry.
6. It remains after 3D morphological cleaning.
7. It is compatible with the archaeological context of the site.

The output should be described as:

```text
possible ancient wall traces
```

or:

```text
probable resistive archaeological targets
```

Do not describe them as confirmed walls unless verified by excavation or independent evidence.

---

# 7. Recommended outputs

A complete program should export:

```text
wall_traces_with_anomalies.png
wall_traces_coordinate_grid_only.png
resistivity_wall_trace_results.xlsx
wall_trace_summary.csv
```

## `wall_traces_with_anomalies.png`

Diagnostic map showing:

- anomaly persistence background;
- wall trace outlines;
- wall labels;
- coordinate axes.

## `wall_traces_coordinate_grid_only.png`

Clean report map showing:

- white background;
- coordinate grid;
- wall trace outlines only;
- labels such as Wall 1, Wall 2, Wall 3.

## `resistivity_wall_trace_results.xlsx`

Workbook containing:

- processed points;
- log-resistivity;
- clipping flags;
- slice summary;
- wall trace table.

## `wall_trace_summary.csv`

CSV table containing:

- Wall ID;
- centroid coordinates;
- bounding coordinates;
- approximate length;
- approximate width;
- orientation;
- persistence;
- depth interval.

---

# 8. Full Python code

```python
# ============================================================
# PROGRAM FOR PROCESSING APPARENT RESISTIVITY DATA
# Archaeological wall-trace detection workflow
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import (
    median_filter,
    gaussian_filter,
    binary_closing,
    binary_opening,
    label
)
from scipy.spatial import cKDTree


# ============================================================
# 1. USER PARAMETERS
# ============================================================

INPUT_FILE = "Lezha_3D.xls"
OUTPUT_FOLDER = "resistivity_output"

GRID_RESOLUTION = 0.5
CLIP_LOW = 2
CLIP_HIGH = 98
MEDIAN_SIZE = 3
GAUSSIAN_SIGMA_M = 2.5
ROBUST_Z_THRESHOLD = 2.0

MAX_EXTRAPOLATION_DISTANCE = 2.6

MIN_COMPONENT_VOXELS = 15
MIN_COMPONENT_SLICES = 2
MIN_PLAN_CELLS = 12


# ============================================================
# 2. COLUMN DETECTION
# ============================================================

def detect_columns(df):
    """
    Automatically detect coordinate, depth, and resistivity columns.
    """

    columns = [c.strip() for c in df.columns]
    df.columns = columns

    possible_x = ["X", "E", "East", "Easting", "Local_X"]
    possible_y = ["Y", "N", "North", "Northing", "Local_Y"]
    possible_depth = ["Thellësia", "Depth", "Z", "Depth_m"]
    possible_rho = [
        "Rho (Ohm.m)",
        "Rd",
        "Resistivity",
        "Apparent_Resistivity",
        "Rho"
    ]

    x_col = next((c for c in possible_x if c in columns), None)
    y_col = next((c for c in possible_y if c in columns), None)
    depth_col = next((c for c in possible_depth if c in columns), None)
    rho_col = next((c for c in possible_rho if c in columns), None)

    if x_col is None:
        raise ValueError("X/E coordinate column was not found.")

    if y_col is None:
        raise ValueError("Y/N coordinate column was not found.")

    if depth_col is None:
        raise ValueError("Depth column was not found.")

    if rho_col is None:
        raise ValueError("Resistivity column was not found.")

    return x_col, y_col, depth_col, rho_col


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

def load_and_prepare_data(input_file):
    """
    Read the database, remove invalid resistivity values, convert
    coordinates to local coordinates, compute log10 resistivity,
    and apply robust percentile clipping.
    """

    df = pd.read_excel(input_file)
    df.columns = [c.strip() for c in df.columns]

    x_col, y_col, depth_col, rho_col = detect_columns(df)

    df = df[np.isfinite(df[rho_col])].copy()
    df = df[df[rho_col] > 0].copy()

    x0 = df[x_col].min()
    y0 = df[y_col].min()

    df["Local_X_m"] = df[x_col] - x0
    df["Local_Y_m"] = df[y_col] - y0

    df["Log10_Rho"] = np.log10(df[rho_col])

    p_low, p_high = np.percentile(
        df["Log10_Rho"],
        [CLIP_LOW, CLIP_HIGH]
    )

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
        "clip_low_value": p_low,
        "clip_high_value": p_high
    }

    return df, metadata


# ============================================================
# 4. CREATE REGULAR GRID
# ============================================================

def create_regular_grid(df):
    """
    Create a regular 2D grid and a valid-area mask based on
    distance to the nearest measured point.
    """

    x_min = df["Local_X_m"].min()
    x_max = df["Local_X_m"].max()
    y_min = df["Local_Y_m"].min()
    y_max = df["Local_Y_m"].max()

    x_grid = np.arange(
        math.floor(x_min / GRID_RESOLUTION) * GRID_RESOLUTION,
        math.ceil(x_max / GRID_RESOLUTION) * GRID_RESOLUTION
        + GRID_RESOLUTION,
        GRID_RESOLUTION
    )

    y_grid = np.arange(
        math.floor(y_min / GRID_RESOLUTION) * GRID_RESOLUTION,
        math.ceil(y_max / GRID_RESOLUTION) * GRID_RESOLUTION
        + GRID_RESOLUTION,
        GRID_RESOLUTION
    )

    XX, YY = np.meshgrid(x_grid, y_grid)

    real_points = df[["Local_X_m", "Local_Y_m"]].drop_duplicates().to_numpy()
    tree = cKDTree(real_points)

    distance_to_nearest, _ = tree.query(
        np.c_[XX.ravel(), YY.ravel()],
        k=1
    )

    valid_mask = (
        distance_to_nearest.reshape(XX.shape)
        <= MAX_EXTRAPOLATION_DISTANCE
    )

    return x_grid, y_grid, XX, YY, valid_mask


# ============================================================
# 5. PROCESS DEPTH SLICES
# ============================================================

def process_depth_slices(df, metadata, x_grid, y_grid, XX, YY, valid_mask):
    """
    Process each depth slice:
    - interpolate clipped log-resistivity;
    - apply median filter;
    - estimate Gaussian background;
    - calculate residual anomaly;
    - calculate robust Z-score;
    - threshold positive anomalies.
    """

    depth_col = metadata["depth_col"]

    depths = sorted(df[depth_col].unique())

    filtered_slices = []
    background_slices = []
    residual_slices = []
    robust_z_slices = []
    binary_slices = []

    slice_summary = []

    gaussian_sigma_cells = GAUSSIAN_SIGMA_M / GRID_RESOLUTION

    for depth in depths:

        slice_df = df[df[depth_col] == depth].copy()

        points = slice_df[["Local_X_m", "Local_Y_m"]].to_numpy()
        values = slice_df["Log10_Rho_Clipped"].to_numpy()

        grid_linear = griddata(
            points,
            values,
            (XX, YY),
            method="linear"
        )

        grid_nearest = griddata(
            points,
            values,
            (XX, YY),
            method="nearest"
        )

        grid_log_rho = np.where(
            np.isfinite(grid_linear),
            grid_linear,
            grid_nearest
        )

        grid_log_rho = np.where(valid_mask, grid_log_rho, np.nan)

        fill_value = np.nanmedian(values)

        grid_filled = np.where(
            np.isfinite(grid_log_rho),
            grid_log_rho,
            fill_value
        )

        grid_median = median_filter(
            grid_filled,
            size=MEDIAN_SIZE,
            mode="nearest"
        )

        background = gaussian_filter(
            grid_median,
            sigma=gaussian_sigma_cells,
            mode="nearest",
            truncate=3.0
        )

        residual = grid_median - background
        residual = np.where(valid_mask, residual, np.nan)

        residual_values = residual[np.isfinite(residual)]

        residual_median = np.nanmedian(residual_values)

        mad = np.nanmedian(
            np.abs(residual_values - residual_median)
        )

        if mad <= 1e-9:
            mad = np.nanstd(residual_values)

        if mad <= 1e-9:
            mad = 1.0

        robust_z = (residual - residual_median) / mad

        binary_anomaly = (
            np.isfinite(robust_z)
            & (robust_z > ROBUST_Z_THRESHOLD)
        )

        filtered_slices.append(grid_median)
        background_slices.append(background)
        residual_slices.append(residual)
        robust_z_slices.append(robust_z)
        binary_slices.append(binary_anomaly)

        slice_summary.append({
            "Depth_m": depth,
            "Num_Data_Points": len(slice_df),
            "Residual_Median": residual_median,
            "Residual_MAD": mad,
            "Threshold_Z": ROBUST_Z_THRESHOLD,
            "Num_Anomaly_Cells": int(binary_anomaly.sum())
        })

    results = {
        "depths": depths,
        "filtered_slices": np.stack(filtered_slices, axis=0),
        "background_slices": np.stack(background_slices, axis=0),
        "residual_slices": np.stack(residual_slices, axis=0),
        "robust_z_slices": np.stack(robust_z_slices, axis=0),
        "binary_slices": np.stack(binary_slices, axis=0),
        "slice_summary": pd.DataFrame(slice_summary)
    }

    return results


# ============================================================
# 6. CLEAN AND LABEL 3D TARGETS
# ============================================================

def clean_and_label_3d_targets(results):
    """
    Morphologically clean the binary anomaly volume and identify
    connected 3D targets.
    """

    binary_volume = results["binary_slices"]

    closed_volume = binary_closing(
        binary_volume,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1
    )

    opened_volume = binary_opening(
        closed_volume,
        structure=np.ones((1, 2, 2), dtype=bool),
        iterations=1
    )

    labels_3d, num_components = label(
        opened_volume,
        structure=np.ones((3, 3, 3), dtype=int)
    )

    kept_component_ids = []

    for component_id in range(1, num_components + 1):

        indices = np.argwhere(labels_3d == component_id)

        num_voxels = len(indices)
        num_slices = len(np.unique(indices[:, 0]))

        if (
            num_voxels >= MIN_COMPONENT_VOXELS
            and num_slices >= MIN_COMPONENT_SLICES
        ):
            kept_component_ids.append(component_id)

    clean_labels = np.zeros_like(labels_3d)

    component_sizes = []

    for component_id in kept_component_ids:
        size = int(np.sum(labels_3d == component_id))
        component_sizes.append((component_id, size))

    component_sizes = sorted(
        component_sizes,
        key=lambda x: x[1],
        reverse=True
    )

    for new_id, (old_id, size) in enumerate(component_sizes, start=1):
        clean_labels[labels_3d == old_id] = new_id

    clean_volume = clean_labels > 0

    return clean_volume, clean_labels


# ============================================================
# 7. PROJECT TARGETS TO PLAN VIEW
# ============================================================

def project_targets_to_plan(clean_labels, results, x_grid, y_grid, metadata):
    """
    Project cleaned 3D targets into plan view and extract wall traces.
    """

    depths = results["depths"]

    persistence = np.sum(clean_labels > 0, axis=0)

    plan_mask = persistence >= MIN_COMPONENT_SLICES

    plan_labels, num_plan_components = label(
        plan_mask,
        structure=np.ones((3, 3), dtype=int)
    )

    wall_rows = []
    final_plan_labels = np.zeros_like(plan_labels)

    wall_id = 1

    x0 = metadata["x0"]
    y0 = metadata["y0"]
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]

    for plan_component_id in range(1, num_plan_components + 1):

        indices = np.argwhere(plan_labels == plan_component_id)

        if len(indices) < MIN_PLAN_CELLS:
            continue

        y_indices = indices[:, 0]
        x_indices = indices[:, 1]

        local_x = x_grid[x_indices]
        local_y = y_grid[y_indices]

        global_x = x0 + local_x
        global_y = y0 + local_y

        center_x = np.mean(global_x)
        center_y = np.mean(global_y)

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

            length = projection_main.max() - projection_main.min()
            width = projection_secondary.max() - projection_secondary.min()

            orientation_deg = np.degrees(
                np.arctan2(main_vector[1], main_vector[0])
            )
        else:
            length = np.nan
            width = np.nan
            orientation_deg = np.nan

        depths_present = []

        for yy, xx in indices:
            depth_indices = np.where(clean_labels[:, yy, xx] > 0)[0]
            for di in depth_indices:
                depths_present.append(depths[di])

        if len(depths_present) > 0:
            depth_top = max(depths_present)
            depth_bottom = min(depths_present)
        else:
            depth_top = np.nan
            depth_bottom = np.nan

        final_plan_labels[plan_labels == plan_component_id] = wall_id

        wall_rows.append({
            "Wall_ID": f"Wall {wall_id}",
            "Center_" + x_col: center_x,
            "Center_" + y_col: center_y,
            x_col + "_Min": np.min(global_x),
            x_col + "_Max": np.max(global_x),
            y_col + "_Min": np.min(global_y),
            y_col + "_Max": np.max(global_y),
            "Area_m2": len(indices) * GRID_RESOLUTION ** 2,
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
            "Depth_Bottom_m": depth_bottom
        })

        wall_id += 1

    wall_table = pd.DataFrame(wall_rows)

    return persistence, final_plan_labels, wall_table


# ============================================================
# 8. PLOT ANOMALY MAP
# ============================================================

def plot_anomaly_map(
    persistence,
    plan_labels,
    wall_table,
    x_grid,
    y_grid,
    metadata,
    output_path
):
    """
    Plot anomaly persistence background plus wall trace outlines.
    """

    x0 = metadata["x0"]
    y0 = metadata["y0"]
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]

    X_global = x0 + x_grid
    Y_global = y0 + y_grid

    fig, ax = plt.subplots(figsize=(11, 8.5))

    anomaly_background = np.where(plan_labels > 0, persistence, np.nan)

    image = ax.imshow(
        anomaly_background,
        origin="lower",
        extent=[
            X_global.min(),
            X_global.max(),
            Y_global.min(),
            Y_global.max()
        ],
        cmap="YlOrRd",
        aspect="equal"
    )

    for _, row in wall_table.iterrows():

        wall_number = int(row["Wall_ID"].split()[-1])
        mask = plan_labels == wall_number

        ax.contour(
            X_global,
            Y_global,
            mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.5
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
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.25"
            )
        )

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


# ============================================================
# 9. PLOT CLEAN COORDINATE MAP
# ============================================================

def plot_clean_coordinate_map(
    plan_labels,
    wall_table,
    x_grid,
    y_grid,
    metadata,
    output_path
):
    """
    Plot a clean report-style map:
    coordinate grid plus wall outlines only.
    """

    x0 = metadata["x0"]
    y0 = metadata["y0"]
    x_col = metadata["x_col"]
    y_col = metadata["y_col"]

    X_global = x0 + x_grid
    Y_global = y0 + y_grid

    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.set_facecolor("white")

    for _, row in wall_table.iterrows():

        wall_number = int(row["Wall_ID"].split()[-1])
        mask = plan_labels == wall_number

        ax.contour(
            X_global,
            Y_global,
            mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.8
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
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.25"
            )
        )

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


# ============================================================
# 10. EXPORT RESULTS
# ============================================================

def export_results_to_excel(df, results, wall_table, output_file):
    """
    Export processed points, slice summary, and wall traces to Excel.
    """

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        df.to_excel(
            writer,
            sheet_name="processed_points",
            index=False
        )

        results["slice_summary"].to_excel(
            writer,
            sheet_name="slice_summary",
            index=False
        )

        wall_table.to_excel(
            writer,
            sheet_name="wall_traces",
            index=False
        )


# ============================================================
# 11. MAIN FUNCTION
# ============================================================

def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Reading database...")
    df, metadata = load_and_prepare_data(INPUT_FILE)

    print("Detected columns:")
    print("X/E coordinate:", metadata["x_col"])
    print("Y/N coordinate:", metadata["y_col"])
    print("Depth:", metadata["depth_col"])
    print("Resistivity:", metadata["rho_col"])

    print("Creating regular grid...")
    x_grid, y_grid, XX, YY, valid_mask = create_regular_grid(df)

    print("Processing depth slices...")
    results = process_depth_slices(
        df,
        metadata,
        x_grid,
        y_grid,
        XX,
        YY,
        valid_mask
    )

    print("Cleaning and labelling 3D targets...")
    clean_volume, clean_labels = clean_and_label_3d_targets(results)

    print("Projecting 3D targets to plan view...")
    persistence, plan_labels, wall_table = project_targets_to_plan(
        clean_labels,
        results,
        x_grid,
        y_grid,
        metadata
    )

    print("Possible wall traces:", len(wall_table))

    anomaly_map_path = os.path.join(
        OUTPUT_FOLDER,
        "wall_traces_with_anomalies.png"
    )

    clean_map_path = os.path.join(
        OUTPUT_FOLDER,
        "wall_traces_coordinate_grid_only.png"
    )

    excel_path = os.path.join(
        OUTPUT_FOLDER,
        "resistivity_wall_trace_results.xlsx"
    )

    csv_path = os.path.join(
        OUTPUT_FOLDER,
        "wall_trace_summary.csv"
    )

    print("Creating anomaly map...")
    plot_anomaly_map(
        persistence,
        plan_labels,
        wall_table,
        x_grid,
        y_grid,
        metadata,
        anomaly_map_path
    )

    print("Creating clean coordinate map...")
    plot_clean_coordinate_map(
        plan_labels,
        wall_table,
        x_grid,
        y_grid,
        metadata,
        clean_map_path
    )

    print("Exporting tables...")
    export_results_to_excel(
        df,
        results,
        wall_table,
        excel_path
    )

    wall_table.to_csv(csv_path, index=False)

    print("Processing completed.")
    print("Output folder:", OUTPUT_FOLDER)
    print("Anomaly map:", anomaly_map_path)
    print("Clean coordinate map:", clean_map_path)
    print("Excel:", excel_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
```

---

# 9. Required Python libraries

Install:

```bash
pip install numpy pandas scipy matplotlib openpyxl xlrd
```

For `.xls` files:

```bash
pip install xlrd
```

For `.xlsx` files:

```bash
pip install openpyxl
```

---

# 10. How to run

Folder example:

```text
project_folder/
├── process_resistivity.py
└── Lezha_3D.xls
```

Set in the script:

```python
INPUT_FILE = "Lezha_3D.xls"
```

Then run:

```bash
python process_resistivity.py
```

Expected output folder:

```text
resistivity_output/
```

Expected files:

```text
wall_traces_with_anomalies.png
wall_traces_coordinate_grid_only.png
resistivity_wall_trace_results.xlsx
wall_trace_summary.csv
```

---

# 11. Notes for coding agents

When adapting or extending this program, preserve the conceptual order:

1. Work on `log10(resistivity)`, not raw resistivity.
2. Clip outliers before interpolation.
3. Interpolate independently for each depth slice.
4. Mask cells outside the measured footprint.
5. Remove regional background before interpreting anomalies.
6. Use residual positive anomalies for wall detection.
7. Use MAD-based robust normalization.
8. Require vertical persistence across depth slices.
9. Use 3D connected-component filtering before plan-view projection.
10. Export both diagnostic anomaly maps and clean coordinate-grid maps.

Useful future improvements:

- interactive Plotly 3D target visualization;
- GeoJSON export;
- shapefile export;
- DXF export for CAD;
- QGIS-compatible layers;
- automatic ranking of top wall candidates;
- smoothing of wall-trace outlines;
- manual interpretation editing;
- report PDF generation.

---

# 12. Report-ready interpretation text

The following text can be used in technical reports:

> The apparent resistivity data were processed using a workflow designed to enhance localized resistive anomalies potentially associated with buried archaeological structures. The original apparent resistivity values were transformed to logarithmic scale using `log10(rho_a)` and clipped between the 2nd and 98th percentiles to reduce the influence of extreme values. For each depth slice, the data were interpolated to a regular 0.5 m grid, filtered with a 3×3 median filter, and separated into regional background and local anomaly components using Gaussian smoothing. The residual anomaly was calculated as the difference between the filtered log-resistivity field and the smoothed background. Positive anomalies were identified using robust MAD-based normalization and a threshold of `Z_robust > 2.0`. The resulting binary anomaly volume was cleaned using 3D morphological operations and connected-component analysis. Only anomalies persistent across at least two depth slices and satisfying minimum size criteria were retained. These targets were then projected onto plan view and interpreted as possible wall traces based on their coherence, persistence, and geometry.

---

# 13. Cautions

This workflow identifies geophysical targets, not archaeological proof.

Final interpretation should consider:

- excavation evidence;
- site history;
- surface remains;
- topography;
- known building orientation;
- alternative causes such as stones, rubble, compacted soil, drains, roots, dry zones, or modern buried objects.

Use conservative language:

```text
possible wall traces
```

or:

```text
probable resistive archaeological targets
```

unless the interpretation is independently verified.
