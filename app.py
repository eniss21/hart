#!/usr/bin/env python3
"""
Small browser UI for testing the resistivity wall-trace workflows.

Two modes are supported:
- 3D: ``process_resistivity.run_workflow`` over a depth-resolved cube.
- 2D: ``process_resistivity_2d.run_workflow`` over a single plan map
  (logic verbatim from ``2d_resistivity_wall_detection_processing_workflow.md``).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from process_resistivity import ProcessingParameters, run_workflow as run_workflow_3d
from process_resistivity_2d import Parameters2D, run_workflow as run_workflow_2d


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "web_runs"
ALLOWED_EXTENSIONS = {".xls", ".xlsx"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def number_field(name: str, default: float) -> float:
    raw = request.form.get(name, "").strip()
    if raw == "":
        return default
    return float(raw)


def integer_field(name: str, default: int) -> int:
    raw = request.form.get(name, "").strip()
    if raw == "":
        return default
    return int(raw)


def parameters_3d_from_form() -> ProcessingParameters:
    return ProcessingParameters(
        grid_resolution=number_field("grid_resolution", 0.55),
        clip_low=number_field("clip_low", 2.0),
        clip_high=number_field("clip_high", 98.0),
        median_size=integer_field("median_size", 3),
        gaussian_sigma_m=number_field("gaussian_sigma_m", 2.5),
        robust_z_threshold=number_field("robust_z_threshold", 2.0),
        max_extrapolation_distance=number_field("max_extrapolation_distance", 2.6),
        min_component_voxels=integer_field("min_component_voxels", 15),
        min_component_slices=integer_field("min_component_slices", 2),
        min_plan_cells=integer_field("min_plan_cells", 12),
    )


def parameters_2d_from_form() -> Parameters2D:
    return Parameters2D(
        grid_resolution_m=number_field("grid_resolution", 0.5),
        clip_low_pct=number_field("clip_low", 2.0),
        clip_high_pct=number_field("clip_high", 98.0),
        median_size_cells=integer_field("median_size", 3),
        gaussian_sigma_m=number_field("gaussian_sigma_m", 2.5),
        max_extrapolation_distance_m=number_field("max_extrapolation_distance", 2.6),
        robust_z_threshold=number_field("robust_z_threshold", 2.0),
        min_plan_cells=integer_field("min_plan_cells", 12),
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/run")
def run_from_upload():
    uploaded_file = request.files.get("input_file")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Upload an .xls or .xlsx resistivity workbook."}), 400
    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Only .xls and .xlsx files are supported."}), 400

    mode = request.form.get("mode", "3d").strip().lower()
    if mode not in {"2d", "3d"}:
        return jsonify({"error": "Mode must be '2d' or '3d'."}), 400

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    upload_dir = run_dir / "input"
    output_dir = run_dir / ("resistivity_output_2d" if mode == "2d" else "resistivity_output")
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(uploaded_file.filename)
    input_path = upload_dir / filename
    uploaded_file.save(input_path)

    try:
        if mode == "3d":
            params = parameters_3d_from_form()
            outputs = run_workflow_3d(input_path, output_dir, params)
            summary_path = outputs["csv"]
            wall_table = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

            files = {
                "wall_traces_with_anomalies.png": f"/runs/{run_id}/resistivity_output/wall_traces_with_anomalies.png",
                "wall_traces_coordinate_grid_only.png": f"/runs/{run_id}/resistivity_output/wall_traces_coordinate_grid_only.png",
                "resistivity_wall_trace_results.xlsx": f"/runs/{run_id}/resistivity_output/resistivity_wall_trace_results.xlsx",
                "wall_trace_summary.csv": f"/runs/{run_id}/resistivity_output/wall_trace_summary.csv",
            }

            return jsonify(
                {
                    "mode": "3d",
                    "run_id": run_id,
                    "wall_count": outputs["wall_count"],
                    "files": files,
                    "maps": [
                        {
                            "caption": "Diagnostic anomaly persistence map",
                            "url": files["wall_traces_with_anomalies.png"],
                        },
                        {
                            "caption": "Clean coordinate-grid trace map",
                            "url": files["wall_traces_coordinate_grid_only.png"],
                        },
                    ],
                    "walls": wall_table.fillna("").to_dict(orient="records"),
                }
            )

        # 2D
        params2 = parameters_2d_from_form()
        outputs = run_workflow_2d(input_path, output_dir, params2)
        summary_path = outputs["summary_csv"]
        wall_table = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

        files = {
            "01_workflow_maps_same_3d_procedure.png": f"/runs/{run_id}/resistivity_output_2d/01_workflow_maps_same_3d_procedure.png",
            "02_final_wall_candidates_same_3d_procedure.png": f"/runs/{run_id}/resistivity_output_2d/02_final_wall_candidates_same_3d_procedure.png",
            "wall_trace_summary_same_3d_workflow.csv": f"/runs/{run_id}/resistivity_output_2d/wall_trace_summary_same_3d_workflow.csv",
            "processed_grid_same_3d_workflow.csv": f"/runs/{run_id}/resistivity_output_2d/processed_grid_same_3d_workflow.csv",
            "processing_report_same_3d_workflow.md": f"/runs/{run_id}/resistivity_output_2d/processing_report_same_3d_workflow.md",
        }

        return jsonify(
            {
                "mode": "2d",
                "run_id": run_id,
                "wall_count": outputs["wall_count"],
                "files": files,
                "maps": [
                    {
                        "caption": "Workflow panels (log10, median, residual, Z-score)",
                        "url": files["01_workflow_maps_same_3d_procedure.png"],
                    },
                    {
                        "caption": "Final wall-candidate map",
                        "url": files["02_final_wall_candidates_same_3d_procedure.png"],
                    },
                ],
                "walls": wall_table.fillna("").to_dict(orient="records"),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/runs/<run_id>/<path:filename>")
def serve_run_file(run_id: str, filename: str):
    if not run_id.isalnum():
        return "Invalid run id", 400
    directory = RUNS_DIR / run_id
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
