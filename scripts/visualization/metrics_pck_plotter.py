# SPDX-FileCopyrightText: 2025-2026 Fraunhofer Italia
# SPDX-License-Identifier: MIT
#
# Author: Francesco Visentin <francesco.visentin@fraunhofer.it>

# GUI functions must be imported before any 'cv2' import to avoid conflicts with PyQt5
from dense_objects_nets.utils.gui_utils import select_runs_folders, selection_window

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from glob import glob
from typing import List, Tuple
from plotly.subplots import make_subplots

import numpy as np

html_base = """
<html>
<head>
<style>
.tabs {{ display: flex; cursor: pointer; }}
.tab {{ padding: 8px; border: 1px solid #ccc; margin-right: 4px; }}
.content {{ display: none; }}
.content.active {{ display: block; }}
</style>
</head>
<body>

<div class="tabs">
{tabs}
</div>

{contents}

<script>
function showTab(name) {{
    // deactivate all
    document.querySelectorAll('.content').forEach(function(c) {{
        c.classList.remove('active');
    }});

    // activate selected
    var el = document.getElementById(name);
    el.classList.add('active');

    // fix layout of plots in this tab
    if (window.Plotly) {{
        el.querySelectorAll('.plotly-graph-div').forEach(function(gd) {{
            Plotly.Plots.resize(gd);
        }});
    }}
}}
</script>

</body>
</html>
"""


def get_metrics_values(runs_folders: List[str], runs_names: List[str], query_file: str | List[str]) -> Tuple[pd.DataFrame, List[str]]:
    if isinstance(query_file, str):
        query_file = [query_file]

    runs_dfs = {
        run_name: {
            qf.rsplit(".", 1)[0]: pd.read_excel(os.path.join(run_folder, qf), sheet_name="Values", header=0)
            for qf in query_file
        }
        for run_folder, run_name in zip(runs_folders, runs_names)
    }

    runs_metrics = []
    metrics_rows = []
    for run_name, query_files_dfs in runs_dfs.items():
        metrics = []
        for qf, df in query_files_dfs.items():
            start = df.index[df["Metric"] == "Iteration best"].tolist()[0]
            end = df.index[df["Metric"] == "Iteration last"].tolist()[0]

            # Assume best and last models to have the same metrics
            metrics_names = df["Metric"].iloc[start + 1 : end].tolist()
            metrics.append(set(metrics_names)) # Set should be redundant

            for i, metric in enumerate(metrics_names):
                best_index, last_index = (start + 1 + i, end + 1 + i) # Indexes for best and last model metrics
                best_model_values = df.iloc[best_index, 1:].values.tolist()
                last_model_values = df.iloc[last_index, 1:].values.tolist()
                for bv, lv in zip(best_model_values, last_model_values):
                    metrics_rows.append({"run": run_name, "query_file": qf, "metric": metric, "model": "best", "value": bv})
                    metrics_rows.append({"run": run_name, "query_file": qf, "metric": metric, "model": "last", "value": lv})

        # Get common metrics across all query files for this run
        runs_metrics.append(set.intersection(*metrics))

    # Get shared metrics across all runs
    shared_metrics = set.intersection(*runs_metrics)
    shared_metrics = sorted(shared_metrics)

    # Select metrics to plot
    selected_metrics = [f"Pixels Metrics/PCK@{k}" for k in range(1, 101)]

    # Filter metrics rows based on selected metrics
    metrics_values = pd.DataFrame(metrics_rows)
    metrics_values = metrics_values[metrics_values["metric"].isin(selected_metrics)]

    return metrics_values, selected_metrics

def get_plot(df: pd.DataFrame, metric: str, use_median: bool = True):
    query_files = sorted(df["query_file"].unique())

    for qf in query_files:
        subset = df[(df["metric"] == metric) & (df["query_file"] == qf)]

        runs = sorted(subset["run"].unique())
        run_name_map = {r: r for r in runs}
        runs_ordered = sorted(runs, key=lambda r: run_name_map[r])

        pck_run_values = []
        for run in runs_ordered:
            run_data = subset[subset["run"] == run]

            # Precompute hover stats
            vals = run_data["value"].dropna().astype(float).values

            def stats(arr):
                if arr.size == 0:
                    return dict(avg=None, med=None, q1=None, q3=None, mn=None, mx=None, n=0)
                q1 = float(pd.Series(arr).quantile(0.25))
                q3 = float(pd.Series(arr).quantile(0.75))
                return dict(
                    avg=float(arr.mean()),
                    med=float(pd.Series(arr).median()),
                    q1=q1,
                    q3=q3,
                    mn=float(arr.min()),
                    mx=float(arr.max()),
                    n=int(arr.size),
                )

            s = stats(vals)
            if use_median:
                pck_run_values.append(s["med"])
            else:
                pck_run_values.append(s["avg"])

    return pck_run_values

def plot_metrics(metrics_values: pd.DataFrame, metrics_names: List[str], run_names: List[str], use_last: bool = True, use_median: bool = True):
    if use_last:
        model_df = metrics_values[metrics_values["model"] == "last"]
    else:
        model_df = metrics_values[metrics_values["model"] == "best"]

    metrics_k = [
        int(metric.split("@")[-1]) for metric in metrics_names if "pck@" in metric.lower()
    ]
    
    # Sort metrics by k value
    metrics_names = [m for _, m in sorted(zip(metrics_k, metrics_names)) if "pck@" in m.lower()]

    # Generate and save plots
    pck_run_values = np.array([
        get_plot(model_df, metric, use_median=use_median)
        for metric in metrics_names
    ])

    pck_fig = go.Figure()
    for i in range(pck_run_values.shape[1]):
        pck_fig.add_trace(go.Scatter(x=metrics_names, y=pck_run_values[:, i], mode='lines+markers', name=run_names[i]))
    pck_fig.update_layout(title="PCK Average Comparison", xaxis_title="Metric", yaxis_title="PCK Average", xaxis_tickangle=-45)

    out_folder = os.path.expandvars("${WS_DIR}/runs/runs_plots")
    os.makedirs(out_folder, exist_ok=True)

    pck_fig_file = os.path.join(out_folder, "pck_comparison_plot.html")
    pck_fig.write_html(pck_fig_file, include_plotlyjs="cdn")
    print(f"PCK comparison plot saved to: {pck_fig_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-bd", default="${WS_DIR}/runs", help="Base directory to search for query files.")
    parser.add_argument("--use_last", "-l", action="store_true", help="Use the last model instead of the best model.")
    parser.add_argument("--use_median", "-m", action="store_true", help="Use median instead of average for PCK values.")
    args = parser.parse_args()

    # Select metric files
    metrics_filenames = glob(os.path.expandvars(os.path.join(args.base_dir, "**/*.xlsx")), recursive=True)
    metrics_filenames = sorted(list(set([os.path.basename(f) for f in metrics_filenames])))
    query_files, _ = selection_window(title="Select Metric Files", values=metrics_filenames)

    runs_folders, runs_names = select_runs_folders(base_dir=args.base_dir, query_file=query_files)
    if not runs_folders:
        print("No runs selected.")
        exit(0)

    metrics_values, metrics_names = get_metrics_values(runs_folders, runs_names, query_file=query_files)
    if metrics_values.empty:
        print("No metrics found in the selected runs.")
        exit(0)
    
    plot_metrics(metrics_values, metrics_names, runs_names, use_last=args.use_last, use_median=args.use_median)