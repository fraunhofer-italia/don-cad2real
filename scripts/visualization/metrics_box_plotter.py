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
    selected_metrics, _ = selection_window(title="Select Metrics", values=shared_metrics)

    # Filter metrics rows based on selected metrics
    metrics_values = pd.DataFrame(metrics_rows)
    metrics_values = metrics_values[metrics_values["metric"].isin(selected_metrics)]

    return metrics_values, selected_metrics

def get_plot(df: pd.DataFrame, model: str, metric: str, x_limits: Tuple[float, float]) -> go.Figure:
    query_files = sorted(df["query_file"].unique())
    num_query_files = len(query_files)

    subplot_titles = [f"{model.capitalize()} - {metric} ({qf})" for qf in query_files]

    fig = make_subplots(rows=num_query_files, cols=1, shared_xaxes=False, vertical_spacing=0.05, subplot_titles=subplot_titles)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for qf_idx, qf in enumerate(query_files):
        idx = qf_idx + 1 # 1-based index for Plotly subplots
        subset = df[(df["metric"] == metric) & (df["query_file"] == qf)]
        if subset.empty:
            fig.add_trace(go.Scatter(x=[], y=[], mode='text', text=["No data"], showlegend=False), row=idx, col=1)
            continue

        runs = sorted(subset["run"].unique())
        for i, run in enumerate(runs):
            run_data = subset[subset["run"] == run]
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Box(
                    x=run_data["value"],
                    name=run,
                    boxpoints="all",
                    jitter=0.3,
                    marker_color="rgba(0,0,0,0)",
                    marker_line_color="rgba(128,128,128,0.5)",
                    marker_line_width=1,
                    line_color=color,
                    orientation="h",
                    pointpos=0,
                    showlegend=(idx == 1),  # Only show once
                    legendgroup=run, # Links all traces with the same run name
                ),
                row=idx,
                col=1
            )
        
        yaxis_name = f'yaxis{idx}'
        fig.update_layout({
            yaxis_name: dict(
                categoryorder='array',
                categoryarray=runs[::-1]
            )
        })

    height_scale = 1000 if num_query_files > 5 else 500
    fig.update_layout(
        height=height_scale * num_query_files,
        title_text=f"{model.capitalize()} Model Metrics",
        showlegend=True,
        margin=dict(l=100, r=50, t=100, b=50),
    )

    # Compute global x-axis limits if not provided
    if x_limits is None:
        if "normalized" in metric.lower():
            x_limits = (0.0, 1.0)
        elif "pixels distance" in metric.lower():
            x_limits = (0.0, 200.0)
        elif "points distance" in metric.lower():
            x_limits = (0.0, 0.15)
        elif "auc" in metric.lower():
            x_limits = (0, 100)
        else:
            # Get min and max from the data
            values = df[(df["metric"] == metric)]["value"]
            x_limits = (values.min(), values.max())

    fig.update_xaxes(range=x_limits)

    return fig

def plot_metrics(metrics_values: pd.DataFrame, metrics_names: List[str], x_limits: Tuple[float, float], autoscale: bool = False):
    df_best = metrics_values[metrics_values["model"] == "best"]
    df_last = metrics_values[metrics_values["model"] == "last"]

    # If autoscale, override x_limits
    if autoscale:
        all_values = pd.concat([df_best["value"], df_last["value"]])
        x_limits = (all_values.min(), all_values.max())

    # Generate and save plots
    for model, df in [("best", df_best), ("last", df_last)]:
        metrics_figs = [
            get_plot(df, model, metric, x_limits)
            for metric in metrics_names
        ]

        # Save the plot to an HTML file
        folder = os.path.expandvars("${WS_DIR}/runs/runs_plots")
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, f"{model}_metrics_plot.html")
        
        tab_headers = ""
        tab_contents = ""
        for idx_metric, (metric, fig) in enumerate(zip(metrics_names, metrics_figs)):
            div_id = f"tab_{metric}"
            tab_headers += (
                f'<div class="tab" onclick="showTab(\'{div_id}\')">{metric}</div>\n'
            )
            tab_contents += (
                f'<div id="{div_id}" class="content {"active" if idx_metric == 0 else ""}">\n'
                f'{fig.to_html(include_plotlyjs="cdn", full_html=False)}\n'
                f'</div>\n'
            )
        final_html = html_base.format(tabs=tab_headers, contents=tab_contents)

        with open(file, "w") as f:
            f.write(final_html)
        
        print(f"Plot saved to: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-bd", default="${WS_DIR}/runs", help="Base directory to search for query files.")
    parser.add_argument("--limits", "-l", nargs=2, type=float, default=None, help="Set x-axis limits as two float values: min max. 'None' uses default limits.")
    parser.add_argument("--autoscale", "-as", action="store_true", help="Automatically scale x-axis limits based on data. Overrides --limits.")
    args = parser.parse_args()

    # Select metric files
    metrics_filenames = glob(os.path.expandvars(os.path.join(args.base_dir, "**/*.xlsx")), recursive=True)
    metrics_filenames = list(set([os.path.basename(f) for f in metrics_filenames]))
    query_files, _ = selection_window(title="Select Metric Files", values=metrics_filenames)

    runs_folders, runs_names = select_runs_folders(base_dir=args.base_dir, query_file=query_files)
    if not runs_folders:
        print("No runs selected.")
        exit(0)

    metrics_values, metrics_names = get_metrics_values(runs_folders, runs_names, query_file=query_files)
    if metrics_values.empty:
        print("No metrics found in the selected runs.")
        exit(0)
    
    plot_metrics(metrics_values, metrics_names, x_limits=args.limits, autoscale=args.autoscale)