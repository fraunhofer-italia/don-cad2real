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
            qf.rsplit(".", 1)[0]: pd.read_excel(os.path.join(run_folder, qf), sheet_name="Metrics Stats", header=0)
            for qf in query_file
        }
        for run_folder, run_name in zip(runs_folders, runs_names)
    }

    runs_metrics = []
    metrics_rows = []
    for run_name, query_files_dfs in runs_dfs.items():
        run_base_names = []
        for qf, df in query_files_dfs.items():
            start = df.index[df["Metric"] == "Iteration best"].tolist()[0]
            end = df.index[df["Metric"] == "Iteration last"].tolist()[0]

            # Assume best and last models to have the same metrics
            metrics_names = df["Metric"].iloc[start + 1 : end].tolist()
            
            # Store metrics for this query file
            metrics_base_names = set()
            for i, metric in enumerate(metrics_names):
                # Consider only object-specific metrics
                if "[obj_" not in metric:
                    continue
                
                best_index, last_index = (start + 1 + i, end + 1 + i) # Indexes for best and last model metrics
                best_model_values = df.iloc[best_index, 2:].values.tolist()
                last_model_values = df.iloc[last_index, 2:].values.tolist()
                
                base_name = metric.split("/", 1)[-1][:-1] # Remove the trailing "]"
                object_name = metric.split("/", 1)[0].replace("[", "").replace("]", "")
                metrics_base_names.add(base_name)
        
                for model, m_stats in [("best", best_model_values), ("last", last_model_values)]:
                    metrics_rows.append({
                        "run": run_name, "query_file": qf, "metric": base_name, "obj": object_name, "model": model, 
                        "avg": m_stats[0], "std": m_stats[1], "median": m_stats[2], "min": m_stats[3], "max": m_stats[4]
                    })

                run_base_names.append(metrics_base_names)

        # Get common metrics across all query files for this run
        runs_metrics.append(set.intersection(*run_base_names))

    # Get shared metrics across all runs
    shared_metrics = set.intersection(*runs_metrics)
    shared_metrics = sorted(shared_metrics)

    # Select metrics to plot
    selected_metrics, _ = selection_window(title="Select Metrics", values=shared_metrics)
    
    # Filter metrics rows based on selected metrics
    metrics_values = pd.DataFrame(metrics_rows)
    metrics_values = metrics_values[metrics_values["metric"].isin(selected_metrics)]
    return metrics_values, selected_metrics

def get_plot(df: pd.DataFrame, model: str, metric: str, target_stat: str = "avg") -> go.Figure:
    query_files = sorted(df["query_file"].unique())
    num_query_files = len(query_files)

    subplot_titles = [f"{model.capitalize()} - {metric} ({qf})" for qf in query_files]

    fig = make_subplots(rows=num_query_files, cols=1, shared_xaxes=False, vertical_spacing=0.05, subplot_titles=subplot_titles)

    # Global y-limits
    metric_subset = df[df["metric"] == metric]
    y_min = (metric_subset[target_stat] - metric_subset["std"]).min()
    y_max = (metric_subset[target_stat] + metric_subset["std"]).max()
    if pd.isna(y_min) or pd.isna(y_max):
        y_limits = (0.0, 1.0)
    else:
        y_limits = (float(y_min), float(y_max))
      
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for qf_idx, qf in enumerate(query_files):
        idx = qf_idx + 1 # 1-based index for Plotly subplots
        qf_subset = metric_subset[metric_subset["query_file"] == qf]
        if qf_subset.empty:
            fig.add_trace(go.Scatter(x=[], y=[], mode='text', text=["No data"], showlegend=False), row=idx, col=1)
            continue

        runs = sorted(qf_subset["run"].unique())
        objects_names = sorted(qf_subset["obj"].unique())

        for r_idx, run in enumerate(runs):
            run_df = qf_subset[(qf_subset["run"] == run)].set_index("obj")

            # Order points by the provided metrics list; drop missing metrics for this run/qf
            x_obj, y_avg, y_std = [], [], []
            for obj in objects_names:
                if obj in run_df.index:
                    y_avg.append(float(run_df.loc[obj, target_stat]))
                    y_std.append(float(run_df.loc[obj, "std"]))
                    x_obj.append(obj)
            if not x_obj:
                continue

            color = colors[r_idx % len(colors)]

            # Error band (avg ± std) as a filled polygon in y-direction
            upper = [a + s for a, s in zip(y_avg, y_std)]
            lower = [a - s for a, s in zip(y_avg, y_std)]

            fig.add_trace(
                go.Scatter(
                    x=x_obj + x_obj[::-1],
                    y=upper + lower[::-1],
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.10,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=run,
                ),
                row=idx,
                col=1,
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=x_obj,
                    y=y_avg,
                    mode="lines+markers",
                    name=run,
                    legendgroup=run,
                    line=dict(color=color),
                    marker=dict(color=color),
                    customdata=list(zip(y_std)),
                    hovertemplate=(
                        "run=%{fullData.name}<br>"
                        "value=%{y}<br>"
                        "std=%{customdata[0]:.6g}<extra></extra>"
                    ),
                    showlegend=(idx == 1),  # legend only once
                ),
                row=idx,
                col=1,
            )

            fig.update_yaxes(range=y_limits, row=idx, col=1)

    fig.update_layout(
        height=max(500 * num_query_files, 420),
        template="plotly_white",
        legend_title_text="Run",
        margin=dict(l=80, r=30, t=60, b=40),
    )

    return fig

def plot_metrics(metrics_values: pd.DataFrame, metrics_names: List[str], use_average: bool = False):
    df_best = metrics_values[metrics_values["model"] == "best"]
    df_last = metrics_values[metrics_values["model"] == "last"]
    
    # Generate and save plots
    for model, df in [("best", df_best), ("last", df_last)]:
        metrics_figs = [
            get_plot(df, model, metric, target_stat="avg" if use_average else "median")
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
    parser.add_argument("--use_average", "-m", action="store_true", help="Visualize average instead of median for metrics.")
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
    
    plot_metrics(metrics_values, metrics_names, use_average=args.use_average)