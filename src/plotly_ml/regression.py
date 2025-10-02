import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from typing import Union

from plotly_ml.utils.metrics import rmse, r2_score, mae, bias, var, count
from plotly_ml.utils.colors import to_rgba


def regression_evaluation_plot(
    data: Union[pl.DataFrame, pd.DataFrame] = None,
    y: list[str] = None,
    split_column: str = "set",
    template="plotly_white",
    colors: list = None,
):
    """Create a comprehensive regression model evaluation plot with multiple subplots.

    This function creates an interactive visualization that includes:
    - Prediction error scatter plot with ideal line
    - Marginal distributions using violin plots
    - Residuals plot
    - Summary metrics table (R², MAE, RMSE, Bias, Variance, Sample size)

    Args:
        data (Union[pl.DataFrame,pd.DataFrame]): DataFrame containing true values, predictions and split information.
            Must contain columns 'y_true' and 'y_pred'.
        y (str, optional): Name of the target variable. Not currently used. Defaults to None.
        split_column (str, optional): Name of the column containing split information (e.g., 'train'/'test').
            Defaults to 'set'.
        template (str, optional): Plotly template to use. Defaults to 'plotly_white'.
        colors (list, optional): List of colors to use for different splits.
            If None, uses Plotly's default D3 qualitative color scale.

    Returns:
        go.Figure: A plotly figure object containing the regression evaluation plots.
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    specs = [
        [
            {"type": "xy"},
            {"type": "xy"},
        ],  # row 1: left = xy, right = domain (table goes here)
        [{"type": "xy"}, {"type": "xy"}],  # row 2
        [{"type": "xy"}, {"type": "xy"}],  # spacer row (still xy but empty)
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "domain"}, {"type": "xy"}],  # bottom row
    ]

    # Build a 4-row layout where row 3 is a spacer to control the gap between rows 2 and 4
    fig = sp.make_subplots(
        rows=6,
        cols=2,
        column_widths=[0.75, 0.25],
        row_heights=[0.1, 0.4, 0.1, 0.2, 0.1, 0.1],
        shared_xaxes=True,
        shared_yaxes="rows",
        vertical_spacing=0,
        horizontal_spacing=0,
        specs=specs,
        subplot_titles=(
            "Prediction Error",
            None,
            None,
            None,
            None,
            None,
            "Residuals",
            None,
        ),
    )

    # Add reference lines
    min_val = min(data["y_true"].min(), data["y_pred"].min())
    max_val = max(data["y_true"].max(), data["y_pred"].max())
    fig.add_trace(
        go.Scattergl(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Line",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=[min_val, max_val],
            y=[0, 0],
            mode="lines",
            name="Zero Error Line",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    colors = px.colors.qualitative.D3 if not colors else colors

    # Prepare containers for metrics
    split_names = []
    r2_list = []
    mae_list = []
    rmse_list = []
    bias_list = []
    var_resid_list = []
    n_list = []

    # --- Add Traces to the Figure and collect metrics ---
    for i, split in enumerate(data.partition_by(split_column)):
        split_name = split[split_column].first()
        y_true = split["y_true"]
        y_pred = split["y_pred"]

        residuals = y_true - y_pred

        # Compute metrics for this split
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = np.nan
        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        bias_val = bias(y_true, y_pred)
        var_resid = var(y_true, y_pred)
        n_val = count(y_true, y_pred)

        # Store metrics (keep numeric values for later formatting)
        split_names.append(str(split_name))
        r2_list.append(r2)
        mae_list.append(mae_val)
        rmse_list.append(rmse_val)
        bias_list.append(bias_val)
        var_resid_list.append(var_resid)
        n_list.append(n_val)

        line_color = colors[i % len(colors)]
        fill_color = to_rgba(line_color, alpha=0.25)
        marker_color = to_rgba(line_color, alpha=0.7)

        # 1. Prediction Error Scatter Plot (middle-left subplot)
        fig.add_trace(
            go.Scattergl(
                x=y_true,
                y=y_pred,
                mode="markers",
                name=f"{split_name}",
                marker=dict(size=8, opacity=0.7, color=marker_color),
                legendgroup=f"{split_name}",
            ),
            row=2,
            col=1,
        )

        # 2. Residuals Scatter Plot (bottom-left subplot)
        fig.add_trace(
            go.Scattergl(
                x=y_true,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(size=8, opacity=0.7, color=marker_color),
                legendgroup=f"{split_name}",
                showlegend=False,
            ),
            row=4,
            col=1,
        )

        # --- Add Marginal Violin Plots ---
        fig.add_trace(
            go.Violin(
                x=y_true,
                orientation="h",
                y=[0] * len(y_true),
                name=f"{split_name}",
                side="positive",
                box_visible=True,
                meanline_visible=True,
                fillcolor=fill_color,
                opacity=0.6,
                line_color=line_color,
                legendgroup=f"{split_name}",
                showlegend=False,
                scalegroup="same",
                scalemode="width",
                width=0.6,
                offsetgroup="overlay",
                alignmentgroup="overlay",
                points=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Violin(
                y=y_pred,
                name=f"{split_name}",
                orientation="v",
                x=[0] * len(y_pred),
                side="positive",
                box_visible=True,
                meanline_visible=True,
                fillcolor=fill_color,
                opacity=0.6,
                line_color=line_color,
                legendgroup=f"{split_name}",
                showlegend=False,
                scalegroup="same",
                scalemode="width",
                width=0.6,
                offsetgroup="overlay",
                alignmentgroup="overlay",
                points=False,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Violin(
                y=residuals,
                name=f"{split_name}",
                x=[0] * len(y_pred),
                side="positive",
                box_visible=True,
                meanline_visible=True,
                fillcolor=fill_color,
                opacity=0.6,
                line_color=line_color,
                legendgroup=f"{split_name}",
                showlegend=False,
                scalegroup="same",
                scalemode="width",
                width=0.6,
                offsetgroup="overlay",
                alignmentgroup="overlay",
                points=False,
            ),
            row=4,
            col=2,
        )

    # --- Add compact metrics table (top-right) ---
    # Format numeric columns to short strings for compact display
    def _short_strings(arr):
        return [
            "{0}".format(
                "{:.3f}".format(x)
                if isinstance(x, (int, float, np.floating, np.integer))
                and not np.isnan(x)
                else "nan"
            )
            for x in arr
        ]

    header_vals = ["Split", "R2", "MAE", "RMSE", "Bias", "VarRes", "N"]
    cell_vals = [
        split_names,
        _short_strings(r2_list),
        _short_strings(mae_list),
        _short_strings(rmse_list),
        _short_strings(bias_list),
        _short_strings(var_resid_list),
        [str(x) for x in n_list],
    ]

    # Add the table into the top-right subplot slot
    fig.add_trace(
        go.Table(
            header=dict(
                values=header_vals,
                fill_color="lightgrey",
                align="left",
                font=dict(size=9),
            ),
            cells=dict(values=cell_vals, align="left", font=dict(size=9)),
        ),
        row=6,
        col=1,
    )

    # --- Update Layout and Axes ---
    fig.update_layout(
        height=900,
        width=700,
        title_text="Regression Model Analysis",
        template=template,
    )

    # Main plot axes
    fig.update_yaxes(title_text="Predicted Values", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=4, col=1)
    fig.update_xaxes(title_text="True Values", row=4, col=1)

    # Hide unnecessary ticks on marginal plots and table subplot
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=True, row=3, col=2)
    fig.update_xaxes(visible=True, showticklabels=True, row=4, col=1)
    fig.update_xaxes(matches="x4", row=2, col=2)
    fig.update_layout(violinmode="overlay")
    return fig
