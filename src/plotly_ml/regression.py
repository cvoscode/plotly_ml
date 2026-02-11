import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from typing import Optional, Union

from plotly_ml.utils.metrics import rmse, r2_score, mae, bias, var, count
from plotly_ml.utils.colors import to_rgba


def _as_polars(data: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    return data


def _clean_regression_xy(data: pl.DataFrame, y_true: str, y_pred: str) -> pl.DataFrame:
    if y_true not in data.columns or y_pred not in data.columns:
        raise ValueError(f"Data must contain columns '{y_true}' and '{y_pred}'")
    return (
        data.select(
            [
                pl.col(y_true).cast(pl.Float64).alias(y_true),
                pl.col(y_pred).cast(pl.Float64).alias(y_pred),
            ]
        )
        .drop_nulls()
        .filter(pl.col(y_true).is_finite() & pl.col(y_pred).is_finite())
    )


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
    data = _as_polars(data)
    if split_column not in data.columns:
        raise ValueError(f"split_column '{split_column}' not found")
    if "y_true" not in data.columns or "y_pred" not in data.columns:
        raise ValueError("Data must contain columns 'y_true' and 'y_pred'")

    # Clean inputs (drop nulls and non-finite) so metrics/plots (esp. R²) are reliable.
    data = (
        data.select(
            [
                pl.col(split_column),
                pl.col("y_true").cast(pl.Float64).alias("y_true"),
                pl.col("y_pred").cast(pl.Float64).alias("y_pred"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("y_true").is_finite() & pl.col("y_pred").is_finite())
    )
    if data.height == 0:
        raise ValueError("No valid rows after cleaning y_true/y_pred")

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
    min_val = float(min(data["y_true"].min(), data["y_pred"].min()))
    max_val = float(max(data["y_true"].max(), data["y_pred"].max()))
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
                marker=dict(size=4, opacity=0.7, color=marker_color),
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
                marker=dict(size=4, opacity=0.7, color=marker_color),
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


def residuals_vs_fitted_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    *,
    split_column: Optional[str] = None,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Residuals vs Fitted",
    width: int = 750,
    height: int = 500,
    marginal_y: Optional[str] = None,
) -> go.Figure:
    """Scatter plot of residuals vs fitted values."""
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: list[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = list(df.partition_by(split_column))

    # Option: use plotly express with marginal distributions on y-axis
    if marginal_y is not None:
        # marginal_y can be 'box', 'violin', 'rug', or 'histogram'
        # build a pandas DataFrame with residuals and fitted values
        if isinstance(data, pl.DataFrame):
            df_pd = data.to_pandas()
        else:
            df_pd = pd.DataFrame(data)
        # compute residual column
        df_pd = df_pd.copy()
        df_pd["__resid__"] = df_pd[y_true] - df_pd[y_pred]
        color_col = (
            split_column
            if (split_column is not None and split_column in df_pd.columns)
            else None
        )
        fig = px.scatter(
            df_pd,
            x=y_pred,
            y="__resid__",
            color=color_col,
            marginal_y=marginal_y,
            template=template,
        )
        # styling: only update scatter traces (histogram traces don't accept marker.size)
        for tr in fig.data:
            if isinstance(tr, (go.Scatter, go.Scattergl)):
                tr.update(marker=dict(size=5, opacity=0.7))
        fig.add_hline(y=0, line=dict(color="black", dash="dash"))
        fig.update_layout(
            title=title,
            xaxis_title="Fitted (y_pred)",
            yaxis_title="Residual (y_true - y_pred)",
            width=width,
            height=height,
            autosize=False,
            legend_title_text=None,
        )
        return fig

    # standard non-marginal scatter
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Zero",
            showlegend=False,
        )
    )

    for i, split_df in enumerate(splits):
        name = (
            "All"
            if split_column is None
            else str(split_df.get_column(split_column).first())
        )
        d = _clean_regression_xy(split_df, y_true, y_pred)
        if d.height == 0:
            continue
        y_t = d.get_column(y_true).to_numpy()
        y_p = d.get_column(y_pred).to_numpy()
        resid = y_t - y_p
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scattergl(
                x=y_p,
                y=resid,
                mode="markers",
                name=name,
                marker=dict(size=5, color=to_rgba(color, 0.65)),
            )
        )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Fitted (y_pred)",
        yaxis_title="Residual (y_true - y_pred)",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    return fig


def qq_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    *,
    split_column: Optional[str] = None,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Q-Q Plot (Residuals)",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Normal Q-Q plot of residuals."""
    if colors is None:
        colors = px.colors.qualitative.D3

    try:
        from scipy.stats import norm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy is required for qq_plot") from e

    df = _as_polars(data)
    if split_column is None:
        splits: list[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = list(df.partition_by(split_column))

    fig = go.Figure()

    for i, split_df in enumerate(splits):
        name = (
            "All"
            if split_column is None
            else str(split_df.get_column(split_column).first())
        )
        d = _clean_regression_xy(split_df, y_true, y_pred)
        if d.height < 3:
            continue
        resid = (d.get_column(y_true) - d.get_column(y_pred)).to_numpy()
        resid = resid[np.isfinite(resid)]
        resid = np.sort(resid)
        n = resid.size
        if n < 3:
            continue

        p = (np.arange(1, n + 1) - 0.5) / n
        theo = norm.ppf(p)

        # Fit a reference line (robust enough for a visual guide)
        slope, intercept = np.polyfit(theo, resid, 1)
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=theo,
                y=resid,
                mode="markers",
                name=name,
                marker=dict(size=6, color=to_rgba(color, 0.7)),
            )
        )
        x_line = np.array([theo.min(), theo.max()])
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{name} · Ref",
                showlegend=False,
                line=dict(color=color, dash="dash"),
            )
        )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Theoretical quantiles (Normal)",
        yaxis_title="Residual quantiles",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    return fig


def binned_actual_vs_pred_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    *,
    split_column: Optional[str] = None,
    n_bins: int = 10,
    ci: float = 0.95,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Binned Actual vs Predicted",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Plot binned mean actual vs mean predicted with a confidence interval."""
    if colors is None:
        colors = px.colors.qualitative.D3

    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1")
    z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

    df = _as_polars(data)
    if split_column is None:
        splits: list[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = list(df.partition_by(split_column))

    # Determine sensible axis limits from the data so the ideal line spans the plotted range
    # Use the whole dataset (not just per-split) to avoid tiny dangling segments.
    try:
        global_min = float(
            np.nanmin(
                np.r_[
                    df.get_column(y_pred).to_numpy(), df.get_column(y_true).to_numpy()
                ]
            )
        )
        global_max = float(
            np.nanmax(
                np.r_[
                    df.get_column(y_pred).to_numpy(), df.get_column(y_true).to_numpy()
                ]
            )
        )
    except Exception:
        global_min, global_max = 0.0, 1.0

    fig = go.Figure()
    # Ideal line spanning data min/max
    fig.add_trace(
        go.Scatter(
            x=[global_min, global_max],
            y=[global_min, global_max],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Ideal",
            showlegend=False,
        )
    )

    for i, split_df in enumerate(splits):
        name = (
            "All"
            if split_column is None
            else str(split_df.get_column(split_column).first())
        )
        d = _clean_regression_xy(split_df, y_true, y_pred)
        if d.height < 3:
            continue

        # Bin by predicted value quantiles
        q = np.linspace(0, 1, int(n_bins) + 1)
        edges = np.quantile(d.get_column(y_pred).to_numpy(), q)
        # Avoid duplicated edges
        edges = np.unique(edges)
        if edges.size < 3:
            continue

        y_p = d.get_column(y_pred).to_numpy()
        y_t = d.get_column(y_true).to_numpy()
        bin_idx = np.digitize(y_p, edges[1:-1], right=True)

        xs: list[float] = []
        ys: list[float] = []
        err: list[float] = []
        for b in range(edges.size - 1):
            mask = bin_idx == b
            if not np.any(mask):
                continue
            xs.append(float(np.mean(y_p[mask])))
            mean = float(np.mean(y_t[mask]))
            ys.append(mean)
            std = float(np.std(y_t[mask], ddof=1)) if np.sum(mask) > 1 else 0.0
            se = std / max(1.0, np.sqrt(float(np.sum(mask))))
            err.append(z * se)

        if not xs:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(color=to_rgba(color, 0.8), size=7),
                error_y=dict(type="data", array=err, visible=True),
            )
        )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Mean predicted (per bin)",
        yaxis_title="Mean actual (per bin)",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    return fig


def residuals_by_group_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    group: str = "group",
    *,
    split_column: Optional[str] = None,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Residuals by Group",
    width: int = 900,
    height: int = 450,
) -> go.Figure:
    """Violin plot of residual distributions by group."""
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if group not in df.columns:
        raise ValueError(f"group column '{group}' not found")

    select_cols = [
        pl.col(group).cast(pl.Utf8).alias(group),
        pl.col(y_true),
        pl.col(y_pred),
    ]
    if split_column is not None:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        select_cols.append(pl.col(split_column).cast(pl.Utf8).alias(split_column))

    base = df.select(select_cols)
    base = base.drop_nulls()

    if split_column is None:
        splits: list[pl.DataFrame] = [base]
    else:
        splits = list(base.partition_by(split_column))

    fig = go.Figure()
    for i, split_df in enumerate(splits):
        split_name = None
        if split_column is not None:
            split_name = str(split_df.get_column(split_column).first())

        d = (
            split_df.with_columns(
                (
                    pl.col(y_true).cast(pl.Float64) - pl.col(y_pred).cast(pl.Float64)
                ).alias("__resid__")
            )
            .drop_nulls()
            .filter(pl.col("__resid__").is_finite())
        )

        color = colors[i % len(colors)]

        # If there are exactly two groups, draw half-violins facing each other for a compact comparison
        groups = d.get_column(group).to_list()
        try:
            unique_groups = list(dict.fromkeys(groups))
        except Exception:
            unique_groups = list(set(groups))

        if len(unique_groups) == 2:
            g0, g1 = unique_groups[0], unique_groups[1]
            vals0 = d.filter(pl.col(group) == g0).get_column("__resid__").to_numpy()
            vals1 = d.filter(pl.col(group) == g1).get_column("__resid__").to_numpy()

            # Use distinct colors per group and draw half-violins that overlay
            color0 = colors[0 % len(colors)]
            color1 = colors[1 % len(colors)]

            # Place both halves at the same categorical x so they directly face each other.
            pair_label = f"{g0} · {g1}"
            fig.add_trace(
                go.Violin(
                    x=[pair_label] * len(vals0),
                    y=vals0,
                    name=str(g0),
                    side="negative",
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.75,
                    line_color=color0,
                    fillcolor=to_rgba(color0, 0.35),
                    points=False,
                    width=0.9,
                    offsetgroup="paired",
                    alignmentgroup="paired",
                )
            )

            fig.add_trace(
                go.Violin(
                    x=[pair_label] * len(vals1),
                    y=vals1,
                    name=str(g1),
                    side="positive",
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.75,
                    line_color=color1,
                    fillcolor=to_rgba(color1, 0.35),
                    points=False,
                    width=0.9,
                    offsetgroup="paired",
                    alignmentgroup="paired",
                )
            )

            # ensure x-axis shows a readable label for the pair
            fig.update_xaxes(
                tickmode="array", tickvals=[pair_label], ticktext=[f"{g0} vs {g1}"]
            )
        else:
            # Fall back to one violin trace that creates separate violins per category
            fig.add_trace(
                go.Violin(
                    x=d.get_column(group).to_list(),
                    y=d.get_column("__resid__").to_numpy(),
                    name="All" if split_name is None else split_name,
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.75,
                    line_color=color,
                    fillcolor=to_rgba(color, 0.35),
                    points=False,
                )
            )

    fig.update_layout(
        template=template,
        title=title,
        width=width,
        height=height,
        autosize=False,
        violinmode="group",
        violingap=0.0,
        xaxis_title=group,
        yaxis_title="Residual (y_true - y_pred)",
        legend_title_text=None,
    )
    return fig
