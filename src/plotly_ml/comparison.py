from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl

from plotly_ml.utils.colors import to_rgba


def _as_polars(data: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    return data


def metrics_by_split_bar_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    *,
    metrics: Sequence[str],
    model_col: str = "model",
    split_col: str = "set",
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Metrics by Split",
    width: int = 1000,
    height: int = 450,
) -> go.Figure:
    """Grouped bar charts comparing one or more metrics across models and splits.

    Expected columns: `model_col`, `split_col`, and each metric in `metrics`.
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    for c in (model_col, split_col, *metrics):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")

    models = df.get_column(model_col).cast(pl.Utf8).unique().to_list()
    splits = df.get_column(split_col).cast(pl.Utf8).unique().to_list()

    fig = sp.make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[str(m) for m in metrics],
        horizontal_spacing=0.08,
    )

    split_color = {s: colors[i % len(colors)] for i, s in enumerate(splits)}

    for metric_i, metric in enumerate(metrics, start=1):
        for split in splits:
            sub = df.filter(pl.col(split_col) == split)
            # Align order
            vals_map = {
                r[model_col]: float(r[metric])
                for r in sub.select([model_col, metric]).to_dicts()
            }
            y = [vals_map.get(m, float("nan")) for m in models]
            fig.add_trace(
                go.Bar(
                    x=[str(m) for m in models],
                    y=y,
                    name=str(split),
                    marker=dict(color=to_rgba(split_color[split], 0.75)),
                    showlegend=(metric_i == 1),
                    legendgroup=str(split),
                ),
                row=1,
                col=metric_i,
            )

        fig.update_xaxes(title_text=model_col, row=1, col=metric_i)
        fig.update_yaxes(title_text=str(metric), row=1, col=metric_i)

    fig.update_layout(
        template=template,
        title=title,
        barmode="group",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    return fig


def learning_curve_plot(
    train_sizes: Sequence[float],
    train_scores: Sequence[float],
    val_scores: Sequence[float],
    *,
    train_std: Optional[Sequence[float]] = None,
    val_std: Optional[Sequence[float]] = None,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Learning Curve",
    width: int = 750,
    height: int = 500,
) -> go.Figure:
    """Plot a learning curve from precomputed scores."""
    if colors is None:
        colors = px.colors.qualitative.D3

    x = np.asarray(train_sizes, dtype=float)
    y_train = np.asarray(train_scores, dtype=float)
    y_val = np.asarray(val_scores, dtype=float)

    fig = go.Figure()

    def _band(
        xv: np.ndarray,
        yv: np.ndarray,
        sv: Optional[Sequence[float]],
        color: str,
        name: str,
    ):
        if sv is None:
            return
        s = np.asarray(sv, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=np.r_[xv, xv[::-1]],
                y=np.r_[yv - s, (yv + s)[::-1]],
                fill="toself",
                fillcolor=to_rgba(color, 0.18),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} ± std",
            )
        )

    _band(x, y_train, train_std, colors[0], "Train")
    _band(x, y_val, val_std, colors[1 % len(colors)], "Validation")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_train,
            mode="lines+markers",
            name="Train",
            line=dict(color=colors[0], width=2),
            marker=dict(color=to_rgba(colors[0], 0.85), size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_val,
            mode="lines+markers",
            name="Validation",
            line=dict(color=colors[1 % len(colors)], width=2),
            marker=dict(color=to_rgba(colors[1 % len(colors)], 0.85), size=7),
        )
    )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Training set size",
        yaxis_title="Score",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    return fig


def validation_curve_plot(
    param_values: Sequence[float],
    train_scores: Sequence[float],
    val_scores: Sequence[float],
    *,
    param_name: str = "param",
    train_std: Optional[Sequence[float]] = None,
    val_std: Optional[Sequence[float]] = None,
    xscale: str = "linear",
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Validation Curve",
    width: int = 750,
    height: int = 500,
) -> go.Figure:
    """Plot a validation curve from precomputed scores."""
    if colors is None:
        colors = px.colors.qualitative.D3

    x = np.asarray(param_values, dtype=float)
    y_train = np.asarray(train_scores, dtype=float)
    y_val = np.asarray(val_scores, dtype=float)

    if xscale not in {"linear", "log"}:
        raise ValueError("xscale must be 'linear' or 'log'")

    fig = go.Figure()

    def _band(
        xv: np.ndarray,
        yv: np.ndarray,
        sv: Optional[Sequence[float]],
        color: str,
        name: str,
    ):
        if sv is None:
            return
        s = np.asarray(sv, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=np.r_[xv, xv[::-1]],
                y=np.r_[yv - s, (yv + s)[::-1]],
                fill="toself",
                fillcolor=to_rgba(color, 0.18),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} ± std",
            )
        )

    _band(x, y_train, train_std, colors[0], "Train")
    _band(x, y_val, val_std, colors[1 % len(colors)], "Validation")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_train,
            mode="lines+markers",
            name="Train",
            line=dict(color=colors[0], width=2),
            marker=dict(color=to_rgba(colors[0], 0.85), size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_val,
            mode="lines+markers",
            name="Validation",
            line=dict(color=colors[1 % len(colors)], width=2),
            marker=dict(color=to_rgba(colors[1 % len(colors)], 0.85), size=7),
        )
    )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title=param_name,
        yaxis_title="Score",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    fig.update_xaxes(type="log" if xscale == "log" else "linear")
    return fig
