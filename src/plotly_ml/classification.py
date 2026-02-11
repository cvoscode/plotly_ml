from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl

from plotly_ml.utils.colors import to_rgba


ArrayLike = Union[Sequence[float], np.ndarray]


_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")


def _apply_square_axes(
    fig: go.Figure,
    *,
    width: Optional[int],
    height: Optional[int],
    xaxis: str = "x",
    yaxis: str = "y",
) -> None:
    if width is not None or height is not None:
        fig.update_layout(
            width=width,
            height=height,
            autosize=False,
        )

    # Keep the plotting area square even if the overall figure is wider
    # (e.g., because the legend sits to the right).
    fig.update_layout({f"{yaxis}axis": {"scaleanchor": xaxis, "scaleratio": 1}})
    fig.update_layout({f"{xaxis}axis": {"constrain": "domain"}})


def _as_polars(data: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    return data


def _clean_binary_targets(y_true: np.ndarray, pos_label: object) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        y = y.reshape(-1)
    return (y == pos_label).astype(int)


def _clean_scores(y_score: np.ndarray) -> np.ndarray:
    s = np.asarray(y_score, dtype=float)
    if s.ndim != 1:
        s = s.reshape(-1)
    mask = np.isfinite(s)
    return s[mask], mask


def _filter_xy(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    s, score_mask = _clean_scores(y_score)
    y = np.asarray(y_true)
    if y.ndim != 1:
        y = y.reshape(-1)
    y = y[score_mask]
    return y, s


@dataclass(frozen=True)
class _Curve:
    x: np.ndarray
    y: np.ndarray
    auc: float


def _roc_curve_binary(y_true01: np.ndarray, y_score: np.ndarray) -> _Curve:
    y_true01, y_score = _filter_xy(y_true01, y_score)
    if y_true01.size == 0:
        return _Curve(x=np.array([]), y=np.array([]), auc=float("nan"))

    y_true01 = np.asarray(y_true01, dtype=int)
    if not set(np.unique(y_true01)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1) after pos_label mapping")

    pos = int(np.sum(y_true01 == 1))
    neg = int(np.sum(y_true01 == 0))
    if pos == 0 or neg == 0:
        return _Curve(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), auc=float("nan"))

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true01[order]
    s_sorted = y_score[order]

    distinct = np.where(np.diff(s_sorted))[0]
    the_idx = np.r_[distinct, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[the_idx]
    fps = (the_idx + 1) - tps

    tpr = tps / pos
    fpr = fps / neg

    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]

    auc = float(np.trapezoid(tpr, fpr))
    return _Curve(x=fpr, y=tpr, auc=auc)


def _pr_curve_binary(y_true01: np.ndarray, y_score: np.ndarray) -> _Curve:
    y_true01, y_score = _filter_xy(y_true01, y_score)
    if y_true01.size == 0:
        return _Curve(x=np.array([]), y=np.array([]), auc=float("nan"))

    y_true01 = np.asarray(y_true01, dtype=int)
    if not set(np.unique(y_true01)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1) after pos_label mapping")

    pos = int(np.sum(y_true01 == 1))
    if pos == 0:
        return _Curve(x=np.array([0.0, 1.0]), y=np.array([1.0, 0.0]), auc=float("nan"))

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true01[order]
    s_sorted = y_score[order]

    distinct = np.where(np.diff(s_sorted))[0]
    the_idx = np.r_[distinct, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[the_idx]
    fps = (the_idx + 1) - tps
    precision = np.divide(
        tps, (tps + fps), out=np.ones_like(tps, dtype=float), where=(tps + fps) != 0
    )
    recall = tps / pos

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    ap = float(np.trapezoid(precision, recall))
    return _Curve(x=recall, y=precision, auc=ap)


def _infer_classes_from_y(y_true: np.ndarray) -> list[object]:
    # Stable, first-seen ordering
    seen: dict[object, None] = {}
    for v in y_true:
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def roc_curve_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_score: str = "y_score",
    *,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "ROC Curve",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Plot ROC curves for binary or multiclass (one-vs-rest) classification.

    Binary usage expects columns `y_true` and `y_score`.
    Multiclass usage expects `proba_columns` containing per-class probabilities.
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)

    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Chance",
            showlegend=True,
        )
    )

    split_dfs = list(splits)
    for split_i, split_df in enumerate(split_dfs):
        split_name = None
        if split_column is not None:
            split_name = str(split_df.get_column(split_column).first())

        y_arr = split_df.get_column(y_true).to_numpy()

        if proba_columns is None:
            s_arr = split_df.get_column(y_score).to_numpy()
            y01 = _clean_binary_targets(y_arr, pos_label=pos_label)
            curve = _roc_curve_binary(y01, s_arr)
            # Keep the same color for the same curve type across splits;
            # differentiate splits via dash style.
            line_color = colors[0]
            dash = _DASHES[split_i % len(_DASHES)]
            name = "ROC" if split_name is None else f"{split_name}"
            auc_txt = "" if np.isnan(curve.auc) else f" (AUC={curve.auc:.3f})"
            fig.add_trace(
                go.Scatter(
                    x=curve.x,
                    y=curve.y,
                    mode="lines",
                    name=f"{name}{auc_txt}",
                    line=dict(color=line_color, width=2, dash=dash),
                )
            )
            continue

        # Multiclass one-vs-rest
        if any(c not in split_df.columns for c in proba_columns):
            missing = [c for c in proba_columns if c not in split_df.columns]
            raise ValueError(f"Missing proba columns: {missing}")

        if classes is None:
            inferred = _infer_classes_from_y(y_arr)
        else:
            inferred = list(classes)
        if len(inferred) != len(proba_columns):
            raise ValueError(
                "classes and proba_columns must have the same length for multiclass"
            )

        class_color_map = {
            label: colors[i % len(colors)] for i, label in enumerate(inferred)
        }

        for class_label, col in zip(inferred, proba_columns):
            s_arr = split_df.get_column(col).to_numpy()
            y01 = _clean_binary_targets(y_arr, pos_label=class_label)
            curve = _roc_curve_binary(y01, s_arr)
            line_color = class_color_map[class_label]
            dash = _DASHES[split_i % len(_DASHES)]

            base = f"{class_label}"
            if split_name is not None:
                base = f"{split_name} · {class_label}"
            auc_txt = "" if np.isnan(curve.auc) else f" (AUC={curve.auc:.3f})"
            fig.add_trace(
                go.Scatter(
                    x=curve.x,
                    y=curve.y,
                    mode="lines",
                    name=f"{base}{auc_txt}",
                    line=dict(color=line_color, width=2, dash=dash),
                )
            )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title_text=None,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _apply_square_axes(fig, width=width, height=height)
    return fig


def precision_recall_curve_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_score: str = "y_score",
    *,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Precision-Recall Curve",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Plot precision-recall curves for binary or multiclass (one-vs-rest) classification."""
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)

    fig = go.Figure()
    split_dfs = list(splits)
    for split_i, split_df in enumerate(split_dfs):
        split_name = None
        if split_column is not None:
            split_name = str(split_df.get_column(split_column).first())
        y_arr = split_df.get_column(y_true).to_numpy()

        if proba_columns is None:
            s_arr = split_df.get_column(y_score).to_numpy()
            y01 = _clean_binary_targets(y_arr, pos_label=pos_label)
            curve = _pr_curve_binary(y01, s_arr)

            prevalence = float(np.mean(y01)) if y01.size else float("nan")
            if np.isfinite(prevalence):
                dash = _DASHES[split_i % len(_DASHES)]
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[prevalence, prevalence],
                        mode="lines",
                        line=dict(color="black", dash=dash),
                        name=(
                            "Baseline"
                            if split_name is None
                            else f"{split_name} · Baseline"
                        ),
                        showlegend=True,
                    )
                )

            line_color = colors[0]
            dash = _DASHES[split_i % len(_DASHES)]
            name = "PR" if split_name is None else f"{split_name}"
            ap_txt = "" if np.isnan(curve.auc) else f" (AP={curve.auc:.3f})"
            fig.add_trace(
                go.Scatter(
                    x=curve.x,
                    y=curve.y,
                    mode="lines",
                    name=f"{name}{ap_txt}",
                    line=dict(color=line_color, width=2, dash=dash),
                )
            )
            continue

        if any(c not in split_df.columns for c in proba_columns):
            missing = [c for c in proba_columns if c not in split_df.columns]
            raise ValueError(f"Missing proba columns: {missing}")

        if classes is None:
            inferred = _infer_classes_from_y(y_arr)
        else:
            inferred = list(classes)
        if len(inferred) != len(proba_columns):
            raise ValueError(
                "classes and proba_columns must have the same length for multiclass"
            )

        class_color_map = {
            label: colors[i % len(colors)] for i, label in enumerate(inferred)
        }

        for class_label, col in zip(inferred, proba_columns):
            s_arr = split_df.get_column(col).to_numpy()
            y01 = _clean_binary_targets(y_arr, pos_label=class_label)
            curve = _pr_curve_binary(y01, s_arr)
            line_color = class_color_map[class_label]
            dash = _DASHES[split_i % len(_DASHES)]

            base = f"{class_label}"
            if split_name is not None:
                base = f"{split_name} · {class_label}"
            ap_txt = "" if np.isnan(curve.auc) else f" (AP={curve.auc:.3f})"
            fig.add_trace(
                go.Scatter(
                    x=curve.x,
                    y=curve.y,
                    mode="lines",
                    name=f"{base}{ap_txt}",
                    line=dict(color=line_color, width=2, dash=dash),
                )
            )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_title_text=None,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _apply_square_axes(fig, width=width, height=height)
    return fig


def discrimination_threshold_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_score: str = "y_score",
    *,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    n_thresholds: int = 101,
    metrics: Sequence[str] = ("precision", "recall", "f1", "tpr", "fpr"),
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Discrimination Threshold",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Plot common classification metrics as a function of the decision threshold.

    - Binary: provide `y_score`.
    - Multiclass: provide `proba_columns` (per-class probabilities). Metrics are computed
        one-vs-rest per class.
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)

    thresholds = np.linspace(0.0, 1.0, int(n_thresholds))
    metrics = tuple(m.lower() for m in metrics)
    allowed = {"precision", "recall", "f1", "tpr", "fpr", "specificity"}
    unknown = [m for m in metrics if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Allowed: {sorted(allowed)}")

    split_dfs = list(splits)

    if proba_columns is None or split_column is None:
        fig: go.Figure = go.Figure()
    else:
        # Multiclass + split: use subplots to avoid overloading dash styles.
        fig = sp.make_subplots(
            rows=1,
            cols=len(split_dfs),
            subplot_titles=[str(d.get_column(split_column).first()) for d in split_dfs],
            horizontal_spacing=0.08,
        )

    metric_label = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "tpr": "TPR",
        "fpr": "FPR",
        "specificity": "Specificity",
    }
    metric_colors = {m: colors[i % len(colors)] for i, m in enumerate(metrics)}

    if proba_columns is not None:
        if any(c not in df.columns for c in proba_columns):
            missing = [c for c in proba_columns if c not in df.columns]
            raise ValueError(f"Missing proba columns: {missing}")

    for split_i, split_df in enumerate(split_dfs, start=1):
        split_name = None
        if split_column is not None:
            split_name = str(split_df.get_column(split_column).first())

        y_arr = split_df.get_column(y_true).to_numpy()
        if proba_columns is None:
            s_arr = split_df.get_column(y_score).to_numpy()
            class_pairs = [(pos_label, s_arr, metric_colors)]
            class_color_map = {pos_label: colors[0]}
        else:
            if classes is None:
                inferred = _infer_classes_from_y(y_arr)
            else:
                inferred = list(classes)
            if len(inferred) != len(proba_columns):
                raise ValueError(
                    "classes and proba_columns must have the same length for multiclass"
                )
            class_color_map = {
                label: colors[i % len(colors)] for i, label in enumerate(inferred)
            }
            class_pairs = [
                (class_label, split_df.get_column(col).to_numpy(), metric_colors)
                for class_label, col in zip(inferred, proba_columns)
            ]

        for class_label, s_arr, _ in class_pairs:
            y01 = _clean_binary_targets(y_arr, pos_label=class_label)
            y01, s_arr = _filter_xy(y01, s_arr)
            y01 = np.asarray(y01, dtype=int)
            if y01.size == 0:
                continue

            pos = float(np.sum(y01 == 1))
            neg = float(np.sum(y01 == 0))
            if pos == 0 or neg == 0:
                continue

            pred_pos = (s_arr[None, :] >= thresholds[:, None]).astype(int)
            tp = (pred_pos * y01[None, :]).sum(axis=1)
            fp = (pred_pos * (1 - y01)[None, :]).sum(axis=1)
            fn = ((1 - pred_pos) * y01[None, :]).sum(axis=1)
            tn = ((1 - pred_pos) * (1 - y01)[None, :]).sum(axis=1)

            precision = np.divide(
                tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
            )
            recall = np.divide(
                tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) != 0,
            )
            tpr = recall
            fpr = np.divide(
                fp, fp + tn, out=np.zeros_like(fp, dtype=float), where=(fp + tn) != 0
            )
            specificity = np.divide(
                tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0
            )

            metric_series = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tpr": tpr,
                "fpr": fpr,
                "specificity": specificity,
            }

            for metric_i, m in enumerate(metrics):
                pretty = metric_label[m]
                line_color = (
                    class_color_map[class_label]
                    if proba_columns is not None
                    else metric_colors[m]
                )
                dash = (
                    _DASHES[metric_i % len(_DASHES)]
                    if proba_columns is not None
                    else _DASHES[(split_i - 1) % len(_DASHES)]
                )
                name = pretty
                if proba_columns is not None:
                    name = f"{class_label} · {pretty}"
                if split_name is not None and proba_columns is None:
                    name = f"{split_name} · {pretty}"
                if (
                    split_name is not None
                    and proba_columns is not None
                    and split_column is None
                ):
                    name = f"{split_name} · {class_label} · {pretty}"

                trace = go.Scatter(
                    x=thresholds,
                    y=metric_series[m],
                    mode="lines",
                    name=name,
                    line=dict(color=line_color, width=2, dash=dash),
                    showlegend=(split_i == 1)
                    if (proba_columns is not None and split_column is not None)
                    else True,
                )
                if proba_columns is None or split_column is None:
                    fig.add_trace(trace)
                else:
                    fig.add_trace(trace, row=1, col=split_i)

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Threshold",
        yaxis_title="Metric Value",
        legend_title_text=None,
        width=width,
        height=height,
        autosize=False,
    )
    if proba_columns is None or split_column is None:
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        _apply_square_axes(fig, width=width, height=height)
    else:
        for c in range(1, len(split_dfs) + 1):
            fig.update_xaxes(range=[0, 1], row=1, col=c)
            fig.update_yaxes(range=[0, 1], row=1, col=c)
    return fig


def calibration_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_score: str = "y_score",
    *,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    n_bins: int = 10,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Calibration Plot",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Plot a reliability diagram (calibration curve) for binary or multiclass (one-vs-rest)."""
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Perfectly calibrated",
        )
    )

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)

    split_dfs = list(splits)
    for split_i, split_df in enumerate(split_dfs):
        split_name = None
        if split_column is not None:
            split_name = str(split_df.get_column(split_column).first())
        y_arr = split_df.get_column(y_true).to_numpy()

        def _add_curve(curve_name: str, y01: np.ndarray, p: np.ndarray) -> None:
            y01, p = _filter_xy(y01, p)
            y01 = np.asarray(y01, dtype=int)
            if y01.size == 0:
                return

            bin_idx = np.digitize(p, edges[1:-1], right=True)
            xs: list[float] = []
            ys: list[float] = []
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                if not np.any(mask):
                    continue
                xs.append(float(np.mean(p[mask])))
                ys.append(float(np.mean(y01[mask])))

            if not xs:
                return
            line_color = colors[0]
            dash = _DASHES[split_i % len(_DASHES)]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=curve_name,
                    line=dict(color=line_color, width=2, dash=dash),
                    marker=dict(color=to_rgba(line_color, alpha=0.8), size=7),
                )
            )

        if proba_columns is None:
            p = split_df.get_column(y_score).to_numpy()
            y01 = _clean_binary_targets(y_arr, pos_label=pos_label)
            base = "Calibration" if split_name is None else f"{split_name}"
            _add_curve(base, y01, p)
            continue

        if any(c not in split_df.columns for c in proba_columns):
            missing = [c for c in proba_columns if c not in split_df.columns]
            raise ValueError(f"Missing proba columns: {missing}")

        if classes is None:
            inferred = _infer_classes_from_y(y_arr)
        else:
            inferred = list(classes)
        if len(inferred) != len(proba_columns):
            raise ValueError(
                "classes and proba_columns must have the same length for multiclass"
            )

        class_color_map = {
            label: colors[i % len(colors)] for i, label in enumerate(inferred)
        }

        for class_label, col in zip(inferred, proba_columns):
            p = split_df.get_column(col).to_numpy()
            y01 = _clean_binary_targets(y_arr, pos_label=class_label)
            base = f"{class_label}"
            if split_name is not None:
                base = f"{split_name} · {class_label}"
            # For multiclass, keep color stable per class and use dash per split.
            # Inline the curve addition to control line color.
            y01_c, p_c = _filter_xy(y01, p)
            y01_c = np.asarray(y01_c, dtype=int)
            if y01_c.size == 0:
                continue
            bin_idx = np.digitize(p_c, edges[1:-1], right=True)
            xs: list[float] = []
            ys: list[float] = []
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                if not np.any(mask):
                    continue
                xs.append(float(np.mean(p_c[mask])))
                ys.append(float(np.mean(y01_c[mask])))
            if not xs:
                continue
            line_color = class_color_map[class_label]
            dash = _DASHES[split_i % len(_DASHES)]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=base,
                    line=dict(color=line_color, width=2, dash=dash),
                    marker=dict(color=to_rgba(line_color, alpha=0.8), size=7),
                )
            )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        legend_title_text=None,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _apply_square_axes(fig, width=width, height=height)
    return fig


def _confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[object]
) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    mask = (
        np.isfinite(y_pred.astype(float))
        if np.issubdtype(y_pred.dtype, np.number)
        else np.ones_like(y_pred, dtype=bool)
    )
    # Keep length aligned if y_true has null-ish values too.
    mask = mask & (y_true == y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1.0
    return cm


def confusion_matrix_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    *,
    y_score: Optional[str] = None,
    threshold: float = 0.5,
    pos_label: object = 1,
    neg_label: object = 0,
    labels: Optional[Sequence[object]] = None,
    normalize: Optional[str] = None,
    split_column: Optional[str] = None,
    template: str = "plotly_white",
    title: str = "Confusion Matrix",
    width: int = 650,
    height: int = 650,
    show_text: bool = True,
) -> go.Figure:
    """Plot a confusion matrix.

    - Binary: provide either `y_pred` or (`y_score` + `threshold`).
    - Multiclass: provide `y_pred`.

    Args:
        normalize: None, 'true' (row), 'pred' (col), or 'all'.
    """
    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)

    normalize = None if normalize is None else str(normalize).lower()
    if normalize not in {None, "true", "pred", "all"}:
        raise ValueError("normalize must be one of: None, 'true', 'pred', 'all'")

    split_dfs = list(splits)

    if len(split_dfs) == 1:
        fig = go.Figure()
    else:
        fig = sp.make_subplots(
            rows=1,
            cols=len(split_dfs),
            subplot_titles=[str(d.get_column(split_column).first()) for d in split_dfs],
            horizontal_spacing=0.08,
        )

    for i, split_df in enumerate(split_dfs, start=1):
        y_t = split_df.get_column(y_true).to_numpy()

        if y_pred in split_df.columns:
            y_p = split_df.get_column(y_pred).to_numpy()
        else:
            if y_score is None or y_score not in split_df.columns:
                raise ValueError(
                    "Provide a y_pred column or set y_score to compute predictions"
                )
            s = split_df.get_column(y_score).to_numpy()
            y01, s = _filter_xy(_clean_binary_targets(y_t, pos_label=pos_label), s)
            y_t = np.where(y01 == 1, pos_label, neg_label)
            y_p = np.where(s >= float(threshold), pos_label, neg_label)

        if labels is None:
            # Use stable order: for binary, show neg then pos.
            uniq = _infer_classes_from_y(y_t)
            if set(uniq) <= {neg_label, pos_label}:
                labs = [neg_label, pos_label]
            else:
                labs = uniq
        else:
            labs = list(labels)

        cm = _confusion_counts(y_t, y_p, labs)

        z = cm.copy()
        if normalize == "true":
            denom = z.sum(axis=1, keepdims=True)
            z = np.divide(z, denom, out=np.zeros_like(z), where=denom != 0)
        elif normalize == "pred":
            denom = z.sum(axis=0, keepdims=True)
            z = np.divide(z, denom, out=np.zeros_like(z), where=denom != 0)
        elif normalize == "all":
            denom = z.sum()
            z = z / denom if denom else z

        text = None
        if show_text:
            if normalize is None:
                text = np.round(cm, 0).astype(int).astype(str)
            else:
                text = np.vectorize(lambda v: f"{v:.2f}")(z)

        # Show raw counts and percent-of-all in hover; render cell labels via
        # annotations to allow per-cell font coloring and a compact two-line label.
        total = float(cm.sum()) if cm.size else 0.0
        pct = (cm / total) if total else np.zeros_like(cm)

        heat = go.Heatmap(
            z=z,
            x=[str(l) for l in labs],
            y=[str(l) for l in labs],
            colorscale=px.colors.sequential.Blues,
            showscale=(len(split_dfs) == 1 and i == 1),
            hovertemplate="True=%{y}<br>Pred=%{x}<br>Count=%{z}<br>Percent=%{customdata:.1%}<extra></extra>",
            customdata=pct,
        )

        if len(split_dfs) == 1:
            fig.add_trace(heat)
        else:
            fig.add_trace(heat, row=1, col=i)

        # Add per-cell annotations (count + percent) with contrast-aware font color
        rows, cols = z.shape
        for r in range(rows):
            for c in range(cols):
                val = cm[r, c]
                pct_val = pct[r, c]
                disp = f"{int(val)}<br>{pct_val:.1%}"
                # choose white text for darker cells
                font_color = "white" if pct_val > 0.45 else "#0b2545"
                if len(split_dfs) == 1:
                    fig.add_annotation(
                        x=str(labs[c]),
                        y=str(labs[r]),
                        text=disp,
                        showarrow=False,
                        font=dict(color=font_color, size=12),
                        align="center",
                    )
                else:
                    fig.add_annotation(
                        x=str(labs[c]),
                        y=str(labs[r]),
                        text=disp,
                        showarrow=False,
                        font=dict(color=font_color, size=12),
                        align="center",
                        row=1,
                        col=i,
                    )

    # Tighten layout and reduce surrounding whitespace. If we have multiple
    # subplots (one per split) adjust subplot-title positions to sit closer
    # to each heatmap so labels/ticks are not far away from the plotted area.
    fig.update_layout(
        template=template,
        title=title,
        width=width,
        height=height,
        autosize=False,
        # aggressively tighten margins to reduce whitespace
        # margin=dict(t=8, b=6, l=12, r=12),
    )

    # When multiple split subplots are present, nudge the subplot titles down
    # (they are rendered as annotations by make_subplots). Also reduce their
    # font size to compact the layout.
    if len(split_dfs) > 1 and hasattr(fig.layout, "annotations"):
        # collect split titles to identify which annotations are subplot titles
        split_titles = [str(d.get_column(split_column).first()) for d in split_dfs]
        new_anns = []
        for ann in fig.layout.annotations:
            try:
                if ann.text in split_titles:
                    # position subplot title closer to the heatmap
                    ann.y = 0.86
                    ann.yanchor = "bottom"
                    ann.font = dict(size=11)
                # leave other annotations (per-cell) unchanged
            except Exception:
                pass
            new_anns.append(ann)
        fig.layout.annotations = tuple(new_anns)
        # Reduce vertical domain of each subplot's axes so ticks are nearer to heatmap
        for c in range(1, len(split_dfs) + 1):
            try:
                # compress top/bottom whitespace around heatmap (tighter)
                fig.update_yaxes(
                    row=1, col=c, domain=[0.25, 0.75], autorange="reversed"
                )
                # keep x-axis constrained to domain to avoid extra horizontal padding
                fig.update_xaxes(row=1, col=c, constrain="domain")
            except Exception:
                pass

    # Square matrix
    if len(split_dfs) == 1:
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    else:
        for c in range(1, len(split_dfs) + 1):
            fig.update_yaxes(
                autorange="reversed", scaleanchor=f"x{c}", scaleratio=1, row=1, col=c
            )
            fig.update_xaxes(constrain="domain", row=1, col=c)

    return fig


def _binary_lift_gains_ks(
    y_true01: np.ndarray, y_score: np.ndarray
) -> dict[str, np.ndarray | float]:
    y_true01, y_score = _filter_xy(y_true01, y_score)
    y_true01 = np.asarray(y_true01, dtype=int)
    n = int(y_true01.size)
    if n == 0:
        return {
            "pop": np.array([]),
            "gains": np.array([]),
            "lift": np.array([]),
            "tpr": np.array([]),
            "fpr": np.array([]),
            "ks": float("nan"),
            "ks_x": float("nan"),
        }

    pos = float(np.sum(y_true01 == 1))
    neg = float(np.sum(y_true01 == 0))
    if pos == 0 or neg == 0:
        return {
            "pop": np.linspace(0, 1, n),
            "gains": np.linspace(0, 1, n),
            "lift": np.ones(n),
            "tpr": np.linspace(0, 1, n),
            "fpr": np.linspace(0, 1, n),
            "ks": float("nan"),
            "ks_x": float("nan"),
        }

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true01[order]
    cum_pos = np.cumsum(y)
    cum_neg = np.cumsum(1 - y)

    pop = (np.arange(1, n + 1) / n).astype(float)
    tpr = cum_pos / pos
    fpr = cum_neg / neg
    gains = tpr
    lift = np.divide(gains, pop, out=np.zeros_like(gains, dtype=float), where=pop != 0)

    diff = np.abs(tpr - fpr)
    ks_idx = int(np.argmax(diff))
    ks = float(diff[ks_idx])
    ks_x = float(pop[ks_idx])

    return {
        "pop": pop,
        "gains": gains,
        "lift": lift,
        "tpr": tpr,
        "fpr": fpr,
        "ks": ks,
        "ks_x": ks_x,
    }


def lift_gain_ks_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_score: str = "y_score",
    *,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Lift / Gains / KS",
    width: int = 1100,
    height: int = 400,
) -> go.Figure:
    """Plot Gains curve, Lift curve, and KS curve.

    - Binary: provide `y_score`.
    - Multiclass: provide `proba_columns` (per-class probabilities). Curves are computed
      one-vs-rest per class.
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)

    fig = sp.make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Gains", "Lift", "KS"),
        horizontal_spacing=0.08,
    )

    split_dfs = list(splits)

    if proba_columns is not None:
        if any(c not in df.columns for c in proba_columns):
            missing = [c for c in proba_columns if c not in df.columns]
            raise ValueError(f"Missing proba columns: {missing}")

    for split_i, split_df in enumerate(split_dfs):
        split_name = None
        if split_column is not None:
            split_name = str(split_df.get_column(split_column).first())

        y_arr = split_df.get_column(y_true).to_numpy()

        if proba_columns is None:
            score_pairs = [(pos_label, split_df.get_column(y_score).to_numpy())]
            class_color_map = {pos_label: colors[0]}
        else:
            if classes is None:
                inferred = _infer_classes_from_y(y_arr)
            else:
                inferred = list(classes)
            if len(inferred) != len(proba_columns):
                raise ValueError(
                    "classes and proba_columns must have the same length for multiclass"
                )
            class_color_map = {
                label: colors[i % len(colors)] for i, label in enumerate(inferred)
            }
            score_pairs = [
                (class_label, split_df.get_column(col).to_numpy())
                for class_label, col in zip(inferred, proba_columns)
            ]

        dash = _DASHES[split_i % len(_DASHES)]
        split_tag = "All" if split_name is None else split_name

        for class_label, s_arr in score_pairs:
            y01 = _clean_binary_targets(y_arr, pos_label=class_label)
            stats = _binary_lift_gains_ks(y01, s_arr)

            pop = stats["pop"]
            if isinstance(pop, np.ndarray) and pop.size == 0:
                continue
            gains = stats["gains"]
            lift = stats["lift"]
            tpr = stats["tpr"]
            fpr = stats["fpr"]
            ks = float(stats["ks"])
            ks_x = float(stats["ks_x"])

            line_color = class_color_map[class_label]
            name = (
                split_tag if proba_columns is None else f"{split_tag} · {class_label}"
            )

            # Gains
            fig.add_trace(
                go.Scatter(
                    x=pop,
                    y=gains,
                    mode="lines",
                    name=name,
                    line=dict(color=line_color, width=2, dash=dash),
                    legendgroup=name,
                ),
                row=1,
                col=1,
            )
            # Lift
            fig.add_trace(
                go.Scatter(
                    x=pop,
                    y=lift,
                    mode="lines",
                    name=name,
                    showlegend=False,
                    line=dict(color=line_color, width=2, dash=dash),
                    legendgroup=name,
                ),
                row=1,
                col=2,
            )

            if proba_columns is None:
                # KS (TPR vs FPR over population)
                fig.add_trace(
                    go.Scatter(
                        x=pop,
                        y=tpr,
                        mode="lines",
                        name=f"{name} · TPR",
                        showlegend=False,
                        line=dict(color=colors[1 % len(colors)], width=2, dash=dash),
                        legendgroup=name,
                    ),
                    row=1,
                    col=3,
                )
                fig.add_trace(
                    go.Scatter(
                        x=pop,
                        y=fpr,
                        mode="lines",
                        name=f"{name} · FPR",
                        showlegend=False,
                        line=dict(color=colors[2 % len(colors)], width=2, dash=dash),
                        legendgroup=name,
                    ),
                    row=1,
                    col=3,
                )
            else:
                # For multiclass, show the KS separation curve (TPR - FPR) for clarity.
                diff = np.asarray(tpr) - np.asarray(fpr)
                fig.add_trace(
                    go.Scatter(
                        x=pop,
                        y=diff,
                        mode="lines",
                        name=f"{name} · TPR-FPR",
                        showlegend=False,
                        line=dict(color=line_color, width=2, dash=dash),
                        legendgroup=name,
                    ),
                    row=1,
                    col=3,
                )

            if np.isfinite(ks) and np.isfinite(ks_x):
                fig.add_trace(
                    go.Scatter(
                        x=[ks_x, ks_x],
                        y=[-1, 1] if proba_columns is not None else [0, 1],
                        mode="lines",
                        name=f"{name} · KS={ks:.3f}",
                        showlegend=False,
                        line=dict(color="black", dash="dot"),
                        legendgroup=name,
                    ),
                    row=1,
                    col=3,
                )

    # Reference baselines
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line=dict(color="black", dash="dash"),
            showlegend=(split_column is None),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[1, 1],
            mode="lines",
            name="Baseline",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template=template,
        title=title,
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    fig.update_xaxes(title_text="Population fraction", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative positives", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Population fraction", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Lift", row=1, col=2)
    fig.update_xaxes(title_text="Population fraction", range=[0, 1], row=1, col=3)
    if proba_columns is None:
        fig.update_yaxes(title_text="Rate", range=[0, 1], row=1, col=3)
    else:
        fig.update_yaxes(title_text="TPR - FPR", row=1, col=3)
    return fig


def score_distribution_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    y_score: str = "y_score",
    *,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Score Distribution",
    nbins: int = 40,
    width: int = 900,
    height: int = 400,
) -> go.Figure:
    """Plot score distributions.

    - Binary: overlays score distributions for positive vs negative.
    - Multiclass: provide `proba_columns` (per-class probabilities). For each class, overlays
      the distribution of predicted probability for samples where that class is the true label
      vs the rest.
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)
    split_dfs = list(splits)

    if len(split_dfs) == 1:
        fig = go.Figure()
    else:
        fig = sp.make_subplots(
            rows=1,
            cols=len(split_dfs),
            subplot_titles=[str(d.get_column(split_column).first()) for d in split_dfs],
            horizontal_spacing=0.08,
        )

    if proba_columns is not None:
        if any(c not in df.columns for c in proba_columns):
            missing = [c for c in proba_columns if c not in df.columns]
            raise ValueError(f"Missing proba columns: {missing}")

    for i, split_df in enumerate(split_dfs, start=1):
        y_arr = split_df.get_column(y_true).to_numpy()

        if proba_columns is None:
            s_arr = split_df.get_column(y_score).to_numpy()
            y01, s_arr = _filter_xy(
                _clean_binary_targets(y_arr, pos_label=pos_label), s_arr
            )
            y01 = np.asarray(y01, dtype=int)
            if y01.size == 0:
                continue

            s_pos = s_arr[y01 == 1]
            s_neg = s_arr[y01 == 0]
            traces = [
                go.Histogram(
                    x=s_neg,
                    nbinsx=nbins,
                    name="Negative",
                    marker=dict(color=to_rgba(colors[1 % len(colors)], alpha=0.5)),
                    opacity=0.65,
                    histnorm="probability density",
                ),
                go.Histogram(
                    x=s_pos,
                    nbinsx=nbins,
                    name="Positive",
                    marker=dict(color=to_rgba(colors[0], alpha=0.5)),
                    opacity=0.65,
                    histnorm="probability density",
                ),
            ]
        else:
            if classes is None:
                inferred = _infer_classes_from_y(y_arr)
            else:
                inferred = list(classes)
            if len(inferred) != len(proba_columns):
                raise ValueError(
                    "classes and proba_columns must have the same length for multiclass"
                )

            traces = []
            for class_i, (class_label, col) in enumerate(zip(inferred, proba_columns)):
                p = split_df.get_column(col).to_numpy()
                y01, p = _filter_xy(
                    _clean_binary_targets(y_arr, pos_label=class_label), p
                )
                y01 = np.asarray(y01, dtype=int)
                if y01.size == 0:
                    continue
                p_true = p[y01 == 1]
                p_rest = p[y01 == 0]
                base_color = colors[class_i % len(colors)]
                traces.extend(
                    [
                        go.Histogram(
                            x=p_rest,
                            nbinsx=nbins,
                            name=f"{class_label} (rest)",
                            marker=dict(color=to_rgba(base_color, alpha=0.25)),
                            opacity=0.5,
                            histnorm="probability density",
                            showlegend=(i == 1),
                        ),
                        go.Histogram(
                            x=p_true,
                            nbinsx=nbins,
                            name=f"{class_label} (true)",
                            marker=dict(color=to_rgba(base_color, alpha=0.55)),
                            opacity=0.7,
                            histnorm="probability density",
                            showlegend=(i == 1),
                        ),
                    ]
                )

        for t in traces:
            if len(split_dfs) == 1:
                fig.add_trace(t)
            else:
                fig.add_trace(t, row=1, col=i)

    fig.update_layout(
        template=template,
        title=title,
        barmode="overlay",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    if len(split_dfs) == 1:
        fig.update_xaxes(title_text="Score", range=[0, 1])
        fig.update_yaxes(title_text="Density")
    else:
        for c in range(1, len(split_dfs) + 1):
            fig.update_xaxes(title_text="Score", range=[0, 1], row=1, col=c)
            fig.update_yaxes(title_text="Density", row=1, col=c)
    return fig


def prediction_confidence_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    y_true: str = "y_true",
    *,
    y_score: Optional[str] = None,
    proba_columns: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[object]] = None,
    split_column: Optional[str] = None,
    pos_label: object = 1,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    title: str = "Prediction Confidence",
    nbins: int = 40,
    width: int = 900,
    height: int = 400,
) -> go.Figure:
    """Plot confidence distributions for correct vs incorrect predictions.

    - Binary: provide `y_score`.
    - Multiclass: provide `proba_columns` (per-class probabilities).
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    df = _as_polars(data)
    if split_column is None:
        splits: Iterable[pl.DataFrame] = [df]
    else:
        if split_column not in df.columns:
            raise ValueError(f"split_column '{split_column}' not found")
        splits = df.partition_by(split_column)
    split_dfs = list(splits)

    if len(split_dfs) == 1:
        fig = go.Figure()
    else:
        fig = sp.make_subplots(
            rows=1,
            cols=len(split_dfs),
            subplot_titles=[str(d.get_column(split_column).first()) for d in split_dfs],
            horizontal_spacing=0.08,
        )

    for i, split_df in enumerate(split_dfs, start=1):
        y_arr = split_df.get_column(y_true).to_numpy()

        if proba_columns is None:
            if y_score is None or y_score not in split_df.columns:
                raise ValueError("For binary confidence, provide y_score")
            s = split_df.get_column(y_score).to_numpy()
            y01, s = _filter_xy(_clean_binary_targets(y_arr, pos_label=pos_label), s)
            y01 = np.asarray(y01, dtype=int)
            pred = (s >= 0.5).astype(int)
            correct = pred == y01
            conf = np.where(pred == 1, s, 1 - s)
        else:
            if any(c not in split_df.columns for c in proba_columns):
                missing = [c for c in proba_columns if c not in split_df.columns]
                raise ValueError(f"Missing proba columns: {missing}")
            if classes is None:
                inferred = _infer_classes_from_y(y_arr)
            else:
                inferred = list(classes)
            if len(inferred) != len(proba_columns):
                raise ValueError(
                    "classes and proba_columns must have the same length for multiclass"
                )
            proba = np.vstack(
                [split_df.get_column(c).to_numpy() for c in proba_columns]
            ).T
            # Filter finite rows
            mask = np.isfinite(proba).all(axis=1)
            proba = proba[mask]
            y_arr = np.asarray(y_arr).reshape(-1)[mask]
            pred_idx = np.argmax(proba, axis=1)
            pred_label = np.array([inferred[j] for j in pred_idx], dtype=object)
            correct = pred_label == y_arr
            conf = np.max(proba, axis=1)

        conf_correct = conf[correct]
        conf_wrong = conf[~correct]

        traces = [
            go.Histogram(
                x=conf_correct,
                nbinsx=nbins,
                name="Correct",
                marker=dict(color=to_rgba(colors[0], alpha=0.55)),
                opacity=0.7,
                histnorm="probability density",
            ),
            go.Histogram(
                x=conf_wrong,
                nbinsx=nbins,
                name="Incorrect",
                marker=dict(color=to_rgba(colors[1 % len(colors)], alpha=0.55)),
                opacity=0.7,
                histnorm="probability density",
            ),
        ]
        for t in traces:
            if len(split_dfs) == 1:
                fig.add_trace(t)
            else:
                fig.add_trace(t, row=1, col=i)

    fig.update_layout(
        template=template,
        title=title,
        barmode="overlay",
        width=width,
        height=height,
        autosize=False,
        legend_title_text=None,
    )
    if len(split_dfs) == 1:
        fig.update_xaxes(title_text="Confidence", range=[0, 1])
        fig.update_yaxes(title_text="Density")
    else:
        for c in range(1, len(split_dfs) + 1):
            fig.update_xaxes(title_text="Confidence", range=[0, 1], row=1, col=c)
            fig.update_yaxes(title_text="Density", row=1, col=c)
    return fig
