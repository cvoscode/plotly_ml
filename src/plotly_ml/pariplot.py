import math
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
import polars as pl
import polars.selectors as cs

from plotly_ml.utils.colors import to_rgba


def _kde_1d(
    values: pl.Series, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    values = values.drop_nulls()
    values = values.filter(values.is_finite())
    if values.len() == 0:
        return np.array([]), np.array([])

    try:
        from scipy.stats import gaussian_kde  # type: ignore

        kde = gaussian_kde(values.to_numpy())
        if grid is None:
            grid = np.linspace(values.min(), values.max(), 200)
        return grid, kde(grid)
    except Exception:
        pass

    std = float(values.std())
    if std == 0:
        std = 1.0
    bw = 1.06 * std * (values.len() ** (-1 / 5))
    bw = max(bw, 1e-6)
    if grid is None:
        grid = np.linspace(values.min(), values.max(), 200)
    values_np = values.to_numpy()
    diffs = (grid[:, None] - values_np[None, :]) / bw
    density = np.exp(-0.5 * diffs**2).sum(axis=1)
    density /= values.len() * bw * math.sqrt(2 * math.pi)
    return grid, density


def _lowess_fallback(
    values: pl.DataFrame, x_col: str, y_col: str, frac: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    if values.height < 2:
        return np.array([]), np.array([])
    sorted_vals = values.sort(x_col)
    window = max(2, int(frac * sorted_vals.height))
    y_smooth = (
        sorted_vals.get_column(y_col)
        .rolling_mean(window_size=window, center=True, min_periods=1)
        .to_numpy()
    )
    return sorted_vals.get_column(x_col).to_numpy(), y_smooth


def _compute_trend(
    values: pl.DataFrame,
    x_col: str,
    y_col: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if values.height < 2:
        return np.array([]), np.array([])

    if method == "ols":
        stats = values.select(
            [
                pl.col(x_col).mean().alias("x_mean"),
                pl.col(y_col).mean().alias("y_mean"),
                pl.col(x_col).var().alias("x_var"),
                pl.cov(x_col, y_col).alias("xy_cov"),
                pl.col(x_col).min().alias("x_min"),
                pl.col(x_col).max().alias("x_max"),
            ]
        ).to_dicts()[0]
        if stats["x_var"] == 0 or stats["x_var"] is None:
            return np.array([]), np.array([])
        slope = float(stats["xy_cov"]) / float(stats["x_var"])
        intercept = float(stats["y_mean"]) - slope * float(stats["x_mean"])
        x_line = np.linspace(float(stats["x_min"]), float(stats["x_max"]), 50)
        y_line = slope * x_line + intercept
        return x_line, y_line

    if method == "lowess":
        x = values.get_column(x_col).to_numpy()
        y = values.get_column(y_col).to_numpy()
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore

            result = lowess(y, x, frac=0.3, return_sorted=True)
            return result[:, 0], result[:, 1]
        except Exception:
            return _lowess_fallback(values, x_col, y_col, frac=0.3)

    raise ValueError("trend must be None, 'ols', or 'lowess'")


def _filtered_xy(values: pl.DataFrame, x_col: str, y_col: str) -> pl.DataFrame:
    return (
        values.select(
            [
                pl.col("__row_id__"),
                pl.col(x_col).cast(pl.Float64).alias(x_col),
                pl.col(y_col).cast(pl.Float64).alias(y_col),
            ]
        )
        .drop_nulls()
        .filter(pl.col(x_col).is_finite() & pl.col(y_col).is_finite())
    )


def pairplot(
    data: Union[pl.DataFrame, pd.DataFrame],
    columns: Optional[list[str]] = None,
    hue: Optional[str] = None,
    diag: str = "kde",
    trend: Optional[str] = None,
    corr: Optional[list[str]] = None,
    sample_size: int = 5000,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    height: int = 800,
    width: int = 800,
    link_selection: bool = False,
    xf_combine: str = "replace",
    xf_show_status: bool = False,
    return_widget: Optional[bool] = None,
    use_webgl: bool = True,
) -> go.Figure:
    """Create a Plotly pairplot for numeric features."""
    if colors is None:
        colors = px.colors.qualitative.D3

    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    if columns is None:
        # Prefer Polars selectors to avoid dtype-group deprecations.
        columns = data.select(cs.numeric()).columns
    if not columns:
        raise ValueError("No numeric columns available for pairplot.")

    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if hue is not None and hue not in data.columns:
        raise ValueError(f"Hue column '{hue}' not found in data.")

    if data.height > sample_size:
        data = data.sample(sample_size, seed=0)

    # Polars deprecated `with_row_count` in favor of `with_row_index`.
    data = data.with_row_index(name="__row_id__")

    # Keep only the columns we actually use. This reduces repeated scans and
    # memory pressure when building many subplots.
    select_cols: list[str] = ["__row_id__", *columns]
    if hue is not None and hue not in select_cols:
        select_cols.append(hue)
    data = data.select(select_cols)

    if hue is not None:
        data = data.with_columns(pl.col(hue).cast(pl.Utf8))
        categories = data.get_column(hue).unique().to_list()
    else:
        categories = [None]

    if diag not in {"kde", "hist"}:
        raise ValueError("diag must be 'kde' or 'hist'")

    if trend not in {None, "ols", "lowess"}:
        raise ValueError("trend must be None, 'ols', or 'lowess'")

    if corr is not None:
        corr = [c.lower() for c in corr]
        for method in corr:
            if method not in {"pearson", "spearman"}:
                raise ValueError("corr must contain 'pearson' and/or 'spearman'")

    xf_combine = (xf_combine or "replace").lower()
    if xf_combine not in {"replace", "union", "intersection", "difference"}:
        raise ValueError(
            "xf_combine must be one of: 'replace', 'union', 'intersection', 'difference'"
        )

    color_map = {
        category: colors[i % len(colors)] for i, category in enumerate(categories)
    }

    # Precompute category subsets once (big win vs. filtering inside every cell).
    if hue is None:
        subset_map: dict[object, pl.DataFrame] = {None: data}
    else:
        subset_map = {cat: data.filter(pl.col(hue) == cat) for cat in categories}

    # Cache correlation pair filtering because corr can request multiple methods.
    corr_filtered_cache: dict[tuple[str, str], pl.DataFrame] = {}
    corr_stats_cache: dict[tuple[str, str], dict[str, float]] = {}
    corr_value_cache: dict[tuple[str, str, str], float] = {}

    def _corr_from_filtered(
        filtered: pl.DataFrame, x_col: str, y_col: str, method: str
    ) -> float:
        if filtered.height < 2:
            return float("nan")
        if method == "pearson":
            corr_df = filtered.select(pl.corr(x_col, y_col))
            return float(corr_df.to_series(0)[0])
        if method == "spearman":
            ranked = filtered.with_columns(
                [
                    pl.col(x_col).rank("average").alias("x_rank"),
                    pl.col(y_col).rank("average").alias("y_rank"),
                ]
            )
            corr_df = ranked.select(pl.corr("x_rank", "y_rank"))
            return float(corr_df.to_series(0)[0])
        raise ValueError("method must be 'pearson' or 'spearman'")

    trace_index = 0

    def _next_uid() -> str:
        nonlocal trace_index
        trace_index += 1
        return f"pairplot-trace-{trace_index}"

    n = len(columns)
    scatter_cls = go.Scatter if (link_selection or not use_webgl) else go.Scattergl
    fig = sp.make_subplots(
        rows=n,
        cols=n,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    for i in range(n):
        for j in range(n):
            x_col = columns[j]
            y_col = columns[i]
            is_diag = i == j

            for category in categories:
                if category is None:
                    label = "All"
                else:
                    label = str(category)

                subset = subset_map[category]

                if is_diag:
                    vals = (
                        subset.select(pl.col(x_col).cast(pl.Float64).alias(x_col))
                        .drop_nulls()
                        .filter(pl.col(x_col).is_finite())
                        .get_column(x_col)
                    )
                    color = color_map[category]
                    fill_color = to_rgba(color, alpha=0.35)

                    if diag == "hist":
                        fig.add_trace(
                            go.Histogram(
                                x=vals,
                                name=label,
                                marker=dict(color=color),
                                opacity=0.55,
                                showlegend=(i == 0 and j == 0),
                                legendgroup=label,
                                histnorm="probability density",
                                uid=_next_uid(),
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                    else:
                        grid, density = _kde_1d(vals)
                        if grid.size:
                            fig.add_trace(
                                go.Scatter(
                                    x=grid,
                                    y=density,
                                    mode="lines",
                                    name=label,
                                    line=dict(color=color),
                                    fill="tozeroy",
                                    fillcolor=fill_color,
                                    opacity=0.8,
                                    showlegend=(i == 0 and j == 0),
                                    legendgroup=label,
                                    uid=_next_uid(),
                                ),
                                row=i + 1,
                                col=j + 1,
                            )
                else:
                    filtered = _filtered_xy(subset, x_col, y_col)
                    x = filtered.get_column(x_col).to_numpy()
                    y = filtered.get_column(y_col).to_numpy()
                    # IMPORTANT: keep `customdata` as a plain Python list.
                    # Plotly 6 serializes NumPy arrays to base64 ({dtype,bdata}),
                    # which breaks client-side crossfilter logic that expects an
                    # array with a `.length`.
                    row_ids = filtered.get_column("__row_id__").to_list()
                    color = color_map[category]
                    marker_color = to_rgba(color, alpha=0.7)
                    show_legend = i == 0 and j == 0

                    fig.add_trace(
                        scatter_cls(
                            x=x,
                            y=y,
                            mode="markers",
                            name=label,
                            marker=dict(size=4, color=marker_color),
                            customdata=row_ids,
                            legendgroup=label,
                            showlegend=show_legend,
                            uid=_next_uid(),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

                    if trend is not None:
                        x_line, y_line = _compute_trend(filtered, x_col, y_col, trend)
                        if x_line.size:
                            fig.add_trace(
                                go.Scatter(
                                    x=x_line,
                                    y=y_line,
                                    mode="lines",
                                    line=dict(color=color),
                                    legendgroup=label,
                                    showlegend=False,
                                    uid=_next_uid(),
                                ),
                                row=i + 1,
                                col=j + 1,
                            )

            if corr and not is_diag:
                filtered = corr_filtered_cache.get((x_col, y_col))
                if filtered is None:
                    filtered = _filtered_xy(data, x_col, y_col)
                    corr_filtered_cache[(x_col, y_col)] = filtered

                if filtered.height >= 2:
                    parts = []
                    if "pearson" in corr:
                        key = ("pearson", x_col, y_col)
                        val = corr_value_cache.get(key)
                        if val is None:
                            val = _corr_from_filtered(filtered, x_col, y_col, "pearson")
                            corr_value_cache[key] = val
                        if not np.isnan(val):
                            parts.append(f"r={val:.2f}")
                    if "spearman" in corr:
                        key = ("spearman", x_col, y_col)
                        val = corr_value_cache.get(key)
                        if val is None:
                            val = _corr_from_filtered(
                                filtered, x_col, y_col, "spearman"
                            )
                            corr_value_cache[key] = val
                        if not np.isnan(val):
                            parts.append(f"rho={val:.2f}")

                    if parts:
                        stats = corr_stats_cache.get((x_col, y_col))
                        if stats is None:
                            raw = filtered.select(
                                [
                                    pl.col(x_col).min().alias("x_min"),
                                    pl.col(x_col).max().alias("x_max"),
                                    pl.col(y_col).min().alias("y_min"),
                                    pl.col(y_col).max().alias("y_max"),
                                ]
                            ).to_dicts()[0]
                            stats = {
                                "x_min": float(raw["x_min"]),
                                "x_max": float(raw["x_max"]),
                                "y_min": float(raw["y_min"]),
                                "y_max": float(raw["y_max"]),
                            }
                            corr_stats_cache[(x_col, y_col)] = stats

                        x_min, x_max = stats["x_min"], stats["x_max"]
                        y_min, y_max = stats["y_min"], stats["y_max"]
                        x_pos = x_min + 0.02 * (x_max - x_min)
                        y_pos = y_max - 0.05 * (y_max - y_min)
                        fig.add_annotation(
                            x=x_pos,
                            y=y_pos,
                            text=", ".join(parts),
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            row=i + 1,
                            col=j + 1,
                        )

            show_x = i == n - 1
            show_y = j == 0
            fig.update_xaxes(showticklabels=show_x, row=i + 1, col=j + 1)
            fig.update_yaxes(showticklabels=show_y, row=i + 1, col=j + 1)

            if show_x:
                fig.update_xaxes(title_text=x_col, row=i + 1, col=j + 1)
            if show_y:
                fig.update_yaxes(title_text=y_col, row=i + 1, col=j + 1)

    fig.update_layout(
        template=template,
        height=height,
        width=width,
        dragmode="lasso" if link_selection else "zoom",
        barmode="overlay",
    )

    # Resolve return_widget default: True when crossfiltering, else False.
    if return_widget is None:
        return_widget = link_selection

    if link_selection:
        fig.update_layout(dragmode="lasso")
        if return_widget:
            from plotly_ml._widget import PairplotWidget

            meta = dict(fig.layout.meta) if fig.layout.meta else {}
            meta.update(
                {
                    "pairplot_xf_combine": xf_combine,
                    "pairplot_xf_show_status": bool(xf_show_status),
                }
            )
            fig.update_layout(meta=meta)
            return PairplotWidget(fig)
        # Plain Figure for Dash / manual use.
        return fig

    if return_widget:
        return go.FigureWidget(fig)

    return fig


def pairplot_html(
    data: Union[pl.DataFrame, pd.DataFrame],
    columns: Optional[list[str]] = None,
    hue: Optional[str] = None,
    diag: str = "kde",
    trend: Optional[str] = None,
    corr: Optional[list[str]] = None,
    sample_size: int = 5000,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    height: int = 800,
    width: int = 800,
    div_id: str = "pairplot",
    include_plotlyjs: str = "cdn",
    use_webgl: bool = False,
) -> str:
    """Create a standalone HTML string with linked lasso selection."""
    fig = pairplot(
        data,
        columns=columns,
        hue=hue,
        diag=diag,
        trend=trend,
        corr=corr,
        sample_size=sample_size,
        template=template,
        colors=colors,
        height=height,
        width=width,
        link_selection=False,
        return_widget=False,
        use_webgl=use_webgl,
    )

    fig.update_layout(dragmode="lasso")

    html = pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs=include_plotlyjs,
        div_id=div_id,
        config={"scrollZoom": True},
    )

    sync_js = f"""
<script>
(function() {{
    var divId = "{div_id}";

    function toKey(v) {{
        return Array.isArray(v) ? String(v[0]) : String(v);
    }}

    function attach(gd) {{
        if (!gd || !gd.on || !gd.data) return false;
        if (gd.__pairplot_xf_attached__) return true;
        gd.__pairplot_xf_attached__ = true;

        var si = [], origOp = [];
        for (var i = 0; i < gd.data.length; i++) {{
            var t = gd.data[i];
            if (t && t.mode && t.mode.indexOf("markers") !== -1 && t.customdata && t.customdata.length > 0) {{
                si.push(i);
                origOp.push((t.marker && t.marker.opacity != null) ? t.marker.opacity : 1.0);
            }}
        }}
        if (si.length === 0) return true;

        var _busy = false;
        function apply(ids) {{
            _busy = true;
            var ops = [], sps = [];
            for (var k = 0; k < si.length; k++) {{
                sps.push(null);
                if (!ids) {{ ops.push(origOp[k]); continue; }}
                var cd = gd.data[si[k]].customdata;
                var b = (typeof origOp[k] === "number" && origOp[k] > 0) ? origOp[k] : 1.0;
                var a = new Array(cd.length);
                for (var j = 0; j < cd.length; j++) a[j] = ids.has(toKey(cd[j])) ? b : b * 0.1;
                ops.push(a);
            }}
            Plotly.restyle(gd, {{"marker.opacity": ops, selectedpoints: sps}}, si)
                .then(function() {{ setTimeout(function() {{ _busy = false; }}, 200); }})
                .catch(function() {{ _busy = false; }});
        }}

        gd.on("plotly_selected", function(ev) {{
            if (_busy) return;
            if (!ev || !ev.points || !ev.points.length) {{ apply(null); return; }}
            var ids = new Set();
            ev.points.forEach(function(p) {{
                if (p.customdata != null) ids.add(toKey(p.customdata));
            }});
            apply(ids.size > 0 ? ids : null);
        }});
        gd.on("plotly_deselect", function() {{ if (!_busy) apply(null); }});
        gd.on("plotly_doubleclick", function() {{ if (!_busy) apply(null); }});

        return true;
    }}

    function boot(attempt) {{
        attempt = attempt || 0;
        var gd = document.getElementById(divId);
        if (!gd) return;

        // Wait for Plotly to fully initialize the graph div.
        // (In some environments gd exists before gd.data/gd.on are ready.)
        var ok = false;
        try {{
            ok = attach(gd);
        }} catch (e) {{
            ok = false;
        }}
        if (!ok && attempt < 200) {{
            setTimeout(function() {{ boot(attempt + 1); }}, 25);
        }}
    }}

    if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", function() {{ boot(0); }});
    }} else {{
        boot(0);
    }}
}})();
</script>
"""

    return html + sync_js


def pairplot_html_file(
    path: str,
    data: Union[pl.DataFrame, pd.DataFrame],
    columns: Optional[list[str]] = None,
    hue: Optional[str] = None,
    diag: str = "kde",
    trend: Optional[str] = None,
    corr: Optional[list[str]] = None,
    sample_size: int = 5000,
    template: str = "plotly_white",
    colors: Optional[list[str]] = None,
    height: int = 800,
    width: int = 800,
    div_id: str = "pairplot",
    include_plotlyjs: str = "cdn",
    use_webgl: bool = False,
) -> str:
    """Write a standalone HTML file with linked lasso selection and return the path."""
    html = pairplot_html(
        data,
        columns=columns,
        hue=hue,
        diag=diag,
        trend=trend,
        corr=corr,
        sample_size=sample_size,
        template=template,
        colors=colors,
        height=height,
        width=width,
        div_id=div_id,
        include_plotlyjs=include_plotlyjs,
        use_webgl=use_webgl,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def pairplot_dash_crossfilter(app, graph_id: str = "pairplot") -> None:
    """Register a Dash clientside callback for pairplot crossfiltering.

    Call this after creating the Dash ``app`` but before ``app.run()``.
    The pairplot figure should be created with
    ``link_selection=True, return_widget=False``.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    graph_id : str
        HTML id of the :class:`dcc.Graph` component that holds the pairplot.

    Example
    -------
    ::

        from dash import Dash, dcc, html
        from plotly_ml import pairplot, pairplot_dash_crossfilter

        fig = pairplot(df, hue="group", link_selection=True, return_widget=False)
        app = Dash(__name__)
        app.layout = html.Div([dcc.Graph(id="pairplot", figure=fig)])
        pairplot_dash_crossfilter(app, "pairplot")
        app.run()
    """
    from dash import Input, Output  # noqa: E402 – lazy import

    js = (
        "function(selectedData) {"
        "  var gd = document.getElementById('" + graph_id + "');"
        "  if (!gd || !gd.data) return window.dash_clientside.no_update;"
        "  var si = [], origOp = [];"
        "  for (var i = 0; i < gd.data.length; i++) {"
        "    var t = gd.data[i];"
        "    if (t && t.mode && t.mode.indexOf('markers') !== -1 && t.customdata && t.customdata.length > 0) {"
        "      si.push(i); origOp.push((t.marker && t.marker.opacity != null) ? t.marker.opacity : 1.0);"
        "    }"
        "  }"
        "  if (si.length === 0) return window.dash_clientside.no_update;"
        "  function tk(v) { return Array.isArray(v) ? String(v[0]) : String(v); }"
        "  var sel = null;"
        "  if (selectedData && selectedData.points && selectedData.points.length) {"
        "    sel = new Set();"
        "    selectedData.points.forEach(function(p) {"
        "      if (p.customdata != null) sel.add(tk(p.customdata));"
        "    });"
        "    if (sel.size === 0) sel = null;"
        "  }"
        "  var ops = [], sps = [];"
        "  for (var k = 0; k < si.length; k++) {"
        "    sps.push(null);"
        "    if (!sel) { ops.push(origOp[k]); continue; }"
        "    var cd = gd.data[si[k]].customdata;"
        "    var b = (typeof origOp[k] === 'number' && origOp[k] > 0) ? origOp[k] : 1.0;"
        "    var a = new Array(cd.length);"
        "    for (var j = 0; j < cd.length; j++) a[j] = sel.has(tk(cd[j])) ? b : b * 0.1;"
        "    ops.push(a);"
        "  }"
        "  Plotly.restyle(gd, {'marker.opacity': ops, selectedpoints: sps}, si);"
        "  return window.dash_clientside.no_update;"
        "}"
    )

    app.clientside_callback(
        js,
        Output(graph_id, "className"),
        Input(graph_id, "selectedData"),
        prevent_initial_call=True,
    )
