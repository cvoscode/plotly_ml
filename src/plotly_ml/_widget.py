"""Anywidget-based Plotly pairplot with JavaScript crossfiltering.

The widget renders a Plotly figure entirely in the browser and attaches
linked lasso / box selection across every scatter subplot.  Because the
crossfiltering logic runs in JavaScript there are no Python round-trips,
which avoids the ``uid`` synchronisation bugs that affect
:class:`plotly.graph_objects.FigureWidget` when ``selectedpoints`` is
modified from Python callbacks.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import anywidget
import traitlets
import plotly.io as pio
from plotly.offline import get_plotlyjs

if TYPE_CHECKING:
    import plotly.graph_objects as go


class PairplotWidget(anywidget.AnyWidget):
    """Interactive pairplot with linked lasso / box selection.

    Selection on any scatter subplot highlights the same data points in
    every other subplot.  The crossfiltering runs entirely in the browser
    (no Python round-trips), so it works reliably in Jupyter, JupyterLab,
    VS Code, Google Colab, and any other ipywidgets-compatible environment.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A Plotly pairplot figure (typically produced by
        :func:`plotly_ml.pairplot`).
    """

    _esm = pathlib.Path(__file__).parent / "_widget.js"

    fig_json = traitlets.Unicode("{}").tag(sync=True)
    plotly_js = traitlets.Unicode("").tag(sync=True)

    def __init__(self, fig: go.Figure, **kwargs):
        # VS Code's Jupyter widget sandbox can block loading Plotly.js from remote
        # CDNs. Embedding Plotly.js here allows the frontend to load it locally.
        super().__init__(
            fig_json=pio.to_json(fig, engine="json"),
            plotly_js=get_plotlyjs(),
            **kwargs,
        )

    # -- public helpers ------------------------------------------------

    def update_figure(self, fig: go.Figure) -> None:
        """Replace the displayed figure (triggers a re-render in the browser)."""
        self.fig_json = pio.to_json(fig, engine="json")

    @property
    def figure(self) -> go.Figure:
        """Reconstruct the current :class:`~plotly.graph_objects.Figure`."""
        return pio.from_json(self.fig_json)
