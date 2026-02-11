"""
plotly_ml: Interactive ML Visualizations with Plotly
"""

from . import regression as regression
from . import univariant as univariant
from . import pariplot as pariplot
from . import classification as classification
from . import comparison as comparison
from .pariplot import (
    pairplot as pairplot,
    pairplot_html as pairplot_html,
    pairplot_html_file as pairplot_html_file,
    pairplot_dash_crossfilter as pairplot_dash_crossfilter,
)
from ._widget import PairplotWidget as PairplotWidget

__all__ = [
    "classification",
    "comparison",
    "regression",
    "univariant",
    "pariplot",
    "pairplot",
    "pairplot_html",
    "pairplot_html_file",
    "pairplot_dash_crossfilter",
    "PairplotWidget",
]
