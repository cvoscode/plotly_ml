import pytest
import pandas as pd
import polars as pl
from plotly_ml import regression, univariant


def _assert_valid_figure(fig):
    assert fig is not None
    # Plotly figures have a to_dict method and at least one trace
    assert hasattr(fig, "to_dict")
    d = fig.to_dict()
    # Ensure data exists and is non-empty
    assert isinstance(d.get("data"), list)
    assert len(d.get("data")) > 0


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
def test_regression_evaluation_plot(df_type):
    # Create a simple regression dataset
    if df_type == "pandas":
        df = pd.DataFrame(
            {
                "y_true": [1, 2, 3, 4, 5],
                "y_pred": [1.1, 1.9, 3.2, 3.8, 5.1],
                "set": ["train", "train", "test", "test", "test"],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "y_true": [1, 2, 3, 4, 5],
                "y_pred": [1.1, 1.9, 3.2, 3.8, 5.1],
                "set": ["train", "train", "test", "test", "test"],
            }
        )

    fig = regression.regression_evaluation_plot(df, y="y_true", split_column="set")
    _assert_valid_figure(fig)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
def test_raincloud_plot_basic(df_type):
    # Create a simple univariate dataset
    if df_type == "pandas":
        df = pd.DataFrame(
            {
                "value": [1, 2, 2, 3, 3, 3, 4, 4, 5],
                "group": ["A", "A", "B", "B", "A", "B", "A", "B", "A"],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "value": [1, 2, 2, 3, 3, 3, 4, 4, 5],
                "group": ["A", "A", "B", "B", "A", "B", "A", "B", "A"],
            }
        )

    fig = univariant.raincloud_plot(df, value="value", group="group")
    _assert_valid_figure(fig)


def test_raincloud_plot_list_values_and_options():
    # Test multiple value columns and different options
    df = pd.DataFrame(
        {
            "v1": [1, 2, 2, 3, 3, 3],
            "v2": [2, 2, 3, 3, 4, 4],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    # List of values
    fig = univariant.raincloud_plot(
        df,
        value=["v1", "v2"],
        group="group",
        show_box=False,
        show_points=False,
        violin_side="negative",
    )
    _assert_valid_figure(fig)


def test_regression_with_custom_colors_and_template():
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 2, 3, 4],
            "y_pred": [0.1, 0.9, 2.1, 3.2, 3.8],
            "set": ["train", "test", "train", "test", "test"],
        }
    )
    fig = regression.regression_evaluation_plot(
        df,
        y="y_true",
        split_column="set",
        template="plotly_dark",
        colors=["#636EFA", "#EF553B"],
    )
    _assert_valid_figure(fig)
