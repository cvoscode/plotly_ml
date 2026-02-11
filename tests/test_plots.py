import pytest
import pandas as pd
import polars as pl
from plotly_ml import regression, univariant, pariplot, classification, comparison
from plotly_ml.utils.metrics import r2_score


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


def test_pairplot_basic_numeric():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2.0, 1.5, 3.2, 2.8, 4.1],
            "c": [5, 4, 3, 2, 1],
        }
    )
    fig = pariplot.pairplot(df)
    _assert_valid_figure(fig)


def test_pairplot_with_hue_hist_and_trend():
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 1, 3, 2, 5, 4],
            "z": [5, 3, 4, 2, 1, 6],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    fig = pariplot.pairplot(
        df,
        hue="group",
        diag="hist",
        trend="ols",
        corr=["pearson", "spearman"],
    )
    _assert_valid_figure(fig)
    annotations = fig.to_dict().get("layout", {}).get("annotations", [])
    assert len(annotations) > 0


def test_r2_score_sanity_and_constant_target():
    y_true = pl.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pl.Series([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y_true, y_pred) == 1.0

    # Constant target: R² is not well-defined; common behavior is 0.0 unless perfect.
    y_true_c = pl.Series([2.0, 2.0, 2.0, 2.0])
    y_pred_c = pl.Series([2.1, 1.9, 2.2, 1.8])
    assert r2_score(y_true_c, y_pred_c) == 0.0


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
def test_classification_binary_plots(df_type):
    if df_type == "pandas":
        df = pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 1, 0, 1, 0],
                "y_score": [0.05, 0.2, 0.8, 0.7, 0.65, 0.3, 0.9, 0.1],
                "set": [
                    "train",
                    "train",
                    "train",
                    "test",
                    "test",
                    "test",
                    "test",
                    "train",
                ],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 1, 0, 1, 0],
                "y_score": [0.05, 0.2, 0.8, 0.7, 0.65, 0.3, 0.9, 0.1],
                "set": [
                    "train",
                    "train",
                    "train",
                    "test",
                    "test",
                    "test",
                    "test",
                    "train",
                ],
            }
        )

    fig_roc = classification.roc_curve_plot(df, split_column="set")
    _assert_valid_figure(fig_roc)

    fig_pr = classification.precision_recall_curve_plot(df, split_column="set")
    _assert_valid_figure(fig_pr)

    fig_the = classification.discrimination_threshold_plot(df, split_column="set")
    _assert_valid_figure(fig_the)

    fig_cal = classification.calibration_plot(df, split_column="set", n_bins=5)
    _assert_valid_figure(fig_cal)


def test_classification_multiclass_plots():
    # 3-class toy example with per-class probabilities
    df = pd.DataFrame(
        {
            "y_true": ["cat", "dog", "fish", "cat", "dog", "fish"],
            "p_cat": [0.8, 0.1, 0.2, 0.7, 0.2, 0.1],
            "p_dog": [0.1, 0.7, 0.2, 0.2, 0.6, 0.2],
            "p_fish": [0.1, 0.2, 0.6, 0.1, 0.2, 0.7],
            "set": ["train", "train", "train", "test", "test", "test"],
        }
    )
    proba_cols = ["p_cat", "p_dog", "p_fish"]
    classes = ["cat", "dog", "fish"]

    fig_roc = classification.roc_curve_plot(
        df, proba_columns=proba_cols, classes=classes
    )
    _assert_valid_figure(fig_roc)

    fig_pr = classification.precision_recall_curve_plot(
        df, proba_columns=proba_cols, classes=classes
    )
    _assert_valid_figure(fig_pr)

    fig_cal = classification.calibration_plot(
        df, proba_columns=proba_cols, classes=classes, n_bins=3
    )
    _assert_valid_figure(fig_cal)

    # Previously binary-only plots should support multiclass via one-vs-rest
    fig_the = classification.discrimination_threshold_plot(
        df,
        proba_columns=proba_cols,
        classes=classes,
        split_column="set",
        n_thresholds=21,
    )
    _assert_valid_figure(fig_the)

    fig_lgk = classification.lift_gain_ks_plot(
        df,
        proba_columns=proba_cols,
        classes=classes,
        split_column="set",
    )
    _assert_valid_figure(fig_lgk)

    fig_sd = classification.score_distribution_plot(
        df,
        proba_columns=proba_cols,
        classes=classes,
        split_column="set",
        nbins=10,
    )
    _assert_valid_figure(fig_sd)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
def test_classification_confusion_matrix_binary(df_type):
    if df_type == "pandas":
        df = pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 1, 0, 1, 0],
                "y_score": [0.05, 0.2, 0.8, 0.7, 0.65, 0.3, 0.9, 0.1],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 1, 0, 1, 0],
                "y_score": [0.05, 0.2, 0.8, 0.7, 0.65, 0.3, 0.9, 0.1],
            }
        )

    fig = classification.confusion_matrix_plot(df, y_score="y_score", threshold=0.5)
    _assert_valid_figure(fig)


def test_classification_confusion_matrix_multiclass():
    df = pd.DataFrame(
        {
            "y_true": ["cat", "dog", "fish", "cat"],
            "y_pred": ["cat", "fish", "fish", "dog"],
        }
    )
    fig = classification.confusion_matrix_plot(df)
    _assert_valid_figure(fig)


def test_classification_lift_gain_ks_and_distributions():
    df = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1, 1, 0, 1, 0],
            "y_score": [0.05, 0.2, 0.8, 0.7, 0.65, 0.3, 0.9, 0.1],
            "set": ["train", "train", "train", "test", "test", "test", "test", "train"],
        }
    )
    fig = classification.lift_gain_ks_plot(df, split_column="set")
    _assert_valid_figure(fig)

    fig = classification.score_distribution_plot(df, split_column="set")
    _assert_valid_figure(fig)

    fig = classification.prediction_confidence_plot(
        df, y_score="y_score", split_column="set"
    )
    _assert_valid_figure(fig)


def test_classification_prediction_confidence_multiclass():
    df = pd.DataFrame(
        {
            "y_true": ["cat", "dog", "fish", "cat", "dog", "fish"],
            "p_cat": [0.8, 0.1, 0.2, 0.7, 0.2, 0.1],
            "p_dog": [0.1, 0.7, 0.2, 0.2, 0.6, 0.2],
            "p_fish": [0.1, 0.2, 0.6, 0.1, 0.2, 0.7],
        }
    )
    fig = classification.prediction_confidence_plot(
        df,
        proba_columns=["p_cat", "p_dog", "p_fish"],
        classes=["cat", "dog", "fish"],
    )
    _assert_valid_figure(fig)


def test_regression_diagnostics_plots():
    df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y_pred": [0.9, 2.1, 3.2, 3.9, 5.1, 5.8],
            "set": ["train", "train", "train", "test", "test", "test"],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    fig = regression.residuals_vs_fitted_plot(df, split_column="set")
    _assert_valid_figure(fig)

    fig = regression.qq_plot(df, split_column="set")
    _assert_valid_figure(fig)

    fig = regression.binned_actual_vs_pred_plot(df, split_column="set", n_bins=3)
    _assert_valid_figure(fig)

    fig = regression.residuals_by_group_plot(df, group="group", split_column="set")
    _assert_valid_figure(fig)


def test_comparison_plots():
    df = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2", "m2"],
            "set": ["train", "test", "train", "test"],
            "auc": [0.9, 0.85, 0.88, 0.86],
            "logloss": [0.3, 0.35, 0.32, 0.34],
        }
    )
    fig = comparison.metrics_by_split_bar_plot(df, metrics=["auc", "logloss"])
    _assert_valid_figure(fig)

    fig = comparison.learning_curve_plot(
        train_sizes=[50, 100, 200],
        train_scores=[0.9, 0.92, 0.93],
        val_scores=[0.82, 0.85, 0.86],
        train_std=[0.02, 0.015, 0.01],
        val_std=[0.03, 0.02, 0.02],
    )
    _assert_valid_figure(fig)

    fig = comparison.validation_curve_plot(
        param_values=[0.1, 1, 10],
        train_scores=[0.95, 0.93, 0.9],
        val_scores=[0.84, 0.86, 0.83],
        param_name="C",
        xscale="log",
        train_std=[0.01, 0.015, 0.02],
        val_std=[0.02, 0.02, 0.03],
    )
    _assert_valid_figure(fig)
