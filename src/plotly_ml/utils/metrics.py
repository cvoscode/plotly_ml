# First define a helper function for R2 score at the start of your main function
import numpy as np
import polars as pl


def _clean_regression_arrays(
    y_true: pl.Series, y_pred: pl.Series
) -> tuple[np.ndarray, np.ndarray]:
    df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred}).with_columns(
        [
            pl.col("y_true").cast(pl.Float64),
            pl.col("y_pred").cast(pl.Float64),
        ]
    )
    df = df.drop_nulls().filter(
        pl.col("y_true").is_finite() & pl.col("y_pred").is_finite()
    )
    if df.height == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return df.get_column("y_true").to_numpy(), df.get_column("y_pred").to_numpy()


def r2_score(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Calculate the R² (coefficient of determination) regression score.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        float: R² score. Best possible score is 1.0, and it can be negative.
    """
    yt, yp = _clean_regression_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))

    # Match common behavior (e.g., sklearn) for constant targets.
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return 1.0 - (ss_res / ss_tot)


def mae(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Calculate Mean Absolute Error between predictions and ground truth.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        float: Mean absolute error.
    """
    return (y_true - y_pred).abs().mean()


def rmse(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Calculate Root Mean Square Error between predictions and ground truth.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        float: Root mean square error.
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def bias(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Calculate the bias (mean error) between predictions and ground truth.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        float: Mean prediction error (bias).
    """
    return (y_true - y_pred).mean()


def var(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Calculate the variance of the prediction errors.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        float: Variance of prediction errors.
    """
    return (y_true - y_pred).var()


def count(y_true: pl.Series, y_pred: pl.Series) -> int:
    """Count the number of samples in the dataset.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        int: Number of samples.
    """
    return len(y_true)
