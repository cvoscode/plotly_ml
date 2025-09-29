# First define a helper function for R2 score at the start of your main function
import polars as pl


def r2_score(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Calculate the R² (coefficient of determination) regression score.

    Args:
        y_true (pl.Series): Ground truth (correct) target values.
        y_pred (pl.Series): Estimated target values.

    Returns:
        float: R² score. Best possible score is 1.0, and it can be negative.
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)


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
    return ((y_true - y_pred) ** 2).mean().sqrt()


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
