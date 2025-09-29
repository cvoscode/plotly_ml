# Plotly ML

Interactive Machine Learning Visualizations with Plotly

## Overview

This project creates plots for machine learning similar to [Yellowbrick](https://www.scikit-yb.org/en/latest/), but using Plotly for interactive visualizations. It provides a scikit-learn style interface that is also AutoGluon ready.

## Features

- Interactive visualizations using Plotly
- Scikit-learn style interface
- AutoGluon compatibility
- Efficient data handling with Polars and Pandas
- Enhanced logging with Loguru

## Installation

We use `uv` for dependency management. To install the required dependencies:

```bash
uv sync
```

## Quick Start

Here's a simple example using the regression evaluation plot:

```python
import plotly_ml as pml
import polars as pl

# Prepare your data
data = pl.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'set': ['train'] * len(y_train) + ['test'] * len(y_test)
})

# Create the evaluation plot
fig = pml.regression_evaluation_plot(data)
fig.show()
```

## Components

The library includes several components for different visualization needs:

- **Regression Analysis**: Comprehensive evaluation plots for regression models
- **Univariant Analysis**: Distribution analysis with raincloud plots
- **Metrics**: Common regression metrics (R², RMSE, MAE, etc.)
- **Colors**: Utilities for color handling and conversion
