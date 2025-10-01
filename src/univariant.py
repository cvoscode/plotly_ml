import polars as pl
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# from colors import to_rgba
from typing import Union, List, Optional
from src.utils.colors import to_rgba


def raincloud_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    value: Union[str, List[str]],
    group: Optional[str] = None,
    template: str = "plotly_white",
    colors: Optional[List[str]] = None,
    show_box: bool = True,
    show_points: bool = True,
    violin_side: str = "positive",
    sample_size=5_000,
    title="Raincloud Plot",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """Create a raincloud plot combining violin plots, box plots, and scatter points.

    A raincloud plot is a hybrid visualization that combines aspects of violin plots,
    box plots, and scatter plots to provide a comprehensive view of data distribution.

    Args:


            data (Union[pl.DataFrame, pd.DataFrame]): Input DataFrame in either Polars or Pandas format.
            value (Union[str, List[str]]): Column name(s) for the variable(s) to plot.
                    Can be a single column name or a list of column names.
            group (Optional[str], optional): Column name for grouping the data.
                    If None, each value column is treated as a separate group. Defaults to None.
            template (str, optional): Plotly template name. Defaults to "plotly_white".
            colors (Optional[List[str]], optional): List of colors for each group.
                    If None, uses Plotly's D3 qualitative color scale. Defaults to None.
            show_box (bool, optional): Whether to show box plots inside violins. Defaults to True.
            show_points (bool, optional): Whether to show individual data points. Defaults to True.
            violin_side (str, optional): Orientation of violin plots - 'both', 'positive',
                    or 'negative'. Defaults to "positive".
            sample_size (int, optional): Maximum number of points to plot if dataset is large.
                    Defaults to 5000.
            title (str, optional): Plot title. Defaults to "Raincloud Plot".
            height (int, optional): Figure height in pixels. Defaults to 600.
            width (int, optional): Figure width in pixels. Defaults to 800.

    Returns:
            go.Figure: Plotly figure object containing the raincloud plot.
    """
    if colors is None:
        colors = px.colors.qualitative.D3

    # Convert to Polars for consistency
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    # Support multiple columns for value
    if isinstance(value, str):
        value_cols = [value]
    else:
        value_cols = value
    if data.height > sample_size:
        data = data.sample(sample_size)
    fig = go.Figure()
    print(f"Value columns: {value_cols}")
    if group is None:
        # Plot each column as a separate group
        print("No group specified, plotting each value column separately.")
        for i, col in enumerate(value_cols):
            vals = data[col]
            color = colors[i % len(colors)]
            fill_color = to_rgba(color, alpha=0.4)
            name = str(col)
            fig.add_trace(
                go.Violin(
                    x=vals,
                    name=name,
                    box_visible=show_box,
                    line_color=color,
                    fillcolor=fill_color,
                    opacity=0.7,
                    points="all" if show_points else None,
                    side=violin_side,
                    width=0.7,
                    legendgroup=name,
                    scalemode="width",
                    showlegend=True,
                    meanline_visible=show_box,
                    spanmode="soft",
                    pointpos=-0.4,
                )
            )

    else:
        for i, df in enumerate(data.partition_by(group)):
            vals = df[value_cols[0]]  # Only first column for grouped
            color = colors[i % len(colors)]
            fill_color = to_rgba(color, alpha=0.4)

            name = df.get_column(group).first()
            fig.add_trace(
                go.Violin(
                    x=vals,
                    name=name,
                    box_visible=show_box,
                    line_color=color,
                    fillcolor=fill_color,
                    opacity=0.7,
                    points="all" if show_points else None,
                    side=violin_side,
                    width=0.7,
                    legendgroup=name,
                    scalemode="width",
                    showlegend=True,
                    meanline_visible=show_box,
                    spanmode="soft",
                    pointpos=-0.4,
                )
            )

    fig.update_layout(
        template=template,
        title=title,
        violinmode="group",
        height=height,
        width=width,
    )
    return fig
