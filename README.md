# plotly-ml — Interactive, code-first ML visualizations

plotly-ml provides a scikit-learn-style, developer-friendly toolkit for building interactive visualizations for machine learning workflows using Plotly. It focuses on producing publication-ready visuals while staying highly scriptable and notebook-first. It styled similar to the inspiration for the project: yellowbricks

Why it matters
- Build interactive diagnostics, model-comparison dashboards, and exploratory visualizations that integrate directly into data pipelines and notebooks.
- Use the same API style you expect from scikit-learn for fast, repeatable analysis.
- Lightweight dependency surface with `plotly`, `polars` (or `pandas`) and optional notebook widgets for richer interactivity.

Key features
- Scikit-learn-like visualizers: familiar, composable call patterns for quick integration.
- Notebook widgets: `AnyWidget`-based components (e.g., crossfilter pairplot) for interactive exploration.
- Polars-first data handling with seamless fallback to Pandas.
- Exportable HTML outputs and embedding-ready Plotly figures.

Quickstart

Install (project uses `uv` for dependency sync):

```bash
uv sync
```

# Minimal example (using `pandas`):
In a Jupyter notebook, use the widget for crossfiltering pairplots:
```python

df_pair_hue = pd.DataFrame(
    {
        "x": np.random.normal(0, 1.0, 400),
        "y": np.random.normal(0.5, 1.1, 400),
        "z": np.random.normal(-0.2, 0.9, 400),
        "group": np.where(np.random.rand(400) > 0.5, "A", "B"),
    }
)
from plotly_ml._widget import PairplotWidget
pw  = pariplot.pairplot(
    df_pair_hue,
    hue="group",
    diag="kde",
    link_selection=True,
    height=700,
    width=700,
)
```
![alt text](/images/image-1.png)

# Other plots:
## Raincloud
![alt text](/images/image-2.png)
# Regression Report:
![alt text](images/image.png)


Documentation & examples
- Full docs and API reference are in the `docs/` site (see `docs/index.md`).
- Example notebooks are in the `notebooks/` folder (`notebooks/all_plots_demo.ipynb`).

Contributing and extending
- Follow the scikit-learn-style API when adding new visualizers.
- Use Polars for data transformations where performance matters and fall back to Pandas for compatibility.
- Tests live in `tests/`; run them with `pytest`.

Need help?
- Open an issue or pull request — contributions, bug reports, and feature requests are welcome.

License
- MIT
