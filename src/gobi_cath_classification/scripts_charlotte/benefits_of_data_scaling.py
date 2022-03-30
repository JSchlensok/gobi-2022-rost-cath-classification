from typing import Dict

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import FixedTicker, Span, LabelSet, Label
from bokeh.layouts import gridplot

from bokeh.models import ColumnDataSource, Whisker
from bokeh.transform import dodge


def apply_style(p):
    p.toolbar.logo = None
    p.background_fill_color = "#fafafa"


def bar_plot(
    title: str,
    a: Dict[str, float],
    b: Dict[str, float],
    a_upper_error,
    b_upper_error,
    a_lower_error,
    b_lower_error,
    legend_label_a: str,
    legend_label_b: str,
    x_axis_label: str,
    y_axis_label: str,
):
    assert a.keys() == b.keys()

    x_labels = list(a.keys())
    source = ColumnDataSource(
        {
            "x_labels": x_labels,
            "a": list(a.values()),
            "b": list(b.values()),
            "a_upper_error": a_upper_error,
            "b_upper_error": b_upper_error,
            "a_lower_error": a_lower_error,
            "b_lower_error": b_lower_error,
        }
    )

    p = figure(
        x_range=x_labels,
        y_range=(0, 100),
        height=350,
        width=900,
        title=title,
    )
    apply_style(p)

    p.vbar(
        x=dodge("x_labels", -0.13, range=p.x_range),
        top="a",
        width=0.2,
        source=source,
        color="navy",
        alpha=0.5,
        legend_label=legend_label_a,
    )

    p.vbar(
        x=dodge("x_labels", 0.13, range=p.x_range),
        top="b",
        width=0.2,
        source=source,
        color="rgb(80, 169, 169)",
        alpha=0.6,
        legend_label=legend_label_b,
    )

    p.add_layout(
        Whisker(
            source=source,
            base=dodge("x_labels", -0.13, range=p.x_range),
            upper="a_upper_error",
            lower="a_lower_error",
            level="overlay",
        )
    )
    p.add_layout(
        Whisker(
            source=source,
            base=dodge("x_labels", +0.13, range=p.x_range),
            upper="b_upper_error",
            lower="b_lower_error",
            level="overlay",
        )
    )

    p.xaxis.axis_label = x_axis_label
    p.yaxis.axis_label = y_axis_label

    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "10pt"

    p.title.text_font_size = "12pt"

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None

    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor tick

    show(p)


def plot_with_error_bars():
    from bokeh.models import ColumnDataSource, Whisker
    from bokeh.transform import factor_cmap

    groups = ["A", "B", "C", "D"]
    counts = [5, 3, 4, 2]
    error = [0.8, 0.4, 0.4, 0.3]
    upper = [x + e for x, e in zip(counts, error)]
    lower = [x - e for x, e in zip(counts, error)]

    source = ColumnDataSource(data=dict(groups=groups, counts=counts, upper=upper, lower=lower))

    p = figure(
        x_range=groups, plot_height=350, toolbar_location=None, title="Values", y_range=(0, 7)
    )
    p.vbar(
        x="groups",
        top="counts",
        width=0.9,
        source=source,
        legend="groups",
        line_color="white",
        fill_color=factor_cmap(
            "groups", palette=["#962980", "#295f96", "#29966c", "#968529"], factors=groups
        ),
    )

    p.add_layout(
        Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay")
    )

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    show(p)


def main():
    bar_plot(
        title="Comparison of Training with Scaled Data and Non-scaled Data",
        a={
            "Set 1": 76.4607,
            "Set 2": 78.4314,
            "Set 3": 79.085,
            "Set 4": 79.085,
            "Set 5": 78.4314,
            "Average over Set 1-5": 78.3,
        },
        b={
            "Set 1": 75.1634,
            "Set 2": 75.1634,
            "Set 3": 75.817,
            "Set 4": 76.4706,
            "Set 5": 73.2026,
            "Average over Set 1-5": 75.1,
        },
        a_upper_error=[79.0, 80.0, 81.0, 82.0, 83.0, 84.0],
        a_lower_error=[76.0, 75.0, 74.0, 73.0, 72.0, 71.0],
        b_upper_error=[76.0, 77.0, 76.0, 77.0, 76.0, 77.0],
        b_lower_error=[75.0, 74.0, 70.0, 71.0, 72.0, 43.0],
        legend_label_a="scaled data",
        legend_label_b="non-scaled data",
        x_axis_label="Training runs with different sets of hyperparameter values",
        y_axis_label="Accuracy on H-level in %",
    )


if __name__ == "__main__":
    main()
