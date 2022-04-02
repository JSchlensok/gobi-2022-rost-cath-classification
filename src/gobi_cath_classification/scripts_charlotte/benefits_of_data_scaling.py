import numpy as np
from pathlib import Path

from typing import Dict

from bokeh.plotting import figure, show

from bokeh.models import ColumnDataSource, Whisker
from bokeh.transform import dodge

from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.prediction import read_in_proba_predictions


def apply_style(p):
    p.toolbar.logo = None
    p.background_fill_color = "#fafafa"


def bar_plot(
    title: str,
    a: Dict[str, float],
    b: Dict[str, float],
    a_errors: Dict[str, float],
    b_errors: Dict[str, float],
    legend_label_a: str,
    legend_label_b: str,
    x_axis_label: str,
    y_axis_label: str,
):
    print(f"a.keys() = {a.keys()}")
    print(f"b.keys() = {b.keys()}")
    print(f"a_errors.keys() = {a_errors.keys()}")
    print(f"b_errors.keys() = {b_errors.keys()}")

    assert a.keys() == b.keys()

    a_upper_error = []
    a_lower_error = []
    for k, v in a.items():
        a_upper_error.append(v + a_errors[k])
        a_lower_error.append(v - a_errors[k])

    b_upper_error = []
    b_lower_error = []
    for k, v in b.items():
        b_upper_error.append(v + b_errors[k])
        b_lower_error.append(v - b_errors[k])

    x_labels = list(a.keys())
    print(f"x_labels = {x_labels}")
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


def main():
    # load dataset
    dataset = load_data(
        data_dir=DATA_DIR,
        rng=np.random.RandomState(1),
        without_duplicates=True,
        reloading_allowed=True,
    )
    dataset.scale()
    evaluations = []

    # load all predictions which should be evaluated
    # run 1
    # directory = Path(
    #     "/Users/x/Downloads/training_function_2022-03-30_19-54-42_scaled_vs_non_scaled/gobi-2022-rost-cath-classification/ray_results/training_function_2022-03-30_19-54-42"
    # )

    # run 2
    directory = Path(
        "/Users/x/Downloads/content/gobi-2022-rost-cath-classification/ray_results/training_function_2022-04-01_17-31-32"
    )

    pathlist = sorted(Path(directory).glob("**/*predictions_val.csv"))
    print(f"pathlist = {pathlist}")
    for path in pathlist:
        model_name = str(path).split("/")[-2]
        print(f"model_name = {model_name}")
        pred = read_in_proba_predictions(path)
        evaluation = Evaluation(
            y_true=dataset.y_val,
            predictions=pred,
            train_labels=dataset.train_labels,
            model_name=model_name,
        )
        evaluation.compute_metrics(accuracy=True)
        evaluation.compute_std_err(bootstrap_n=100)
        evaluations.append(evaluation)

    results_acc_scaled = {}
    results_error_scaled = {}
    results_acc_non_scaled = {}
    results_error_non_scaled = {}

    num_sets = int(len(evaluations) / 2)

    for i in range(num_sets):
        set_i = f"Set {i + 1}"

        results_acc_non_scaled[set_i] = evaluations[i].eval_dict["accuracy"]["accuracy_h"] * 100
        results_error_non_scaled[set_i] = evaluations[i].error_dict["accuracy"]["accuracy_h"] * 100

        results_acc_scaled[set_i] = (
            evaluations[i + num_sets].eval_dict["accuracy"]["accuracy_h"] * 100
        )
        results_error_scaled[set_i] = (
            evaluations[i + num_sets].error_dict["accuracy"]["accuracy_h"] * 100
        )

    results_acc_non_scaled["Mean of Set 1-5"] = sum(results_acc_non_scaled.values()) / num_sets
    results_error_non_scaled["Mean of Set 1-5"] = sum(results_error_non_scaled.values()) / num_sets

    results_acc_scaled["Mean of Set 1-5"] = sum(results_acc_scaled.values()) / num_sets
    results_error_scaled["Mean of Set 1-5"] = sum(results_error_scaled.values()) / num_sets

    bar_plot(
        title="Comparison of Training with Scaled Data and Non-scaled Data",
        a=results_acc_scaled,
        b=results_acc_non_scaled,
        a_errors=results_error_scaled,
        b_errors=results_error_non_scaled,
        legend_label_a="scaled data",
        legend_label_b="non-scaled data",
        x_axis_label="Training runs with different sets of hyperparameter values",
        y_axis_label="Accuracy on H-level in %",
    )


if __name__ == "__main__":
    main()
