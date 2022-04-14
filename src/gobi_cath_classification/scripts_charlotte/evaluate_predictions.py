from collections import Counter
from typing import List

import torch
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Label
from bokeh.plotting import figure, show
from matplotlib_venn import venn3

from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.prediction import Prediction
from gobi_cath_classification.pipeline.utils import torch_utils, CATHLabel
from gobi_cath_classification.scripts_charlotte.benefits_of_data_scaling import apply_style

CATH_COLOURS = {
    "C": "rgb(182, 70, 71)",
    "A": "rgb(64, 131, 69)",
    "T": "rgb(45, 132, 167)",
    "H": "rgb(243, 142, 26)",
}


def plot_label_frequency_in_train_for_each_level(
    y_true: List[CATHLabel], y_pred: List[CATHLabel], y_train: List[str], train_labels: List[str],
):
    plt_dict = {}

    for cath_level in ["C", "A", "T", "H"]:
        train_labels = train_labels
        label2count = Counter([str(CATHLabel(y)[:cath_level]) for y in y_train])

        class_names_for_level = list(set([label[:cath_level] for label in train_labels]))
        y_true_for_level = [str(label[:cath_level]) for label in y_true]
        y_pred_for_level = [str(label[:cath_level]) for label in y_pred]

        # delete all entries where the ground truth label does not occur in training class names.
        n = len(y_true_for_level) - 1
        for i in range(n, -1, -1):
            if y_true_for_level[i] not in class_names_for_level:
                del y_true_for_level[i]
                del y_pred_for_level[i]

        plt_dict[f"x_{cath_level}"] = []
        plt_dict[f"y_{cath_level}"] = []
        for i, y in enumerate(y_true_for_level):
            x = label2count[y]
            y = True if y == y_pred_for_level[i] else False
            plt_dict[f"x_{cath_level}"].append(x)
            plt_dict[f"y_{cath_level}"].append(y)

    scatter_plots = []
    x_start = 1
    x_end = max(plt_dict["x_C"]) * 1.1

    for cath_level in ["C", "A", "T", "H"]:
        width = 1200
        height = 120 if cath_level is not "H" else 145

        x = plt_dict[f"x_{cath_level}"]
        y = plt_dict[f"y_{cath_level}"]

        x_red = []
        y_red = []
        x_green = []
        y_green = []

        for i, y_i in enumerate(y):
            if y_i == True:
                x_green.append(x[i])
                y_green.append(0)
            else:
                x_red.append(x[i])
                y_red.append(1)

        # --- Scatter plot ---
        p_scatter = figure(height=height, width=width, y_axis_location=None, x_axis_type="log",)
        apply_style(p_scatter)

        p_scatter.scatter(
            x=x_red, y=y_red, color="red", alpha=0.3, size=4, legend_label="Incorrect prediction",
        )
        p_scatter.scatter(
            x=x_green,
            y=y_green,
            color="green",
            alpha=0.3,
            size=4,
            legend_label="Correct prediction",
        )
        if cath_level is not "C":
            p_scatter.legend.items = []
        else:
            p_scatter.legend.location = "top_left"
        if cath_level == "H":
            p_scatter.xaxis.axis_label = "Number of occurences of groundtruth label in training set"

        p_scatter.x_range.start = x_start
        p_scatter.x_range.end = x_end
        p_scatter.y_range.start = -1
        p_scatter.y_range.end = 2
        p_scatter.ygrid.grid_line_color = "#fafafa"
        p_scatter.toolbar.logo = None

        median_label = Label(
            x=x_end - 27000,
            x_units="data",
            y=1.4,
            y_units="data",
            text=f" Predictions for {cath_level}-level ",
            background_fill_color=CATH_COLOURS[cath_level],
            background_fill_alpha=0.6,
            text_font_size="14px",
        )
        p_scatter.add_layout(median_label)

        scatter_plots.append([p_scatter])

    layout = gridplot([sp for sp in scatter_plots], merge_tools=False)

    show(layout)


def plot_overlap_of_two_predictions(
    y_true: List[CATHLabel],
    pred_1: Prediction,
    pred_2: Prediction,
    train_labels: List[CATHLabel],
    pred_1_legend_name: str,
    pred_2_legend_name: str,
    cath_levels: List,
    which_set: str,
) -> None:
    for cath_level in cath_levels:
        class_names_for_level = list(set([str(y[:cath_level]) for y in train_labels]))

        y_pred_1 = [str(CATHLabel(y)[:cath_level]) for y in pred_1.argmax_labels()]
        y_pred_2 = [str(CATHLabel(y)[:cath_level]) for y in pred_2.argmax_labels()]
        y_true_str = [str(y[:cath_level]) for y in y_true]
        assert len(y_pred_1) == len(y_true_str)
        assert len(y_pred_2) == len(y_true_str)

        n = len(y_true_str) - 1
        for i in range(n, -1, -1):
            if y_true_str[i] not in class_names_for_level:
                del y_true_str[i]
                del y_pred_1[i]
                del y_pred_2[i]

        y_true_venn = []
        y_pred_1_venn = []
        y_pred_2_venn = []

        for i in range(len(y_true_str)):
            y_true_venn.append(str(y_true_str[i]) + f"_{i}")
            if y_pred_1[i] != y_true_str[i]:
                y_pred_1_venn.append(str(y_pred_1[i]) + f"_{i}_false")
            else:
                y_pred_1_venn.append(str(y_pred_1[i]) + f"_{i}")

            if y_pred_2[i] != y_true_str[i]:
                y_pred_2_venn.append(str(y_pred_2[i]) + f"_{i}_false")
            else:
                y_pred_2_venn.append(str(y_pred_2[i]) + f"_{i}")

        c = venn3(
            [set(y_true_venn), set(y_pred_1_venn), set(y_pred_2_venn)],
            ("Ground truth on " + which_set, pred_1_legend_name, pred_2_legend_name),
        )

        patch_id2color = {
            "100": "#ACDFB2",
            "010": "#ABCCE8",
            "001": "#F9F39D",
            "101": "#d8e680",
            "011": "#D8DF95",
            "110": "#81BCC2",
            "111": "#C7D785",
        }
        for patch_id, color in patch_id2color.items():
            patch = c.get_patch_by_id(patch_id)
            if patch is not None:
                patch.set_color(color)
                patch.set_alpha(1.0)

        plt.title(f"Predictions for all samples on {cath_level}-level")
        plt.show()


def main():
    dataset = load_data(
        data_dir=DATA_DIR,
        rng=np.random.RandomState(1),
        without_duplicates=True,
        shuffle_data=False,
        reloading_allowed=True,
        load_tmp_holdout_set=True,
    )
    dataset.scale()

    model_path = sorted(
        Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models").glob("**/*model_object.model")
    )
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.device = torch_utils.get_device()
    model_name = model.__class__.__name__
    print(f"model_name = {model_name}")

    y_true = dataset.y_test
    y_pred_str = model.predict(embeddings=dataset.X_test).argmax_labels()
    y_pred = [CATHLabel(y) for y in y_pred_str]

    y_train = [str(y) for y in dataset.y_train]

    plot_label_frequency_in_train_for_each_level(
        y_true=y_true, y_pred=y_pred, y_train=y_train, train_labels=dataset.train_labels
    )


if __name__ == "__main__":
    main()
