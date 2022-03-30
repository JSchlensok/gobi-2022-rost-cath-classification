from typing import List, Dict
from collections import Counter
import os
import uuid
import warnings
from typing_extensions import Literal
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
)
from tabulate import tabulate
from matplotlib import pyplot as plt
from plotnine import (
    ggplot,
    aes,
    geom_col,
    position_dodge2,
    position_dodge,
    labs,
    geom_errorbar,
    scale_y_continuous,
    theme,
    element_text,
)

from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.pipeline.prediction import Prediction
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds
from gobi_cath_classification.pipeline.data.data_loading import REPO_ROOT_DIR


METRICS = ["accuracy", "mcc", "f1", "kappa", "bacc"]
LEVELS: List[str] = ["c-level", "a-level", "t-level", "h-level", "mean"]
ERRORS = ["c-error", "a-error", "t-error", "h-error", "mean-error"]


class Evaluation:
    """
    The Evaluation class:
    - use this class to compute metrics for your model
    - supported metrics are: Accuracy, Matthews correlation coefficient, F1-score and Cohen's kappa score

    -> in the __init__ pass the given parameters
    -> with compute_metrics() you can specify the metrics you want to compute
        -> the metrics are then saved in the eval_dict of the Evaluation object in this structure
            eval_dict = {"accuracy": {"accuracy_C": acc_C, "accuracy_A": acc_A, ... , "accuracy_avg": acc_avg"},
                        "mcc": {"mcc_C": mcc_C, "mcc_A": mcc_A, ... , "mcc_avg": mcc_avg"},
                        "f1": {"f1_C": f1_C, "f1_A": acc_A, ... , "f1_avg": f1_avg"},
                        "kappa": {"kappa_C": kappa_C, "kappa_A": kappa_A, ... , "kappa_avg": kappa_avg},
                        "bacc" : {"bacc_C": bacc_C, "bacc_A": bacc_A, ..., "bacc_avg": bacc_avg}}
    -> with compute_std_err() the standard error for all the previously computed metrics is calculated
    -> with print_evaluation() the computed results can be printed to the stdout

    Note: only the metrics that were specified in the compute_metric() function are available in the dict
    """

    def __init__(
        self,
        y_true: List[CATHLabel],
        predictions: Prediction,
        train_labels: List[CATHLabel],
        model_name: str,
    ):
        """
        Create a Evaluation class with the true labels, the Prediction object from the predict method of the model
        and all the labels that wre used in the training

        Args:
            y_true: True labels
            predictions: Prediction object returned from the predict() method of the model
            train_labels: all labels that were used in the training
        """

        self.y_true = y_true

        # indexing the rows with corresponding CATH labels
        self.yhat_probabilities = predictions.probabilities
        idx = pd.Index([str(label) for label in y_true])
        self.yhat_probabilities = self.yhat_probabilities.set_index(idx)

        self.yhats = [CATHLabel(label) for label in predictions.argmax_labels()]
        self.train_labels = train_labels
        self.model_name = model_name
        self.eval_dict = None
        self.error_dict = None

    def compute_metrics(
        self,
        accuracy: bool = False,
        mcc: bool = False,
        f1: bool = False,
        kappa: bool = False,
        bacc: bool = False,
        _y_true: List = None,
        _y_hats: List = None,
    ):
        """
        Usage: set the desired metrics to "TRUE"
        -> creates an evaluation dictionary containing all the specified metrics

        Supported Metrics
        Args:
            accuracy: accuracy for each level in CATH (default = False)
            mcc: matthews correlation coefficient for each level in CATH (default = False)
            f1: the f1 score for each level in CATH (default = False)
            kappa: cohen's kappa for each level in CATH (default = False)
            bacc: The balanced Accuracy for each level in CATH (default = False)
            _y_true: default=None (ignore: only for internal usage)
            _y_hats: default=None (ignore: only for internal usage)

        Returns:
            updates the evaluation dict with the chosen metrics

        """

        CATH_levels = ["C", "A", "T", "H"]
        assert (accuracy or mcc or f1 or kappa or bacc) is True, print(
            "At least on metric must be selected"
        )

        # set class parameters if no external lists are given
        bootstrap = True
        if _y_true is None and _y_hats is None:
            _y_true = self.y_true
            _y_hats = self.yhats
            bootstrap = False

        eval_dict = dict()
        bootstrap_dict = dict()

        # calculate the accuracy for each level
        if accuracy:
            # calculating accuracy using dict comprehension
            accuracy_dict = {
                f"accuracy_{i.lower()}": metric_for_level(
                    _y_true, _y_hats, self.train_labels, i, "acc"
                )
                for i in CATH_levels
            }

            # average over all levels
            accuracy_dict["accuracy_avg"] = sum(accuracy_dict.values()) / len(CATH_levels)

            eval_dict["accuracy"] = accuracy_dict
            bootstrap_dict["accuracy"] = accuracy_dict

        # calculate f1 for each level
        if mcc:
            mcc_dict = {
                f"mcc_{i.lower()}": metric_for_level(_y_true, _y_hats, self.train_labels, i, "mcc")
                for i in CATH_levels
            }
            # average over all levels
            mcc_dict["mcc_avg"] = sum(mcc_dict.values()) / len(CATH_levels)

            eval_dict["mcc"] = mcc_dict
            bootstrap_dict["mcc"] = mcc_dict

        # calculating f1 for each level
        if f1:
            f1_dict = {
                f"f1_{i.lower()}": metric_for_level(_y_true, _y_hats, self.train_labels, i, "f1")
                for i in CATH_levels
            }
            # average over all levels
            f1_dict["f1_avg"] = sum(f1_dict.values()) / len(CATH_levels)

            eval_dict["f1"] = f1_dict
            bootstrap_dict["f1"] = f1_dict

        # calculating cohen's kappa for each level
        if kappa:
            kappa_dict = {
                f"kappa_{i.lower()}": metric_for_level(
                    _y_true, _y_hats, self.train_labels, i, "kappa"
                )
                for i in CATH_levels
            }
            # average over all levels
            kappa_dict["kappa_avg"] = sum(kappa_dict.values()) / len(CATH_levels)

            eval_dict["kappa"] = kappa_dict
            bootstrap_dict["kappa"] = kappa_dict

        if bacc:
            bacc_dict = {
                f"bacc_{i.lower()}": metric_for_level(
                    _y_true, _y_hats, self.train_labels, i, "bacc"
                )
                for i in CATH_levels
            }
            # average over all levels
            bacc_dict["bacc_avg"] = sum(bacc_dict.values()) / len(CATH_levels)

            eval_dict["bacc"] = bacc_dict
            bootstrap_dict["bacc"] = bacc_dict

        # if the bootstrap dict is required, return it
        if bootstrap:
            return bootstrap_dict
        else:
            self.eval_dict = eval_dict

    def compute_std_err(self, bootstrap_n: int = 1000, random_seed: int = 42) -> None:
        """
        compute the standard error for the metrics currently in the evaluation dict using 1000 bootstrap intervals
        in each bootstrap interval, choose samples with replacement and compute the given metric
        Bootstrapping takes a lot of time, so I would not recommend to compute the standard error in every epoch and
        only calling the function at the end of the training
        - on the validation set with 219 sequences and only the accuracy metric it takes
        around 3 seconds for 10 bootstrap iterations -> approx. 300 seconds for all 1000 iterations

        Args:
            bootstrap_n: number of bootstrap intervals
            random_seed: set the random seed for reproducible results

        Returns:
            creates a new dict with the 95% confidence intervals for the available metrics
        """

        # setting random seed for reproducible results
        set_random_seeds(random_seed)

        # what metrics are available and what metrics are there in general
        available_metrics = [
            "accuracy" in self.eval_dict,
            "mcc" in self.eval_dict,
            "f1" in self.eval_dict,
            "kappa" in self.eval_dict,
            "bacc" in self.eval_dict,
        ]

        # if no metrics are available, do not compute the standard error
        if not any(available_metrics):
            return

        n_pred = len(self.y_true)
        indexes = range(n_pred)
        bootstrap_dicts = list()

        # only if the eval_dict contains metrics we can calculate the error
        if self.eval_dict is not None:
            for _ in range(bootstrap_n):
                # choose n_pred indices with replacement
                sample = np.random.choice(indexes, n_pred, replace=True)
                y_true = [self.y_true[idx] for idx in sample]
                y_hats = [self.yhats[idx] for idx in sample]
                tmp_eval = self.compute_metrics(*available_metrics, _y_true=y_true, _y_hats=y_hats)

                bootstrap_dicts.append(tmp_eval)

            # extract the bootstrapped metrics
            error_dict = {}

            for i, available in enumerate(available_metrics):
                if available:
                    error_dict[METRICS[i]] = {}
                    for level in ["c", "a", "t", "h", "avg"]:

                        # for all available metrics calculate the standard error for all levels
                        error_dict[f"{METRICS[i]}"][f"{METRICS[i]}_{level}"] = 1.96 * np.std(
                            np.array(
                                [
                                    boot[METRICS[i]][f"{METRICS[i]}_{level}"]
                                    for boot in bootstrap_dicts
                                ]
                            ),
                            ddof=1,
                        )

            self.error_dict = error_dict
        else:
            raise ValueError("The eval_dict does not contain any metrics yet")

    def print_evaluation(self):
        """
        If metrics were computed, prints out all the computed metrics with standard error (if available)
        to the STD-OUT
        """
        if self.eval_dict is not None:
            print("\n##################### EVALUATION RESULTS #####################\n")
            for metric in METRICS:
                if metric in self.eval_dict:

                    df = pd.DataFrame(
                        data=self.eval_dict[metric],
                        index=[self.model_name if self.model_name is not None else "Performance"],
                    )

                    # only multiply by 100 if the metric is accuracy
                    if metric == "accuracy" or metric == "bacc":
                        df = df.multiply(100).round(2).astype(str)

                    else:
                        df = df.round(2).astype(str)

                    # assign a name to the dataframe
                    df.name = (
                        f"{self.model_name}: {metric}"
                        if self.model_name is not None
                        else f"Metric: {metric}"
                    )

                    # add the standard error if available
                    if self.error_dict is not None:
                        # only multiply standard error if the metric is accuracy
                        if metric == "accuracy" or metric == "bacc":
                            errors = (np.array(list(self.error_dict[metric].values())) * 100).round(
                                2
                            )
                        else:
                            errors = np.array(list(self.error_dict[metric].values())).round(2)

                        df = df + " +/- " + [str(err) for err in errors]

                        # assign a name to the dataframe
                        df.name = (
                            f"{self.model_name}: {metric} with errors"
                            if self.model_name is not None
                            else f"Metric: {metric}"
                        )

                    # print the evaluation results to std-out
                    print(df.name)
                    print(tabulate(df, headers="keys", tablefmt="psql"))
                    print()

            print("##################### EVALUATION RESULTS #####################\n")

        else:
            raise TypeError("No results were computed yet")


def plot_metric_bars(
    different_evals: List[Evaluation],
    metric: Literal["accuracy", "mcc", "f1", "kappa", "bacc"],
    levels: List[str] = None,
    save: bool = False,
) -> None:
    """
    For each of the Evaluation objects, we plot 5 bars. For each level one and one for the mean for the given metric.

    Args:
        different_evals: List of 1 or more Evaluation objects with the desired metric already computed
                        The Evaluation.model_name of the objects should be different
        metric: The metric that is plotted. Should be computed in all Evaluation objects, otherwise it
        will not show in the final plot
        levels: choose the different levels you want to show in the plot ["c-level", "a-level", "t-level", "c-level", "mean"] are possible
        save: set to True if you want to save the plot in REPO_ROOT_DIR/plots (default: False)

    Returns:
        prints a plot and/or saves it to REPO_ROOT_DIR/plots

    """
    if levels is None:
        levels = LEVELS

    for lvl in levels:
        if lvl not in LEVELS:
            warnings.warn(
                f"{lvl} is not a valid level. Please choose out of the following: {LEVELS}"
            )

    frames = list()
    for evaluation in different_evals:
        # for each evaluation object, compute the data frame add it to the list of all data frames
        frames.append(Evaluation_to_frame(evaluation, metric, levels))

    # concatenate all the accumulated data frames containing all the necessary data
    df = pd.concat(frames, ignore_index=True)

    # check for dataframes with the same model_name and change the name for models with the same name
    models = list()
    double_count = 0
    for i, row in df.iterrows():
        tmp_name = df.loc[i, "model"]
        c = Counter(models)
        if tmp_name in models and c[tmp_name] + 1 > 5:
            double_count += 1
            warnings.warn(f"Model {tmp_name} occurs more than once")
            for j in range(i, i + 5):
                df.loc[j, ["model"]] = f"{tmp_name}{double_count}"

        models.append(df.loc[i, "model"])

    # scales differ for different metrics -> no cleaner solution found
    if metric == "accuracy" or metric == "bacc":
        plot = (
            ggplot(df, aes("model", "metric", fill="level"))
            + geom_col(width=0.6, position=position_dodge2(padding=0.3))
            + geom_errorbar(
                aes(ymin="metric-metric_error", ymax="metric+metric_error"),
                width=0.15,
                position=position_dodge(0.6),
            )
            + labs(title=f"Performance measured in {metric.upper()}", x="", y=f"{metric}")
            + theme(axis_text_x=element_text(angle=45, hjust=1))
            + scale_y_continuous(limits=[0, 1])
        )
    else:
        plot = (
            ggplot(df, aes("model", "metric", fill="level"))
            + geom_col(width=0.6, position=position_dodge2(padding=0.3))
            + geom_errorbar(
                aes(ymin="metric-metric_error", ymax="metric+metric_error"),
                width=0.15,
                position=position_dodge(0.6),
            )
            + labs(title=f"Performance measured in {metric.upper()}", x="", y=f"{metric}")
            + theme(axis_text_x=element_text(angle=45, hjust=1))
            + scale_y_continuous(limits=[-1, 1])
        )

    # show the plot
    plot.draw(show=True)

    # if wanted finally save the plot in REPO_ROOT_DIR/plots
    if save:
        plot_directory = REPO_ROOT_DIR / "plots"
        if not os.path.exists(plot_directory):
            try:
                os.mkdir(plot_directory)
            except OSError:
                print(f"Creation of directory {plot_directory} failed")

        # use unique id as filename
        filename = str(uuid.uuid4())

        # width of plot should be dynamic to the number of evaluations shown
        plot.save(
            filename="barplot_" + filename,
            path=plot_directory,
            height=6,
            width=7 + len(different_evals) * 2,
        )


def plot_metric_line(
    different_evals: List[Dict],
    metric: Literal["accuracy", "mcc", "f1", "kappa", "bacc"],
    levels: List[Literal["C", "A", "T", "H", "avg"]] = None,
    save: bool = False,
) -> None:
    """
    During the training, you can input a list of eval_dicts for the train or evaluation dataset which were already
    computed and print out a line plot for a specific metric to see the course of the metric

    Args:
        different_evals: a list of eval_dicts with the specified metric computed
        metric: a metric for which the plot should be drawn
        levels: the level which is to be plotted
        save: set to True if you want to save the plot in REPO_ROOT_DIR/plots (default: False)

    Returns:
        shows the plot and if wanted saves it to the REPO_ROOT_DIR/plots folder
    """
    if len(different_evals) < 2:
        warnings.warn("More than one evaluation object should be passed")

    all_frames = list()
    for i, eval_dict in enumerate(different_evals):
        if metric in eval_dict:
            tmp = pd.DataFrame(data=eval_dict[metric], index=[i])
            all_frames.append(tmp)

    df = pd.concat(all_frames)

    # if no level is selected, display all levels
    if levels is None:
        levels = ["C", "A", "T", "H", "avg"]

    df[[f"{metric}_{lvl}" for lvl in levels]].plot.line()
    if metric == "accuracy" or metric == "bacc":
        plt.ylim(0, 1)
    else:
        plt.ylim(-1, 1)

    plt.xlabel("epochs")
    plt.ylabel(f"{metric}")
    plt.legend(loc=2)
    plt.title(f"{metric} over the course of {len(different_evals)} epochs")

    if save:
        plot_directory = REPO_ROOT_DIR / "plots"
        if not os.path.exists(plot_directory):
            try:
                os.mkdir(plot_directory)
            except OSError:
                print(f"Creation of directory {plot_directory} failed")

        # use date as unique filename and replace spaces and ":" with "_"
        filename = str(uuid.uuid4())

        plt.savefig(f"{plot_directory}/lineplot_{filename}.png")

    plt.show()


def metric_for_level(
    y_true: List[CATHLabel],
    y_hat: List[CATHLabel],
    train_labels: List[CATHLabel],
    cath_level: Literal["C", "A", "T", "H"],
    metric: Literal["acc", "mcc", "f1", "kappa", "bacc"],
) -> float:
    """
    Calculates the specific metric according to a given CATH-level.
    Args:
        y_true:
            List of labels (List[CATHLabel]): ground truth
        y_hat:
            List of labels (List[str]): prediction
        train_labels:
            Alphabetically sorted list of all labels that occurred in training
        cath_level:
            Literal["C", "A", "T", "H"]
        metric:
            the chosen metric is calculated for the specified level in the CATH hierarchy
            ["acc", "mcc", "f1", "kappa", "bacc"]
    Returns:
        chosen metric score for each level in CATH and the mean
    """
    class_names_for_level = list(set([label[:cath_level] for label in train_labels]))
    y_true_for_level = [str(label[:cath_level]) for label in y_true]
    y_pred_for_level = [str(label[:cath_level]) for label in y_hat]

    # delete all entries where the ground truth label does not occur in training class names.
    n = len(y_true_for_level) - 1
    for i in range(n, -1, -1):
        if y_true_for_level[i] not in class_names_for_level:
            del y_true_for_level[i]
            del y_pred_for_level[i]

    assert len(y_true_for_level) == len(y_pred_for_level)
    # compute the specified metric
    if metric == "acc":
        return accuracy_score(
            y_true=y_true_for_level,
            y_pred=y_pred_for_level,
        )
    if metric == "mcc":
        return matthews_corrcoef(y_true=y_true_for_level, y_pred=y_pred_for_level)
    if metric == "f1":
        return f1_score(
            y_true=y_true_for_level,
            y_pred=y_pred_for_level,
            average="weighted",
        )
    if metric == "kappa":
        return cohen_kappa_score(y1=y_true_for_level, y2=y_pred_for_level)

    if metric == "bacc":
        return balanced_accuracy_score(y_true=y_true_for_level, y_pred=y_pred_for_level)


def Evaluation_to_frame(
    evaluation: Evaluation, metric: Literal["accuracy", "mcc", "f1", "kappa", "bacc"], levels=None
) -> pd.DataFrame:
    """
    Converts the data from the Evaluation dict into a Dataframe which can then be used for plotting
    Args:
        evaluation: an Evaluation object
        metric: the metric for which the DataFrame should be generated
        levels: which levels should be used

    Returns:
        A pandas Dataframe with the columns: [metric, level, model_name, metric_error, error]
        if errors are available

    """
    if levels is None:
        levels = LEVELS
    # check if the evaluation contains the specified metric
    if evaluation.eval_dict is not None and metric in evaluation.eval_dict:
        # convert the metric dict in the eval_dict to a DataFrame used for plotting
        tmp = pd.DataFrame(
            data={
                "metric": evaluation.eval_dict[metric].values(),
                "level": LEVELS,
                "model": evaluation.model_name,
            }
        )
        df = tmp.assign(level=pd.Categorical(tmp["level"], categories=levels))

        # if the errors are available, add the errors to the DataFrame otherwise add the error 0
        if evaluation.error_dict is not None and metric in evaluation.error_dict:
            tmp = df.assign(metric_error=evaluation.error_dict[metric].values(), error=ERRORS)
        else:
            warnings.warn(f"There are no std_err for: {evaluation.model_name}")
            tmp = df.assign(metric_error=np.repeat(0, 5), error=ERRORS)

        df = tmp.assign(error=pd.Categorical(tmp["error"], categories=ERRORS))

        # only return the wanted levels
        return df[df.level.isin(levels)]
    else:
        warnings.warn(f"The requested metric was not computed for {evaluation.model_name}")
        return pd.DataFrame()
