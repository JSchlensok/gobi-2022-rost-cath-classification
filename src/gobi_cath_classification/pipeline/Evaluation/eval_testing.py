import numpy as np
import warnings
from timeit import timeit

from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.scripts_finn.baseline_models import RandomBaseline, ZeroRate
from gobi_cath_classification.pipeline.Evaluation.Evaluation import (
    Evaluation,
    plot_metric_bars,
    plot_metric_line,
)


def main():
    # suppress warnings
    # warnings.filterwarnings("ignore", category=UserWarning)

    def testing_bar_chart():
        random_seed = RANDOM_SEED
        set_random_seeds(seed=random_seed)
        rng = np.random.RandomState(random_seed)
        print(f"rng = {rng}")

        data_dir = DATA_DIR

        data_set = load_data(
            data_dir=data_dir,
            without_duplicates=True,
            shuffle_data=False,
            rng=rng,
            load_only_small_sample=False,
            reloading_allowed=True,
            load_tmp_holdout_set=True,
        )
        model1 = RandomBaseline(
            data=data_set, class_balance=False, rng=rng, random_seed=random_seed
        )
        model2 = ZeroRate(data=data_set, rng=rng, random_seed=random_seed)

        predictions1 = model1.predict(model1.data.X_test)
        predictions2 = model2.predict(model2.data.X_test)

        eval1 = Evaluation(
            y_true=data_set.y_test,
            predictions=predictions1,
            train_labels=data_set.train_labels,
            model_name="Random Baseline",
        )
        eval1.compute_metrics(accuracy=True)
        eval1.compute_std_err(bootstrap_n=2)

        eval2 = Evaluation(
            y_true=data_set.y_test,
            predictions=predictions2,
            train_labels=data_set.train_labels,
            model_name="Zero Rate",
        )
        eval2.compute_metrics(accuracy=True)
        eval2.compute_std_err(bootstrap_n=2)

        plot_metric_bars([eval1, eval2], metric="accuracy", levels=["h-level", "mean"])

    def testing_line_chart():
        # test for accuracy
        all_dicts = list()
        n = 1
        for i in range(100):
            # simulate metric dict for n epochs with rising accuracy for each epoch
            tmp = {
                "accuracy": {
                    "accuracy_C": np.random.randint(low=i, high=i + 10) / (i + 10),
                    "accuracy_A": np.random.randint(low=i, high=i + 20) / (i + 20),
                    "accuracy_T": np.random.randint(low=i, high=i + 30) / (i + 30),
                    "accuracy_H": np.random.randint(low=i, high=i + 40) / (i + 40),
                    "accuracy_avg": np.random.randint(low=i, high=i + 50) / (i + 50),
                }
            }
            all_dicts.append(tmp)

        plot_metric_line(different_evals=all_dicts, metric="accuracy", levels=["C", "H"], save=True)

    def testing_print_eval():
        random_seed = RANDOM_SEED
        set_random_seeds(seed=random_seed)
        rng = np.random.RandomState(random_seed)
        print(f"rng = {rng}")

        data_dir = DATA_DIR

        data_set = load_data(
            data_dir=data_dir,
            without_duplicates=True,
            shuffle_data=False,
            rng=rng,
            load_only_small_sample=False,
            reloading_allowed=True,
        )
        x = data_set.train_labels

        model1 = RandomBaseline(
            data=data_set, class_balance=False, rng=rng, random_seed=random_seed
        )
        model2 = RandomBaseline(data=data_set, class_balance=True, rng=rng, random_seed=random_seed)
        model3 = ZeroRate(data=data_set, rng=rng, random_seed=random_seed)

        predictions1 = model1.predict(model1.data.X_test)
        predictions2 = model2.predict(model2.data.X_test)
        predictions3 = model3.predict(model3.data.X_val)

        eval1 = Evaluation(
            y_true=data_set.y_test,
            predictions=predictions1,
            train_labels=data_set.train_labels,
            model_name="Random Baseline",
        )

        eval1.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True, bacc=True)

        eval1.compute_std_err()

        eval1.print_evaluation()

        eval2 = Evaluation(
            y_true=data_set.y_test,
            predictions=predictions2,
            train_labels=data_set.train_labels,
            model_name="Random Baseline with weights",
        )

        eval2.compute_metrics(accuracy=True, bacc=True)

        eval2.print_evaluation()

    testing_line_chart()
    testing_bar_chart()
    testing_print_eval()


if __name__ == "__main__":
    main()
