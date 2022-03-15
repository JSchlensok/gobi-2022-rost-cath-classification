import numpy as np
import time

from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.scripts_finn.baseline_models import RandomBaseline
from gobi_cath_classification.pipeline.Evaluation.Evaluation import Evaluation


def main():
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

    count = 0
    for label in data_set.y_test:
        if label not in data_set.train_labels:
            count += 1

    print(count)

    model1 = RandomBaseline(data=data_set, class_balance=False, rng=rng, random_seed=random_seed)

    predictions1 = model1.predict(model1.data.X_test)

    eval1 = Evaluation(
        y_true=data_set.y_test,
        predictions=predictions1,
        train_labels=data_set.train_labels,
        model_name="Random Baseline",
    )

    # eval1.foldseek_metric()


if __name__ == "__main__":
    main()
