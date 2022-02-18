import numpy as np
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    DATA_DIR,
    scale_dataset,
)
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED, set_random_seeds


def training_function(config: dict) -> None:

    random_seed = config["random_seed"]
    set_random_seeds(seed=random_seed)
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    # load in the data
    data_dir = DATA_DIR
    data_set = scale_dataset(
        load_data(
            data_dir=data_dir,
            without_duplicates=True,
            shuffle_data=False,
            rng=rng
        )
    )




