import numpy as np
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    DATA_DIR,
    scale_dataset,
)
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.scripts_finn.baseline_models import RandomBaseline, ZeroRate
from gobi_cath_classification.pipeline.evaluation import evaluate

random_seed = RANDOM_SEED
set_random_seeds(seed=random_seed)
rng = np.random.RandomState(random_seed)
print(f"rng = {rng}")

data_dir = DATA_DIR

data_set = scale_dataset(
    load_data(
        data_dir=data_dir,
        without_duplicates=True,
        shuffle_data=False,
        rng=rng,
        load_only_small_sample=False,
    )
)

model1 = RandomBaseline(data=data_set, class_balance=False, rng=rng, random_seed=random_seed)
model2 = RandomBaseline(data=data_set, class_balance=True, rng=rng, random_seed=random_seed)
model3 = ZeroRate(data=data_set, rng=rng, random_seed=random_seed)

predictions1 = model1.predict(model1.data.X_val)
predictions2 = model2.predict(model2.data.X_val)
predictions3 = model3.predict(model3.data.X_val)

evaluation1 = evaluate(data_set.y_val, predictions1, data_set.all_labels_train_sorted)
evaluation2 = evaluate(data_set.y_val, predictions2, data_set.all_labels_train_sorted)
evaluation3 = evaluate(data_set.y_val, predictions3, data_set.all_labels_train_sorted)

print(f"the accuracys for the random baseline without class balance are: {evaluation1}")
print(f"the accuracys for the random baseline with class balance are: {evaluation2}")
print(f"the accuracys for the Zero Rate model is: {evaluation3}")
