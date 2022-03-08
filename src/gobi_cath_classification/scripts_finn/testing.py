import numpy as np
import time

from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.scripts_finn.baseline_models import RandomBaseline, ZeroRate
from gobi_cath_classification.pipeline.Evaluation.Evaluation import Evaluation


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


model1 = RandomBaseline(data=data_set, class_balance=False, rng=rng, random_seed=random_seed)
model2 = RandomBaseline(data=data_set, class_balance=True, rng=rng, random_seed=random_seed)
model3 = ZeroRate(data=data_set, rng=rng, random_seed=random_seed)

predictions1 = model1.predict(model1.data.X_test)
predictions2 = model2.predict(model2.data.X_val)
predictions3 = model3.predict(model3.data.X_val)

eval1 = Evaluation(
    y_true=data_set.y_test, predictions=predictions1, train_labels=data_set.train_labels
)
start = time.perf_counter()
eval1.compute_metrics(accuracy=True, mcc=True)
end = time.perf_counter()
print(f"time to compute the metrics: {end-start}")
print(f"the accuracy for the random baseline without class balance are: {eval1.eval_dict}")

start = time.perf_counter()
eval1.compute_std_err(bootstrap_n=10)
end = time.perf_counter()
print(f"time to compute the standard error: {end-start}")

"""
evaluation1 = evaluate(data_set.y_test, predictions1, data_set.train_labels)
evaluation2 = evaluate(data_set.y_val, predictions2, data_set.train_labels)
evaluation3 = evaluate(data_set.y_val, predictions3, data_set.train_labels)

print(f"the accuracy for the random baseline without class balance are: {evaluation1}")
print(f"the accuracy for the random baseline with class balance are: {evaluation2}")
print(f"the accuracy for the Zero Rate model is: {evaluation3}")
"""
