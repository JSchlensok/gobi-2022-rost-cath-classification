import sys

from gobi_cath_classification.contrastive_learning_arcface import ArcFaceModel
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir

filename = sys.argv[1]

ROOT_DIR = get_base_dir()
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

model = ArcFaceModel()
model.load_model_from_checkpoint(ROOT_DIR / "arcface_2022-03-20_epoch_25.pth")
pred = model.predict(None)

print(f"Writing Prediction instance to {DATA_DIR / filename}...")
pred.save(DATA_DIR / filename)
