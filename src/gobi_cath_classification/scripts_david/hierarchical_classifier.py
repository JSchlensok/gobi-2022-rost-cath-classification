# Different approaches to hierarchically classify the CATH H-domain

# IMPORT STATEMENTS:
import torch
import decimal
import numpy as np
from typing import List, Optional, Dict

from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds
from gobi_cath_classification.pipeline.evaluation import accuracy_for_level


class HierarchicalClassifier:
    ########################################################################################
    # CLASS NAME        : ClassifierPerLevel
    # IMPLEMENTS        : Nothing
    # DESCRIPTION       : Class to implement different hierarchical classification methods
    #                     LCL
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 08.03.2022
    # UPDATE            : ---
    ########################################################################################

    def __init__(self, models: List[str], classifier_type: str, classification_cutoff: decimal):
        ########################################################################################
        # FUNCTION NAME     : __init__()
        # INPUT PARAMETERS  : models: List[Path], classifier_type: str, classification_cutoff: decimal
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Constructor
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 09.03.2022
        # UPDATE            : ---
        ########################################################################################
        # Necessary variables for the class
        self.models = []
        self.dataset = None
        self.thresh = classification_cutoff
        self.classifier_type = classifier_type
        # Read in the models from the checkpoint_files
        try:
            print("Attempting to read in the models from list 'models'...")
            for model_path in models:
                self.models.append(torch.load(model_path))
        except:
            raise ValueError("Failed to read models from list 'models'!")

        if classifier_type == "LCL":
            print("Commencing classification with 'Local Classifier Per Level'")
            if len(self.models) != 4:
                raise ValueError("Wrong number of models supplied! Required 4")

    def get_data(self, random_seed: int = 42):
        ########################################################################################
        # FUNCTION NAME     : get_data()
        # INPUT PARAMETERS  : random_seed: int
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Read in the data in the required format
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 09.03.2022
        # UPDATE            : ---
        ########################################################################################
        print("Reading in data...")
        set_random_seeds(seed=random_seed)
        rng = np.random.RandomState(random_seed)
        print(f"rng = {rng}")
        # load data
        self.dataset = load_data(
            data_dir=DATA_DIR,
            rng=rng,
            without_duplicates=True,
            shuffle_data=True,
            reloading_allowed=False,
        )
        self.dataset.scale()
        print("Data successfully read in!")

    def predict_lcl(self):
        ########################################################################################
        # FUNCTION NAME     : predict_lcl()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Predict each level with a different model and concatenate the results
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 10.03.2022
        # UPDATE            : ---
        ########################################################################################
        # Load the model for each CATH level
        model_C = None
        model_A = None
        model_T = None
        model_H = None
        try:
            print("Assigning models")
            model_C = self.models[0]
            print("Model for level C ready and standing by!")
            model_A = self.models[1]
            print("Model for level A ready and standing by!")
            model_T = self.models[2]
            print("Model for level T ready and standing by!")
            model_H = self.models[3]
            print("Model for level H ready and standing by!")
        except:
            print("FAILED to assign models...")
            exit(0)
        # Predict with each model
        print("Commencing to predict...")
        prediction_C = model_C.predict(embeddings=self.dataset.X_val)
        prediction_A = model_A.predict(embeddings=self.dataset.X_val)
        prediction_T = model_T.predict(embeddings=self.dataset.X_val)
        prediction_H = model_H.predict(embeddings=self.dataset.X_val)
        print("Probabilities available!")
        # Get the argmax_labels() for every prediction
        print("Reading out argument maxima for probabilities...")
        labels_C = prediction_C.argmax_labels()
        labels_A = prediction_A.argmax_labels()
        labels_T = prediction_T.argmax_labels()
        labels_H = prediction_H.argmax_labels()
        # Concatenate every level to create a prediction for every level
        print("Concatenating predictions to form full CATH labels")
        labels_CATH = []
        for i in range(len(labels_C)):
            labels_CATH.append(f"{labels_C[i]}.{labels_A[i]}.{labels_T[i]}.{labels_H[i]}")
        # Evaluate the prediction
        eval_dict = {
            "accuracy_c": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_CATH,
                class_names_training=self.dataset.train_labels,
                cath_level="C",
            ),
            "accuracy_a": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_CATH,
                class_names_training=self.dataset.train_labels,
                cath_level="A",
            ),
            "accuracy_t": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_CATH,
                class_names_training=self.dataset.train_labels,
                cath_level="T",
            ),
            "accuracy_h": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_CATH,
                class_names_training=self.dataset.train_labels,
                cath_level="H",
            ),
        }
        eval_dict["accuracy_avg"] = ((
                                    eval_dict["accuracy_c"]
                                    + eval_dict["accuracy_a"]
                                    + eval_dict["accuracy_t"]
                                    + eval_dict["accuracy_h"]
                                    ) / 4)
        # Print out the results:
        print("\nRESULTS - FOR PREDICTIONS USING LOCAL CLASSIFIERS PER LEVEL")
        print("-----------------------------------------------------------")
        print("USED MODELS:")
        print(f"Model for level C : {self.models[0]}")
        print(f"Model for level A : {self.models[1]}")
        print(f"Model for level T : {self.models[2]}")
        print(f"Model for level H : {self.models[3]}")
        print("-----------------------------------------------------------")
        print("STATISTICS:")
        print(f"Accuracy C   : {eval_dict['accuracy_c']}")
        print(f"Accuracy A   : {eval_dict['accuracy_a']}")
        print(f"Accuracy T   : {eval_dict['accuracy_t']}")
        print(f"Accuracy H   : {eval_dict['accuracy_h']}")
        print(f"Accuracy AVG : {eval_dict['accuracy_avg']}")


if __name__ == "__main__":
    set_random_seeds(seed=1)
    rng = np.random.RandomState(1)
    print(f"rng = {rng}")
    # load data
    dataset = load_data(
        data_dir=DATA_DIR,
        rng=rng,
        without_duplicates=True,
        shuffle_data=True,
        reloading_allowed=True,
        specific_level="T"
    )
    dataset.scale()
    print(dataset.train_labels[:10])

