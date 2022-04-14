# Different approaches to hierarchically classify the CATH H-domain

# IMPORT STATEMENTS:
import torch
import decimal
import numpy as np
from typing_extensions import Literal
from typing import List

from gobi_cath_classification.pipeline.utils.CATHLabel import CATHLabel
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds
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
        except Exception as e:
            raise ValueError("Failed to read models from list 'models'!\n{e}")

        if classifier_type == "LCL":
            print("Commencing classification with 'Local Classifier Per Level'")
            if len(self.models) != 4:
                raise ValueError("Wrong number of models supplied! Required 4")
        elif classifier_type == "LCPN":
            print("Commencing classification with 'Local Classifier Per Parent Node'")
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
            reloading_allowed=True,
        )
        self.dataset.scale()
        print("Data successfully read in!")

    def predict_lcl(self):
        ########################################################################################
        # FUNCTION NAME     : predict_lcl()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Predict each level with a local classifier per level
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
        except Exception as e:
            print(f"FAILED to assign models...\n{e}")
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
        eval_dict = self.return_evaluation(labels_CATH)
        # Print out the results:
        print("\n\nRESULTS - FOR PREDICTIONS USING LOCAL CLASSIFIERS PER LEVEL")
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
        print("-----------------------------------------------------------\n\n")

    def predict_lcpn(self, threshold, prediction: Literal["AVG", "H"]):
        ########################################################################################
        # FUNCTION NAME     : predict_lcpn()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Predict each level with a local classifier per parent node
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 12.03.2022
        # UPDATE            : ---
        ########################################################################################
        print(f"PREDICTING CATH-LABELS BASED ON LABEL PROBABILITY - {prediction}")
        # Load the model for each CATH level
        model_C = None
        model_A = None
        model_T = None
        model_H = None
        try:
            print("Assigning models...")
            model_C = self.models[0]
            print("Model for level C ready and standing by...")
            model_A = self.models[1]
            print("Model for level A ready and standing by...")
            model_T = self.models[2]
            print("Model for level T ready and standing by...")
            model_H = self.models[3]
            print("Model for level H ready and standing by...")
        except Exception as e:
            print("FAILED to assign models!\n{e}")
            exit(0)
        # Predict with each model
        print("Commencing to predict...")
        prediction_C = model_C.predict(embeddings=self.dataset.X_val)
        prediction_A = model_A.predict(embeddings=self.dataset.X_val)
        prediction_T = model_T.predict(embeddings=self.dataset.X_val)
        prediction_H = model_H.predict(embeddings=self.dataset.X_val)
        print("Predictions available...")
        # Check if all models returned the same number of predictions
        if (
            len(prediction_C.probabilities)
            == len(prediction_A.probabilities)
            == len(prediction_T.probabilities)
            == len(prediction_H.probabilities)
        ):
            print("Same number of predictions for every embedding available...")
        else:
            print("Not all models returned the same number of predictions!")
            exit(-1)
        labels_CATH = []
        # Loop over every line in every probabilities dataframe
        for index in range(len(prediction_C.probabilities)):
            current_prediction_C = prediction_C.probabilities.iloc[[index]]
            current_prediction_A = prediction_A.probabilities.iloc[[index]]
            current_prediction_T = prediction_T.probabilities.iloc[[index]]
            current_prediction_H = prediction_H.probabilities.iloc[[index]]
            # Give the probabilities for every embedding to the prediction function
            (
                label_prediction_AVG,
                label_probability_AVG,
                label_prediction_H,
                label_probability_H,
            ) = self.return_prediction(
                current_prediction_C,
                current_prediction_A,
                current_prediction_T,
                current_prediction_H,
                threshold,
            )
            if prediction == "AVG":
                labels_CATH.append(label_prediction_AVG.__str__())
            if prediction == "H":
                labels_CATH.append(label_prediction_H.__str__())
        # Evaluate the predictions
        eval_dict = self.return_evaluation(labels_CATH)
        # Print out the results:
        print("\n\nRESULTS - FOR PREDICTIONS USING LOCAL CLASSIFIERS PER PARENT NODE")
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
        print("-----------------------------------------------------------\n\n")

    def return_prediction(
        self,
        probs_C,
        probs_A,
        probs_T,
        probs_H,
        threshold,
    ) -> (CATHLabel, float):
        ########################################################################################
        # FUNCTION NAME     : return_prediction()
        # INPUT PARAMETERS  : probs_C, probs_A, probs_T, probs_H, treshhold
        # OUTPUT PARAMETERS : CATHlabels, probabilities
        # DESCRIPTION       : Predict each level with knowledge about the prediction of the parent node
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 15.03.2022
        # UPDATE            : ---
        ########################################################################################
        # Assign variables to default
        current_best_label_AVG = "0.0.0.0"
        proba_of_current_best_label_AVG = 0
        current_best_label_H = "0.0.0.0"
        proba_of_current_best_label_H = 0
        current_label = "0.0.0.0"
        mean_proba_of_current_label = 0
        # Get all column headers for Level C
        columns_C = probs_C.columns.values
        # C-LEVEL ------------------------------------------------------------------------------------------------------
        # Loop over all probabilities of every prediction for level C
        for current_column_C in columns_C:
            # Update current label
            current_label = f"{current_column_C}.0.0.0"
            # Get probability for the current prediction
            current_proba_C = probs_C[current_column_C].values[0]
            # Calculate mean probability over every level
            mean_proba_of_current_label = (current_proba_C + 0 + 0 + 0) / 4
            # Break if mean probability does not exceed the threshold
            if not mean_proba_of_current_label >= threshold:
                break
            # If best probability is less then the actual --> Update the best values
            elif mean_proba_of_current_label > proba_of_current_best_label_AVG:
                proba_of_current_best_label_AVG = mean_proba_of_current_label
                current_best_label_AVG = current_label
            # Get every column header that matches with the previous predictions
            columns_A = [
                column
                for column in probs_A.columns.values
                if current_column_C == CATHLabel(f"{column}.0.0")["C"]
            ]
            # A-LEVEL ------------------------------------------------------------------------------------------------------
            # Loop over all probabilities of every prediction for level A
            for current_column_A in columns_A:
                # Update current label
                current_label = f"{current_column_A}.0.0"
                # Get probability for the current prediction
                current_proba_A = probs_A[current_column_A].values[0]
                # Calculate mean probability over every level
                mean_proba_of_current_label = (current_proba_C + current_proba_A + 0 + 0) / 4
                # Break if mean probability does not exceed the threshold
                if not mean_proba_of_current_label >= threshold:
                    break
                # If best probability is less then the actual --> Update the best values
                elif mean_proba_of_current_label > proba_of_current_best_label_AVG:
                    proba_of_current_best_label_AVG = mean_proba_of_current_label
                    current_best_label_AVG = current_label
                # Get every column header that matches with the previous predictions
                columns_T = [
                    column
                    for column in probs_T.columns.values
                    if current_column_A == CATHLabel(f"{column}.0")[:"A"].__str__()
                ]
                # T-LEVEL ------------------------------------------------------------------------------------------------------
                # Loop over all probabilities of every prediction for level T
                for current_column_T in columns_T:
                    # Update current label
                    current_label = f"{current_column_T}.0"
                    # Get probability for the current prediction
                    current_proba_T = probs_T[current_column_T].values[0]
                    # Calculate mean probability over every level
                    mean_proba_of_current_label = (
                        current_proba_C + current_proba_A + current_proba_T + 0
                    ) / 4
                    # Break if mean probability does not exceed the threshold
                    if not mean_proba_of_current_label >= threshold:
                        break
                    # If best probability is less then the actual --> Update the best values
                    elif mean_proba_of_current_label > proba_of_current_best_label_AVG:
                        proba_of_current_best_label_AVG = mean_proba_of_current_label
                        current_best_label_AVG = current_label
                    # Get every column header that matches with the previous predictions
                    columns_H = [
                        column
                        for column in probs_H.columns.values
                        if current_column_T == CATHLabel(column)[:"T"].__str__()
                    ]
                    # H-LEVEL ------------------------------------------------------------------------------------------------------
                    # Loop over all probabilities of every prediction for level H
                    for current_column_H in columns_H:
                        # Update current label
                        current_label = f"{current_column_H}"
                        # Get probability for the current prediction
                        current_proba_H = probs_H[current_column_H].values[0]
                        # Calculate mean probability over every level
                        mean_proba_of_current_label = (
                            current_proba_C + current_proba_A + current_proba_T + current_proba_H
                        ) / 4
                        # Break if mean probability does not exceed the threshold
                        if not mean_proba_of_current_label >= threshold:
                            break
                        # If best probability is less then the actual --> Update the best values
                        elif mean_proba_of_current_label > proba_of_current_best_label_AVG:
                            proba_of_current_best_label_AVG = mean_proba_of_current_label
                            current_best_label_AVG = current_label
                        if current_proba_H > proba_of_current_best_label_H:
                            proba_of_current_best_label_H = current_proba_H
                            current_best_label_H = current_label
        # Return the best label
        return (
            CATHLabel(current_best_label_AVG),
            proba_of_current_best_label_AVG,
            CATHLabel(current_best_label_H),
            proba_of_current_best_label_H,
        )

    def return_evaluation(self, labels_cath: List[str]):
        ########################################################################################
        # FUNCTION NAME     : return_evaluation()
        # INPUT PARAMETERS  : labels_CATH: List[str]
        # OUTPUT PARAMETERS : eval_dict
        # DESCRIPTION       : Evaluate the predictions
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 15.03.2022
        # UPDATE            : ---
        ########################################################################################
        eval_dict = {
            "accuracy_c": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_cath,
                class_names_training=self.dataset.train_labels,
                cath_level="C",
            ),
            "accuracy_a": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_cath,
                class_names_training=self.dataset.train_labels,
                cath_level="A",
            ),
            "accuracy_t": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_cath,
                class_names_training=self.dataset.train_labels,
                cath_level="T",
            ),
            "accuracy_h": accuracy_for_level(
                y_true=self.dataset.y_val,
                y_pred=labels_cath,
                class_names_training=self.dataset.train_labels,
                cath_level="H",
            ),
        }
        eval_dict["accuracy_avg"] = (
            eval_dict["accuracy_c"]
            + eval_dict["accuracy_a"]
            + eval_dict["accuracy_t"]
            + eval_dict["accuracy_h"]
        ) / 4
        return eval_dict
