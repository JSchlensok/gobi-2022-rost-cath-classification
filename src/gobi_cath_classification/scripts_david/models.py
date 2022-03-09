# Import basic functionalities
import os.path
from pathlib import Path
from typing import List, Optional, Dict

# Import classes specifically needed for machine learning
import numpy as np
import pandas as pd
import torch
import uuid

# Import own classes needed in this script
from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction

# Import libraries for machine learning from scikit learn and torch
from sklearn import svm


class SupportVectorMachine(ModelInterface):
    ########################################################################################
    # CLASS NAME        : SupportVectorMachine
    # IMPLEMENTS        : ModelInterface
    # DESCRIPTION       : Class to implement a Support Vector machine as machine learning
    #                     model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 18.02.2022
    # UPDATE            : ---
    ########################################################################################

    def __init__(self, c=1.0, kernel="rbf", degree=3, gamma="scale"):
        # ####################################################################################### FUNCTION NAME     :
        # __init__() INPUT PARAMETERS  : self OUTPUT PARAMETERS : none DESCRIPTION       : Constructor to create
        # instance of class SupportVectorMachine AUTHOR            : D. Mauder CREATE DATE       : 18.02.2022 UPDATE
        # : 20.02.2022 - probability = True eingefügt damit predict_proba ausgeführt werden kann : 21.02.2022 Model
        # Cache auf 1000 MB erweitert
        # ####################################################################################### Parameter
        # description for SVMs Kernel: The main  function of  the  kernel is to transform the given dataset input
        # data into the required form. There are various types of functions such as linear, polynomial, and radial
        # basis function(RBF).Polynomial and RBF are useful for non - linear hyperplane.Polynomial and RBF kernels
        # compute the separation line in the higher dimension.In some of the applications, it is suggested to use a
        # more complex kernel to separate the classes that are curved or nonlinear.This transformation can lead to
        # more accurate classifiers.
        #
        # Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain
        # regularization. Here C is the penalty parameter, which represents mis classification or error term. The mis
        # classification or error term tells the SVM optimization how much error is bearable. This is how you can
        # control the trade-off between decision boundary and mis classification term. A smaller value of C creates a
        # small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
        #
        # Gamma: A lower value  of Gamma will loosely fit the training dataset, whereas a higher value of gamma will
        # exactly fit the training dataset, which causes over - fitting.In other words, you can say a low value of gamma
        # considers only nearby  points in calculating the separation line, while the a value of gamma considers all the
        # data points in the calculation of the separation line.

        self.model = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, probability=True)
        self.model.cache_size = 1000  # Make more MBs of RAM available for model cache

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        ########################################################################################
        # FUNCTION NAME     : train_one_epoch()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Train one single epoch in SVM-model
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        self.model.fit(X=embeddings, y=labels)
        model_specific_metrics = {}
        return model_specific_metrics

    def predict(self, embeddings: np.ndarray) -> Prediction:
        ########################################################################################
        # FUNCTION NAME     : predict()
        # INPUT PARAMETERS  : self
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Create predictions with SVM-model given correct input parameters
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : 20.02.2022 - predict durch predict_proba ersetzt
        ########################################################################################

        predictions = self.model.predict_proba(X=embeddings)
        df = pd.DataFrame(data=predictions, columns=self.model.classes_)
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        ########################################################################################
        # FUNCTION NAME     : save_checkpoint()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Save the current state of the model to prevent information loss
        #                     in case of disturbances in program flow
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 07.03.2022
        # UPDATE            : ---
        ########################################################################################
        print(f"Attempting to save model 'SupportVectorMachine' in file model_object.model")
        print(f"Saving into directory: '{save_to_dir}'")
        checkpoint_file_path = os.path.join(save_to_dir, "model_object.model")
        try:
            torch.save(self, checkpoint_file_path)
            print(f"Checkpoint saved to: {checkpoint_file_path}")
        except:
            print(f"Failed to save model 'SupportVectorMachine'")

    def load_model_from_checkpoint(self, checkpoint_dir: Path):
        ########################################################################################
        # FUNCTION NAME     : load_model_from_checkpoint()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Load a specific, previously saved, state of the model
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 07.03.2022
        # UPDATE            : ---
        ########################################################################################
        print(f"Attempting to reload model 'SupportVectorMachine' from file: {checkpoint_dir}")
        try:
            model_file_path = os.path.join(checkpoint_dir, "model_object.model")
            model = torch.load(model_file_path)
            print(f"Successfully read in model!")
            return model
        except:
            print(f"Failed to read in model!")
            return None
