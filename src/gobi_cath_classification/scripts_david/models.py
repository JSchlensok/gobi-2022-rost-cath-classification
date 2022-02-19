# Import basic functionalities
from pathlib import Path
from typing import List, Optional
# Import classes specifically needed for machine learning
import numpy as np
import pandas as pd
import torch
# Import own classes needed in this script
from src.gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
# Import library for support vector machines from scikit learn
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

    def __init__(
            self,
            c=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale'
    ):
        ########################################################################################
        # FUNCTION NAME     : __init__()
        # INPUT PARAMETERS  : self
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Constructor to create instance of class SupportVectorMachine
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################
        # Parameter description for SVMs
        # GAMMA             -   Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        # REGULARIZATION C  -   The strength of the regularization is inversely proportional to C. Must be strictly
        #                       positive. The penalty is a squared l2 penalty.
        # KERNEL            -   Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will
        #                       be used. If a callable is given it is used to pre-compute the kernel matrix from data
        #                       matrices;

        self.model = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)


    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> None:
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

    def predict(self, embeddings: np.ndarray) -> Prediction:
        ########################################################################################
        # FUNCTION NAME     : predict()
        # INPUT PARAMETERS  : self
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Create predictions with SVM-model given correct input parameters
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        predictions = self.model.predict(X=embeddings)
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
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        ########################################################################################
        # FUNCTION NAME     : load_model_from_checkpoint()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Load a specific, previously saved, state of the model
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        raise NotImplementedError