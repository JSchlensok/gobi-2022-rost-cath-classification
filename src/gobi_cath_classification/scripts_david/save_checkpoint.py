# Import basic functionalities
import os
import uuid
import datetime
from os import listdir
from pathlib import Path
from os.path import isfile, join

# Import classes specifically needed for machine learning
import torch

# Import own classes needed in this script
from gobi_cath_classification.scripts_charlotte.models import (
    RandomForestModel,
    NeuralNetworkModel,
    GaussianNaiveBayesModel,
)
from gobi_cath_classification.scripts_david.models import SupportVectorMachine


def save_model_configuration(
        model_class: str,
        unique_ID: uuid,
        dict_config: dict,
):
    ########################################################################################
    # FUNCTION NAME     : save_model_configuration()
    # INPUT PARAMETERS  : str_modeltype: str,dict_config: dict
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Saves the configuration of a model into a file
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 01.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Get current datetime to make new file-name unique
    dt_datetime = datetime.datetime.now()
    # Concatenate filename for new checkpoint file
    str_filename = f"{model_class.replace(' ', '')}(-{str(unique_ID)}-)_Saved_Model_Configuration - {str(dt_datetime.strftime('%c'))}".replace(
        ":", "-")
    # Direct the directory path to the correct folder and give the folder a distinguishable name
    path = Path(os.path.abspath(__file__)).parent.parent.absolute()
    directory_path = f"{str(path)}\\model checkpoints\\{str(model_class)} {str(unique_ID)} {str(datetime.datetime.now().strftime('%F'))}"
    # Create the folder if not existent
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    # Concatenate the file path
    file_path = directory_path + f"\\{str_filename}.dt"
    # Create and open the file
    checkpoint_file = open(file_path, "w")
    # Write the file header
    checkpoint_file.write(f"CHECKPOINT FILE - SAVED CONFIGURATION\nConfiguration for : {model_class}\nCreated on: "
                          f"{datetime.datetime.now()}\nDo not alter the structure of this file to ensure no disturbance"
                          f" in program flow while reading in the checkpoint!\n--->>><<<---\n")
    # Write the given configuration into the file
    for parameter in dict_config:
        checkpoint_file.write(f"{parameter} -:- {dict_config[parameter]}\n")
    # Write footer and close the file
    checkpoint_file.write("--->>><<<---")
    checkpoint_file.close()

def save_model_results(
        model_class: str,
        unique_ID: uuid,
        eval_dict: dict,
):
    ########################################################################################
    # FUNCTION NAME     : save_model_results()
    # INPUT PARAMETERS  : model_class: str, unique_ID: uuid, eval_dict: dict,
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Saves the results of a model after training
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Find the correct directory in the folder
    path = Path(os.path.abspath(__file__)).parent.parent.absolute()
    allFilesInDirectory = [file for file in listdir(f"{str(path)}\\model checkpoints")]
    # Get the folder with the correct UUID
    myDirectory = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{unique_ID}"):
            myDirectory = file
    # Create filename
    str_filename = f"{model_class} Model Results {str(datetime.datetime.now().strftime('%F'))}"
    # Concatenate the file path
    file_path = f"{path}\\model checkpoints\\{myDirectory}\\{str_filename}.rs"
    # Create and open the file
    checkpoint_file = open(file_path, "w")
    # Write the file header
    checkpoint_file.write(f"RESULT FILE\nResults for : {model_class}\nCreated on: "
                          f"{datetime.datetime.now()}\n\n")
    # Write the given configuration into the file
    for result in eval_dict:
        checkpoint_file.write(f"{result} -:- {eval_dict[result]}\n")
    # close the file
    checkpoint_file.close()



def save_model(
        model,
        model_class: str,
        unique_ID: uuid,
        epoch: int,
):
    ########################################################################################
    # FUNCTION NAME     : save_model()
    # INPUT PARAMETERS  : model, model_class: str
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Saves the models current state
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 01.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Direct the filepath into the model checkpoint folder and append the file name
    path = Path(os.path.abspath(__file__)).parent.parent.absolute()
    # Format the filepath to a distinguishable name
    file_path = f"{str(path)}\\model checkpoints\\{str(model_class)} {str(unique_ID)} {str(datetime.datetime.now().strftime('%F'))}"
    # Create the directory if not existent
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    # Append the actual filename to the path
    file_path = file_path + f"\\Checkpoint_Epoch_{str(epoch)}.pt"
    print(f"Attempting to save {model_class} as intermediate checkpoint...")
    # Saving the model as intermediate checkpoint using torch
    if model_class == NeuralNetworkModel.__name__:  # CLASS - Neural Network
        torch.save({
            'epoch': model.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': model.loss,
        }, file_path)
        print(f"Successfully saved the models state on epoch : {str(epoch)} - to : {file_path}")
    elif model_class == RandomForestModel.__name__:  # CLASS - Random Forest
        torch.save(model, file_path)
        print(f"Successfully saved the models state on epoch : {str(epoch)} - to : {file_path}")
    elif model_class == GaussianNaiveBayesModel.__name__:  # CLASS - Gaussian Naive Bayes
        torch.save(model, file_path)
        print(f"Successfully saved the models state on epoch : {str(epoch)} - to : {file_path}")
    elif model_class == SupportVectorMachine.__name__:  # CLASS - Support Vector Machine
        torch.save(model, file_path)
        print(f"Successfully saved the models state on epoch : {str(epoch)} - to : {file_path}")
    else:
        raise ValueError(f"Model class {model_class} does not exist and can not be saved.")


def load_model(
        unique_ID: uuid
):
    ########################################################################################
    # FUNCTION NAME     : load_model()
    # INPUT PARAMETERS  : str_modeltype: str, int_modelkey: int
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Find the correct directory in the folder
    path = Path(os.path.abspath(__file__)).parent.parent.absolute()
    allFilesInDirectory = [file for file in listdir(f"{str(path)}\\model checkpoints")]
    # Get the folder with the correct UUID
    myDirectory = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{unique_ID}"):
            myDirectory = file
    if myDirectory is None:
        raise ValueError("Unique Key could not be found")
    # Find the correct model file
    allFilesInDirectory = [file for file in listdir(f"{str(path)}\\model checkpoints\\{myDirectory}")]
    myModelFile = None
    highest_epoch = -1
    for file in allFilesInDirectory:
        if str(file).__contains__(f"Checkpoint"):
            if int(str(file).split("Epoch_")[1].split(".pt")[0]) > highest_epoch:
                myModelFile = file
    if myModelFile is None:
        raise ValueError("Assigned folder does not contain a checkpoint file")
    # Return the model
    return torch.load(f"{path}\\model checkpoints\\{myDirectory}\\{myModelFile}")