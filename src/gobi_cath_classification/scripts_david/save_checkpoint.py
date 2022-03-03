# Import basic functionalities
import os
import re
import uuid
import datetime
from os import listdir
from pathlib import Path
from decimal import *

# Import classes specifically needed for machine learning
import torch

# Import own classes needed in this script
from gobi_cath_classification.scripts_david.models import (
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
    # INPUT PARAMETERS  : model_class: str, unique_ID: uuid, dict_config: dict,
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Saves the configuration of a model into a file
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 03.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Direct the filepath into the model checkpoint folder and append the file name
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    # Format the filepath to a distinguishable name
    directory_path = f"{str(directory)}\\model checkpoints\\{str(model_class)} {str(unique_ID)}"
    # Create the directory if not existent
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    # Concatenate filename for new checkpoint file
    str_filename = f"{model_class} Model Configuration"
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
        epoch: int,
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
    # Remove all previous checkpoints
    remove_files(filetype="Model Results", unique_ID=unique_ID)
    # Direct the filepath into the model checkpoint folder and append the file name
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    # Format the filepath to a distinguishable name
    directory_path = f"{str(directory)}\\model checkpoints\\{str(model_class)} {str(unique_ID)}"
    # Create the directory if not existent
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    # Create filename
    str_filename = f"{model_class} Model Results - Epoch {epoch}"
    # Concatenate the file path
    file_name = f"{directory_path}\\{str_filename}.rs"
    # Create and open the file
    checkpoint_file = open(file_name, "w")
    # Write the file header
    checkpoint_file.write(f"RESULT FILE\nResults for : {model_class}\nCreated on: "
                          f"{datetime.datetime.now()}\n\nSaved on Epoch - {str(epoch)}\n")
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
    # Remove all previous checkpoints
    remove_files(filetype="Model Checkpoint", unique_ID=unique_ID)
    # Direct the filepath into the model checkpoint folder and append the file name
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    # Format the filepath to a distinguishable name
    directory_path = f"{str(directory)}\\model checkpoints\\{str(model_class)} {str(unique_ID)}"
    # Create the directory if not existent
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    # Append the actual filename to the path
    file_path = directory_path + f"\\{model_class} Model Checkpoint - Epoch {str(epoch)}.pt"
    print(f"Attempting to save {model_class} as intermediate checkpoint...")
    # Saving the model as intermediate checkpoint using torch
    if model_class == NeuralNetworkModel.__name__:  # CLASS - Neural Network
        torch.save(model, file_path)
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

def load_configuration(
        unique_ID: uuid
):
    ########################################################################################
    # FUNCTION NAME     : load_model()
    # INPUT PARAMETERS  : unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model configuration
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Find the correct directory in the folder
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints")]
    # Get the folder with the correct UUID
    myDirectory = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{unique_ID}"):
            myDirectory = file
    if myDirectory is None:
        raise ValueError("Unique Key could not be found")
    # Find the correct model file
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints\\{myDirectory}")]
    modelConfiguration = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"Model Configuration"):
            modelConfiguration = file
    if modelConfiguration is None:
        raise ValueError("Assigned folder does not contain a configuration file")
    # Return the configuration
    configurationfile = open(f"{str(directory)}\\model checkpoints\\{myDirectory}\\{modelConfiguration}", "r")
    configurationcontent = configurationfile.read()
    config_lines = configurationcontent.split("--->>><<<---")[1].split("\n")
    config_dict = {}
    for line in config_lines:
        if line != "":
            if re.match(r"model -:- .*",line):
                model_dict = line.split("{")[1].split("}")[0].split(", '")
                for config in model_dict:
                    config = config.replace("'","")
                    config_dict[config.split(": ")[0]] = config.split(": ")[1]
            else:
                config_dict[line.split(" -:- ")[0]] = line.split(" -:- ")[1]
    configurationfile.close()
    return(config_dict)

def load_results(
        unique_ID: uuid
):
    ########################################################################################
    # FUNCTION NAME     : load_results()
    # INPUT PARAMETERS  : unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model result
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Find the correct directory in the folder
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints")]
    # Get the folder with the correct UUID
    myDirectory = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{unique_ID}"):
            myDirectory = file
    if myDirectory is None:
        raise ValueError("Unique Key could not be found")
    # Find the correct model file
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints\\{myDirectory}")]
    modelResults = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"Model Results"):
            modelResults = file
    if modelResults is None:
        raise ValueError("Assigned folder does not contain a result file")
        # Return the configuration
    resultfile = open(f"{str(directory)}\\model checkpoints\\{myDirectory}\\{modelResults}", "r")
    resultcontent = resultfile.read().split("\n")
    epoch = 0
    eval_dict = {}
    for line in resultcontent:
        if line != "":
            if line.__contains__(" -:- "):
                eval_dict[line.split(" -:- ")[0]] = line.split(" -:- ")[1]
    resultfile.close()
    return eval_dict, Decimal(eval_dict["accuracy_avg"])

def load_model(
        unique_ID: uuid
):
    ########################################################################################
    # FUNCTION NAME     : load_model()
    # INPUT PARAMETERS  : unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Find the correct directory in the folder
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints")]
    # Get the folder with the correct UUID
    myDirectory = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{unique_ID}"):
            myDirectory = file
    if myDirectory is None:
        raise ValueError("Unique Key could not be found")
    # Find the correct model file
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints\\{myDirectory}")]
    myModelFile = None
    highest_epoch = -1
    for file in allFilesInDirectory:
        if str(file).__contains__(f"Model Checkpoint"):
            if int(str(file).split("Epoch ")[1].split(".pt")[0]) > highest_epoch:
                myModelFile = file
                highest_epoch = int(str(file).split("Epoch ")[1].split(".pt")[0])
    if myModelFile is None:
        raise ValueError("Assigned folder does not contain a checkpoint file")
    # Return the model and the current epoch
    return torch.load(f"{directory}\\model checkpoints\\{myDirectory}\\{myModelFile}"), int(highest_epoch), unique_ID

def remove_files(
        filetype: str,
        unique_ID: uuid
):
    ########################################################################################
    # FUNCTION NAME     : remove_files()
    # INPUT PARAMETERS  : filetype: str, unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Removes all files of a specific purpose from a directory
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # Find the correct directory in the folder
    directory = Path(os.path.abspath(__file__)).parent.parent.absolute()
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints")]
    # Get the folder with the correct UUID
    myDirectory = None
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{unique_ID}"):
            myDirectory = file
    if myDirectory is None:
        raise ValueError("Unique Key could not be found")
    # Find the correct model file
    allFilesInDirectory = [file for file in listdir(f"{str(directory)}\\model checkpoints\\{myDirectory}")]
    for file in allFilesInDirectory:
        if str(file).__contains__(f"{filetype}"):
            os.remove(f"{str(directory)}\\model checkpoints\\{myDirectory}\\{str(file)}")