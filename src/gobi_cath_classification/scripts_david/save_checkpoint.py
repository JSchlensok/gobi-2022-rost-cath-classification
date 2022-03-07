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


def load_configuration(directory: Path) -> dict:
    ########################################################################################
    # FUNCTION NAME     : load_model()
    # INPUT PARAMETERS  : unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model configuration
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # TODO: use json.load() to load config dict from params.json
    modelConfiguration = None
    for file in listdir(str(directory)):
        if str(file).__contains__(f"Model Configuration"):
            modelConfiguration = file
    if modelConfiguration is None:
        raise ValueError("Assigned folder does not contain a configuration file")
    # Return the configuration
    configurationfile = open(directory / modelConfiguration, "r")
    configurationcontent = configurationfile.read()
    config_lines = configurationcontent.split("--->>><<<---")[1].split("\n")
    config_dict = {}
    for line in config_lines:
        if line != "":
            if re.match(r"model -:- .*", line):
                model_dict = line.split("{")[1].split("}")[0].split(", '")
                for config in model_dict:
                    config = config.replace("'", "")
                    config_dict[config.split(": ")[0]] = config.split(": ")[1]
            else:
                config_dict[line.split(" -:- ")[0]] = line.split(" -:- ")[1]
    configurationfile.close()
    return config_dict


def load_results(directory: Path):
    ########################################################################################
    # FUNCTION NAME     : load_results()
    # INPUT PARAMETERS  : unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model result
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    # TODO: use json.load() to load results dict from result.json and maybe iterate over dict
    #  to get the highest accuracy and the according epoch.
    modelResults = None
    for file in listdir(str(directory)):
        if str(file).__contains__(f"Model Results"):
            modelResults = file
    if modelResults is None:
        raise ValueError("Assigned folder does not contain a result file")
    # Return the configuration
    resultfile = open(directory / modelResults, "r")
    resultcontent = resultfile.read().split("\n")

    eval_dict = {}
    for line in resultcontent:
        if line != "":
            if line.__contains__(" -:- "):
                eval_dict[line.split(" -:- ")[0]] = line.split(" -:- ")[1]
    resultfile.close()
    return eval_dict, Decimal(eval_dict["accuracy_h"])


def load_model(directory: Path):
    ########################################################################################
    # FUNCTION NAME     : load_model()
    # INPUT PARAMETERS  : unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : ---
    ########################################################################################
    myModelFile = None
    highest_epoch = -1
    for file in listdir(str(directory)):
        if str(file).__contains__(f"Model Checkpoint"):
            if int(str(file).split("Epoch ")[1].split(".pt")[0]) > highest_epoch:
                myModelFile = file
                highest_epoch = int(str(file).split("Epoch ")[1].split(".pt")[0])
    if myModelFile is None:
        raise ValueError("Assigned folder does not contain a checkpoint file")
    # Return the model and the current epoch
    return (
        torch.load(directory / myModelFile),
        int(highest_epoch),
    )


def remove_files(filetype: str, directory: Path):
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
    allFilesInDirectory = [file for file in listdir(str(directory))]
    # Get the folder with the correct UUID
    # TODO: instead of searching for correct directory I would recommend to forward the correct
    #  directory in the input parameter "directory"
    # myDirectory = None
    # for file in allFilesInDirectory:
    #     if str(file).__contains__(f"{unique_ID}"):
    #         myDirectory = file
    # Find the correct model file
    # if myDirectory is not None:
    #     allFilesInDirectory = [file for file in listdir(str(directory / str(myDirectory)))]
    #     for file in allFilesInDirectory:
    #         if str(file).__contains__(f"{filetype}"):
    #             os.remove(directory / file)
    for file in listdir(str(directory)):
        if str(file).__contains__(f"{filetype}"):
            os.remove(directory / file)
