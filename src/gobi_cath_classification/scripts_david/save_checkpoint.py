# Import basic functionalities
import os
import json
from os import listdir
from pathlib import Path


def load_results(checkpoint_dir: Path):
    ########################################################################################
    # FUNCTION NAME     : load_results()
    # INPUT PARAMETERS  : checkpoint_dir: Path
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Finds and reads in a previously saved model result
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : Switch on json format
    ########################################################################################
    # Create file path
    file_path = os.path.join(checkpoint_dir, "result.json")
    print(f"Reading results from: {file_path}")
    # read file
    with open(file_path, "r") as result_file:
        for line in result_file.read().split("\n"):
            if line != "":
                result_content = line

    # parse file
    if "result_content" not in locals():
        raise ValueError("No results available!")
    return json.loads(result_content)


def load_configuration(checkpoint_dir: Path):
    ########################################################################################
    # FUNCTION NAME     : load_configuration()
    # INPUT PARAMETERS  : checkpoint_dir: Path
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Reads in the configuration file from a directory
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 07.03.2022
    # UPDATE            : Switch on json format
    ########################################################################################
    # Create file path
    file_path = os.path.join(checkpoint_dir, "params.json")
    print(f"Reading configuration from: {file_path}")
    # read file
    with open(file_path, "r") as config_file:
        config_content = config_file.read()
    # parse file
    if "config_content" not in locals():
        raise ValueError("No configuration available!")
    return json.loads(config_content)


def remove_files(checkpoint_dir: Path, filetype: str):
    ########################################################################################
    # FUNCTION NAME     : remove_files()
    # INPUT PARAMETERS  : filetype: str, unique_ID: uuid
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Removes all files of a specific purpose from a directory
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 02.03.2022
    # UPDATE            : Switch on json format
    ########################################################################################
    # Get all files out of the model directory
    allFilesInCheckpointDirectory = [file for file in listdir(str(checkpoint_dir))]
    for file in allFilesInCheckpointDirectory:
        # Loop over all files
        if str(file).__contains__(f"{filetype}"):
            # If the filename contains the specified filetype --> delete
            os.remove(os.path.join(checkpoint_dir, file))
