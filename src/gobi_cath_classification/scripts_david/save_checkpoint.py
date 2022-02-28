# Import classes specifically needed for machine learning
import os
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint


def generate_output_dir(out_dir, file_name):
    ########################################################################################
    # FUNCTION NAME     : generate_output_dir()
    # INPUT PARAMETERS  : outdir, run_desc
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Create a new directory to save a model checkpoint
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 28.02.2022
    # UPDATE            : ---
    ########################################################################################
    # Create list for existing directories
    prev_run_dirs = []
    # Save all previous running directories
    if os.path.isdir(out_dir):
        prev_run_dirs = [x for x in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, x))]
    # Use regular expressions to match all IDs of previous runs
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    # Filter out None list entries (regular expression reported no matches) and only return complete matches
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    # Save the highest ID, incremented by 1, as current ID (-1 if not existent)
    cur_run_id = max(prev_run_ids, default=-1) + 1
    # Concatenate new file name as running directory
    run_dir = os.path.join(out_dir, f'{cur_run_id:05d}-{file_name}')
    # Check if the filename is already taken and raise assert AssertionError if true
    assert not os.path.exists(run_dir)
    # Create new path
    os.makedirs(run_dir)
    # return the pathname
    return run_dir


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    ########################################################################################
    # FUNCTION NAME     : step_decay_schedule()
    # INPUT PARAMETERS  : initial_lr, decay_factor, step_size
    # OUTPUT PARAMETERS : LearningRateScheduler
    # DESCRIPTION       : Return a LearningRateScheduler containing step size, learning rate and decay factor
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 28.02.2022
    # UPDATE            : ---
    ########################################################################################
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


def load_model_data(model_path, opt_path, lst_params):
    ########################################################################################
    # FUNCTION NAME     : load_model_data()
    # INPUT PARAMETERS  : model_path, opt_path, lst_params
    # OUTPUT PARAMETERS : model, dict_params
    # DESCRIPTION       : Read a previously saved model from a file
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 28.02.2022
    # UPDATE            : ---
    ########################################################################################
    # load model from model path
    model = load_model(model_path)
    # open pickle file
    with open(opt_path, 'rb') as fp:
        d = pickle.load(fp)
        # Extract parameters from dictionary into a dictionary
        dict_params = {}
        for parameter in lst_params:
            try:
                # Try to extend the dictionary with the next parameter
                new_value = d[parameter]
                dict_params.update({parameter: new_value})
            except:
                # If the parameter can not be found in the dictionary, do not append to the dictionary and print an
                # informational message
                print(f"Failed to retrieve parameter: {parameter} from the given checkpoint! Check spelling or "
                      f"availability")
        # Return the model and the parameters
        return model, dict_params


class SaveModelCheckpoint(ModelCheckpoint):
    ########################################################################################
    # CLASS NAME        : SaveModelCheckpoint
    # IMPLEMENTS        : ModelCheckpoint
    # DESCRIPTION       : Class to implement the ability to save a model at a current state
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 28.02.2022
    # UPDATE            : ---
    ########################################################################################
    def __init__(self, *args, **kwargs):
        ########################################################################################
        # FUNCTION NAME     : __init__()
        # INPUT PARAMETERS  : self, *args, **kwargs
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Constructor to create instance of class SaveModelCheckpoint
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 28.02.2022
        # UPDATE            : ---
        ########################################################################################
        # Call the constructor of the parent class
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, dict_params, logs=None):
        ########################################################################################
        # FUNCTION NAME     : on_epoch_end()
        # INPUT PARAMETERS  : self, epoch, logs
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Saves a model at its current state
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 28.02.2022
        # UPDATE            : ---
        ########################################################################################
        # Print current state of the function
        print('\nEpoch %05d: Attempting to save current model state')
        # Call the on_epoch_end function of the parent class
        super().on_epoch_end(epoch, logs)
        # Get the current file path from _get_file_path
        filepath = self._get_file_path(epoch, logs)
        # Print current state of the function
        print('\nEpoch %05d: Attempting to save hyper parameters to %s' % (epoch + 1, filepath))
        # Exchange the filetype from xxx to .pkl
        filepath = filepath.rsplit(".", 1)[0]
        filepath += ".pkl"
        # Create new file at filepath
        with open(filepath, 'wb') as fp:
            # Save the model configuration within a dictionary given to the model, using pickle
            pickle.dump(dict_params, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # Print current state of the function
        print('\nEpoch %05d: Saved hyper parameters to %s' % (epoch + 1, filepath))
