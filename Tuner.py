"""
This module defines a custom grid search tuner with the ability to save and reload
each epoch without loosing any data
"""

from keras_tuner import tuners
from keras_tuner.src.engine import tuner_utils
import copy
from Callback import MyCallback
import os
import pickle
from tensorflow.keras.optimizers import Adam
import json
import collections

class MyTuner(tuners.GridSearch):
    def __init__(self,
                 hypermodel=None,
                 objective=None,
                 max_trials=None,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 max_retries_per_trial=0,
                 max_consecutive_failed_trials=3,
                 **kwargs):

        """
        Instantiates the custom tuner which inherits from GridSearch

        :param hypermodel: see superclass method for more information
        :param objective: see superclass method for more information
        :param max_trials: see superclass method for more information
        :param seed: see superclass method for more information
        :param hyperparameters: see superclass method for more information
        :param tune_new_entries: see superclass method for more information
        :param allow_new_entries: see superclass method for more information
        :param max_retries_per_trial: see superclass method for more information
        :param max_consecutive_failed_trials: see superclass method for more information
        :param kwargs: see superclass method for more information
        """

        self.reloaded = False

        super(MyTuner, self).__init__(hypermodel,
                                      objective,
                                      max_trials,
                                      seed,
                                      hyperparameters,
                                      tune_new_entries,
                                      allow_new_entries,
                                      max_retries_per_trial,
                                      max_consecutive_failed_trials,
                                      **kwargs)


    def run_trial(self, trial, *args, **kwargs):
        """
        Calls methods to build model, tuner, callbacks and begin training

        This method is copied from the superclass methods run_trial and _build_and_fit_model
        the two methods have been combined into one with some additional code added to test if
        the program has been reloaded after being interrupted previously. If it has then the model
        and optimiser states are retrieved and the loaded back into the current trial. Training
        then resumes at the last fully completed epoch

        :param trial: The current trial
        :param args: See superclass method for more information
        :param kwargs: See superclass method for more information
        :return: The return value of `model.fit(), a dictionary or a float
        """

        hp = trial.hyperparameters
        model = self._try_build(hp)
        save_directory = os.path.join(self.get_trial_dir(trial.trial_id), "saves")
        if self.reloaded:
            epoch = self.find_latest_epoch(save_directory)

            with open(os.path.join(save_directory, f"generator_optimiser_epoch_{epoch}.pkl"), 'rb') as f:
                generator_optimizer_config = pickle.load(f)
            model.generator_optimizer = Adam(**generator_optimizer_config)

            with open(os.path.join(save_directory, f"discriminator_optimiser_epoch_{epoch}.pkl"), 'rb') as f:
                discriminator_optimizer_config = pickle.load(f)
            model.discriminator_optimizer = Adam(**discriminator_optimizer_config)

            model.generator.load_weights(os.path.join(save_directory, f"generator_epoch_{epoch}.h5"))
            model.discriminator.load_weights(os.path.join(save_directory, f"discriminator_epoch_{epoch}.h5"))

            kwargs['epochs'] -= epoch
            print("Loaded model weights")
            print(f"Resuming at the end of epoch: {epoch}")
            print(f"Remaining epochs: {kwargs['epochs']}")
            self.reloaded = False
        self.save()
        model_checkpoint = MyCallback(save_directory)
        original_callbacks = kwargs.pop("callbacks", [])
        copied_kwargs = copy.copy(kwargs)
        callbacks = self._deepcopy_callbacks(original_callbacks)
        self._configure_tensorboard_dir(callbacks, trial, 0)
        callbacks.append(tuner_utils.TunerCallback(self, trial))
        callbacks.append(model_checkpoint)
        copied_kwargs["callbacks"] = callbacks
        results = self.hypermodel.fit(hp, model, *args, **copied_kwargs)
        return results

    def find_latest_epoch(self, save_directory):
        """
        Searches the save_directory and returns the last epoch that successfully saved

        :param save_directory: The directory to be searched
        :return: The last completed epoch
        """
        epochs_seen = []
        for file in os.listdir(save_directory):
            if file.startswith("generator_epoch_"):
                epochs_seen.append(int(file.split("_")[2].split(".")[0]))
        if epochs_seen:
            return max(epochs_seen)
        else:
            return 0

    def reload(self):
        """
        Reloads the custom tuner and all necessary values

        Due to an oversight in keras_tuner this is needed to reload a GridSearch tuner
        as _ordered_ids and _populate_next are not reloaded in the superclass
        version of this method
        """
        self.reloaded = True
        directory = os.path.join(self.project_dir, "GridSearch.json", )
        with open(directory, "r") as file:
            loaded_data = json.load(file)
        self.from_dict(loaded_data["_ordered_ids"])
        self.oracle._populate_next = loaded_data["_populate_next"]
        super(MyTuner, self).reload()

    def save(self):
        """
        Saves the current state of _ordered_ids and _populate_next

        Due to an oversight in keras_tuner this is needed to reload a GridSearch tuner
        as _ordered_ids and _populate_next are not saved in the superclass
        version of this method
        """

        data = {
            '_ordered_ids': self.to_dict(),
            '_populate_next': self.oracle._populate_next
        }
        directory = os.path.join(self.project_dir, "GridSearch.json",)
        with open(directory, "w") as file:
            json.dump(data, file)
        super(MyTuner, self).save()

    def to_dict(self):
        """
        Serialises the _ordered_ids linked list

        Since only serialised objects can be stored in a.json format this is needed
        to extract and serialise the _ordered_ids before saving to disk

        :return: A dictionary containing relevant values of the linked list
        _memory, _data_to_index, _next_index, _last_index
        """
        linked_list = self.oracle._ordered_ids
        next_index_dict = dict(linked_list._next_index)
        return {
            "_memory":linked_list._memory,
            "_data_to_index": linked_list._data_to_index,
            "_next_index": next_index_dict,
            "_last_index": linked_list._last_index
        }

    def from_dict(self, data):
        """
        Reloads the json data back into the linked list _ordered_ids
        
        :param data: The json data fetched from GridSearch.json
        """
        self.oracle._ordered_ids._memory= data["_memory"]
        self.oracle._ordered_ids._data_to_index = data["_data_to_index"]
        self.oracle._ordered_ids._next_index = collections.defaultdict(lambda: None, data["_next_index"])
        self.oracle._ordered_ids._last_index = data["_last_index"]