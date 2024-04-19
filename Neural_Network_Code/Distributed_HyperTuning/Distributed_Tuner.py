"""
This module details a class to handle distributed hyperparameter tuning for using
Distributed_Oracle for each trial
"""

from keras_tuner import tuners
from keras_tuner.src.engine import tuner_utils
import copy
from Neural_Network_Code.Solo_HyperTuning.Callback import MyCallback
import os
import pickle
from tensorflow.keras.optimizers import Adam
import Distributed_Oracle



class Distributed_Tuner(tuners.RandomSearch):
    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=10,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        trial_id=None,
        **kwargs
    ):
        """
        Instantiates a Distributed_Tuner object

        Method copied and modified from superclass to set reloaded flag and use my
        Distributed_Oracle class as the oracle. Passes to the grandfather class method

        :param hypermodel: Refer to superclass DocString
        :param objective: Refer to superclass DocString
        :param max_trials: Refer to superclass DocString
        :param seed: Refer to superclass DocString
        :param hyperparameters: The hyperparameters to use for the one trial to run
        :param tune_new_entries: Refer to superclass DocString
        :param allow_new_entries: Refer to superclass DocString
        :param max_retries_per_trial: Refer to superclass DocString
        :param max_consecutive_failed_trials: Refer to superclass DocString
        :param trial_id: The trial id to be used for the one trial to run
        :param kwargs: Refer to superclass DocString
        """
        self.reloaded = False
        self.seed = seed
        oracle = Distributed_Oracle.Distributed_Oracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            trial_id=trial_id,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super(tuners.RandomSearch, self).__init__(oracle, hypermodel, **kwargs)

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
            if os.path.exists(save_directory):
                epoch = self.find_latest_epoch(save_directory)

                with open(os.path.join(save_directory, f"generator_optimiser_epoch_{epoch}.pkl"), 'rb') as f:
                    generator_optimizer_config = pickle.load(f)
                model.generator_optimizer = Adam(**generator_optimizer_config)

                with open(os.path.join(save_directory, f"discriminator_optimiser_epoch_{epoch}.pkl"), 'rb') as f:
                    discriminator_optimizer_config = pickle.load(f)
                model.discriminator_optimizer = Adam(**discriminator_optimizer_config)

                model.generator.load_weights(os.path.join(save_directory, f"generator_epoch_{epoch}.weights.h5"))
                model.discriminator.load_weights(os.path.join(save_directory, f"discriminator_epoch_{epoch}.weights.h5"))

                kwargs['epochs'] -= epoch
                print("Loaded model weights")
                print(f"Resuming at the end of epoch: {epoch}")
                print(f"Remaining epochs: {kwargs['epochs']}")
                self.reloaded = False
        self.save()
        model_checkpoint = MyCallback(save_directory)
        #callback2 = MyCallback2(noise_dim=128, output_path=save_directory)
        original_callbacks = kwargs.pop("callbacks", [])
        copied_kwargs = copy.copy(kwargs)
        callbacks = self._deepcopy_callbacks(original_callbacks)
        self._configure_tensorboard_dir(callbacks, trial, 0)
        callbacks.append(tuner_utils.TunerCallback(self, trial))
        callbacks.append(model_checkpoint)
        #callbacks.append(callback2)
        copied_kwargs["callbacks"] = callbacks
        results = self.hypermodel.fit(hp, model, *args, **copied_kwargs)
        return results

    def find_latest_epoch(self, save_directory):
        """
        Searches save_directory to find last saved epoch

        :param save_directory: The directory to be searched
        :return: Int - The last saved epoch
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
        Sets reloaded flag to true then passes to superclass method
        """
        self.reloaded = True
        super().reload()
