from keras_tuner import tuners
from keras_tuner.src.engine import tuner_utils
import copy
import Callback

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
        self.reloaded = False

    def reload(self):
        self.reloaded = True
        super(MyTuner, self).reload()

    def save(self):
        super(MyTuner, self).save()


#TODO: - finish implementing run trial and custom callback
    def run_trial(self, trial, *args, **kwargs):
        """
        Overwrites the superclass run_trial to allow reloading mid-trial

        Majority of code copied from superclass with some modifications to allow models
        to be reloaded after a crash or termination. The model is loaded with all the
        weights and gradients of the last completed epoch.

        :param trial:
        :param args:
        :param kwargs:
        :return:
        """
        model_checkpoint = Callback.MyCallback()
        original_callbacks = kwargs.pop("callbacks", [])
        copied_kwargs = copy.copy(kwargs)
        callbacks = self._deepcopy_callbacks(original_callbacks)
        callbacks.append(tuner_utils.TunerCallback(self, trial))
        self._configure_tensorboard_dir(callbacks, trial, 0)
        callbacks.append(model_checkpoint)
        copied_kwargs["callbacks"] = callbacks
        results = self._build_and_fit_model(trial, *args, **copied_kwargs)
        return results