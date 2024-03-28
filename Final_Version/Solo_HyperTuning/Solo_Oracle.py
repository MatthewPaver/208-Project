"""
This module sets up a custom oracle to handle trials when its just one machine running it
"""

import keras_tuner.src.tuners.gridsearch as gs
from keras_tuner.src.engine import trial as trial_module


class MyOracle(gs.GridSearchOracle):
    def __init__(
            self,
            trial_id=None,
            objective=None,
            max_trials=None,
            seed=None,
            hyperparameters=None,
            allow_new_entries=True,
            tune_new_entries=True,
            max_retries_per_trial=0,
            max_consecutive_failed_trials=3,
    ):
        """
        Instantiates a Distributed Oracle to handle a distributed trial

        Sets self.trial_id and then passes to superclass

        :param objective: Refer to superclass DocString
        :param max_trials: Refer to superclass DocString
        :param seed: Refer to superclass DocString
        :param hyperparameters: The set of values to run this trial with
        :param allow_new_entries: Refer to superclass DocString
        :param tune_new_entries: Refer to superclass DocString
        :param max_retries_per_trial: Refer to superclass DocString
        :param max_consecutive_failed_trials: Refer to superclass DocString
        :param trial_id: The trial ID to set the trial as
        """
        self.trial_id = trial_id
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )

    def create_trial(self, tuner_id):
        """
        Creates a trial with trial_id and hyperparameters specified

        This method copies and modifies the superclass method to take control of the trial
        creation process in order to set the trial id for a trial.

        :param tuner_id: The id of the tuner requesting a trial.
        :return: A trial instance
        """
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        self.tuner_ids.add(tuner_id)

        if len(self._retry_queue) > 0:
            trial = self.trials[self._retry_queue.pop()]
            trial.status = trial_module.TrialStatus.RUNNING
            self.ongoing_trials[tuner_id] = trial
            self.save()
            self._display.on_trial_begin(trial)
            return trial

        if self.trial_id:
            trial_id = self.trial_id
        else:
            trial_id = f"{{:0{len(str(self.max_trials))}d}}"
        trial_id = trial_id.format(len(self.trials))

        if self.max_trials and len(self.trials) >= self.max_trials:
            status = trial_module.TrialStatus.STOPPED
            values = None
        else:
            response = self.populate_space(trial_id)
            status = response["status"]
            values = response["values"] if "values" in response else None

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values or {}

        trial = trial_module.Trial(
            hyperparameters=hyperparameters, trial_id=trial_id, status=status
        )

        if status == trial_module.TrialStatus.RUNNING:
            self._record_values(trial)

            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial
            self.start_order.append(trial_id)
            self._save_trial(trial)
            self.save()
            self._display.on_trial_begin(trial)

        if status == trial_module.TrialStatus.STOPPED:
            self.tuner_ids.remove(tuner_id)

        return trial
