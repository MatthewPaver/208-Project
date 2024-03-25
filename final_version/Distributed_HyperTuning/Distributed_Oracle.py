"""
This module sets up an oracle to handle distributed hyperparameter trials.
"""

import collections
import keras_tuner.src.tuners.randomsearch as rs
from keras_tuner.src.engine import trial as trial_module


class Distributed_Oracle(rs.RandomSearchOracle):
    def __init__(
        self,
        objective=None,
        max_trials=10,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        trial_id=None
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
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )

    def create_trial(self, tuner_id):
        """
        Creates a trial with trial_id and hyperparameters specified

        This method copies and modifies the superclass method to take control of the trial
        creation process in order to set the trial id and hyperparameters for a single
        trial to those passed in. If this method is called with the same trial id it
        stops the tuning process.

        :param tuner_id: The id of the tuner requesting a trial.
        :return: A trial instance
        """
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]
        self.tuner_ids.add(tuner_id)

        if self.trial_id in self._retry_queue:
            idx = self._retry_queue.index(self.trial_id)
            trial = self.trials[self._retry_queue.pop(idx)]
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

        status = trial_module.TrialStatus.RUNNING
        values = self.hyperparameters.values
        if self.end_order:
            if self.trial_id == self.end_order[-1]:
                status = trial_module.TrialStatus.STOPPED
                values = None

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

    def set_state(self, state):
        """
        Sets the state of the oracle

        Copied and modified the superclass method to stop hyperparameter space from
        being reloaded and instead use those passed as parameters in __init__

        :param state: The state of the oracle to be set to
        """
        self.ongoing_trials = {
            tuner_id: self.trials[trial_id]
            for tuner_id, trial_id in state["ongoing_trials"].items()
        }
        self.start_order = state["start_order"]
        self.end_order = state["end_order"]
        self._run_times = collections.defaultdict(lambda: 0)
        self._run_times.update(state["run_times"])
        self._retry_queue = state["retry_queue"]
        self.seed = state["seed"]
        self._seed_state = state["seed_state"]
        self._tried_so_far = set(state["tried_so_far"])
        self._id_to_hash = collections.defaultdict(lambda: None)
        self._id_to_hash.update(state["id_to_hash"])
        self._display.set_state(state["display"])
