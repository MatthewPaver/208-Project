import keras_tuner.src.tuners.randomsearch as rs
from keras_tuner.src.engine import trial as trial_module
import datetime

class myotheroracle(rs.RandomSearchOracle):
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
        trial_id = None
    ):
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