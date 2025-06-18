import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from scipy.stats import chi2, multivariate_t
from concurrent.futures import ProcessPoolExecutor
from utils.DualProcess import generate_random_trial_sequence
import time
import ast
from scipy.special import psi
from scipy.stats import dirichlet
import copy

# Mapping of choices to pairs of options for model recovery
mapping = {
    'SetSeen': {
        ('A', 'B'): 0,
        ('C', 'D'): 1,
        ('C', 'A'): 2,
        ('C', 'B'): 3,
        ('A', 'D'): 4,
        ('B', 'D'): 5,
    },
    'KeyResponse': {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3
    }
}


def fit_participant(model, participant_id, pdata, model_type, num_iterations=1000,
                    beta_lower=-1, beta_upper=1):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_n = len(pdata['reward'])

    # get the number of parameters for the model
    k = model._PARAM_COUNT.get(model_type)

    model.iteration = 0
    best_nll = 100000  # Initialize best negative log likelihood to a large number
    best_initial_guess = None
    best_parameters = None
    best_EV = None

    for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))

        if model_type in ('decay', 'delta', 'decay_choice', 'decay_win'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999))
        elif model_type in ('delta_PVL', 'delta_PVL_relative', 'decay_PVL', 'decay_PVL_relative'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 4.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 4.9999))
        elif model_type in ('decay_fre'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(beta_lower, beta_upper)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (beta_lower, beta_upper))
        elif model_type in ('delta_asymmetric'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                         np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999))
        elif model_type in ('mean_var_utility'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 123.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 123.9999))
        elif model_type in ('delta_decay', 'sampler_decay', 'sampler_decay_PE', 'sampler_decay_AV'):
            if model.num_params == 2:
                initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
                bounds = ((0.0001, 4.9999), (0.0001, 0.9999))
            elif model.num_params == 3:
                initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                                 np.random.uniform(0.0001, 0.9999)]
                bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999))
        elif model_type == 'decay_PVPE':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 4.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 4.9999))
        elif model_type == 'WSLS':
            initial_guess = [np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 0.9999), (0.0001, 0.9999))
        elif model_type == 'WSLS_delta':
            initial_guess = [np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999))
        elif model_type in ('WSLS_delta_weight', 'WSLS_decay_weight'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999))
        elif model_type == 'ACTR':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(-1.9999, -0.0001)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (-1.9999, -0.0001))
        elif model_type == 'ACTR_Ori':
            initial_guess = [np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(-1.9999, -0.0001)]
            bounds = ((0.0001, 0.9999), (0.0001, 0.9999), (-1.9999, -0.0001))

        if model.task == 'ABCD':
            result = minimize(model.negative_log_likelihood, initial_guess,
                              args=(pdata['reward'], pdata['choice'], pdata['choiceset']),
                              bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
        elif model.task == 'IGT_SGT':
            result = minimize(model.negative_log_likelihood, initial_guess,
                              args=(pdata['reward'], pdata['choice']),
                              bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

        if result.fun < best_nll:
            best_nll = result.fun
            best_initial_guess = initial_guess
            best_parameters = result.x
            best_EV = model.EVs.copy()

    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_nll': best_nll,
        'best_initial_guess': best_initial_guess,
        'best_parameters': best_parameters,
        'best_EV': best_EV,
        'AIC': aic,
        'BIC': bic
    }

    print(f"Participant {participant_id} fitted in {(time.time() - start_time) / 60} minutes.")

    return result_dict


class ComputationalModels:
    def __init__(self, model_type, task='ABCD', num_params=2):
        """
        Initialize the Model.

        Parameters:
        - reward_means: List of mean rewards for each option.
        - reward_sd: List of standard deviations for each option.
        - model_type: Type of the model.
        - condition: Condition of the model.
        - num_trials: Number of trials for the simulation.
        - num_params: Number of parameters for the model.
        """

        self.negative_log_likelihood = None
        self.num_options = 4 # This is only a placeholder, it will be set in the fit method
        self.num_training_trials = None
        self.num_exp_restart = None
        self.num_params = num_params
        self.initial_EV = None
        self.choices_count = np.zeros(self.num_options)
        self.possible_options = [0, 1, 2, 3]
        self.memory_weights = []
        self.choice_history = []
        self.reward_history = []
        self.chosen_history = {option: [] for option in self.possible_options}
        self.reward_history_by_option = {option: [] for option in self.possible_options}
        self.AllProbs = []
        self.PE = []
        self.iteration = 0

        # define for each model_type a dict of { attr_name: param_index }
        self._PARAM_MAP = {
            'delta': {'t': 0, 'a': 1},
            'delta_asymmetric': {'t': 0, 'a': 1, 'b': 2},
            'delta_PVL': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'delta_PVL_relative': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'decay': {'t': 0, 'a': 1},
            'decay_fre': {'t': 0, 'a': 1, 'b': 2},
            'decay_choice': {'t': 0, 'a': 1},
            'decay_win': {'t': 0, 'a': 1},
            'decay_PVL': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'decay_PVL_relative': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'decay_PVPE': {'t': 0, 'a': 1, 'w': 2, 'lamda': 3},
            'delta_decay': {'t': 0, 'a': 1, 'b': 2},
            'mean_var_utility': {'t': 0, 'a': 1, 'lamda': 2},
            'sampler_decay': {'t': 0, 'a': 1},
            'sampler_decay_PE': {'t': 0, 'a': 1},
            'sampler_decay_AV': {'t': 0, 'a': 1},
            'WSLS': {'p_ws': 0, 'p_ls': 1},
            'WSLS_delta': {'a': 1, 'p_ws': 0, 'p_ls': 2},
            'WSLS_decay_weight': {'t': 0, 'a': 1, 'p_ws': 2, 'p_ls': 3, 'w': 4},
            'WSLS_delta_weight': {'t': 0, 'a': 1, 'p_ws': 2, 'p_ls': 3, 'w': 4},
            'ACTR_Ori': {'a': 1, 's': 0, 'tau': 2},
            'ACTR': {'t': 0, 'a': 1, 'tau': 2},
        }

        # any attributes you always want to have, even if None
        self._DEFAULT_ATTRS = [
            'RT_initial_suboptimal', 'RT_initial_optimal', 'RT_initial',
            'k', 's', 't', 'a', 'b', 'tau', 'lamda',
            'p_ws', 'p_ls', 'w',
        ]

        # initialize default b overrides
        if num_params == 2:
            self._B_OVERRIDES = {
                'sampler_decay': 'a',
                'sampler_decay_PE': 'a',
                'sampler_decay_AV': 'a',
                'delta_decay': 'a',
            }
        elif num_params == 3:
            self._B_OVERRIDES = {}

        # initialize all attributes to None
        for attr in self._DEFAULT_ATTRS:
            setattr(self, attr, None)

        self._PARAM_COUNT = {
            **dict.fromkeys(
                ('decay', 'delta', 'decay_choice', 'decay_win', 'WSLS'),
                2
            ),
            **dict.fromkeys(
                ('delta_asymmetric', 'decay_fre', 'ACTR', 'ACTR_Ori', 'WSLS_delta', 'mean_var_utility'),
                3
            ),
            **dict.fromkeys(
                ('delta_PVL', 'delta_PVL_relative', 'decay_PVL', 'decay_PVL_relative', 'decay_PVPE'),
                4
            ),
            **dict.fromkeys(
                ('WSLS_delta_weight', 'WSLS_decay_weight'),
                5
            ),
            **dict.fromkeys(
                ('sampler_decay', 'sampler_decay_PE', 'sampler_decay_AV', 'delta_decay'),
                None  # will use model.num_params instead
            ),
        }

        self.EVs = None
        self.Probs = None
        self.mean = None
        self.var = None
        self.AV = None

        # Model type
        self.model_type = model_type
        self.task = task

        # Mapping of updating functions to model types
        self.updating_mapping = {
            'delta': self.delta_update,
            'delta_asymmetric': self.delta_asymmetric_update,
            'delta_PVL': self.delta_PVL_update,
            'delta_PVL_relative': self.delta_PVL_relative_update,
            'decay': self.decay_update,
            'decay_fre': self.decay_fre_update,
            'decay_choice': self.decay_choice_update,
            'decay_win': self.decay_win_update,
            'delta_decay': self.delta_update,
            'decay_PVL': self.decay_PVL_update,
            'decay_PVL_relative': self.decay_PVL_relative_update,
            'decay_PVPE': self.decay_PVPE_update,
            'mean_var_utility': self.mean_var_utility,
            'sampler_decay': self.sampler_decay_update,
            'sampler_decay_PE': self.sampler_decay_PE_update,
            'sampler_decay_AV': self.sampler_decay_AV_update,
            'WSLS': self.WSLS_update,
            'WSLS_delta': self.WSLS_delta_update,
            'WSLS_delta_weight': self.WSLS_delta_weight_update,
            'WSLS_decay_weight': self.WSLS_decay_weight_update,
            'ACTR': self.ACTR_update,
            'ACTR_Ori': self.ACTR_update
        }

        self.updating_function = self.updating_mapping[self.model_type]

        # Mapping of nll functions to model types
        self.nll_mapping_ABCD = {
            'delta': self.standard_nll,
            'delta_asymmetric': self.standard_nll,
            'delta_PVL': self.standard_nll,
            'delta_PVL_relative': self.standard_nll,
            'decay': self.standard_nll,
            'decay_fre': self.standard_nll,
            'decay_choice': self.standard_nll,
            'decay_win': self.standard_nll,
            'decay_PVL': self.standard_nll,
            'decay_PVL_relative': self.standard_nll,
            'decay_PVPE': self.standard_nll,
            'delta_decay': self.standard_nll,
            'mean_var_utility': self.standard_nll,
            'sampler_decay': self.standard_nll,
            'sampler_decay_PE': self.standard_nll,
            'sampler_decay_AV': self.standard_nll,
            'WSLS': self.WSLS_nll,
            'WSLS_delta': self.WSLS_nll,
            'WSLS_delta_weight': self.WSLS_nll,
            'ACTR': self.ACTR_nll,
            'ACTR_Ori': self.ACTR_nll
        }

        self.nll_mapping_IGT_SGT = {
            'delta': self.igt_nll,
            'delta_asymmetric': self.igt_nll,
            'delta_PVL': self.igt_nll,
            'delta_PVL_relative': self.igt_nll,
            'decay': self.igt_nll,
            'decay_fre': self.igt_nll,
            'decay_choice': self.igt_nll,
            'decay_win': self.igt_nll,
            'decay_PVL': self.igt_nll,
            'decay_PVL_relative': self.igt_nll,
            'delta_decay': self.igt_nll,
            'decay_PVPE': self.igt_nll,
            'mean_var_utility': self.igt_nll,
            'sampler_decay': self.igt_nll,
            'sampler_decay_PE': self.igt_nll,
            'sampler_decay_AV': self.igt_nll,
            'WSLS': self.WSLS_nll,
            'WSLS_delta': self.WSLS_nll,
            'WSLS_delta_weight': self.WSLS_nll,
            'WSLS_decay_weight': self.WSLS_nll,
            'ACTR': self.ACTR_nll,
            'ACTR_Ori': self.ACTR_nll
        }

        if task == 'ABCD':
            self.nll_function = self.nll_mapping_ABCD[self.model_type]
        elif task == 'IGT_SGT':
            self.nll_function = self.nll_mapping_IGT_SGT[self.model_type]

        # Mapping of choice sets to pairs of options
        self.choiceset_mapping = [
            {
            0: (0, 1),
            1: (2, 3),
            2: (2, 0),
            3: (2, 1),
            4: (0, 3),
            5: (1, 3)
            },
            {
            'AB': (0, 1),
            'CD': (2, 3),
            'CA': (2, 0),
            'CB': (2, 1),
            'BD': (1, 3),
            'AD': (0, 3)
            }
        ]

        self.activation_function_mapping = {
            'ACTR_Ori': self.calculate_activation_ori,
            'ACTR': self.calculate_activation
        }

        self.softmax_mapping = {
            'delta': self.softmax,
            'delta_asymmetric': self.softmax,
            'delta_PVL': self.softmax,
            'delta_PVL_relative': self.softmax,
            'decay': self.softmax,
            'decay_fre': self.softmax,
            'decay_choice': self.softmax,
            'decay_win': self.softmax,
            'decay_PVL': self.softmax,
            'decay_PVL_relative': self.softmax,
            'decay_PVPE': self.softmax,
            'delta_decay': self.softmax,
            'mean_var_utility': self.softmax,
            'sampler_decay': self.softmax,
            'sampler_decay_PE': self.softmax,
            'sampler_decay_AV': self.softmax,
            'WSLS': self.softmax,
            'WSLS_delta': self.softmax,
            'WSLS_delta_weight': self.softmax,
            'WSLS_decay_weight': self.softmax,
            'ACTR': self.softmax,
            'ACTR_Ori': self.softmax_ACTR
        }

    def reset(self):
        """
        Reset the model and empty all the lists.
        """
        self.choices_count = np.zeros(self.num_options)
        self.memory_weights = []
        self.choice_history = []
        self.reward_history = []
        self.chosen_history = {option: [] for option in self.possible_options}
        self.reward_history_by_option = {option: [] for option in self.possible_options}
        self.AllProbs = []
        self.PE = []

        self.EVs = self.initial_EV.copy()
        self.Probs = np.full(self.num_options, 0.25)
        self.mean = np.mean(self.initial_EV.copy())
        self.AV = np.mean(self.initial_EV.copy())
        self.var = np.full(self.num_options, 1 / 12)

    def softmax(self, x):
        c = 3 ** self.t - 1
        e_x = np.exp(np.clip(c * x, -700, 700))
        return np.clip(e_x / e_x.sum(), 1e-12, 1 - 1e-12)

    def softmax_ACTR(self, x):
        """
        This is the softmax function used in the ACT-R model (Erev et al., 2010).
        It is different from the standard softmax rule used in most papers from our lab.
        We have adapted the ACT-R model to use our own implementation of softmax, but will keep an original version
        of the ACT-R model for comparison.
        """
        t = np.sqrt(2) * self.s

        e_x = np.exp(np.clip(x / t, -700, 700))
        return np.clip(e_x / e_x.sum(), 1e-12, 1 - 1e-12)

    def calculate_activation(self, appearances, current_trial):
        """
        Calculate the activation level for each previous experience for each option in the current trial.
        This version is adapted from the ACT-R model in Erev et al. (2010) using our own implementation of softmax.
        """
        s = 1 / (np.sqrt(2) * (3 ** self.t - 1))
        sum_tk = np.sum([(current_trial - t) ** (-self.a) for t in appearances])
        activation = np.log(sum_tk) + np.random.logistic(0, s)
        return activation

    def calculate_activation_ori(self, appearances, current_trial):
        """
        This is the original version of the activation calculation function used in the ACT-R model.
        """
        sum_tk = np.sum([(current_trial - t) ** (-self.a) for t in appearances])
        activation = np.log(sum_tk) + np.random.logistic(0, self.s)
        return activation

    def activation_calculation(self, option, current_trial, exclude_current_trial=False):
        # initialize lists to store activations and corresponding rewards
        activations = []
        corresponding_rewards = []

        chosen_history = self.chosen_history[option][:-1] if exclude_current_trial else self.chosen_history[option]

        for previous_trial_index, previous_trial in enumerate(chosen_history):
            activation_level = self.activation_function_mapping[self.model_type](chosen_history, current_trial)
            if activation_level > self.tau:
                activations.append(activation_level)
                corresponding_rewards.append(self.reward_history_by_option[option][previous_trial_index])

        return activations, corresponding_rewards

    def EV_calculation(self, option, rewards, activations):
        """
        Calculate the expected value for the given option based on the rewards and activations.
        In the ACT-R model, the EV is calculated as the sum of the rewards weighted by their activation probabilities
        """
        # Proceed only if there is history for the chosen option and there are activated memories
        if len(self.chosen_history[option]) > 0 and len(activations) > 0:
            probabilities = self.softmax_mapping[self.model_type](np.array(activations))
            # Calculate the expected value as the weighted mean of rewards
            self.EVs[option] = np.dot(probabilities, rewards)
        else:
            # If there are no activated memories or no history at all, set the EV to the default value
            self.EVs[option] = self.condition_initialization[self.condition]

    def determine_alt_option(self, choiceset, chosen):
        """
        Sometimes the trial type is encoded as a single digit (e.g., 0 or 1) and sometimes as a tuple (e.g., (0, 1)).
        This function determines the alternative option based on the encoding.
        """
        # Infer whether it's "OneDigit" or "TwoDigit" based on choiceset
        if isinstance(choiceset, int):
            # This implies "OneDigit"
            alt_option = self.choiceset_mapping[0][choiceset][1] if chosen == self.choiceset_mapping[0][choiceset][
                0] else \
                self.choiceset_mapping[0][choiceset][0]
        elif isinstance(choiceset, (tuple, list)) and len(choiceset) == 2:
            # This implies "TwoDigit"
            alt_option = choiceset[1] if chosen == choiceset[0] else choiceset[0]
        else:
            print(choiceset)
            raise ValueError("Unknown choiceset format")

        return alt_option

    def simulation_unpacker(self, dict):
        all_sims = []

        for res in dict:
            sim_num = res['simulation_num']
            a_val = res['a']
            b_val = res['b']
            t_val = res['t']
            lambda_val = res['lamda']
            tau_val = res['tau']
            for trial_idx, trial_detail in zip(res['trial_indices'], res['trial_details']):
                data_row = {
                    'simulation_num': sim_num,
                    'trial_index': trial_idx,
                    'a': a_val,
                    'b': b_val,
                    't': t_val,
                    'tau': tau_val,
                    'lambda': lambda_val,
                    'pair': trial_detail['pair'],
                    'choice': trial_detail['choice'],
                    'reward': trial_detail['reward'],
                }
                all_sims.append(data_row)

        return pd.DataFrame(all_sims)

    # ==================================================================================================================
    # Now we define the updating function for each model
    # ==================================================================================================================
    def delta_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def delta_asymmetric_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += (prediction_error > 0) * self.a * prediction_error + (prediction_error < 0) * self.b * \
                            prediction_error

    def delta_PVL_update(self, chosen, reward, trial, choiceset=None):
        utility = (np.abs(reward) ** self.b) * (reward >= 0) + (-self.lamda * (np.abs(reward) ** self.b)) * (reward < 0)
        prediction_error = utility - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def delta_PVL_relative_update(self, chosen, reward, trial, choiceset=None):
        reward_diff = reward - self.AV
        utility = (np.abs(reward_diff) ** self.b) * (reward_diff >= 0) + (-self.lamda * (np.abs(reward_diff) ** self.b)) * (reward_diff < 0)
        prediction_error = utility - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error
        self.AV += self.a * reward_diff

    def decay_update(self, chosen, reward, trial, choiceset=None):
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += reward

    def decay_fre_update(self, chosen, reward, trial, choiceset=None):
        multiplier = self.choices_count[chosen] ** (-self.b)
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += reward * multiplier

    def decay_choice_update(self, chosen, reward, trial, choiceset=None):
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += 1

    def decay_win_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.AV
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += (prediction_error > 0)
        self.AV += prediction_error * self.a

    def decay_PVL_update(self, chosen, reward, trial, choiceset=None):
        utility = (np.abs(reward) ** self.b) * (reward >= 0) + (-self.lamda * (np.abs(reward) ** self.b)) * (reward < 0)
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += utility

    def decay_PVL_relative_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.AV
        utility = ((np.abs(prediction_error) ** self.b) * (prediction_error >= 0) +
                   (-self.lamda * (np.abs(prediction_error) ** self.b)) * (prediction_error < 0))
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += utility
        self.AV += self.a * prediction_error

    def decay_PVPE_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.AV
        utility = ((prediction_error >= 0) * (1 - self.w) * (np.abs(prediction_error) ** self.lamda) +
                            (prediction_error < 0) * self.w * (np.abs(prediction_error) ** self.lamda))
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.EVs[chosen] += utility
        self.AV += self.a * prediction_error

    def delta_decay_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error
        self.EVs = [x * (1 - self.b) for x in self.EVs]

    def mean_var_utility(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.mean[chosen]
        self.mean[chosen] += self.a * prediction_error
        self.var[chosen] += self.a * (prediction_error ** 2 - self.var[chosen])
        self.EVs[chosen] = self.mean[chosen] - (self.lamda * self.var[chosen]) / 2

    def sampler_decay_update(self, chosen, reward, trial, choiceset=None):
        """
        This sampler model is similar to the ACT-R model, but it takes all past trials into account.
        The parameter b consistently decays the weights of past trials over time.
        """

        self.reward_history.append(reward)
        self.choice_history.append(chosen)
        self.memory_weights.append(1)

        # Decay weights of past trials and EVs
        self.EVs = self.EVs * (1 - self.a)
        self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

        # Compute the probabilities from memory weights
        total_weight = sum(self.memory_weights)
        self.AllProbs = [w / total_weight for w in self.memory_weights]

        # Update EVs based on the samples from memory
        for j in range(len(self.reward_history)):
            self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.reward_history[j]

    def sampler_decay_PE_update(self, chosen, reward, trial, choiceset=None):
        """
        In this version of the sampler model, we use prediction errors between the actual reward and the EV of
        the chosen option as reward instead of actual rewards.
        """

        self.reward_history.append(reward)
        self.choice_history.append(chosen)
        self.memory_weights.append(1)

        # use the following code if you want to use prediction errors as reward instead of actual rewards
        prediction_error = self.a * (reward - self.EVs[chosen])
        self.PE.append(prediction_error)

        # Decay weights of past trials and EVs
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

        # Compute the probabilities from memory weights
        total_weight = sum(self.memory_weights)
        self.AllProbs = [w / total_weight for w in self.memory_weights]

        # Update EVs based on the samples from memory
        for j in range(len(self.reward_history)):
            self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.PE[j]

    def sampler_decay_AV_update(self, chosen, reward, trial, choiceset=None):
        """
        In this version of the sampler model, we use prediction errors between the actual reward and the average
        reward over all options as reward instead of actual rewards.
        """

        self.reward_history.append(reward)
        self.choice_history.append(chosen)
        self.memory_weights.append(1)

        # use the following code if you want to use average reward as reward instead of actual rewards
        prediction_error = self.a * (reward - self.AV)
        self.AV += prediction_error
        self.PE.append(prediction_error)

        # Decay weights of past trials and EVs
        self.EVs = [x * (1 - self.a) for x in self.EVs]
        self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

        # Compute the probabilities from memory weights
        total_weight = sum(self.memory_weights)
        self.AllProbs = [w / total_weight for w in self.memory_weights]

        # Update EVs based on the samples from memory
        for j in range(len(self.reward_history)):
            self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.PE[j]

    def WSLS_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.AV
        self.AV += prediction_error / (trial + 1)

        pos = int(prediction_error > 0)
        chosen_prob = pos * self.p_ws + (1 - pos) * (1 - self.p_ls)
        other_prob = pos * (1 - self.p_ws) + (1 - pos) * self.p_ls
        self.Probs[:] = [other_prob] * self.num_options
        self.Probs[chosen] = chosen_prob

    def WSLS_delta_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.AV
        self.AV += prediction_error * self.a

        pos = int(prediction_error > 0)
        chosen_prob = pos * self.p_ws + (1 - pos) * (1 - self.p_ls)
        other_prob = pos * (1 - self.p_ws) + (1 - pos) * self.p_ls
        self.Probs[:] = [other_prob] * self.num_options
        self.Probs[chosen] = chosen_prob

    def WSLS_delta_weight_update(self, chosen, reward, trial, choiceset=None):
        # WSLS
        prediction_error = reward - self.AV
        self.AV += prediction_error / (trial + 1)

        pos = int(prediction_error > 0)
        chosen_prob = pos * self.p_ws + (1 - pos) * (1 - self.p_ls)
        other_prob = pos * (1 - self.p_ws) + (1 - pos) * self.p_ls
        self.Probs[:] = [other_prob] * self.num_options
        self.Probs[chosen] = chosen_prob

        # Decay
        prediction_error_per_option = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error_per_option

    def WSLS_decay_weight_update(self, chosen, reward, trial, choiceset=None):
        # WSLS
        prediction_error = reward - self.AV
        self.AV += prediction_error / (trial + 1)

        pos = int(prediction_error > 0)
        chosen_prob = pos * self.p_ws + (1 - pos) * (1 - self.p_ls)
        other_prob = pos * (1 - self.p_ws) + (1 - pos) * self.p_ls
        self.Probs[:] = [other_prob] * self.num_options
        self.Probs[chosen] = chosen_prob

        # Decay
        self.EVs[chosen] += reward
        self.EVs = self.EVs * (1 - self.a)

    def ACTR_update(self, chosen, reward, trial, choiceset=None):

        self.reward_history.append(reward)
        self.choice_history.append(chosen)

        # Update the appearance and reward history for the current combination
        current_trial = len(self.choice_history)
        self.chosen_history[chosen].append(current_trial)
        self.reward_history_by_option[chosen].append(reward)

        # determine the alternative option
        alt_option = self.determine_alt_option(choiceset, chosen)

        # Calculate activation levels for previous occurrences of the current combination
        activations, corresponding_rewards = self.activation_calculation(chosen, current_trial,
                                                                         exclude_current_trial=True)
        alt_activations, alt_corresponding_rewards = self.activation_calculation(alt_option, current_trial,
                                                                                 exclude_current_trial=False)

        # Calculate probabilities using softmax rule if there are valid activations and update EVs
        self.EV_calculation(chosen, corresponding_rewards, activations)
        self.EV_calculation(alt_option, alt_corresponding_rewards, alt_activations)

    def update(self, chosen, reward, trial, choiceset=None):
        """
        Update EVs based on the choice, received reward, and trial number.

        Parameters:
        - chosen: Index of the chosen option.
        - reward: Reward received for the current trial.
        - trial: Current trial number.
        """

        self.choices_count[chosen] += 1

        self.updating_function(chosen, reward, trial, choiceset)

    def simulate(self, reward_means, reward_sd, AB_freq, CD_freq, num_trials=250, num_iterations=1000,
                 beta_lower=-1, beta_upper=1):
        """
        Simulate the EV updates for a given number of trials and specified number of iterations.

        Parameters:
        - num_trials: Number of trials for the simulation.
        - AB_freq: Frequency of appearance for the AB pair in the first 150 trials.
        - CD_freq: Frequency of appearance for the CD pair in the first 150 trials.
        - num_iterations: Number of times to repeat the simulation.
        - beta_lower: Lower bound for the beta parameter.
        - beta_upper: Upper bound for the beta parameter.

        Returns:
        - A list of simulation results.
        """
        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.reset()

            self.s = np.random.uniform(0.0001, 0.9999) if self.model_type == 'ACTR_Ori' else None
            self.t = np.random.uniform(0, 4.9999)
            self.a = np.random.uniform()  # Randomly set decay parameter between 0 and 1
            self.b = np.random.uniform(beta_lower, beta_upper) if self.num_params == 3 else self.a
            self.tau = np.random.uniform(-1.9999, -0.0001) if self.model_type in ('ACTR', 'ACTR_Ori') else None
            self.lamda = np.random.uniform(0.0001, 0.9999) if self.model_type == 'mean_var_utility' else None
            self.choices_count = np.zeros(self.num_options)

            EV_history = np.zeros((num_trials, self.num_options))
            trial_details = []
            trial_indices = []

            training_trials = [(0, 1), (2, 3)]
            training_trial_sequence = [training_trials[0]] * AB_freq + [training_trials[1]] * CD_freq
            np.random.shuffle(training_trial_sequence)

            # Distributing the next 100 trials equally among the four pairs (AC, AD, BC, BD)
            transfer_trials = [(2, 0), (1, 3), (0, 3), (2, 1)]
            transfer_trial_sequence = transfer_trials * 25
            np.random.shuffle(transfer_trial_sequence)

            for trial in range(num_trials):
                trial_indices.append(trial + 1)

                if trial < 150:
                    pair = training_trial_sequence[trial]
                    optimal, suboptimal = (pair[0], pair[1])
                    prob_optimal = self.softmax_mapping[self.model_type](np.array([self.EVs[optimal],
                                                                                   self.EVs[suboptimal]]))[0]
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal
                else:
                    pair = transfer_trial_sequence[trial - 150]
                    optimal, suboptimal = (pair[0], pair[1])
                    prob_optimal = self.softmax_mapping[self.model_type](np.array([self.EVs[optimal],
                                                                                  self.EVs[suboptimal]]))[0]
                    chosen = optimal if np.random.rand() < prob_optimal else suboptimal

                reward = np.random.normal(reward_means[chosen], reward_sd[chosen])
                trial_details.append(
                    {"trial": trial + 1, "pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
                     "reward": reward})

                self.update(chosen, reward, trial + 1, pair)
                EV_history[trial] = self.EVs

            all_results.append({
                "simulation_num": iteration + 1,
                "trial_indices": trial_indices,
                "s": self.s,
                "t": self.t,
                "a": self.a,
                "b": self.b,
                "tau": self.tau,
                "lamda": self.lamda,
                "weight": self.w,
                "trial_details": trial_details,
                "EV_history": EV_history
            })

        return self.simulation_unpacker(all_results)

    # ==================================================================================================================
    # Now we define the negative log likelihood function for the ACT-R model because it requires updating EVs before
    # calculating the likelihood
    # ==================================================================================================================
    def ACTR_nll(self, reward, choice, trial, epsilon, choiceset=None):

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):

            # if the trial is not a training trial, we skip the update
            if t % self.num_exp_restart > self.num_training_trials:
                pass
            else:
                # Otherwise, we update the model
                # in the ACTR model, we need to update the EVs before calculating the likelihood
                self.update(ch, r, t, cs)

            cs_mapped = self.choiceset_mapping[0][cs]
            prob_choice = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[0]],
                                                                          self.EVs[cs_mapped[1]]]))[0]
            prob_choice_alt = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[1]],
                                                                              self.EVs[cs_mapped[0]]]))[0]
            nll += -np.log(max(epsilon, prob_choice if ch == cs_mapped[0] else prob_choice_alt))

            if t % self.num_exp_restart == 0:
                self.reset()

        return nll

    def WSLS_nll(self, reward, choice, trial, epsilon, choiceset=None):

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            prob_choice = self.Probs[ch]
            nll += -np.log(max(epsilon, prob_choice))
            # if the experiment is restarted, we reset the model
            if t % self.num_exp_restart == 0:
                self.reset()
                continue

            # if the current trial is the starting trial of the new experiment and if the initialization is set to 'first_trial',
            # we initialize the model with the first trial
            if t % self.num_exp_restart == 1 and self.negative_log_likelihood == self.nll_first_trial_init:
                self.update(ch, r, t, cs)

                # Populate the EVs for the first trial
                EV_new_exp = self.EVs[ch]
                self.EVs = np.full(self.num_options, EV_new_exp)
                self.AV = EV_new_exp

            # if the trial is not a training trial, we skip the update
            if t % self.num_exp_restart > self.num_training_trials:
                continue

            # Otherwise, we update the model
            self.update(ch, r, t, cs)

        return nll

    def standard_nll(self, reward, choice, trial, epsilon, choiceset=None):

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]
            prob_choice = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[0]],
                                                                          self.EVs[cs_mapped[1]]]))[0]
            prob_choice_alt = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[1]],
                                                                              self.EVs[cs_mapped[0]]]))[0]
            nll += -np.log(max(epsilon, prob_choice if ch == cs_mapped[0] else prob_choice_alt))

            # print(f'Trial {t}, Choice: {ch}, Choiceset: {cs}, Reward: {r}, EVs: {self.EVs}, AV: {self.AV}, '
            #       f'alpha: {self.a}')

            # if the experiment is restarted, we reset the model
            if t % self.num_exp_restart == 0:
                self.reset()
                continue

            # if the current trial is the starting trial of the new experiment and if the initialization is set to 'first_trial',
            # we initialize the model with the first trial
            if t % self.num_exp_restart == 1 and self.negative_log_likelihood == self.nll_first_trial_init:
                self.update(ch, r, t, cs)

                # Populate the EVs for the first trial
                EV_new_exp = self.EVs[ch]
                self.EVs = np.full(self.num_options, EV_new_exp)
                self.AV = EV_new_exp
                continue

            # if the trial is not a training trial, we skip the update
            if t % self.num_exp_restart > self.num_training_trials:
                continue

            # Otherwise, we update the model
            self.update(ch, r, t, cs)

        return nll

    def igt_nll(self, reward, choice, trial, epsilon, choiceset=None):

        nll = 0

        for r, ch, t in zip(reward, choice, trial):
            prob_choice = self.softmax_mapping[self.model_type](np.array(self.EVs))[ch]
            nll += -np.log(max(epsilon, prob_choice))

            # if the experiment is restarted, we reset the model
            if t % self.num_exp_restart == 0:
                self.reset()
                continue

            # if the current trial is the starting trial of the new experiment and if the initialization is set to 'first_trial',
            # we initialize the model with the first trial
            if t % self.num_exp_restart == 1 and self.negative_log_likelihood == self.nll_first_trial_init:
                self.update(ch, r, t)

                # Populate the EVs for the first trial
                EV_new_exp = self.EVs[ch]
                self.EVs = np.full(self.num_options, EV_new_exp)
                self.AV = EV_new_exp

            # if the trial is not a training trial, we skip the update
            if t % self.num_exp_restart > self.num_training_trials:
                continue

            # Otherwise, we update the model
            self.update(ch, r, t)

        return nll

    def nll(self, reward, choice, trial, choiceset=None, epsilon=1e-12):
        """
        :param reward:
        :param choice:
        :param trial:
        :param choiceset:
        :param epsilon:
        :return:
        """
        return self.nll_function(reward, choice, trial, epsilon, choiceset)

    def nll_fixed_init(self, params, reward, choice, choiceset=None):
        self.reset()

        cfg = self._PARAM_MAP.get(self.model_type, {})
        for attr, idx in cfg.items():
            setattr(self, attr, params[idx])

        trial = np.arange(1, len(reward) + 1)

        self.b = getattr(self, self._B_OVERRIDES.get(self.model_type, 'b'))

        return self.nll(reward, choice, trial, choiceset)

    def nll_first_trial_init(self, params, reward, choice, choiceset=None):
        """
        Compute the negative log likelihood for the given parameters and data, initializing the model on the first trial.

        Parameters:
        - params: Parameters of the model.
        - reward: List or array of observed rewards.
        - choiceset: List or array of available choicesets for each trial.
        - choice: List or array of chosen options for each trial.
        """
        self.reset()

        cfg = self._PARAM_MAP.get(self.model_type, {})
        for attr, idx in cfg.items():
            setattr(self, attr, params[idx])

        self.b = getattr(self, self._B_OVERRIDES.get(self.model_type, 'b'))

        epsilon = 1e-12
        trial = np.arange(1, len(reward) + 1)

        # Initialize the model on the first trial
        self.update(choice[0], reward[0], trial[0], choiceset[0] if choiceset is not None else None)

        # Populate the EVs for the first trial
        EV_trial1 = self.EVs[choice[0]]
        self.EVs = np.full(self.num_options, EV_trial1)
        self.AV = EV_trial1

        return self.nll(reward[1:], choice[1:], trial[1:], choiceset[1:] if choiceset is not None else None, epsilon)

    def nll_first_trial_no_alpha_init(self, params, reward, choice, choiceset=None):
        """
        Compute the negative log likelihood for the given parameters and data, initializing the model on the first trial.

        Parameters:
        - params: Parameters of the model.
        - reward: List or array of observed rewards.
        - choiceset: List or array of available choicesets for each trial.
        - choice: List or array of chosen options for each trial.
        """
        self.reset()

        cfg = self._PARAM_MAP.get(self.model_type, {})
        for attr, idx in cfg.items():
            setattr(self, attr, params[idx])

        self.b = getattr(self, self._B_OVERRIDES.get(self.model_type, 'b'))

        epsilon = 1e-12
        trial = np.arange(1, len(reward) + 1)

        # Temporarily force alpha=1 so that update adds full PE
        orig_a = self.a
        self.a = 1.0

        # Initialize the model on the first trial
        self.update(choice[0], reward[0], trial[0], choiceset[0] if choiceset is not None else None)

        # Restore the original alpha value
        self.a = orig_a

        # Populate the EVs for the first trial
        EV_trial1 = self.EVs[choice[0]]
        self.EVs = np.full(self.num_options, EV_trial1)
        self.AV = EV_trial1

        # print(f'Trial 1, Choice: {choice[0]}, Choiceset: {choiceset[0] if choiceset is not None else None}, '
        #       f'Reward: {reward[0]}, EVs: {self.EVs}, AV: {self.AV}, alpha: {self.a}')

        return self.nll(reward[1:], choice[1:], trial[1:], choiceset[1:] if choiceset is not None else None, epsilon)

    def fit(self, data, num_training_trials=150, num_exp_restart=999, initial_EV=None, initial_mode='fixed',
            num_iterations=100, beta_lower=-1, beta_upper=1):

        self.num_training_trials = num_training_trials
        self.num_exp_restart = num_exp_restart
        self.num_options = len(initial_EV) if initial_EV is not None else 4
        self.initial_EV = np.array(initial_EV) if initial_EV is not None else [0.0, 0.0, 0.0, 0.0]

        if initial_mode == 'fixed':
            self.negative_log_likelihood = self.nll_fixed_init
        elif initial_mode == 'first_trial':
            self.negative_log_likelihood = self.nll_first_trial_init
        elif initial_mode == 'first_trial_no_alpha':
            self.negative_log_likelihood = self.nll_first_trial_no_alpha_init

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, self.model_type,
                                         num_iterations, beta_lower, beta_upper)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        return pd.DataFrame(results)

    def post_hoc_simulation(self, fitting_result, original_data, reward_means,
                            reward_sd, num_iterations=1000, AB_freq=100, CD_freq=50, use_random_sequence=True,
                            sim_trials=250, summary=False):

        num_parameters = len(fitting_result['best_parameters'][0].strip('[]').split())

        parameter_sequences = []
        for i in range(num_parameters):
            sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[i]) if isinstance(x, str) else np.nan
            )
            parameter_sequences.append(sequence)

        if not use_random_sequence:
            trial_index = original_data.groupby('Subnum')['trial_index'].apply(list)
            trial_sequence = original_data.groupby('Subnum')['TrialType'].apply(list)
        else:
            trial_index, trial_sequence = None, None

        # start the simulation
        all_results = []

        for participant in fitting_result['participant_id']:

            # extract the best-fitting parameters for the current participant
            self.s = parameter_sequences[0][participant - 1] if self.model_type == 'ACTR_Ori' else None
            self.t = parameter_sequences[0][participant - 1]
            self.a = parameter_sequences[1][participant - 1]
            self.b = parameter_sequences[2][participant - 1] if num_parameters == 3 else self.a
            self.tau = parameter_sequences[2][participant - 1] if self.model_type in ('ACTR', 'ACTR_Ori') else None
            self.lamda = parameter_sequences[2][participant - 1] if self.model_type == 'mean_var_utility' else None

            for _ in range(num_iterations):

                print(f"Participant {participant} - Iteration {_ + 1} of {num_iterations}")

                self.reset()

                self.iteration = 0

                trial_details = []
                trial_indices = []

                if use_random_sequence:

                    training_trial_sequence, transfer_trial_sequence = generate_random_trial_sequence(AB_freq, CD_freq)

                    for trial in range(sim_trials):
                        trial_indice = trial + 1
                        trial_indices.append(trial_indice)

                        if trial_indice < 151:
                            pair = training_trial_sequence[trial_indice - 1]
                        else:
                            pair = transfer_trial_sequence[trial_indice - 151]

                        optimal, suboptimal = pair[0], pair[1]

                        prob_optimal = self.softmax_mapping[self.model_type](np.array([self.EVs[optimal],
                                                                                       self.EVs[suboptimal]]))[0]
                        chosen = 1 if np.random.rand() < prob_optimal else 0

                        reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                                  reward_sd[optimal if chosen == 1 else suboptimal])

                        trial_details.append(
                            {"pair": pair, "choice": chosen, "reward": reward}
                        )

                        self.updating_function(optimal if chosen == 1 else suboptimal, reward, trial,
                                               pair)

                else:
                    for trial, pair in zip(trial_index[participant], trial_sequence[participant]):
                        trial_indices.append(trial)

                        optimal, suboptimal = self.choiceset_mapping[1][pair]

                        prob_optimal = self.softmax_mapping[self.model_type](np.array([self.EVs[optimal],
                                                                                       self.EVs[suboptimal]]))[0]
                        chosen = 1 if np.random.rand() < prob_optimal else 0

                        reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                                  reward_sd[optimal if chosen == 1 else suboptimal])

                        trial_details.append(
                            {"pair": pair, "choice": chosen, "reward": reward}
                        )

                        self.update(optimal if chosen == 1 else suboptimal, reward, trial,
                                    self.choiceset_mapping[1][pair])

                all_results.append({
                    "Subnum": participant,
                    "s": self.s,
                    "t": self.t,
                    "a": self.a,
                    "b": self.b,
                    "lamda": self.lamda if self.model_type == 'mean_var_utility' else None,
                    "tau": self.tau if self.model_type in ('ACTR', 'ACTR_Ori') else None,
                    "trial_indices": trial_indices,
                    "trial_details": trial_details
                })

        unpacked_results = []

        for result in all_results:
            sim_num = result['Subnum']

            s = result["s"]
            t = result["t"]
            a = result["a"]
            b = result["b"]
            lamda = result["lamda"] if self.model_type == 'mean_var_utility' else None
            tau = result["tau"] if self.model_type == 'ACTR' else None

            for trial_idx, trial_detail in zip(result['trial_indices'], result['trial_details']):
                var = {
                    "Subnum": sim_num,
                    "trial_index": trial_idx,
                    "s": s,
                    "t": t,
                    "a": a,
                    "b": b,
                    "lamda": lamda,
                    "tau": tau,
                    "pair": trial_detail['pair'],
                    "choice": trial_detail['choice'],
                    "reward": trial_detail['reward'],
                }

                unpacked_results.append(var)

        df = pd.DataFrame(unpacked_results)
        df.dropna(how='all', inplace=True, axis=1)
        if all(df['a'] == df['b']):
            df.drop('b', axis=1, inplace=True)

        if summary:
            if use_random_sequence:
                # remove the trial index column as it is not needed for the summary
                df.drop('trial_index', axis=1, inplace=True)
                summary_df = df.groupby(['Subnum', 'pair']).mean().reset_index()
            else:
                summary_df = df.groupby(['Subnum', 'trial_index']).mean().reset_index()

            return summary_df

        else:
            return df

    def bootstrapping_post_hoc_simulation(self, fitting_result, reward_means, reward_sd, num_iterations=1000,
                                          AB_freq=100, CD_freq=50, sim_trials=250, summary=False):

        num_parameters = len(fitting_result['best_parameters'][0].strip('[]').split())

        parameter_sequences = []
        for i in range(num_parameters):
            sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[i]) if isinstance(x, str) else np.nan
            )
            parameter_sequences.append(sequence)

        # start the simulation
        all_results = []

        for n in range(num_iterations):
            print(f"Iteration {n + 1} of {num_iterations}")

            # randomly sample with replacement from the original data
            random_idx = random.randint(0, len(fitting_result) - 1)
            self.t = parameter_sequences[0][random_idx]

            self.s = parameter_sequences[0][random_idx] if self.model_type == 'ACTR_Ori' else None
            self.t = parameter_sequences[0][random_idx]
            self.a = parameter_sequences[1][random_idx]
            self.b = parameter_sequences[2][random_idx] if num_parameters == 3 else self.a
            self.tau = parameter_sequences[2][random_idx] if self.model_type in ('ACTR', 'ACTR_Ori') else None
            self.lamda = parameter_sequences[2][random_idx] if self.model_type == 'mean_var_utility' else None

            self.reset()

            trial_details = []
            trial_indices = []

            training_trial_sequence, transfer_trial_sequence = generate_random_trial_sequence(AB_freq, CD_freq)

            for trial in range(sim_trials):
                trial_indice = trial + 1
                trial_indices.append(trial_indice)

                if trial_indice < 151:
                    pair = training_trial_sequence[trial_indice - 1]
                else:
                    pair = transfer_trial_sequence[trial_indice - 151]

                optimal, suboptimal = pair[0], pair[1]

                prob_optimal = self.softmax_mapping[self.model_type](np.array([self.EVs[optimal],
                                                                               self.EVs[suboptimal]]))[0]
                chosen = 1 if np.random.rand() < prob_optimal else 0

                reward = np.random.normal(reward_means[optimal if chosen == 1 else suboptimal],
                                          reward_sd[optimal if chosen == 1 else suboptimal])

                trial_details.append(
                    {"pair": pair, "choice": chosen, "reward": reward}
                )

                self.updating_function(optimal if chosen == 1 else suboptimal, reward, trial,
                                       pair)

            all_results.append({
                "Subnum": n + 1,
                "s": self.s,
                "t": self.t,
                "a": self.a,
                "b": self.b,
                "lamda": self.lamda if self.model_type == 'mean_var_utility' else None,
                "tau": self.tau if self.model_type in ('ACTR', 'ACTR_Ori') else None,
                "trial_indices": trial_indices,
                "trial_details": trial_details
            })

        unpacked_results = []

        for result in all_results:
            subnum = result['Subnum']
            s = result["s"]
            t = result["t"]
            a = result["a"]
            b = result["b"]
            lamda = result["lamda"] if self.model_type == 'mean_var_utility' else None
            tau = result["tau"] if self.model_type == 'ACTR' else None

            for trial_idx, trial_detail in zip(result['trial_indices'], result['trial_details']):
                var = {
                    "Subnum": subnum,
                    "trial_index": trial_idx,
                    "s": s,
                    "t": t,
                    "a": a,
                    "b": b,
                    "lamda": lamda,
                    "tau": tau,
                    "pair": trial_detail['pair'],
                    "choice": trial_detail['choice'],
                    "reward": trial_detail['reward'],
                }

                unpacked_results.append(var)

        df = pd.DataFrame(unpacked_results)
        df.dropna(how='all', inplace=True, axis=1)
        if all(df['a'] == df['b']):
            df.drop('b', axis=1, inplace=True)

        if summary:
            df.drop('trial_index', axis=1, inplace=True)
            summary_df = df.groupby(['Subnum', 'pair']).mean().reset_index()
            return summary_df
        else:
            return df

# ======================================================================================================================
# End of the ComputationalModels class
# ======================================================================================================================

# ======================================================================================================================
# Model Comparison Functions
# ======================================================================================================================
def likelihood_ratio_test(null_results, alternative_results, df):
    """
    Perform a likelihood ratio test.

    Parameters:
    - null_nll: Negative log-likelihood of the simpler (null) model.
    - alternative_nll: Negative log-likelihood of the more complex (alternative) model.
    - df: Difference in the number of parameters between the two models.

    Returns:
    - p_value: p-value of the test.
    """
    # locate the nll values for the null and alternative models
    null_nll = null_results['best_nll']

    alternative_nll = alternative_results['best_nll']

    # Compute the likelihood ratio statistic
    lr_stat = 2 * np.mean((null_nll - alternative_nll))

    # Get the p-value
    p_value = chi2.sf(lr_stat, df)

    return p_value


def bayes_factor(null_results, alternative_results):
    """
    Compute the Bayes factor.

    Parameters:
    - null_BIC: BIC of the simpler (null) model.
    - alternative_BIC: BIC of the more complex (alternative) model.

    Returns:
    - bayes_factor: Bayes factor of the test.
    """
    # locate the nll values for the null and alternative models
    null_BIC = null_results['BIC']
    alternative_BIC = alternative_results['BIC']

    # Compute the Bayes factor
    BIC_diff_array = 0.5 * (null_BIC - alternative_BIC)
    # log_BIC_array = np.exp(BIC_diff_array)
    # BF = np.prod(log_BIC_array)
    mean_bic_diff = np.mean(BIC_diff_array)
    BF = np.exp(mean_bic_diff)

    return BF

# ----------------------------------------------------------------------------------------------------------------------
# Implement Bayesian Model Selection (BMS)
# ----------------------------------------------------------------------------------------------------------------------
def vb_model_selection(log_evidences, alpha0=None, tol=1e-6, max_iter=1000):
    """
    Variational Bayesian Model Selection for multiple models and multiple subjects.

    Implements the iterative VB algorithm described by:
    - Equations 3, 7, 9, 11, 12, 13, and the final pseudo-code in Equation 14.

    Parameters
    ----------
    log_evidences : np.ndarray, shape (N, K)
        Matrix of log model evidences for N subjects and K models.
        log_evidences[n, k] = ln p(y_n | m_{nk})
    alpha0 : np.ndarray, shape (K,)
        Initial Dirichlet prior parameters. If None, set to 1 for all models.
    tol : float
        Tolerance for convergence based on changes in alpha.
    max_iter : int
        Maximum number of VB iterations.

    Returns
    -------
    alpha : np.ndarray, shape (K,)
        Final Dirichlet parameters of the approximate posterior q(r).
    g : np.ndarray, shape (N, K)
        Posterior model assignment probabilities per subject.
    """

    N, K = log_evidences.shape
    if alpha0 is None:
        alpha0 = np.ones(K)

    # Initialize alpha
    alpha = alpha0.copy()

    for iteration in range(max_iter):
        alpha_sum = np.sum(alpha)

        # Compute unnormalized posterior assignments u_nk
        # u_nk = exp(ln p(y_n | m_nk) + Psi(alpha_k) - Psi(alpha_sum))
        u = np.exp(log_evidences + psi(alpha) - psi(alpha_sum))  # shape (N,K)

        # Normalize to get g_nk
        u_sum = np.sum(u, axis=1, keepdims=True)
        g = u / u_sum  # shape (N,K)

        # Update beta_k = sum_n g_nk
        beta = np.sum(g, axis=0)  # shape (K,)

        # Update alpha
        alpha_new = alpha0 + beta

        # Check convergence
        diff = np.linalg.norm(alpha_new - alpha)
        alpha = alpha_new
        if diff < tol:
            break

    return alpha, g


def compute_exceedance_prob(alpha, n_samples=100000):
    """
    Compute exceedance probabilities for each model by Monte Carlo approximation.

    Parameters
    ----------
    alpha : array-like of shape (K,)
        The Dirichlet parameters for the posterior q(r).
    n_samples : int
        Number of samples to draw from Dirichlet.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    exceedance_probs : np.ndarray of shape (K,)
        The exceedance probability for each model.
    """
    samples = dirichlet.rvs(alpha, size=n_samples)
    winners = np.argmax(samples, axis=1)  # indices of best model per draw

    K = len(alpha)
    exceedance_probs = np.bincount(winners, minlength=K) / n_samples
    return exceedance_probs

# ======================================================================================================================
# End of the Model Comparison Functions; now we define the preparatory functions
# ======================================================================================================================
def dict_generator(df, task='ABCD'):
    """
    Convert a dataframe into a dictionary.

    Parameters:
    - df: Dataframe to be converted.

    Returns:
    - A dictionary of the dataframe.
    """
    def find_col(candidates):
        """Return first candidate that’s in df.columns, else error."""
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"None of {candidates!r} found in DataFrame columns")

    # define for each task which output‐keys map to which column‐name candidates
    COLUMN_MAP = {
        'ABCD': {
            'reward':   ['Reward', 'points'],
            'choiceset':['SetSeen.', 'SetSeen ', 'setSeen'],
            'choice':   ['KeyResponse', 'choice'],
        },
        'IGT_SGT': {
            'reward': ['outcome.1', 'outcome', 'Reward', 'SGTReward', 'OutcomeValue'],
            'choice': ['choice', 'keyResponse', 'SGTBinChoice', 'Optimal_Choice'],
        },
        'VS': {
            'reward': ['OutcomeValue'],
            'choice': ['Optimal_Choice'],
            'react_time': ['RT']
        }
    }

    if task not in COLUMN_MAP:
        raise ValueError(f"Unsupported task {task!r}")

    # optional: allow different grouping columns too
    group_col = find_col(['subjID', 'Subnum', 'subnum', 'SubjectID', 'ID', 'subject', 'SubNo'])

    d = {}
    for subject_id, group in df.groupby(group_col):
        entry = {}
        for key, candidates in COLUMN_MAP[task].items():
            actual_col = find_col(candidates)
            entry[key] = group[actual_col].tolist()
        d[subject_id] = entry

    return d

def extract_all_parameters(param_str):
    """
    Extracts all numerical values from a parameter string and returns them as a list of floats.

    Parameters:
    param_str (str): A string containing numerical values.

    Returns:
    list: A list of floats extracted from the string.
    """
    if isinstance(param_str, str):
        return [float(x) for x in param_str.strip('[]').split()]
    return []


def parameter_extractor(df, param_name=['t', 'alpha', "subj_weight"]):
    """
    Extracts all parameter values from a dataframe and returns them as a list of lists.

    Parameters:
    df (pd.DataFrame): A dataframe containing parameter values.

    Returns:
    list: A list of lists containing parameter values.
    """
    all_params = df['best_parameters'].apply(extract_all_parameters).tolist()
    all_params_transposed = list(map(list, zip(*all_params)))
    params_dict = {param: all_params_transposed[i] for i, param in enumerate(param_name)}

    # attach back to the dataframe
    for i, param in enumerate(param_name):
        df[param] = params_dict[param]

    # remove the best_parameters column
    df.drop(columns=['best_parameters'], inplace=True)

    return df


def safely_evaluate(x):
    if isinstance(x, list):
        return x
    try:
        # Try to safely evaluate the string to a list using ast.literal_eval
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        # If it's not evaluable (e.g., a number), just return as is
        return [x]  # Wrap in a list for consistency


def clean_list_string(s):
    if isinstance(s, str):
        # Remove unwanted characters and ensure proper formatting
        s = s.strip("[],")  # Remove leading/trailing brackets and commas
        s = s.replace(" ", ",")  # Replace spaces with commas
        s = s.replace(",,", ",")  # Replace consecutive commas with a single comma
        s = s.strip(",")  # Remove leading/trailing commas again after cleanup
        # Ensure the string is wrapped in brackets
        s = f"[{s}]"
    return s


def trial_exploder(data, col):
    return data.apply(lambda x: safely_evaluate(x[col]), axis=1).explode().reset_index(drop=True)

# ======================================================================================================================
# Model Recovery & Parameter Recovery Functions
# ======================================================================================================================
def model_recovery(model_names, models, reward_means, reward_var, AB_freq=100, CD_freq=50, n_iterations=100,
                   n_fitting_iterations=100, metric='BIC'):

    n_models = len(models)
    all_best_fitting_model = pd.DataFrame()
    all_sim = pd.DataFrame()
    all_fit = pd.DataFrame()

    for i in range(n_iterations):

        print(f"Running the {i + 1}th iteration")
        print("=============================================")

        start_time = time.time()

        i_sim = pd.DataFrame()
        i_fit = pd.DataFrame()

        # simulate the data
        for j in range(n_models):
            sim = models[j].simulate(reward_means, reward_var, AB_freq=AB_freq, CD_freq=CD_freq, num_iterations=1)
            sim['simulated_model'] = model_names[j]

            i_sim = pd.concat([i_sim, sim]).reset_index(drop=True)

        i_sim.iloc[:, 0] = (i_sim.index // 250) + 1
        i_sim['pair'] = i_sim['pair'].map(mapping['SetSeen'])
        i_sim['choice'] = i_sim['choice'].map(mapping['KeyResponse'])
        i_sim.rename(columns={'simulation_num': 'Subnum', 'pair': 'SetSeen.',
                                'choice': 'KeyResponse', 'reward': 'Reward'}, inplace=True)
        sim_dict = dict_generator(i_sim)
        all_sim = pd.concat([all_sim, i_sim]).reset_index(drop=True)

        # fit the data
        for k in range(n_models):
            fit = models[k].fit(sim_dict, num_iterations=n_fitting_iterations)
            fit['fit_model'] = model_names[k]
            fit['simulated_model'] = model_names
            i_fit = pd.concat([i_fit, fit]).reset_index(drop=True)

        all_fit = pd.concat([all_fit, i_fit]).reset_index(drop=True)

        # find the best fitting model
        best_fitting_model = (i_fit.loc[i_fit.groupby('participant_id')[metric].idxmin()].reset_index(drop=True))

        # append the best fitting model to the all_best_fitting_model
        all_best_fitting_model = pd.concat([all_best_fitting_model, best_fitting_model]).reset_index(drop=True)

        print(f"Time taken for iteration {i + 1}: {(time.time() - start_time) / 60} minutes")

    # reset the participant id
    all_best_fitting_model['participant_id'] = all_best_fitting_model.index + 1

    return all_sim, all_fit, all_best_fitting_model

# ======================================================================================================================
# Moving window approach
# ======================================================================================================================
def create_sliding_windows(data, window_size, id_col, filter_fn=None):
    """Create sliding windows of specified size from the data per participant."""
    max_window_steps = max(len(group) - window_size + 1 for _, group in data.groupby(id_col))

    for step in range(max_window_steps):
        window_data = []
        for participant_id, participant_data in data.groupby(id_col):
            if step < len(participant_data) - window_size + 1:
                window = participant_data.iloc[step:step + window_size].copy()
                # Apply filter_fn, if provided:
                if filter_fn is not None:
                    window = filter_fn(window)
                if not window.empty:
                    window_data.append(window)
                else:
                    print(f"Warning: Empty window for participant {participant_id} at window step {step + 1}")
        yield pd.concat(window_data, ignore_index=True)


def _fit_one_subject(args):
    model, pid, pdata, init_ev, num_iterations, fit_kwargs = args
    # clone your model so each process has its own copy
    local_model = copy.deepcopy(model)
    # fit just this one subject
    df = local_model.fit(
        {pid: pdata},
        num_iterations=num_iterations,
        initial_EV=init_ev,
        **fit_kwargs
    )
    # grab the single‐row result
    row = df.iloc[0].to_dict()
    return pid, row


def moving_window_model_fitting(data, model, task='ABCD', num_iterations=100, window_size=10,
                                id_col='Subnum', filter_fn=None, restart_EV=False, **kwargs):
    """
    Fit the model to the data using a moving window approach.
    
    Parameters:
        data: DataFrame containing the behavioral data
        model: ComputationalModel instance
        task: String identifying the task type ('ABCD' or 'IGT_SGT')
        num_iterations: Number of iterations for model fitting
        window_size: Size of the sliding window
        id_col: Column name for participant ID
        **kwargs: Additional keyword arguments to pass to model.fit()
    """

    # Initialize the result list and previous EVs dictionary
    window_results = []
    prev_EVs = {}
    max_window_steps = max(len(group) - window_size + 1 for _, group in data.groupby(id_col))

    for i, window_data in enumerate(create_sliding_windows(data, window_size, id_col, filter_fn)):

        # detect how many participants have how many trials in the current window
        rows_per_sub = window_data.groupby(id_col).size()
        freq_dist = rows_per_sub.value_counts().sort_index()
        df_freq_dist = freq_dist.reset_index(name='n_participants')
        df_freq_dist.columns = ['n_trials', 'n_participants']
        print(f'Fitting window {i + 1}/{max_window_steps}')
        print()
        print('Frequency distribution:')
        print(df_freq_dist.to_string(index=False))
        print(f'=' * 50)

        window_dict = dict_generator(window_data, task)
        args_list = []

        for pid, pdata in window_dict.items():
            init_ev = None if restart_EV else prev_EVs.get(pid)
            args_list.append((
                model,
                pid,
                pdata,
                init_ev,
                num_iterations,
                kwargs
            ))

        # parallel‐map over subjects in this window
        with ProcessPoolExecutor() as executor:
            for pid, row in executor.map(_fit_one_subject, args_list):
                # annotate window info
                row.update({
                    'window_id': i + 1,
                    'window_start': i + 1,
                    'window_end': i + window_size
                })
                window_results.append(row)

                # Extract the EVs for the current participant
                final_EV = row.get('best_EV', None)
                if final_EV is not None:
                    # Store the final EV for the participant
                    prev_EVs[pid] = final_EV

    return pd.DataFrame(window_results)

# ======================================================================================================================
# Testing Code
# ======================================================================================================================
# # testing
# test_results = []
# test_data = dict_generator(HV_df[:250])
# for model_type in ['delta', 'decay', 'decay_fre', 'decay_choice', 'decay_win', 'delta_decay', 'sampler_decay',
#               'sampler_decay_PE', 'sampler_decay_AV', 'ACTR', 'ACTR_Ori']:
#     print(f'Fitting {model_type} model')
#     model_spec = ComputationalModels(model_type)
#     sim_results = model_spec.simulate([0.65, 0.35, 0.75, 0.25], [0.43, 0.43, 0.43, 0.43],
#                                  AB_freq=100, CD_freq=50, num_iterations=10)
#
#     # Here you can save the simulation results
#
#     result = model_spec.fit(test_data, num_iterations=2)
#
#     # Here you can save the fitting results
#
#
# for i, model_type in enumerate(['delta', 'decay', 'decay_fre', 'decay_choice', 'decay_win', 'delta_decay', 'sampler_decay',
#               'sampler_decay_PE', 'sampler_decay_AV', 'ACTR', 'ACTR_Ori']):
#
#     print(f'Post-hoc simulation for {model_type}')
#
#     # You will need the best-fitting parameters from the fitting results, so you load the results from previous
#     # steps and use them here
#     post_hoc_results = pd.read_csv(f'./{model_type}_HV_results.csv')
#     model_spec = ComputationalModels(model_type)
#     result = model_spec.post_hoc_simulation(post_hoc_results, HV_df[:250], reward_means=[0.65, 0.35, 0.75, 0.25],
#                                             reward_sd=[0.43, 0.43, 0.43, 0.43], num_iterations=10, summary=True)
#
#     # Here you can save the post-hoc results
