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


def fit_participant(model, participant_id, pdata, model_type, num_iterations=100,
                    beta_lower=-1, beta_upper=1):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_n = len(pdata['reward'])  # Total number of trials

    # get the number of parameters for the model
    k = model._PARAM_COUNT.get(model_type)

    model.iteration = 0
    best_nll = 100000  # Initialize best negative log likelihood to a large number
    best_initial_guess = None
    best_parameters = None

    for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))

        if model_type in ('decay', 'delta', 'decay_choice', 'decay_win', 'delta_RPUT', 'decay_RPUT'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999))
        elif model_type in ('delta_perseveration'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 9.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 9.9999))
        elif model_type in ('delta_PVL', 'delta_PVL_relative', 'decay_PVL_relative'):
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
        elif model_type == 'ACTR':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(-1.9999, -0.0001)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (-1.9999, -0.0001))
        elif model_type == 'ACTR_Ori':
            initial_guess = [np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(-1.9999, -0.0001)]
            bounds = ((0.0001, 0.9999), (0.0001, 0.9999), (-1.9999, -0.0001))
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
        elif model_type == 'RT_exp_basic':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 23.9999),
                             np.random.uniform(0.0001, 23.9999), np.random.uniform(0.0, 0.01)]
            bounds = ((0.0001, 4.9999),(0.0, 0.01), (0.0001, 23.9999), (0.0001, 23.9999))
        if model_type in ('RT_delta', 'RT_decay'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 23.9999), np.random.uniform(0.0001, 23.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 23.9999), (0.0001, 23.9999))
        elif model_type in ('RT_exp_delta', 'RT_exp_decay'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 23.9999), np.random.uniform(0.0001, 23.9999),
                             np.random.uniform(0.0, 0.01)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 23.9999), (0.0001, 23.9999), (0.0, 0.01))
        elif model_type in ('RT_delta_PVL', 'RT_delta_PVL', 'RT_decay_PVL'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 23.9999), np.random.uniform(0.0001, 23.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 4.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 23.9999), (0.0001, 23.9999), (0.0001, 0.9999),
                      (0.0001, 4.9999))
        elif model_type in ('hybrid_delta_delta', 'hybrid_decay_delta'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 23.9999), np.random.uniform(0.0001, 23.9999),
                             np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 23.9999), (0.0001, 23.9999), (0.0001, 0.9999))
        elif model_type in ('hybrid_delta_delta_3', 'hybrid_decay_delta_3'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 23.9999),
                             np.random.uniform(0.0001, 23.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 23.9999), (0.0001, 23.9999), (0.0001, 0.9999))

        result = minimize(model.negative_log_likelihood, initial_guess,
                          args=(pdata['reward'], pdata['choice'], pdata['react_time']),
                          bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

        if result.fun < best_nll:
            best_nll = result.fun
            best_initial_guess = initial_guess
            best_parameters = result.x

    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_nll': best_nll,
        'best_initial_guess': best_initial_guess,
        'best_parameters': best_parameters,
        'AIC': aic,
        'BIC': bic
    }

    print(f"Participant {participant_id} fitted in {(time.time() - start_time) / 60} minutes.")

    return result_dict


class VisualSearchModels:
    def __init__(self, model_type, task='VS', condition="Gains", num_params=2, num_options=2):
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
        self.num_options = num_options
        self.num_params = num_params
        self.choices_count = np.zeros(self.num_options)
        self.condition = condition
        self.possible_options = [0, 1]
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
            'delta_perseveration': {'t': 0, 'a': 1, 'b': 2},
            'delta_asymmetric': {'t': 0, 'a': 1, 'b': 2},
            'delta_RPUT': {'t': 0, 'a': 1},
            'delta_PVL': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'delta_PVL_relative': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'decay': {'t': 0, 'a': 1},
            'decay_fre': {'t': 0, 'a': 1, 'b': 2},
            'decay_PVL_relative': {'t': 0, 'a': 1, 'b': 2, 'lamda': 3},
            'decay_choice': {'t': 0, 'a': 1},
            'decay_win': {'t': 0, 'a': 1},
            'decay_RPUT': {'t': 0, 'a': 1},
            'delta_decay': {'t': 0, 'a': 1, 'b': 2},
            'mean_var_utility': {'t': 0, 'a': 1, 'lamda': 2},
            'sampler_decay': {'t': 0, 'a': 1},
            'sampler_decay_PE': {'t': 0, 'a': 1},
            'sampler_decay_AV': {'t': 0, 'a': 1},
            'ACTR_Ori': { 'a': 1, 's': 0, 'tau': 2},
            'ACTR': {'t': 0, 'a': 1, 'tau': 2},
            'WSLS': {'p_ws': 0, 'p_ls': 1},
            'WSLS_delta': {'a': 1, 'p_ws': 0, 'p_ls': 2},
            'WSLS_decay_weight': {'t': 0, 'a': 1, 'p_ws': 2, 'p_ls': 3, 'w': 4},
            'WSLS_delta_weight': {'t': 0, 'a': 1, 'p_ws': 2, 'p_ls': 3, 'w': 4},
            'RT_exp_basic': {'t': 0, 'k': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3},
            'RT_delta': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3},
            'RT_decay': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3},
            'RT_exp_delta': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3, 'k': 4},
            'RT_exp_decay': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3, 'k': 4},
            'RT_delta_PVL': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3, 'b': 4, 'lamda': 5},
            'RT_decay_PVL': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3, 'b': 4, 'lamda': 5},
            'hybrid_delta_delta': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3, 'w': 4},
            'hybrid_delta_delta_3': {'t': 0, 'a': 1, 'b': 2, 'RT_initial_suboptimal': 3, 'RT_initial_optimal': 4, 'w': 5},
            'hybrid_decay_delta': {'t': 0, 'a': 1, 'RT_initial_suboptimal': 2, 'RT_initial_optimal': 3, 'w': 4},
            'hybrid_decay_delta_3': {'t': 0, 'a': 1, 'b': 2, 'RT_initial_suboptimal': 3, 'RT_initial_optimal': 4, 'w': 5},
        }

        # any attributes you always want to have, even if None
        self._DEFAULT_ATTRS = [
            'RT_initial_suboptimal', 'RT_initial_optimal', 'RT_initial',
            'k', 's', 't', 'a', 'b', 'tau', 'lamda',
            'p_ws', 'p_ls', 'w',
        ]

        # initialize default b overrides
        self._B_OVERRIDES = {
            'hybrid_delta_delta': 'a',
            'hybrid_decay_delta': 'a'
        }

        # initialize all attributes to None
        for attr in self._DEFAULT_ATTRS:
            setattr(self, attr, None)

        self._PARAM_COUNT = {
            **dict.fromkeys(
                ('decay', 'delta', 'delta_RPUT', 'decay_RPUT', 'decay_choice', 'decay_win', 'WSLS'),
                2
            ),
            **dict.fromkeys(
                ('delta_asymmetric', 'decay_fre', 'ACTR', 'ACTR_Ori', 'WSLS_delta', 'mean_var_utility',
                 'delta_perseveration'),
                3
            ),
            **dict.fromkeys(
                ('delta_PVL', 'delta_PVL_relative', 'decay_PVL_relative', 'RT_exp_basic', 'RT_delta',
                 'RT_decay'),
                4
            ),
            **dict.fromkeys(
                ('WSLS_delta_weight', 'WSLS_decay_weight', 'RT_exp_delta', 'RT_exp_decay',
                 'hybrid_delta_delta', 'hybrid_decay_delta'),
                5
            ),
            **dict.fromkeys(
                ('RT_delta_PVL', 'RT_decay_PVL', 'hybrid_delta_delta_3', 'hybrid_decay_delta_3'),
                6
            ),
            **dict.fromkeys(
                ('sampler_decay', 'sampler_decay_PE', 'sampler_decay_AV', 'delta_decay'),
                None  # will use model.num_params instead
            ),
        }

        self.condition_initialization = {
            "Gains": 0.5,
            "Losses": -0.5,
            "Both": 0.0
        }

        self.EVs = np.full(self.num_options, self.condition_initialization[self.condition])
        self.RTs = np.full(self.num_options, 3.0)
        self.Probs = np.full(self.num_options, 0.25)
        self.mean = np.full(self.num_options, self.condition_initialization[self.condition])
        self.var = np.full(self.num_options, 1 / 12)
        self.AV = self.condition_initialization[self.condition]
        self.RT_AV = None

        # Model type
        self.model_type = model_type
        self.task = task

        # Mapping of updating functions to model types
        self.updating_mapping = {
            'delta': self.delta_update,
            'delta_perseveration': self.delta_perseveration_update,
            'delta_asymmetric': self.delta_asymmetric_update,
            'delta_RPUT': self.delta_reward_per_RT_update,
            'delta_PVL': self.delta_PVL_update,
            'delta_PVL_relative': self.delta_PVL_relative_update,
            'decay': self.decay_update,
            'decay_fre': self.decay_fre_update,
            'decay_PVL_relative': self.decay_PVL_relative_update,
            'decay_choice': self.decay_choice_update,
            'decay_win': self.decay_win_update,
            'decay_RPUT': self.decay_reward_per_RT_update,
            'delta_decay': self.delta_update,
            'mean_var_utility': self.mean_var_utility,
            'sampler_decay': self.sampler_decay_update,
            'sampler_decay_PE': self.sampler_decay_PE_update,
            'sampler_decay_AV': self.sampler_decay_AV_update,
            'WSLS': self.WSLS_update,
            'WSLS_delta': self.WSLS_delta_update,
            'WSLS_delta_weight': self.WSLS_delta_weight_update,
            'WSLS_decay_weight': self.WSLS_decay_weight_update,
            'RT_exp_basic': self.RT_exp_basic,
            'RT_delta': self.RT_delta,
            'RT_decay': self.RT_decay,
            'RT_exp_delta': self.RT_exp_delta,
            'RT_exp_decay': self.RT_exp_decay,
            'RT_delta_PVL': self.RT_delta_PVL,
            'RT_decay_PVL': self.RT_decay_PVL,
            'hybrid_delta_delta': self.hybrid_delta_delta,
            'hybrid_delta_delta_3': self.hybrid_delta_delta,
            'hybrid_decay_delta': self.hybrid_decay_delta,
            'hybrid_decay_delta_3': self.hybrid_decay_delta,
        }

        self.updating_function = self.updating_mapping[self.model_type]

        # Mapping of nll functions to model types
        self.nll_mapping_VS = {
            'delta': self.standard_nll,
            'delta_perseveration': self.standard_nll,
            'delta_asymmetric': self.standard_nll,
            'delta_RPUT': self.standard_nll,
            'delta_PVL': self.standard_nll,
            'delta_PVL_relative': self.standard_nll,
            'decay': self.standard_nll,
            'decay_fre': self.standard_nll,
            'decay_PVL_relative': self.standard_nll,
            'decay_choice': self.standard_nll,
            'decay_win': self.standard_nll,
            'decay_RPUT': self.standard_nll,
            'delta_decay': self.standard_nll,
            'mean_var_utility': self.standard_nll,
            'sampler_decay': self.standard_nll,
            'sampler_decay_PE': self.standard_nll,
            'sampler_decay_AV': self.standard_nll,
            'WSLS': self.WSLS_nll,
            'WSLS_delta': self.WSLS_nll,
            'WSLS_delta_weight': self.WSLS_nll,
            'WSLS_decay_weight': self.weight_nll,
            'RT_exp_basic': self.standard_nll,
            'RT_delta': self.standard_nll,
            'RT_decay': self.standard_nll,
            'RT_exp_delta': self.standard_nll,
            'RT_exp_decay': self.standard_nll,
            'RT_delta_PVL': self.standard_nll,
            'RT_decay_PVL': self.standard_nll,
            'hybrid_delta_delta': self.hybrid_nll,
            'hybrid_delta_delta_3': self.hybrid_nll,
            'hybrid_decay_delta': self.hybrid_nll,
            'hybrid_decay_delta_3': self.hybrid_nll,
        }

        self.nll_function = self.nll_mapping_VS[self.model_type]

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

        self.EVs = np.full(self.num_options, self.condition_initialization[self.condition])
        self.RTs = np.full(self.num_options, 3.0)
        self.Probs = np.full(self.num_options, 0.25)
        self.AV = self.condition_initialization[self.condition]
        self.RT_AV = None
        self.mean = np.full(self.num_options, self.condition_initialization[self.condition])
        self.var = np.full(self.num_options, 1 / 12)

    def softmax(self, x):
        c = 3 ** self.t - 1
        e_x = np.exp(np.clip(c * x, -700, 700))
        return np.clip(e_x / e_x.sum(), 1e-12, 1 - 1e-12)

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
    def delta_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def delta_asymmetric_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += (prediction_error > 0) * self.a * prediction_error + (prediction_error < 0) * self.b * \
                            prediction_error

    def delta_perseveration_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error + self.b

    def delta_reward_per_RT_update(self, chosen, reward, rt, trial):
        reward_per_RT = reward / np.clip(rt, 0.0001, None)  # Avoid division by zero
        prediction_error = reward_per_RT - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def delta_PVL_update(self, chosen, reward, rt, trial):
        utility = (np.abs(reward) ** self.b) * (reward >= 0) + (-self.lamda * (np.abs(reward) ** self.b)) * (reward < 0)
        prediction_error = utility - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def delta_PVL_relative_update(self, chosen, reward, rt, trial):
        reward_diff = reward - self.AV
        self.AV += reward_diff / (trial + 1)
        utility = (np.abs(reward) ** self.b) * (reward_diff >= 0) + ((1 / self.lamda) * (np.abs(reward) ** self.b)) * (reward_diff < 0)
        prediction_error = utility - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def decay_update(self, chosen, reward, rt, trial):
        self.EVs[chosen] += reward
        self.EVs = self.EVs * (1 - self.a)

    def decay_fre_update(self, chosen, reward, rt, trial):
        multiplier = self.choices_count[chosen] ** (-self.b)
        self.EVs[chosen] += reward * multiplier
        self.EVs = self.EVs * (1 - self.a)

    def decay_choice_update(self, chosen, reward, rt, trial):
        self.EVs[chosen] += 1
        self.EVs = self.EVs * (1 - self.a)

    def decay_PVL_relative_update(self, chosen, reward, rt, trial):
        reward_diff = reward - self.AV
        self.AV += reward_diff / (trial + 1)
        utility = (np.abs(reward) ** self.b) * (reward_diff >= 0) + ((1 / self.lamda) * (np.abs(reward) ** self.b)) * (reward_diff < 0)
        self.EVs[chosen] += utility
        self.EVs = self.EVs * (1 - self.a)

    def decay_reward_per_RT_update(self, chosen, reward, rt, trial):
        reward_per_RT = reward / np.clip(rt, 0.0001, None)  # Avoid division by zero
        self.EVs[chosen] += self.a * reward_per_RT
        self.EVs = self.EVs * (1 - self.a)

    def decay_win_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.AV
        self.AV += prediction_error * self.a
        self.EVs[chosen] += (prediction_error > 0)
        self.EVs = self.EVs * (1 - self.a)

    def delta_decay_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error
        self.EVs = self.EVs * (1 - self.b)

    def mean_var_utility(self, chosen, reward, rt, trial):
        prediction_error = reward - self.mean[chosen]
        self.mean[chosen] += self.a * prediction_error
        self.var[chosen] += self.a * (prediction_error ** 2 - self.var[chosen])
        self.EVs[chosen] = self.mean[chosen] - (self.lamda * self.var[chosen]) / 2

    def sampler_decay_update(self, chosen, reward, rt, trial):
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

    def sampler_decay_PE_update(self, chosen, reward, rt, trial):
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
        self.EVs = self.EVs * (1 - self.a)
        self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

        # Compute the probabilities from memory weights
        total_weight = sum(self.memory_weights)
        self.AllProbs = [w / total_weight for w in self.memory_weights]

        # Update EVs based on the samples from memory
        for j in range(len(self.reward_history)):
            self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.PE[j]

    def sampler_decay_AV_update(self, chosen, reward, rt, trial):
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
        self.EVs = self.EVs * (1 - self.a)
        self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

        # Compute the probabilities from memory weights
        total_weight = sum(self.memory_weights)
        self.AllProbs = [w / total_weight for w in self.memory_weights]

        # Update EVs based on the samples from memory
        for j in range(len(self.reward_history)):
            self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.PE[j]

    def WSLS_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.AV
        self.AV += prediction_error / (trial + 1)

        pos = int(prediction_error > 0)
        chosen_prob = pos * self.p_ws + (1 - pos) * (1 - self.p_ls)
        other_prob = pos * (1 - self.p_ws) + (1 - pos) * self.p_ls
        self.Probs[:] = [other_prob] * self.num_options
        self.Probs[chosen] = chosen_prob

    def WSLS_delta_update(self, chosen, reward, rt, trial):
        prediction_error = reward - self.AV
        self.AV += prediction_error * self.a

        pos = int(prediction_error > 0)
        chosen_prob = pos * self.p_ws + (1 - pos) * (1 - self.p_ls)
        other_prob = pos * (1 - self.p_ws) + (1 - pos) * self.p_ls
        self.Probs[:] = [other_prob] * self.num_options
        self.Probs[chosen] = chosen_prob

    def WSLS_delta_weight_update(self, chosen, reward, rt, trial):
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

    def WSLS_decay_weight_update(self, chosen, reward, rt, trial):
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

    # ------------------------------------------------------------------------------------------------------------------
    # Now we define RT-related models specifically for the visual search task
    # ------------------------------------------------------------------------------------------------------------------
    def RT_exp_basic(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial
        self.RTs[chosen] = self.RT_initial[chosen] * np.exp(-1 * self.k * trial)
        self.EVs = [-x for x in self.RTs]
    
    def RT_delta(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial

        self.RTs[chosen] += self.a * (rt - self.RTs[chosen])
        self.EVs = [-x for x in self.RTs]

    def RT_delta_PVL(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial
            self.RT_AV = np.mean(self.RTs)

        RT_diff = rt - self.RT_AV
        self.RT_AV += RT_diff / (trial + 1)
        utility = (self.lamda * np.abs(rt) ** self.b) * (RT_diff >= 0) + (np.abs(rt) ** self.b) * (RT_diff < 0)
        prediction_error = utility - self.RTs[chosen]
        self.RTs[chosen] += self.a * prediction_error
        self.EVs = [-x for x in self.RTs]

    def RT_decay(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial

        self.RTs[chosen] += rt
        self.RTs = [x * (1 - self.a) for x in self.RTs]
        self.EVs = [-x for x in self.RTs]

    def RT_decay_PVL(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial
            self.RT_AV = np.mean(self.RTs)

        RT_diff = rt - self.RT_AV
        self.RT_AV += RT_diff / (trial + 1)
        utility = (self.lamda * np.abs(rt) ** self.b) * (RT_diff >= 0) + (np.abs(rt) ** self.b) * (RT_diff < 0)
        self.RTs[chosen] += utility
        self.RTs = [x * (1 - self.a) for x in self.RTs]
        self.EVs = [-x for x in self.RTs]

    def RT_exp_delta(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial

        pred = self.RT_initial[chosen] * np.exp(-1 * self.k * trial)
        prediction_error = rt - pred
        self.RTs[chosen] += self.a * prediction_error
            # prediction_error = rt - self.RTs[chosen]
            # self.RTs[chosen] = self.RT_initial[chosen] * np.exp(-1 * self.k * rt) + self.a * prediction_error
        self.EVs = [-x for x in self.RTs]

    def RT_exp_decay(self, chosen, reward, rt, trial):
        if trial == 1:
            self.RTs = self.RT_initial

        self.RTs[chosen] = self.RT_initial[chosen] * np.exp(-1 * self.k * trial) + rt
        self.RTs = [x * (1 - self.a) for x in self.RTs]
        self.EVs = [-x for x in self.RTs]

    # ------------------------------------------------------------------------------------------------------------------
    # Now we define hybrid models
    # ------------------------------------------------------------------------------------------------------------------
    def hybrid_delta_delta(self, chosen, reward, rt, trial):
        # Reward update
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

        # RT update
        if trial == 1:
            self.RTs = self.RT_initial

        self.RTs[chosen] += self.b * (rt - self.RTs[chosen])

    def hybrid_decay_delta(self, chosen, reward, rt, trial):
        # Reward update
        self.EVs[chosen] += reward
        self.EVs = self.EVs * (1 - self.a)

        # RT update
        if trial == 1:
            self.RTs = self.RT_initial

        self.RTs[chosen] += self.b * (rt - self.RTs[chosen])

    def update(self, chosen, reward, rt, trial):
        """
        Update EVs based on the choice, received reward, and trial number.

        Parameters:
        - chosen: Index of the chosen option.
        - reward: Reward received for the current trial.
        - trial: Current trial number.
        """

        self.choices_count[chosen] += 1

        self.updating_function(chosen, reward, rt, trial)

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
    def WSLS_nll(self, reward, choice, trial, react_time, epsilon):

        nll = 0

        for r, ch, t, rt in zip(reward, choice, trial, react_time):
            prob_choice = self.Probs[ch]
            nll += -np.log(max(epsilon, prob_choice))
            self.update(ch, r, rt, t)

        return nll

    def standard_nll(self, reward, choice, trial, react_time, epsilon):

        nll = 0

        for r, ch, t, rt in zip(reward, choice, trial, react_time):
            prob_choice = self.softmax(np.array(self.EVs))[ch]
            nll += -np.log(max(epsilon, prob_choice))
            self.update(ch, r, rt, t)
            # print(f'Trial: {t}, Chosen: {ch}, Reward: {r}, alpha:{self.a}, RT: {rt}, Ex_RT: {self.RTs}, EV: {self.EVs}, Prob: {prob_choice}')

        return nll

    def weight_nll(self, reward, choice, trial, react_time, epsilon):

        nll = 0

        for r, ch, t, rt in zip(reward, choice, trial, react_time):
            prob_choice = self.softmax(np.array(self.EVs))[ch] * self.w + (1 - self.w) * self.Probs[ch]
            nll += -np.log(max(epsilon, prob_choice))
            self.update(ch, r, rt, t)

        return nll

    def hybrid_nll(self, reward, choice, trial, react_time, epsilon):

        nll = 0

        for r, ch, t, rt in zip(reward, choice, trial, react_time):
            prob_choice_reward = self.softmax(np.array(self.EVs))[ch]
            RT_EVs = [-rt for rt in self.RTs]
            prob_choice_RT = self.softmax(np.array(RT_EVs))[ch]
            prob_choice = self.w * prob_choice_reward + (1 - self.w) * prob_choice_RT
            nll += -np.log(max(epsilon, prob_choice))
            self.update(ch, r, rt, t)

        return nll

    def negative_log_likelihood(self, params, reward, choice, react_time):
        """
        Compute the negative log likelihood for the given parameters and data.

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

        # set the initial RTs
        if self.RT_initial_suboptimal is not None:
            self.RT_initial = [
                self.RT_initial_suboptimal,
                self.RT_initial_optimal
            ]
        else:
            self.RT_initial = None

        self.b = getattr(self, self._B_OVERRIDES.get(self.model_type, 'b'))

        epsilon = 1e-12

        trial_onetask = np.arange(1, len(reward) + 1)

        return self.nll_function(reward, choice, trial_onetask, react_time, epsilon)

    def fit(self, data, num_iterations=1000, beta_lower=-1, beta_upper=1):

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
# End of the VisualSearchModels class (Other assistant classes can be found in ComputationalModeling.py)
# ======================================================================================================================