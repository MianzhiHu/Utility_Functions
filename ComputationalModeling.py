import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, multivariate_t
from concurrent.futures import ProcessPoolExecutor
from DualProcess import generate_random_trial_sequence
import time


def fit_participant(model, participant_id, pdata, model_type, num_iterations=1000,
                    beta_lower=-1, beta_upper=1):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_n = model.num_trials

    if model_type in ('decay', 'delta', 'decay_choice', 'decay_win'):
        k = 2  # Initialize the cumulative number of parameters
    elif model_type in ('decay_fre', 'ACTR', 'ACTR_Ori'):
        k = 3
    elif model_type in ('sampler_decay', 'sampler_decay_PE', 'sampler_decay_AV', 'delta_decay'):
        k = model.num_params

    model.iteration = 0

    best_nll = 100000  # Initialize best negative log likelihood to a large number
    best_initial_guess = None
    best_parameters = None

    for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))

        if model_type in ('decay', 'delta', 'decay_choice', 'decay_win'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999))
        elif model_type in ('decay_fre'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(beta_lower, beta_upper)]
            bounds = ((0.0001, 4.9999), (0.0001, 0.9999), (beta_lower, beta_upper))
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

        if model.task == 'ABCD':
            result = minimize(model.negative_log_likelihood, initial_guess,
                              args=(pdata['reward'], pdata['choiceset'], pdata['choice']),
                              bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
        elif model.task == 'IGT_SGT':
            result = minimize(model.negative_log_likelihood, initial_guess,
                              args=(pdata['reward'], pdata['choice']),
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


class ComputationalModels:
    def __init__(self, model_type, task='ABCD', condition="Gains", num_trials=250, num_params=2):
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
        self.num_options = 4
        self.num_trials = num_trials
        self.num_params = num_params
        self.choices_count = np.zeros(self.num_options)
        self.condition = condition
        self.possible_options = [0, 1, 2, 3]
        self.memory_weights = []
        self.choice_history = []
        self.reward_history = []
        self.chosen_history = {option: [] for option in self.possible_options}
        self.reward_history_by_option = {option: [] for option in self.possible_options}
        self.AllProbs = []
        self.PE = []

        self.t = None
        self.a = None
        self.b = None
        self.s = None
        self.tau = None
        self.w = None  # This is for the addition of the IRL model which is not implemented yet
        self.iteration = 0

        self.condition_initialization = {
            "Gains": 0.5,
            "Losses": -0.5,
            "Both": 0.0
        }

        self.EVs = np.full(self.num_options, self.condition_initialization[self.condition])
        self.AV = self.condition_initialization[self.condition]

        # Model type
        self.model_type = model_type
        self.task = task

        # Mapping of updating functions to model types
        self.updating_mapping = {
            'delta': self.delta_update,
            'decay': self.decay_update,
            'decay_fre': self.decay_fre_update,
            'decay_choice': self.decay_choice_update,
            'decay_win': self.decay_win_update,
            'delta_decay': self.delta_update,
            'sampler_decay': self.sampler_decay_update,
            'sampler_decay_PE': self.sampler_decay_PE_update,
            'sampler_decay_AV': self.sampler_decay_AV_update,
            'ACTR': self.ACTR_update,
            'ACTR_Ori': self.ACTR_update
        }

        self.updating_function = self.updating_mapping[self.model_type]

        # Mapping of nll functions to model types
        self.nll_mapping_ABCD = {
            'delta': self.standard_nll,
            'decay': self.standard_nll,
            'decay_fre': self.standard_nll,
            'decay_choice': self.standard_nll,
            'decay_win': self.standard_nll,
            'delta_decay': self.standard_nll,
            'sampler_decay': self.standard_nll,
            'sampler_decay_PE': self.standard_nll,
            'sampler_decay_AV': self.standard_nll,
            'ACTR': self.ACTR_nll,
            'ACTR_Ori': self.ACTR_nll
        }

        self.nll_mapping_IGT_SGT = {
            'delta': self.igt_nll,
            'decay': self.igt_nll,
            'decay_fre': self.igt_nll,
            'decay_choice': self.igt_nll,
            'decay_win': self.igt_nll,
            'delta_decay': self.igt_nll,
            'sampler_decay': self.igt_nll,
            'sampler_decay_PE': self.igt_nll,
            'sampler_decay_AV': self.igt_nll
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
            'decay': self.softmax,
            'decay_fre': self.softmax,
            'decay_choice': self.softmax,
            'decay_win': self.softmax,
            'delta_decay': self.softmax,
            'sampler_decay': self.softmax,
            'sampler_decay_PE': self.softmax,
            'sampler_decay_AV': self.softmax,
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

        self.EVs = np.full(self.num_options, self.condition_initialization[self.condition])
        self.AV = self.condition_initialization[self.condition]

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

    # ==================================================================================================================
    # Now we define the updating function for each model
    # ==================================================================================================================
    def delta_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error

    def decay_update(self, chosen, reward, trial, choiceset=None):
        self.EVs[chosen] += reward
        self.EVs = self.EVs * (1 - self.a)

    def decay_fre_update(self, chosen, reward, trial, choiceset=None):
        multiplier = self.choices_count[chosen] ** (-self.b)
        self.EVs[chosen] += reward * multiplier
        self.EVs = self.EVs * (1 - self.a)

    def decay_choice_update(self, chosen, reward, trial, choiceset=None):
        self.EVs[chosen] += 1
        self.EVs = self.EVs * (1 - self.a)

    def decay_win_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.AV
        weighted_PE = prediction_error * self.a
        self.AV += weighted_PE
        if prediction_error > 0:
            self.EVs[chosen] += 1
        self.EVs = self.EVs * (1 - self.a)

    def delta_decay_update(self, chosen, reward, trial, choiceset=None):
        prediction_error = reward - self.EVs[chosen]
        self.EVs[chosen] += self.a * prediction_error
        self.EVs = self.EVs * (1 - self.b)

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
        self.EVs = self.EVs * (1 - self.a)
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
        self.EVs = self.EVs * (1 - self.a)
        self.memory_weights = [w * (1 - self.b) for w in self.memory_weights]

        # Compute the probabilities from memory weights
        total_weight = sum(self.memory_weights)
        self.AllProbs = [w / total_weight for w in self.memory_weights]

        # Update EVs based on the samples from memory
        for j in range(len(self.reward_history)):
            self.EVs[self.choice_history[j]] += self.AllProbs[j] * self.PE[j]

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

        return self.EVs

    def update(self, chosen, reward, trial, choiceset=None):
        """
        Update EVs based on the choice, received reward, and trial number.

        Parameters:
        - chosen: Index of the chosen option.
        - reward: Reward received for the current trial.
        - trial: Current trial number.
        """
        if trial > 150:
            return self.EVs

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
                "weight": self.w,
                "trial_details": trial_details,
                "EV_history": EV_history
            })

        return all_results

    # ==================================================================================================================
    # Now we define the negative log likelihood function for the ACT-R model because it requires updating EVs before
    # calculating the likelihood
    # ==================================================================================================================
    def ACTR_nll(self, reward, choiceset, choice, trial, epsilon):

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            # in the ACTR model, we need to update the EVs before calculating the likelihood
            self.update(ch, r, t, cs)
            cs_mapped = self.choiceset_mapping[0][cs]
            prob_choice = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[0]],
                                                                          self.EVs[cs_mapped[1]]]))[0]
            prob_choice_alt = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[1]],
                                                                              self.EVs[cs_mapped[0]]]))[0]
            nll += -np.log(max(epsilon, prob_choice if ch == cs_mapped[0] else prob_choice_alt))

        return nll

    def standard_nll(self, reward, choiceset, choice, trial, epsilon):

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]
            prob_choice = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[0]],
                                                                          self.EVs[cs_mapped[1]]]))[0]
            prob_choice_alt = self.softmax_mapping[self.model_type](np.array([self.EVs[cs_mapped[1]],
                                                                              self.EVs[cs_mapped[0]]]))[0]
            nll += -np.log(max(epsilon, prob_choice if ch == cs_mapped[0] else prob_choice_alt))
            self.update(ch, r, t)

        return nll

    def igt_nll(self, reward, choice, trial, epsilon):

        nll = 0

        for r, ch, t in zip(reward, choice, trial):
            prob_choice = self.softmax_mapping[self.model_type](np.array(self.EVs))[ch]
            nll += -np.log(max(epsilon, prob_choice))
            self.update(ch, r, t)

        return nll

    def negative_log_likelihood(self, params, reward, choiceset, choice):
        """
        Compute the negative log likelihood for the given parameters and data.

        Parameters:
        - params: Parameters of the model.
        - reward: List or array of observed rewards.
        - choiceset: List or array of available choicesets for each trial.
        - choice: List or array of chosen options for each trial.
        """
        self.reset()

        self.s = params[0] if self.model_type == 'ACTR_Ori' else None
        self.t = params[0]
        self.a = params[1]
        self.b = params[2] if self.num_params == 3 else self.a
        self.tau = params[2] if self.model_type in ('ACTR', 'ACTR_Ori') else None

        epsilon = 1e-12

        trial = np.arange(1, self.num_trials + 1)

        return self.nll_function(reward, choiceset, choice, trial, epsilon)

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
            tau = result["tau"] if self.model_type == 'ACTR' else None

            for trial_idx, trial_detail in zip(result['trial_indices'], result['trial_details']):
                var = {
                    "Subnum": sim_num,
                    "trial_index": trial_idx,
                    "s": s,
                    "t": t,
                    "a": a,
                    "b": b,
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
    log_BF_array = -0.5 * (alternative_BIC - null_BIC)
    mean_log_BF = np.mean(log_BF_array)
    BF = np.exp(mean_log_BF)

    return BF


def dict_generator(df, task='ABCD'):
    """
    Convert a dataframe into a dictionary.

    Parameters:
    - df: Dataframe to be converted.

    Returns:
    - A dictionary of the dataframe.
    """
    d = {}
    if task == 'ABCD':
        for name, group in df.groupby('Subnum'):
            d[name] = {
                'reward': group['Reward'].tolist(),
                'choiceset': group['SetSeen.'].tolist(),
                'choice': group['KeyResponse'].tolist(),
            }
    elif task == 'IGT_SGT':
        for name, group in df.groupby('Subnum'):
            d[name] = {
                'reward': group['outcome'].tolist(),
                'choice': group['choice'].tolist(),
            }

    return d


def best_param_generator(df, param):
    """

    :param df:
    :param param:
    :return:
    """
    if param == 't':
        t_best = df['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[0]) if isinstance(x, str) else np.nan
        )
        return t_best

    elif param == 'a':
        a_best = df['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[1]) if isinstance(x, str) else np.nan
        )
        return a_best

    elif param == 'b':
        b_best = df['best_parameters'].apply(
            lambda x: float(x.strip('[]').split()[2]) if isinstance(x, str) else np.nan
        )
        return b_best
