import numpy as np
import pandas as pd
import time
import ast
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import dirichlet, multivariate_normal, entropy, norm
from concurrent.futures import ProcessPoolExecutor


# This function is used to build and fit a dual-process model
# The idea of the dual-process model is that decision-making, particularly decision-making in the ABCD task,
# is potentially driven by two processes: a Dirichlet process and a Gaussian process.
# When the variance of the underlying reward distribution is small, the Gaussian process (average) dominates the
# decision-making process, whereas when the variance is large, the Dirichlet process (frequency) dominates.


def fit_participant(model, participant_id, pdata, model_type, weight_fun, num_iterations=1000):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_n = model.num_trials

    if model_type in ('Dir', 'Gau', 'Recency', 'Param_Dynamic', 'Param_Dynamic', 'Param_Dynamic_Recency', 'Entropy_Recency',
                      'Confidence_Recency', 'Threshold', 'Entropy_Dis', 'Dual_Dis'):
        k = 2
    elif model_type in ('Threshold_Recency', 'Param', 'Entropy_Dis_ID'):
        k = 3
    elif model_type == 'Multi_Param':
        k = 7
    else:
        k = 1

    if weight_fun == 'pure_weight':
        k = k - 1

    model.iteration = 0

    best_nll = 100000
    best_initial_guess = None
    best_parameters = None
    best_weight = None
    best_obj_weight = None
    best_EV_Gau = None
    best_EV_Dir = None
    best_EV = None

    for _ in range(num_iterations):

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))

        if model_type in ('Dir', 'Gau', 'Recency', 'Param_Dynamic', 'Param_Dynamic', 'Param_Dynamic_Recency',
                          'Entropy_Recency', 'Confidence_Recency', 'Entropy_Dis', 'Dual_Dis'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999)]
        elif model_type in ('Entropy_Dis_ID', 'Param'):
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999)]
        elif model_type == 'Threshold':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.6931)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.6931)]
        elif model_type == 'Threshold_Recency':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.6931)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.6931)]
        elif model_type == 'Multi_Param':
            initial_guess = [np.random.uniform(0.0001, 4.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999), np.random.uniform(0.0001, 0.9999),
                             np.random.uniform(0.0001, 0.9999)]
            bounds = [(0.0001, 4.9999), (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999),
                      (0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999)]
        else:
            initial_guess = [np.random.uniform(0.0001, 4.9999)]
            bounds = [(0.0001, 4.9999)]

        if weight_fun == 'pure_weight':
            # remove the first parameter from the initial guess and bounds
            initial_guess = initial_guess[1:]
            bounds = bounds[1:]
            if len(initial_guess) == 0:
                result = model.negative_log_likelihood_weight(initial_guess, pdata['reward'], pdata['choiceset'],
                                                              pdata['choice'])
            else:
                result = minimize(model.negative_log_likelihood_weight, initial_guess,
                                  args=(pdata['reward'], pdata['choiceset'], pdata['choice']),
                                  bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})
        else:
            result = minimize(model.negative_log_likelihood, initial_guess,
                              args=(pdata['reward'], pdata['choiceset'], pdata['choice']),
                              bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

        if isinstance(result, OptimizeResult):
            if result.fun < best_nll:
                best_nll = result.fun
                best_initial_guess = initial_guess
                best_parameters = result.x
                best_process_chosen = model.process_chosen
                best_weight = model.weight_history
                best_obj_weight = model.obj_weight_history
                best_EV_Gau = model.EV_Gau
                best_EV_Dir = model.EV_Dir
                best_EV = model.EVs
        elif isinstance(result, (float, np.float64)):
            best_nll = result
            best_initial_guess = initial_guess
            best_parameters = initial_guess  # Since no optimization, best_parameters are initial guess
            best_process_chosen = model.process_chosen
            best_weight = model.weight_history

    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_initial_guess': best_initial_guess,
        'best_parameters': best_parameters,
        'EV_Gau': best_EV_Gau,
        'EV_Dir': best_EV_Dir,
        'EV': best_EV,
        'best_process_chosen': best_process_chosen if model_type in ('Dual', 'Recency', 'Threshold',
                                                                     'Threshold_Recency', 'Dual_Dis') else None,
        'best_weight': best_weight if model_type in ('Entropy', 'Entropy_Recency', 'Confidence', 'Confidence_Recency',
                                                     'Threshold', 'Threshold_Recency', 'Entropy_Dis', 'Entropy_Dis_ID'
                                                     ) else None,
        'best_obj_weight': best_obj_weight if model_type in ('Entropy_Dis', 'Entropy_Dis_ID') else None,
        'total_nll': best_nll,
        'AIC': aic,
        'BIC': bic
    }

    print(f"Participant {participant_id} fitted in {(time.time() - start_time) / 60} minutes.")

    return result_dict


def EV_calculation(EV_Dir, EV_Gau, weight):
    EVs = weight * EV_Dir + (1 - weight) * EV_Gau
    return EVs


def generate_random_trial_sequence(AB_freq, CD_freq):
    training_trials = [(0, 1), (2, 3)]
    training_trial_sequence = [training_trials[0]] * AB_freq + [training_trials[1]] * CD_freq
    np.random.shuffle(training_trial_sequence)

    transfer_trials = [(2, 0), (1, 3), (0, 3), (2, 1)]
    transfer_trial_sequence = transfer_trials * 25
    np.random.shuffle(transfer_trial_sequence)

    return training_trial_sequence, transfer_trial_sequence


class DualProcessModel:
    def __init__(self, n_samples=1000, num_trials=250):
        self.iteration = None
        self.num_trials = num_trials
        self.EVs = np.full(4, 0.5)
        self.default_EVs = np.full(4, 0.5)
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 1 / 12)
        self.M2 = np.full(4, 0.0)
        self.alpha = np.full(4, 1.0)
        self.gamma_a = np.full(4, 0.5)
        self.gamma_b = np.full(4, 0.0)
        self.n_samples = n_samples
        self.reward_history = [[0] for _ in range(4)]
        self.process_chosen = []
        self.weight_history = []
        self.obj_weight_history = []

        self.t = None
        self.a = None
        self.tau = None
        self.weight = None
        self.model = None
        self.sim_type = None
        self.arbitration_function = None
        self.Gau_update_fun = None
        self.Dir_update_fun = None
        self.action_selection_Gau = None
        self.action_selection_Dir = None
        self.weight_fun = None
        self.param_start = 0

        self.prior_mean = 0.5
        self.prior_var = 1 / 12

        # Define the mapping between model parameters and input features
        self.choiceset_mapping = [
            {0: (0, 1),
             1: (2, 3),
             2: (2, 0),
             3: (2, 1),
             4: (0, 3),
             5: (1, 3)
             },
            {'AB': (0, 1),
             'CD': (2, 3),
             'CA': (2, 0),
             'CB': (2, 1),
             'BD': (1, 3),
             'AD': (0, 3)
             }
        ]

        self.model_mapping = {
            'Dir': self.dir_nll,
            'Gau': self.gau_nll,
            'Dual': self.dual_nll,
            'Recency': self.recency_nll,
            'Dual_Dis': self.distribution_dual_nll,
            'Entropy': self.entropy_nll,
            'Entropy_Recency': self.entropy_recency_nll,
            'Entropy_Dis': self.distribution_entropy_nll,
            'Entropy_Dis_ID': self.distribution_entropy_id_nll,
            'Confidence': self.confidence_nll,
            'Confidence_Recency': self.confidence_recency_nll,
            'Threshold': self.threshold_nll,
            'Threshold_Recency': self.threshold_recency_nll,
            'Param': self.param_nll,
            'Multi_Param': self.multi_param_nll
        }

        self.sim_function_mapping = {
            'Dir': self.single_process_sim,
            'Gau': self.single_process_sim,
            'Dual': self.dual_process_sim,
            'Recency': self.dual_process_sim,
            'Dual_Dis': self.distribution_dual_sim,
            'Entropy': self.entropy_sim,
            'Entropy_Dis': self.distribution_entropy_sim,
            'Entropy_Dis_ID': self.distribution_entropy_id_sim,
            'Threshold': self.threshold_sim,
            'Threshold_Recency': self.threshold_sim,
            'Threshold_Dis': self.distribution_threshold_sim,
            'Param': self.param_sim,
        }

        self.Gau_fun_mapping = {
            'Dir': self.Gau_naive_update,
            'Gau': self.Gau_bayesian_update_with_recency,
            'Dual': self.Gau_bayesian_update,
            'Recency': self.Gau_bayesian_update_with_recency,
            'Dual_Dis': self.Gau_bayesian_update_with_recency,
            'Entropy': self.Gau_bayesian_update,
            'Entropy_Recency': self.Gau_bayesian_update_with_recency,
            'Entropy_Dis': self.Gau_bayesian_update,
            'Entropy_Dis_ID': self.Gau_bayesian_update_with_recency,
            'Confidence': self.Gau_bayesian_update,
            'Confidence_Recency': self.Gau_bayesian_update_with_recency,
            'Threshold': self.Gau_bayesian_update,
            'Threshold_Recency': self.Gau_bayesian_update_with_recency,
            'Threshold_Dis': self.Gau_bayesian_update,
            'Param': self.Gau_bayesian_update_with_recency,
            'Multi_Param': self.Gau_bayesian_update_with_recency
        }

        self.Dir_fun_mapping = {
            'Dir': self.Dir_update,
            'Gau': self.Dir_update,
            'Dual': self.Dir_update,
            'Recency': self.Dir_update_with_linear_recency,
            'Dual_Dis': self.Dir_update_with_linear_recency,
            'Entropy': self.Dir_update,
            'Entropy_Recency': self.Dir_update_with_linear_recency,
            'Entropy_Dis': self.Dir_update,
            'Entropy_Dis_ID': self.Dir_update_with_linear_recency,
            'Confidence': self.Dir_update,
            'Confidence_Recency': self.Dir_update_with_linear_recency,
            'Threshold': self.Dir_update,
            'Threshold_Recency': self.Dir_update_with_linear_recency,
            'Threshold_Dis': self.Dir_update_with_linear_recency,
            'Param': self.Dir_update_with_linear_recency,
            'Multi_Param': self.Dir_update_with_linear_recency
        }

        self.arbitration_mapping = {
            'Original': self.original_arbitration_mechanism,
            'Max Prob': self.max_prob_arbitration_mechanism,
            'Entropy': self.entropy_arbitration_mechanism
        }

        self.selection_mapping_Dir = {
            'softmax': self.softmax,
            'weight': self.weight_prop,
            'weight_softmax': self.weight_softmax
        }

        self.selection_mapping_Gau = {
            'softmax': self.softmax,
            'weight': self.weight_prop,
            'weight_Gau': self.weight_Gau,
            'weight_softmax': self.weight_softmax
        }

        self.Gau_fun_customized = {
            'Naive': self.Gau_naive_update,
            'Bayesian': self.Gau_bayesian_update,
            'Naive_Recency': self.Gau_naive_update_with_recency,
            'Bayesian_Recency': self.Gau_bayesian_update_with_recency
        }

        self.Dir_fun_customized = {
            'Normal': self.Dir_update,
            'Linear_Recency': self.Dir_update_with_linear_recency,
            'Exp_Recency': self.Dir_update_with_exp_recency
        }

    def reset(self):
        self.EV_Dir = np.full(4, 0.5)
        self.EV_Gau = np.full(4, 0.5)
        self.AV = np.full(4, 0.5)
        self.var = np.full(4, 1 / 12)
        self.M2 = np.full(4, 0.0)
        self.alpha = np.full(4, 1.0)
        self.gamma_a = np.full(4, 0.5)
        self.gamma_b = np.full(4, 0.0)
        self.reward_history = [[] for _ in range(4)]
        self.process_chosen = []
        self.weight_history = []
        self.obj_weight_history = []

    def softmax(self, chosen_prob, alt1_prob, chosen, alt1, n1, n2):
        c = 3 ** self.t - 1
        num = np.exp(min(700, c * chosen_prob))
        denom = np.clip(num + np.exp(min(700, c * alt1_prob)), 0.0001, 9999)
        return np.clip(num / denom, 0.0001, 0.9999)

    def weight_prop(self, chosen_prob, alt1_prob, chosen, alt1, n1, n2):
        weight = chosen_prob / (chosen_prob + alt1_prob)
        return np.clip(weight, 0.0001, 0.9999)

    def weight_Gau(self, chosen_prob, alt1_prob, chosen, alt1, n1, n2):
        n1 = np.clip(n1, 1, 9999)
        n2 = np.clip(n2, 1, 9999)
        mu_D = self.AV[chosen] - self.AV[alt1]
        var_D = np.sqrt((self.var[chosen] * n1 + self.var[alt1] * n2) / (n1 + n2))
        weight = norm.cdf(mu_D / var_D)
        return np.clip(weight, 0.0001, 0.9999)

    def weight_softmax(self, chosen_prob, alt1_prob, chosen, alt1, n1, n2):
        num = np.exp(chosen_prob)
        denom = num + np.exp(alt1_prob)
        return np.clip(num / denom, 0.0001, 0.9999)

    def original_arbitration_mechanism(self, max_prob, dir_prob, dir_prob_alt, gau_prob, gau_prob_alt, trial_type=None,
                                       chosen=None):
        # Define the mapping of trial types to their respective probabilities
        trial_mapping = {
            trial_type[0]: (dir_prob, gau_prob),
            trial_type[1]: (dir_prob_alt, gau_prob_alt)
        }

        # Retrieve the appropriate probabilities based on the chosen trial plot_type
        dir_prob_selected, gau_prob_selected = trial_mapping[chosen]

        # Determine the chosen process and probabilities
        chosen_process = 'Dir' if dir_prob_selected > gau_prob_selected else 'Gau'
        prob_choice = max(dir_prob_selected, gau_prob_selected)
        prob_choice_alt = 1 - prob_choice

        return chosen_process, prob_choice, prob_choice_alt

    def max_prob_arbitration_mechanism(self, max_prob, dir_prob, dir_prob_alt, gau_prob, gau_prob_alt, trial_type=None,
                                       chosen=None):
        # Mapping of processes to their probabilities
        process_probs = {
            'Dir': (dir_prob, dir_prob_alt),
            'Gau': (gau_prob, gau_prob_alt)
        }

        # Finding the chosen process and associated probabilities
        chosen_process = next((process for process, probs in process_probs.items() if max_prob in probs), None)

        # Assigning the probabilities based on the chosen process
        prob_choice, prob_choice_alt = process_probs[chosen_process]

        return chosen_process, prob_choice, prob_choice_alt

    def entropy_arbitration_mechanism(self, max_prob, dir_prob, dir_prob_alt, gau_prob, gau_prob_alt, trial_type=None,
                                      chosen=None):

        entropies = {
            'Dir': (entropy([dir_prob, dir_prob_alt]), dir_prob, dir_prob_alt),
            'Gau': (entropy([gau_prob, gau_prob_alt]), gau_prob, gau_prob_alt)
        }

        chosen_process = min(entropies, key=lambda k: entropies[k][0])
        prob_choice, prob_choice_alt = entropies[chosen_process][1], entropies[chosen_process][2]

        return chosen_process, prob_choice, prob_choice_alt

    def Gau_bayesian_update(self, prior_mean, prior_var, reward, chosen, n=1):
        # since we are conducting sequential Bayesian updating with a batch size of 1, the sample variance needs to be
        # estimated with an inverse gamma distribution

        self.gamma_a[chosen] += n / 2
        self.gamma_b[chosen] += (reward - prior_mean) ** 2 / 2

        if self.gamma_a[chosen] <= 1:
            self.AV[chosen] = reward
            self.var[chosen] = prior_var
        else:
            # sample variance can be directly calculated using a / (b - 1)
            sample_var = self.gamma_b[chosen] / (self.gamma_a[chosen] - 1)

            self.AV[chosen] = (prior_mean * sample_var + reward * n * prior_var) / (prior_var * n + sample_var)
            self.var[chosen] = (prior_var * sample_var) / (n * prior_var + sample_var)

    def Gau_bayesian_update_with_recency(self, prior_mean, prior_var, reward, chosen, n=1):

        self.gamma_a[chosen] += n / 2
        self.gamma_b[chosen] += (reward - prior_mean) ** 2 / 2

        if self.gamma_a[chosen] <= 1:
            sample_var = (reward - self.AV[chosen]) ** 2
        else:
            # sample variance can be directly calculated using a / (b - 1)
            sample_var = self.gamma_b[chosen] / (self.gamma_a[chosen] - 1)

        self.AV[chosen] = ((prior_mean * sample_var + self.a * reward * n * prior_var) /
                           (self.a * prior_var * n + sample_var))
        self.var[chosen] = (prior_var * sample_var) / (self.a * n * prior_var + sample_var)

    def Gau_naive_update(self, prior_mean, prior_var, reward, chosen, n=1):
        self.AV[chosen] = np.mean(self.reward_history[chosen])
        self.var[chosen] = np.var(self.reward_history[chosen], ddof=1) if len(self.reward_history[chosen]) > 1 else (
                                                                                                                            reward - self.prior_mean) ** 2

    def Gau_naive_update_with_recency(self, prior_mean, prior_var, reward, chosen, n=1):
        delta = self.a * (reward - prior_mean)
        self.AV[chosen] += delta
        self.M2[chosen] += delta * (reward - self.AV[chosen])
        self.var[chosen] = self.M2[chosen] / np.clip((len(self.reward_history[chosen]) - 1), 1, 9999)

    def Dir_update(self, chosen, reward, AV_total, trial):
        if (reward > AV_total and trial > 1) or (reward > self.prior_mean and trial == 1):
            self.alpha[chosen] += 1
        else:
            pass

    def Dir_update_with_linear_recency(self, chosen, reward, AV_total, trial):
        if (reward > AV_total and trial > 1) or (reward > self.prior_mean and trial == 1):
            self.alpha[chosen] += 1
        else:
            pass

        self.alpha = [np.clip(i * (1 - self.a), 1, 9999) for i in self.alpha]

    def Dir_update_with_exp_recency(self, chosen, reward, AV_total, trial):
        if (reward > AV_total and trial > 1) or (reward > self.prior_mean and trial == 1):
            self.alpha[chosen] += 1
        else:
            pass

        self.alpha = [np.clip(i ** (1 - self.a), 1, 9999) for i in self.alpha]

    def Dir_update_learning(self, chosen, reward, AV_total, trial):
        if (reward > AV_total and trial > 1) or (reward > self.prior_mean and trial == 1):
            self.alpha[chosen] += 1 * self.a
        else:
            pass

    def update(self, chosen, reward, trial):

        if trial > 150:
            return self.EV_Dir, self.EV_Gau

        else:
            self.reward_history[chosen].append(reward)

            # for every trial, we need to update the EV for both the Dirichlet and Gaussian processes
            # Gaussian process
            self.Gau_update_fun(self.AV[chosen], self.var[chosen], reward, chosen)

            # # The four options are independent, so the covariance matrix is diagonal
            # cov_matrix = np.diag(self.var)
            #
            # # Check for non-positive diagonal elements and set them to a default variance value
            # for i in range(len(self.var)):
            #     if cov_matrix[i, i] <= 0:
            #         cov_matrix[i, i] = (reward - self.prior_mean) ** 2

            # Sample from the posterior distribution to get the expected value
            # self.EV_Gau = np.mean(multivariate_normal.rvs(self.AV, cov_matrix, size=self.n_samples), axis=0)
            # posterior_Gau = multivariate_normal(self.AV, cov_matrix)
            # self.EV_Gau = posterior_Gau.mean

            self.EV_Gau = self.AV

            # Dirichlet process
            AV_total = np.mean(self.EV_Gau)

            self.Dir_update_fun(chosen, reward, AV_total, trial)

            # Use the updated parameters to get the posterior Dirichlet distribution
            # Sample from the posterior distribution to get the expected value
            # self.EV_Dir = np.mean(dirichlet.rvs(self.alpha, size=self.n_samples), axis=0)
            self.EV_Dir = dirichlet(self.alpha).mean()

        return self.EV_Dir, self.EV_Gau

    # =============================================================================
    # Define the simulation function for each single model
    # This is to reduce the number of if-else statements in the main function and improve time complexity
    # =============================================================================
    def single_process_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                           process=None):
        action_selection_fun = getattr(self, f'action_selection_{process}')
        prob_optimal = action_selection_fun(getattr(self, process)[optimal], getattr(self, process)[suboptimal],
                                            optimal, suboptimal)

        chosen = optimal if np.random.rand() < prob_optimal else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen), "reward": reward})

        self.update(chosen, reward, trial)

    def param_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, process=None):

        EVs = EV_calculation(self.EV_Dir, self.EV_Gau, self.weight)

        prob_optimal = self.softmax(EVs[optimal], EVs[suboptimal], optimal, suboptimal)

        chosen = optimal if np.random.rand() < prob_optimal else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen), "reward": reward})

        self.update(chosen, reward, trial)

    def dual_process_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                         process=None):

        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_suboptimal_dir = 1 - prob_optimal_dir
        prob_suboptimal_gau = 1 - prob_optimal_gau

        max_prob = max(prob_optimal_dir, prob_suboptimal_dir, prob_optimal_gau, prob_suboptimal_gau)

        chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, prob_optimal_dir,
                                                                                 prob_suboptimal_dir,
                                                                                 prob_optimal_gau,
                                                                                 prob_suboptimal_gau,
                                                                                 pair, None)

        chosen = optimal if np.random.rand() < prob_choice else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "process": chosen_process})

        self.update(chosen, reward, trial)

    def distribution_dual_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                              process=None):
        trial_av = [self.AV[optimal], self.AV[suboptimal]]
        trial_var = [self.var[optimal], self.var[suboptimal]]
        trial_cov = np.diag(trial_var)

        gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

        trial_alpha = [self.alpha[optimal], self.alpha[suboptimal]]

        dir_entropy = 2 ** (dirichlet.entropy(trial_alpha))

        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))

        entropies = {
            'Dir': (dir_entropy, prob_optimal_dir, 1 - prob_optimal_dir),
            'Gau': (gau_entropy, prob_optimal_gau, 1 - prob_optimal_gau)
        }

        chosen_process = min(entropies, key=lambda k: entropies[k][0])
        prob_choice = entropies[chosen_process][1]

        chosen = optimal if np.random.rand() < prob_choice else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "process": chosen_process})

        self.update(chosen, reward, trial)

    def entropy_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, process=None):
        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_suboptimal_dir = 1 - prob_optimal_dir
        prob_suboptimal_gau = 1 - prob_optimal_gau

        dir_entropy = entropy([prob_optimal_dir, prob_suboptimal_dir])
        gau_entropy = entropy([prob_optimal_gau, prob_suboptimal_gau])

        weight_dir = gau_entropy / (dir_entropy + gau_entropy)

        prob_optimal = EV_calculation(prob_optimal_dir, prob_optimal_gau, weight_dir)

        chosen = optimal if np.random.rand() < prob_optimal else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "weight": weight_dir})

        self.update(chosen, reward, trial)

    def distribution_entropy_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                                 process=None):

        trial_av = [self.AV[optimal], self.AV[suboptimal]]
        trial_var = [self.var[optimal], self.var[suboptimal]]
        trial_cov = np.diag(trial_var)

        gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

        trial_alpha = [self.alpha[optimal], self.alpha[suboptimal]]

        dir_entropy = 2 ** (dirichlet.entropy(trial_alpha))

        weight_dir = gau_entropy / (dir_entropy + gau_entropy)

        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))

        prob_optimal = EV_calculation(prob_optimal_dir, prob_optimal_gau, weight_dir)

        chosen = optimal if np.random.rand() < prob_optimal else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "weight": weight_dir})

        self.update(chosen, reward, trial)

    def distribution_entropy_id_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                                    process=None):

        trial_av = [self.AV[optimal], self.AV[suboptimal]]
        trial_var = [self.var[optimal], self.var[suboptimal]]
        trial_cov = np.diag(trial_var)

        gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

        trial_alpha = [self.alpha[optimal], self.alpha[suboptimal]]

        dir_entropy = 2 ** (dirichlet.entropy(trial_alpha))

        obj_weight = gau_entropy / (dir_entropy + gau_entropy)
        weight_dir = (self.weight * obj_weight) / (self.weight * obj_weight + (1 - self.weight) * (1 - obj_weight))

        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))

        prob_optimal = EV_calculation(prob_optimal_dir, prob_optimal_gau, weight_dir)

        chosen = optimal if np.random.rand() < prob_optimal else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "weight": weight_dir, 'obj_weight': obj_weight})
        self.update(chosen, reward, trial)

    def threshold_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, process=None):
        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_suboptimal_dir = 1 - prob_optimal_dir
        prob_suboptimal_gau = 1 - prob_optimal_gau

        dir_entropy = entropy([prob_optimal_dir, prob_suboptimal_dir])
        gau_entropy = entropy([prob_optimal_gau, prob_suboptimal_gau])

        # max_prob = max(prob_optimal_dir, prob_suboptimal_dir, prob_optimal_gau, prob_suboptimal_gau)

        min_entropy = min(dir_entropy, gau_entropy)

        if min_entropy < self.tau:
            chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(None,
                                                                                     prob_optimal_dir,
                                                                                     prob_suboptimal_dir,
                                                                                     prob_optimal_gau,
                                                                                     prob_suboptimal_gau,
                                                                                     pair, None)
            weight_dir = 0 if chosen_process == 'Gau' else 1
        else:
            chosen_process = 'Parametric'

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)

            prob_choice = EV_calculation(prob_optimal_dir, prob_optimal_gau, weight_dir)

        chosen = optimal if np.random.rand() < prob_choice else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "process": chosen_process, 'weight': weight_dir})

        self.update(chosen, reward, trial)

    def distribution_threshold_sim(self, optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial,
                                   process=None):

        trial_av = [self.AV[optimal], self.AV[suboptimal]]
        trial_var = [self.var[optimal], self.var[suboptimal]]
        trial_cov = np.diag(trial_var)

        gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

        trial_alpha = [self.alpha[optimal], self.alpha[suboptimal]]

        dir_entropy = 2 ** (dirichlet.entropy(trial_alpha))

        min_entropy = min(dir_entropy, gau_entropy)

        prob_optimal_dir = self.action_selection_Dir(self.EV_Dir[optimal], self.EV_Dir[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        prob_optimal_gau = self.action_selection_Gau(self.EV_Gau[optimal], self.EV_Gau[suboptimal],
                                                     optimal, suboptimal, len(self.reward_history[optimal]),
                                                     len(self.reward_history[suboptimal]))
        if min_entropy < self.tau:
            entropies = {
                'Dir': (dir_entropy, prob_optimal_dir, 1 - prob_optimal_dir),
                'Gau': (gau_entropy, prob_optimal_gau, 1 - prob_optimal_gau)
            }

            chosen_process = min(entropies, key=lambda k: entropies[k][0])
            prob_choice, prob_choice_alt = entropies[chosen_process][1], entropies[chosen_process][2]
            weight_dir = 0 if chosen_process == 'Gau' else 1
        else:
            chosen_process = 'Parametric'

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)

            prob_choice = EV_calculation(prob_optimal_dir, prob_optimal_gau, weight_dir)

        chosen = optimal if np.random.rand() < prob_choice else suboptimal

        reward = np.random.normal(reward_means[chosen], reward_sd[chosen])

        trial_details.append(
            {"pair": (chr(65 + pair[0]), chr(65 + pair[1])), "choice": chr(65 + chosen),
             "reward": reward, "process": chosen_process, 'weight': weight_dir})

        self.update(chosen, reward, trial)

    def unpack_simulation_results(self, results, use_random_sequences=None):
        unpacked_results = []

        for result in results:
            if 'simulation_num' not in result:  # that means this is a post-hoc simulation
                self.sim_type = 'post-hoc'
                sim_num = result['Subnum']
                t = result["t"]
                a = result["a"]
                param_weight = result["weight"] if self.model in ('Param', 'Entropy_Dis_ID') else None
                tau = result["tau"] if self.model in ('Threshold', 'Threshold_Recency') else None

                for trial_idx, trial_detail in zip(result['trial_indices'], result['trial_details']):
                    var = {
                        "Subnum": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
                        "a": a,
                        "param_weight": param_weight if self.model in ('Param', 'Entropy_Dis_ID') else None,
                        'obj_weight': trial_detail.get('obj_weight') if self.model in ('Entropy_Dis_ID') else None,
                        "weight_Dir": trial_detail.get('weight') if self.model in ('Entropy', 'Threshold',
                                                                                   'Entropy_Dis', 'Threshold_Dis',
                                                                                   'Entropy_Dis_ID') else None,
                        "tau": tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                        "pair": trial_detail['pair'],
                        "choice": trial_detail['choice'],
                        "reward": trial_detail['reward'],
                        "process": trial_detail.get('process') if self.model in (
                            'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                    }
                    unpacked_results.append(var)
            else:
                self.sim_type = 'a priori'
                use_random_sequences = use_random_sequences
                sim_num = result["simulation_num"]
                t = result["t"]
                a = result["a"] if self.model in ('Recency', 'Threshold_Recency', 'Entropy_Dis_ID') else None
                param_weight = result["param_weight"] if self.model in ('Param', 'Entropy_Dis_ID') else None
                tau = result["tau"] if self.model in ('Threshold', 'Threshold_Recency') else None

                for trial_idx, trial_detail, ev_dir, ev_gau in zip(result['trial_indices'], result['trial_details'],
                                                                   result['EV_history_Dir'], result['EV_history_Gau']):
                    var = {
                        "simulation_num": sim_num,
                        "trial_index": trial_idx,
                        "t": t,
                        "a": a if self.model in (
                            'Recency', 'Threshold_Recency', 'Entropy_Dis', 'Threshold_Dis', 'Dual_Dis',
                            'Entropy_Dis_ID') else None,
                        "param_weight": param_weight if self.model in ('Param', 'Entropy_Dis_ID') else None,
                        'obj_weight': trial_detail.get('obj_weight') if self.model in ('Entropy_Dis_ID') else None,
                        "weight_Dir": trial_detail.get('weight') if self.model in (
                            'Entropy', 'Threshold', 'Entropy_Dis', 'Threshold_Dis', 'Entropy_Dis_ID') else None,
                        "tau": tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                        "pair": trial_detail['pair'],
                        "choice": trial_detail['choice'],
                        'process': trial_detail.get('process') if self.model in (
                            'Dual', 'Recency', 'Threshold', 'Threshold_Recency', 'Threshold_Dis', 'Dual_Dis') else None,
                        "reward": trial_detail['reward'],
                        "EV_A_Dir": ev_dir[0] if self.model in (
                            'Dir', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_B_Dir": ev_dir[1] if self.model in (
                            'Dir', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_C_Dir": ev_dir[2] if self.model in (
                            'Dir', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_D_Dir": ev_dir[3] if self.model in (
                            'Dir', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_A_Gau": ev_gau[0] if self.model in (
                            'Gau', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_B_Gau": ev_gau[1] if self.model in (
                            'Gau', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_C_Gau": ev_gau[2] if self.model in (
                            'Gau', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_D_Gau": ev_gau[3] if self.model in (
                            'Gau', 'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None,
                        "EV_A": EV_calculation(ev_dir, ev_gau, param_weight)[0] if self.model in ('Param') else None,
                        "EV_B": EV_calculation(ev_dir, ev_gau, param_weight)[1] if self.model in ('Param') else None,
                        "EV_C": EV_calculation(ev_dir, ev_gau, param_weight)[2] if self.model in ('Param') else None,
                        "EV_D": EV_calculation(ev_dir, ev_gau, param_weight)[3] if self.model in ('Param') else None,
                    }
                    unpacked_results.append(var)

        df = pd.DataFrame(unpacked_results)

        if self.sim_type == 'a priori':
            df = df.dropna(axis=1, how='all')
            return df
        elif self.sim_type == 'post-hoc':
            df['pair'] = df['pair'].map(lambda x: ''.join(x))
            best_option_dict = {'AB': 'A', 'CA': 'C', 'AD': 'A',
                                'CB': 'C', 'BD': 'B', 'CD': 'C'}
            df['BestOption'] = df['pair'].map(best_option_dict)
            df['BestOptionChosen'] = df['choice'] == df['BestOption']
            if not use_random_sequences:
                df['process'] = df['process'].map({'Gau': 0, 'Dir': 1}) if self.model in (
                    'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None
                df['Param_Process'] = df['process'].isna().astype(int) if self.model in (
                    'Threshold', 'Threshold_Recency') else None
                summary = df.groupby(['Subnum', 'trial_index']).agg(
                    pair=('pair', 'first'),
                    reward=('reward', 'mean'),
                    t=('t', 'mean'),
                    a=('a', 'mean'),
                    param_weight=('param_weight', 'mean'),
                    obj_weight=('obj_weight', 'mean'),
                    weight_Dir=('weight_Dir', 'mean'),
                    tau=('tau', 'mean'),
                    choice=('BestOptionChosen', 'mean'),
                    process=('process', 'mean'),
                    param_process=('Param_Process', 'mean')
                ).reset_index()
            else:
                df['process'] = df['process'].map({'Gau': 0, 'Dir': 1}) if self.model in (
                    'Dual', 'Recency', 'Threshold', 'Threshold_Recency') else None
                df['Param_Process'] = df['process'].isna().astype(int) if self.model in (
                    'Threshold', 'Threshold_Recency') else None
                summary = df.groupby(['Subnum', 'pair']).agg(
                    reward=('reward', 'mean'),
                    t=('t', 'mean'),
                    a=('a', 'mean'),
                    param_weight=('param_weight', 'mean'),
                    obj_weight=('obj_weight', 'mean'),
                    weight_Dir=('weight_Dir', 'mean'),
                    tau=('tau', 'mean'),
                    choice=('BestOptionChosen', 'mean'),
                    process=('process', 'mean'),
                    param_process=('Param_Process', 'mean')
                ).reset_index()
            summary = summary.dropna(axis=1, how='all')
            return summary

    def simulate(self, reward_means, reward_sd, model, AB_freq=None, CD_freq=None,
                 sim_trials=250, num_iterations=1000, arbi_option='Entropy', weight_Dir='weight',
                 weight_Gau='softmax', Gau_fun=None, Dir_fun=None):

        self.model = model

        sim_func = self.sim_function_mapping[self.model]

        self.arbitration_function = self.arbitration_mapping[arbi_option]

        # Assign the methods based on the provided strings
        self.action_selection_Dir = self.selection_mapping_Dir.get(weight_Dir)
        self.action_selection_Gau = self.selection_mapping_Gau.get(weight_Gau)

        self.Gau_update_fun = self.Gau_fun_customized.get(Gau_fun, self.Gau_fun_mapping[self.model])
        self.Dir_update_fun = self.Dir_fun_customized.get(Dir_fun, self.Dir_fun_mapping[self.model])

        print(f'============================================================')
        print(f'In the current model, the Dirichlet process is updated using {self.Dir_update_fun.__name__} '
              f'and the Gaussian process is updated using {self.Gau_update_fun.__name__}')
        print(f'Dirichlet process is selected using {self.action_selection_Dir.__name__} and '
              f'Gaussian process is selected using {self.action_selection_Gau.__name__}')
        print(f'The arbitration mechanism used is {self.arbitration_function.__name__}')
        print(f'============================================================')

        all_results = []

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1} of {num_iterations}")

            self.reset()

            if self.model in ('Dir', 'Gau'):
                process = f'EV_{self.model}'
            else:
                process = None

            # Randomly sample the parameters for the model
            self.t = np.random.uniform(0.0001, 4.9999)

            if self.Dir_update_fun in (self.Dir_update_with_linear_recency, self.Dir_update_with_exp_recency) or \
                    self.Gau_update_fun in (self.Gau_bayesian_update_with_recency, self.Gau_naive_update_with_recency):
                self.a = np.random.uniform(0.0001, .9999)

            self.weight = np.random.uniform(0.0001, 0.9999) if self.model in ('Param',
                                                                              'Entropy_Dis_ID') else None

            if self.model in ('Threshold', 'Threshold_Recency'):
                self.tau = np.random.uniform(0.0001, 0.6931)
            elif self.model in ('Threshold_Dis'):
                self.tau = np.random.uniform(0.0001, 2.9999)

            # Initialize the EVs for the Dirichlet and Gaussian processes
            EV_history_Dir = np.zeros((sim_trials, 4))
            EV_history_Gau = np.zeros((sim_trials, 4))
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

                optimal, suboptimal = (pair[0], pair[1])

                sim_func(optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial_indice, process)

                EV_history_Dir[trial] = self.EV_Dir
                EV_history_Gau[trial] = self.EV_Gau

            all_results.append({
                "simulation_num": iteration + 1,
                "trial_indices": trial_indices,
                "t": self.t,
                "a": self.a if self.model in ('Recency', 'Threshold_Recency', 'Entropy_Dis', 'Threshold_Dis',
                                              'Dual_Dis', 'Entropy_Dis_ID') else None,
                "param_weight": self.weight if self.model in ('Param', 'Entropy_Dis_ID') else None,
                "tau": self.tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                "trial_details": trial_details,
                "EV_history_Dir": EV_history_Dir,
                "EV_history_Gau": EV_history_Gau
            })

        return self.unpack_simulation_results(all_results)

    # =============================================================================
    # Define the negative log likelihood function for each single model
    # This is to reduce the number of if-else statements in the main function and improve time complexity
    # =============================================================================
    """
    Use the following print statement to debug the negative log likelihood functions
    
    # print(f'Trial: {t}, Trial Type: {cs_mapped}, Choice: {ch}, Reward: {r}')
    # print(f'Dir_EV: {self.EV_Dir}')
    # print(f'Gau_EV: {self.EV_Gau}')
    # print(f'Dir_Prob: {dir_prob}, Dir_Prob_Alt: {dir_prob_alt}')
    # print(f'Gau_Prob: {gau_prob}, Gau_Prob_Alt: {gau_prob_alt}')
    # print(f'Dir_Entropy: {dir_entropy}, Gau_Entropy: {gau_entropy}')
    # print(f'Weight: {weight_dir}, Prob_Choice: {prob_choice}')
    
    """

    def param_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[self.param_start + 1]
        self.weight = params[self.param_start + 2]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))

            # Calculate the probability of choosing the optimal option
            prob_choice = EV_calculation(dir_prob, gau_prob, self.weight)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            # Update the EVs
            self.update(ch, r, t)

        return nll

    def multi_param_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        # Decide which trial plot_type it is
        weight_mapping = {
            0: params[1],
            1: params[2],
            2: params[3],
            3: params[4],
            4: params[5],
            5: params[6]
        }

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            if np.std(self.EV_Dir) == 0 and np.std(self.EV_Gau) == 0:
                self.EVs = self.default_EVs
            else:
                # Standardize the EVs
                EV_Dir = (self.EV_Dir - np.mean(self.EV_Dir)) / np.std(self.EV_Dir)
                EV_Gau = (self.EV_Gau - np.mean(self.EV_Gau)) / np.std(self.EV_Gau)

                weight = weight_mapping[cs]

                # Calculate the expected value of the model
                self.EVs = EV_calculation(EV_Dir, EV_Gau, weight)

            prob_choice = self.softmax(self.EVs[cs_mapped[0]], self.EVs[cs_mapped[1]])
            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def dir_nll(self, params, reward, choiceset, choice, trial):

        self.a = params[self.param_start + 1]

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            prob_choice = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                    cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                    len(self.reward_history[cs_mapped[1]]))
            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def gau_nll(self, params, reward, choiceset, choice, trial):

        self.a = params[self.param_start + 1]

        nll = 0

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            prob_choice = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                    cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                    len(self.reward_history[cs_mapped[1]]))
            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def dual_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.process_chosen = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                     dir_prob_alt,
                                                                                     gau_prob, gau_prob_alt,
                                                                                     cs_mapped, ch)
            self.process_chosen.append(chosen_process)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)

            self.update(ch, r, t)

        return nll

    def recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[self.param_start + 1]

        self.process_chosen = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], None, None)
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                     dir_prob_alt,
                                                                                     gau_prob, gau_prob_alt,
                                                                                     cs_mapped, ch)
            self.process_chosen.append(chosen_process)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)

            self.update(ch, r, t)

        return nll

    def entropy_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            dir_entropy = entropy([dir_prob, dir_prob_alt])
            gau_entropy = entropy([gau_prob, gau_prob_alt])

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def entropy_recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        self.a = params[self.param_start + 1]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            dir_entropy = entropy([dir_prob, dir_prob_alt])
            gau_entropy = entropy([gau_prob, gau_prob_alt])

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def distribution_entropy_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        self.a = params[self.param_start + 1]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            trial_av = [self.AV[cs_mapped[0]], self.AV[cs_mapped[1]]]
            trial_var = [self.var[cs_mapped[0]], self.var[cs_mapped[1]]]
            trial_cov = np.diag(trial_var)

            gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

            trial_alphas = [self.alpha[cs_mapped[0]], self.alpha[cs_mapped[1]]]
            dir_entropy = 2 ** (dirichlet.entropy(trial_alphas))

            weight_dir = gau_entropy / (dir_entropy + gau_entropy)
            self.weight_history.append(weight_dir)

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def distribution_dual_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.process_chosen = []

        self.a = params[self.param_start + 1]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            trial_av = [self.AV[cs_mapped[0]], self.AV[cs_mapped[1]]]
            trial_var = [self.var[cs_mapped[0]], self.var[cs_mapped[1]]]
            trial_cov = np.diag(trial_var)

            gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

            trial_alphas = [self.alpha[cs_mapped[0]], self.alpha[cs_mapped[1]]]
            dir_entropy = 2 ** (dirichlet.entropy(trial_alphas))

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], None, None)
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))

            entropies = {
                'Dir': (dir_entropy, dir_prob, 1 - dir_prob),
                'Gau': (gau_entropy, gau_prob, 1 - gau_prob)
            }

            chosen_process = min(entropies, key=lambda k: entropies[k][0])

            prob_choice, prob_choice_alt = entropies[chosen_process][1], entropies[chosen_process][2]

            self.process_chosen.append(chosen_process)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else prob_choice_alt)

            self.update(ch, r, t)

        return nll

    def distribution_entropy_id_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []
        self.obj_weight_history = []

        self.a = params[self.param_start + 1]
        self.weight = params[self.param_start + 2]

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            trial_av = [self.AV[cs_mapped[0]], self.AV[cs_mapped[1]]]
            trial_var = [self.var[cs_mapped[0]], self.var[cs_mapped[1]]]
            trial_cov = np.diag(trial_var)

            gau_entropy = 2 ** (multivariate_normal.entropy(trial_av, trial_cov))

            trial_alphas = [self.alpha[cs_mapped[0]], self.alpha[cs_mapped[1]]]
            dir_entropy = 2 ** (dirichlet.entropy(trial_alphas))

            obj_weight = gau_entropy / (dir_entropy + gau_entropy)
            weight_dir = (self.weight * obj_weight) / (self.weight * obj_weight + (1 - self.weight) * (1 - obj_weight))

            self.obj_weight_history.append(obj_weight)
            self.weight_history.append(weight_dir)

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def confidence_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            confidence_dir = np.max([dir_prob, dir_prob_alt])
            confidence_gau = np.max([gau_prob, gau_prob_alt])

            weight_dir = confidence_dir / (confidence_dir + confidence_gau)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def confidence_recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[self.param_start + 1]

        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            confidence_dir = np.max([dir_prob, dir_prob_alt])
            confidence_gau = np.max([gau_prob, gau_prob_alt])

            weight_dir = confidence_dir / (confidence_dir + confidence_gau)
            self.weight_history.append(weight_dir)

            prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def threshold_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.tau = params[self.param_start + 1]

        self.process_chosen = []
        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1])
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            if max_prob > self.tau:
                chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(max_prob, dir_prob,
                                                                                         dir_prob_alt,
                                                                                         gau_prob,
                                                                                         gau_prob_alt)
                self.process_chosen.append(chosen_process)
                self.weight_history.append(1.0)
            else:
                chosen_process = 'Param'
                self.process_chosen.append(chosen_process)

                dir_entropy = entropy([dir_prob, dir_prob_alt])
                gau_entropy = entropy([gau_prob, gau_prob_alt])

                weight_dir = gau_entropy / (dir_entropy + gau_entropy)
                self.weight_history.append(weight_dir)

                prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def threshold_recency_nll(self, params, reward, choiceset, choice, trial):

        nll = 0

        self.a = params[self.param_start + 1]
        self.tau = params[self.param_start + 2]

        self.process_chosen = []
        self.weight_history = []

        for r, cs, ch, t in zip(reward, choiceset, choice, trial):
            cs_mapped = self.choiceset_mapping[0][cs]

            dir_prob = self.action_selection_Dir(self.EV_Dir[cs_mapped[0]], self.EV_Dir[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            dir_prob_alt = 1 - dir_prob
            gau_prob = self.action_selection_Gau(self.EV_Gau[cs_mapped[0]], self.EV_Gau[cs_mapped[1]],
                                                 cs_mapped[0], cs_mapped[1], len(self.reward_history[cs_mapped[0]]),
                                                 len(self.reward_history[cs_mapped[1]]))
            gau_prob_alt = 1 - gau_prob

            # max_prob = max(dir_prob, dir_prob_alt, gau_prob, gau_prob_alt)

            dir_entropy = entropy([dir_prob, dir_prob_alt])
            gau_entropy = entropy([gau_prob, gau_prob_alt])

            min_entropy = min(dir_entropy, gau_entropy)

            if min_entropy < self.tau:
                chosen_process, prob_choice, prob_choice_alt = self.arbitration_function(None, dir_prob,
                                                                                         dir_prob_alt,
                                                                                         gau_prob,
                                                                                         gau_prob_alt)
                self.process_chosen.append(chosen_process)
                self.weight_history.append(1.0 if chosen_process == 'Dir' else 0.0)
            else:
                chosen_process = 'Param'
                self.process_chosen.append(chosen_process)

                weight_dir = gau_entropy / (dir_entropy + gau_entropy)
                self.weight_history.append(weight_dir)

                prob_choice = EV_calculation(dir_prob, gau_prob, weight_dir)

            nll += -np.log(prob_choice if ch == cs_mapped[0] else 1 - prob_choice)

            self.update(ch, r, t)

        return nll

    def negative_log_likelihood(self, params, reward, choiceset, choice):

        self.reset()

        self.t = params[self.param_start]

        trial = np.arange(1, self.num_trials + 1)

        return self.model_mapping[self.model](params, reward, choiceset, choice, trial)

    def negative_log_likelihood_weight(self, params, reward, choiceset, choice):

        self.reset()

        self.param_start = -1

        trial = np.arange(1, self.num_trials + 1)

        return self.model_mapping[self.model](params, reward, choiceset, choice, trial)

    def fit(self, data, model, num_iterations=100, arbi_option='Max Prob', Gau_fun=None, Dir_fun=None,
            weight_Gau='weight', weight_Dir='weight'):

        self.model = model
        self.arbitration_function = self.arbitration_mapping[arbi_option]

        # Assign the methods based on the provided strings
        self.action_selection_Dir = self.selection_mapping_Dir.get(weight_Dir)
        self.action_selection_Gau = self.selection_mapping_Gau.get(weight_Gau)

        # Check if both selections are 'weight'
        if self.action_selection_Dir == self.weight_prop and self.action_selection_Gau in (self.weight_prop,
                                                                                           self.weight_Gau,
                                                                                           self.weight_softmax):
            self.weight_fun = 'pure_weight'
        else:
            self.weight_fun = 'mixed_weight'

        self.Gau_update_fun = self.Gau_fun_customized.get(Gau_fun, self.Gau_fun_mapping[self.model])
        self.Dir_update_fun = self.Dir_fun_customized.get(Dir_fun, self.Dir_fun_mapping[self.model])

        print(f'============================================================')
        print(f'In the current model, the Dirichlet process is updated using {self.Dir_update_fun.__name__} '
              f'and the Gaussian process is updated using {self.Gau_update_fun.__name__}')
        print(f'Dirichlet process is selected using {self.action_selection_Dir.__name__} and '
              f'Gaussian process is selected using {self.action_selection_Gau.__name__}')
        print(f'============================================================')

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, model,
                                         self.weight_fun, num_iterations)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        results_df = pd.DataFrame(results).dropna(how='all', axis=1)

        return results_df

    def post_hoc_simulation(self, fitting_result, original_data, model, reward_means, reward_sd, AB_freq=100,
                            CD_freq=50, sim_trials=250, num_iterations=1000, arbi_option='Entropy',
                            Gau_fun=None, Dir_fun=None, weight_Gau='softmax', weight_Dir='softmax',
                            use_random_sequence=True, recency=True):

        self.model = model
        self.arbitration_function = self.arbitration_mapping[arbi_option]

        # Assign the methods based on the provided strings
        self.action_selection_Dir = self.selection_mapping_Dir.get(weight_Dir)
        self.action_selection_Gau = self.selection_mapping_Gau.get(weight_Gau)

        self.Gau_update_fun = self.Gau_fun_customized.get(Gau_fun, self.Gau_fun_mapping[self.model])
        self.Dir_update_fun = self.Dir_fun_customized.get(Dir_fun, self.Dir_fun_mapping[self.model])

        post_hoc_func = self.sim_function_mapping[self.model]

        if self.model in ('Dir', 'Gau'):
            process = f'EV_{self.model}'
        else:
            process = None

        print(f'============================================================')
        print(f'In the current model, the Dirichlet process is updated using {self.Dir_update_fun.__name__} '
              f'and the Gaussian process is updated using {self.Gau_update_fun.__name__}')
        print(f'Dirichlet process is selected using {self.action_selection_Dir.__name__} and '
              f'Gaussian process is selected using {self.action_selection_Gau.__name__}')
        print(f'============================================================')

        # extract the trial sequence for each participant
        if not use_random_sequence:
            trial_index = original_data.groupby('Subnum')['trial_index'].apply(list)
            trial_sequence = original_data.groupby('Subnum')['TrialType'].apply(list)
        else:
            trial_index, trial_sequence = None, None

        num_parameters = len(fitting_result['best_parameters'][0].strip('[]').split())

        parameter_sequences = []
        for i in range(num_parameters):
            sequence = fitting_result['best_parameters'].apply(
                lambda x: float(x.strip('[]').split()[i]) if isinstance(x, str) else np.nan
            )
            parameter_sequences.append(sequence)

        # start the simulation
        all_results = []

        for participant in fitting_result['participant_id']:
            print(f"Participant {participant}")
            start_time = time.time()

            self.t = None
            self.a = None
            self.tau = None
            self.weight = None

            self.t = parameter_sequences[0][participant - 1]

            if recency:
                self.a = parameter_sequences[1][participant - 1]
                self.weight = parameter_sequences[2][participant - 1] if self.model in ('Param', 'Threshold',
                                                                                        'Threshold_Recency',
                                                                                        'Entropy_Dis_ID') else None
                self.tau = parameter_sequences[2][participant - 1] if self.model in ('Threshold',
                                                                                     'Threshold_Recency') else None
            else:
                self.weight = parameter_sequences[1][participant - 1] if self.model in ('Param', 'Threshold',
                                                                                        'Threshold_Recency',
                                                                                        'Entropy_Dis_ID') else None
                self.tau = parameter_sequences[1][participant - 1] if self.model in ('Threshold',
                                                                                     'Threshold_Recency') else None

            for _ in range(num_iterations):

                print(f"Iteration {_ + 1} of {num_iterations}")

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
                        post_hoc_func(optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial_indice,
                                      process)
                else:
                    for trial, pair in zip(trial_index[participant], trial_sequence[participant]):
                        trial_indices.append(trial)
                        optimal, suboptimal = self.choiceset_mapping[1][pair]
                        pair = (optimal, suboptimal)
                        post_hoc_func(optimal, suboptimal, reward_means, reward_sd, trial_details, pair, trial, process)

                all_results.append({
                    "Subnum": participant,
                    "t": self.t,
                    "a": self.a if recency else None,
                    "weight": self.weight if self.model in ('Param', 'Threshold', 'Threshold_Recency',
                                                            'Entropy_Dis_ID') else None,
                    "tau": self.tau if self.model in ('Threshold', 'Threshold_Recency') else None,
                    "trial_indices": trial_indices,
                    "trial_details": trial_details
                })

            print(f"Post-hoc simulation for participant {participant} finished in {(time.time() - start_time) / 60} "
                  f"minutes")

        return self.unpack_simulation_results(all_results, use_random_sequence)
