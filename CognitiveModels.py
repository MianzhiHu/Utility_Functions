import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, roots_legendre, log_ndtr
from scipy.stats import exponnorm, invgauss
from scipy.integrate import quad


# Parameter bounds for each model type. Each tuple is (lower_bound, upper_bound).
model_bounds = {
    'cct_pt': [(0.01, 20.0), (0.0, 1.0), (0.01, 3.0)],
    'cct_pt_prob': [(0.01, 20.0), (0.0, 1.0), (0.01, 3.0), (0.01, 5.0), (0.01, 5.0)],
    'cct_pt_loss_shape': [(0.01, 20.0), (0.0, 1.0), (0.01, 3.0), (0.0, 1.0)],
    'dd_exponential': [(0.01, 20.0), (0.0, 1.0)],
    'dd_hyperbolic': [(0.01, 20.0), (0.0, 1.0)],
    'dd_hyperboloid': [(0.01, 20.0), (0.0, 1.0), (0.0, 5.0)],
    'ss_logistic': [(-2.0, 5.0), (0.001, 5.0)] + [(0.000001, 0.999999)] * 2,
    'motor_logistic': [(-2.0, 5.0), (0.001, 5.0)] + [(0.000001, 0.999999)] * 3,
    'ss_hr_exgau': [(0.0, 1.5), (0.01, 0.5), (0.01, 1.5)] * 3 + [(0.000001, 0.999999)],
    'motor_hr_exgau': [(0.0, 1.5), (0.01, 0.5), (0.01, 1.5)] * 3 + [(0.000001, 0.999999), (0.5, 2.0)],
    'ss_rdex': [(0.05, 10.0), (0.05, 10.0), (0.1, 5.0), (0.0, 1.0), (0.0, 1.5), (0.01, 0.5), (0.01, 1.5),
                (0.000001, 0.999999), (0.000001, 0.999999)],
    'motor_rdex': [(0.05, 10.0), (0.05, 10.0), (0.1, 5.0), (0.0, 1.0), (0.0, 1.5), (0.01, 0.5), (0.01, 1.5),
                   (0.000001, 0.999999), (0.000001, 0.999999), (0.5, 2.0)],
}


def random_initial_guess(bounds):
    return [np.random.uniform(low, high) for (low, high) in bounds]


def fit_participant(model, participant_id, pdata, model_type, num_iterations=1000):
    print(f"Fitting participant {participant_id}...")
    start_time = time.time()

    total_n = len(pdata['choice'])

    model.iteration = 0
    best_nll = np.inf  # Initialize best negative log likelihood to be positive infinity
    best_parameters = None

    for _ in range(num_iterations):  # Randomly initiate the starting parameter for 1000 times

        model.iteration += 1

        print('Participant {} - Iteration [{}/{}]'.format(participant_id, model.iteration,
                                                          num_iterations))
        bounds = model_bounds[model_type]

        # generate initial guesses from dictionary bounds
        initial_guess = random_initial_guess(bounds)

        result = minimize(model.negative_log_likelihood, initial_guess,
                          args=tuple(pdata[column] for column in model.input_columns),
                          bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000})

        if result.fun < best_nll:
            best_nll = result.fun
            best_parameters = result.x

    k = len(best_parameters)  # Number of parameters
    aic = 2 * k + 2 * best_nll
    bic = k * np.log(total_n) + 2 * best_nll

    result_dict = {
        'participant_id': participant_id,
        'best_nll': best_nll,
        'best_parameters': best_parameters,
        'AIC': aic,
        'BIC': bic
    }

    if isinstance(model, StopSignal):
        result_dict['likelihood'] = 'joint_rt' if model.use_rt else 'choice'

    print(f"Participant {participant_id} fitted in {(time.time() - start_time) / 60} minutes.")

    return result_dict


class DelayedDiscounting:
    def __init__(self, model_type):
        if model_type not in ['dd_exponential', 'dd_hyperbolic', 'dd_hyperboloid']:
            raise ValueError('Use exponential, hyperbolic, or hyperboloid')
        self.model_type = model_type
        self.input_columns = ['large_amount', 'later_delay', 'small_amount', 'choice']

        # Initialize potential parameters
        self._default_attrs = ['t', 'k', 's']
        for attr in self._default_attrs:
            setattr(self, attr, None)

        self._param_map = {
            'dd_exponential': {'t': 0, 'k': 1},
            'dd_hyperbolic': {'t': 0, 'k': 1},
            'dd_hyperboloid': {'t': 0, 'k': 1, 's': 2},
        }

        self._function_map = {
            'dd_exponential': self.exponential_function,
            'dd_hyperbolic': self.hyperbolic_function,
            'dd_hyperboloid': self.hyperboloid_function,
        }

    def softmax(self, x):
        # Apply the original custom softmax separately to each trial's options.
        x = np.asarray(x, dtype=float)
        x_norm = x - np.min(x, axis=-1, keepdims=True)
        e_x = np.exp(np.minimum(self.t * x_norm, 700))
        return np.maximum(e_x / e_x.sum(axis=-1, keepdims=True), 1e-64)

    def exponential_function(self, large_amout, later_delay):
        subjective_value = large_amout * np.exp(-self.k * later_delay)
        return subjective_value

    def hyperbolic_function(self, large_amout, later_delay):
        subjective_value = large_amout / (1 + self.k * later_delay)
        return subjective_value

    def hyperboloid_function(self, large_amout, later_delay):
        subjective_value = large_amout / (1 + self.k * later_delay)**self.s
        return subjective_value

    def negative_log_likelihood(self, params, large_amount, later_delay, small_amount, choice):
        cfg = self._param_map.get(self.model_type, {})
        for attr, idx in cfg.items():
            setattr(self, attr, params[idx])

        # Make sure inputs are numeric numpy arrays
        large_amount = np.asarray(large_amount, dtype=float)
        later_delay = np.asarray(later_delay, dtype=float)
        small_amount = np.asarray(small_amount, dtype=float)
        choice = np.asarray(choice)

        # Compute subjective values for the larger-later option and stack with smaller-sooner values
        # 0 = smaller-sooner; 1 = larger-later
        subjective_values = self._function_map[self.model_type](large_amount, later_delay)
        values = np.column_stack([small_amount, subjective_values])

        # Convert values to choice probabilities; floor the observed probability before taking its log.
        probabilities = self.softmax(values)
        choice = np.asarray(choice, dtype=int)
        observed_probability = probabilities[np.arange(len(choice)), choice]
        trial_nll = -np.log(np.maximum(observed_probability, 1e-64))
        total_nll = trial_nll.sum()

        return float(total_nll)

    def fit(self, data, num_iterations=20, max_workers=None):
        # Detect how many works we have
        workers = max_workers or os.cpu_count()

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, self.model_type,
                                         num_iterations)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        return pd.DataFrame(results)

    def evaluate(self, params, data):
        cfg = self._param_map[self.model_type]

        results = []

        for participant_id, participant_data in data.items():

            participant_params = np.asarray(params.loc[participant_id], dtype=float)

            # Set model parameters
            for attr, idx in cfg.items():
                setattr(self, attr, participant_params[idx])

            # Get THIS participant's test data
            large_amount = np.asarray(participant_data['large_amount'], dtype=float)
            later_delay = np.asarray(participant_data['later_delay'], dtype=float)
            small_amount = np.asarray(participant_data['small_amount'], dtype=float)
            choice = np.asarray(participant_data['choice'],dtype=int)

            # Subjective values
            subjective_values = self._function_map[self.model_type](large_amount, later_delay)
            values = np.column_stack([small_amount, subjective_values])

            # Convert values to choice probabilities; floor the observed probability before taking its log.
            probabilities = self.softmax(values)
            choice = np.asarray(choice, dtype=int)
            observed_probability = probabilities[np.arange(len(choice)), choice]
            trial_nll = -np.log(np.maximum(observed_probability, 1e-64))
            total_nll = trial_nll.sum()
            mean_nll = trial_nll.mean()

            # Accuracy
            predicted_choice = probabilities.argmax(axis=1)
            n_correct = int((predicted_choice == choice).sum())
            accuracy = n_correct / len(choice)

            results.append({
                'participant_id': participant_id,
                'total_nll': float(total_nll),
                'mean_nll': float(mean_nll),
                'accuracy': float(accuracy),
                'n_trials': len(choice),
                'n_correct': n_correct
            })

        return pd.DataFrame(results)


class ColumbiaCardTask:
    """Wuellhorst et al. (2024), Models 1-3; doi:10.1523/JNEUROSCI.1337-23.2024.

    Input valid decisions only, unscaled points and trial-wise probabilities.
    choice: 0 = end_round, 1 = draw_card. loss_amount is negative in our data.
    All models fit loss aversion. Model 3 separates gain/loss curvature and
    does NOT include Model 2's probability weighting. Stop utility is zero.
    As in DD, inverse temperature = 3**t - 1 (paper uses temperature instead).
    """
    def __init__(self, model_type):
        if model_type not in ['cct_pt', 'cct_pt_prob', 'cct_pt_loss_shape']:
            raise ValueError('Use cct_pt, cct_pt_prob, or cct_pt_loss_shape')
        self.model_type = model_type
        self.input_columns = ['gain_amount', 'loss_amount', 'gain_probability', 'loss_probability', 'choice']
        self._default_attrs = ['t', 'rho', 'loss_aversion', 'eta', 'delta', 'rho_loss']
        for attr in self._default_attrs:
            setattr(self, attr, None)
        self._param_map = {
            'cct_pt': {'t': 0, 'rho': 1, 'loss_aversion': 2},
            'cct_pt_prob': {'t': 0, 'rho': 1, 'loss_aversion': 2, 'eta': 3, 'delta': 4},
            'cct_pt_loss_shape': {'t': 0, 'rho': 1, 'loss_aversion': 2, 'rho_loss': 3},
        }
        self._function_map = {
            'cct_pt': self.pt_function,
            'cct_pt_prob': self.pt_prob_function,
            'cct_pt_loss_shape': self.pt_loss_shape_function,
        }
        
    def softmax(self, x):
        # Apply the original custom softmax separately to each trial's options.
        x = np.asarray(x, dtype=float)
        x_norm = x - np.min(x, axis=-1, keepdims=True)
        e_x = np.exp(np.minimum(self.t * x_norm, 700))
        return np.maximum(e_x / e_x.sum(axis=-1, keepdims=True), 1e-64)

    def pt_function(self, gain_amount, loss_amount, gain_probability, loss_probability):
        subjective_utility = (gain_probability * gain_amount**self.rho - loss_probability * self.loss_aversion * np.abs(loss_amount)**self.rho)
        return subjective_utility

    def pt_prob_function(self, gain_amount, loss_amount, gain_probability, loss_probability):
        # Prelec weights are applied separately, without renormalizing their sum.
        # Limits: w(0)=0 and w(1)=1.
        gain_weight = np.exp(-self.delta * (-np.log(gain_probability))**self.eta)
        loss_weight = np.exp(-self.delta * (-np.log(loss_probability))**self.eta)
        return self.pt_function(gain_amount, loss_amount, gain_weight, loss_weight)

    def pt_loss_shape_function(self, gain_amount, loss_amount, gain_probability, loss_probability):
        return (gain_probability * gain_amount**self.rho - loss_probability * self.loss_aversion * np.abs(loss_amount)**self.rho_loss)

    def negative_log_likelihood(self, params, gain_amount, loss_amount, gain_probability, loss_probability, choice):
        for attr, idx in self._param_map[self.model_type].items():
            setattr(self, attr, params[idx])
        gain_amount = np.asarray(gain_amount, dtype=float)
        loss_amount = np.asarray(loss_amount, dtype=float)
        gain_probability = np.asarray(gain_probability, dtype=float)
        loss_probability = np.asarray(loss_probability, dtype=float)
        choice = np.asarray(choice, dtype=int)

        subjective_values = self._function_map[self.model_type](gain_amount, loss_amount, gain_probability, loss_probability)
        values = np.column_stack([np.zeros(len(choice)), subjective_values])
        probabilities = self.softmax(values)
        choice = np.asarray(choice, dtype=int)
        observed_probability = probabilities[np.arange(len(choice)), choice]
        trial_nll = -np.log(np.maximum(observed_probability, 1e-64))
        total_nll = trial_nll.sum()
        return float(total_nll)

    def fit(self, data, num_iterations=20, max_workers=None):
        # Detect how many works we have
        workers = max_workers or os.cpu_count()

        # Creating a list to hold the future results
        futures = []
        results = []

        # Starting a pool of workers with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submitting jobs to the executor for each participant
            for participant_id, participant_data in data.items():
                # fit_participant is the function to be executed in parallel
                future = executor.submit(fit_participant, self, participant_id, participant_data, self.model_type,
                                         num_iterations)
                futures.append(future)

            # Collecting results as they complete
            for future in futures:
                results.append(future.result())

        return pd.DataFrame(results)

    def evaluate(self, params, data):
        cfg = self._param_map[self.model_type]
        results = []

        for participant_id, participant_data in data.items():

            participant_params = np.asarray(params.loc[participant_id], dtype=float )

            # Set model parameters
            for attr, idx in cfg.items():
                setattr(self, attr, participant_params[idx])

            # Get this participant's test data
            gain_amount = np.asarray(participant_data['gain_amount'], dtype=float)
            loss_amount = np.asarray(participant_data['loss_amount'], dtype=float)
            gain_probability = np.asarray(participant_data['gain_probability'], dtype=float)
            loss_probability = np.asarray(participant_data['loss_probability'], dtype=float)
            choice = np.asarray(participant_data['choice'], dtype=int)

            # Subjective value
            subjective_values = self._function_map[self.model_type](gain_amount, loss_amount, gain_probability, loss_probability)

            # Option 0 = reject gamble, value = 0
            # Option 1 = accept gamble, value = subjective gamble value
            values = np.column_stack([np.zeros(len(choice)), subjective_values])

            # Convert values to choice probabilities; floor the observed probability before taking its log.
            probabilities = self.softmax(values)
            choice = np.asarray(choice, dtype=int)
            observed_probability = probabilities[np.arange(len(choice)), choice]
            trial_nll = -np.log(np.maximum(observed_probability, 1e-64))
            total_nll = trial_nll.sum()
            mean_nll = trial_nll.mean()

            # Accuracy
            predicted_choice = probabilities.argmax(axis=1)
            n_correct = int((predicted_choice == choice).sum())
            accuracy = n_correct / len(choice)

            results.append({
                'participant_id': participant_id,
                'total_nll': float(total_nll),
                'mean_nll': float(mean_nll),
                'accuracy': float(accuracy),
                'n_trials': len(choice),
                'n_correct': n_correct
            })

        return pd.DataFrame(results)


class _PositiveExGaussian:
    """Ex-Gaussian conditioned on finishing after zero seconds."""
    def __init__(self, mu, sigma, tau, scale=1.0):
        # Multiplying all three time parameters scales the entire finishing time.
        # SciPy then handles density normalization: pdf_scaled(t) = pdf(t/scale)/scale.
        if scale <= 0:
            raise ValueError('Time scale must be positive')
        mu, sigma, tau = mu * scale, sigma * scale, tau * scale
        self.mu, self.sigma, self.tau = mu, sigma, tau
        self.distribution = exponnorm(tau / sigma, loc=mu, scale=sigma)
        self.positive_probability = self.distribution.sf(0)

    def pdf(self, t):
        # Truncation removes negative times and renormalizes the remaining density.
        t = np.asarray(t)
        density = self.distribution.pdf(t) / self.positive_probability
        return np.where(t > 0, density, 0.)

    def sf(self, t):
        # Before time zero, the runner is certainly unfinished.
        t = np.asarray(t)
        survival = self.distribution.sf(t) / self.positive_probability
        return np.where(t > 0, survival, 1.)

    def logpdf(self, t):
        # Direct log density avoids rounding a tiny PDF to zero first.
        t = np.asarray(t)
        log_density = self.distribution.logpdf(t) - self.distribution.logsf(0)
        return np.where(t > 0, log_density, -np.inf)

    def logsf(self, t):
        # Ex-Gaussian survival has two terms: add their logs without underflow.
        t = np.maximum(np.asarray(t), 0)
        z = (t - self.mu) / self.sigma
        ratio = self.sigma / self.tau
        log_survival = np.logaddexp(log_ndtr(-z),
                                   ratio * (ratio / 2 - z) + log_ndtr(z - ratio))
        # Normalize for truncation; survival is one at/before zero.
        return np.minimum(log_survival - self.distribution.logsf(0), 0.)


class StopSignal:
    """One independent horse-race model with three zero-truncated ex-Gaussian runners.

    Parameters: mu_correct, sigma_correct, tau_correct, mu_error, sigma_error,
    tau_error, mu_stop, sigma_stop, tau_stop, p_trigger_failure; then critical_scale
    for motor only. All times are seconds. Nondecision time is fixed at zero.
    Trigger failure means the stop runner is not initiated on that trial.
    Motor critical_scale multiplies both go-runner times on critical trials.
    block_duration is treated as the trial's response deadline. Keep nonresponses.
    use_rt=False fits correctness; use_rt=True fits joint RT/outcome likelihood.
    RT-conditional correctness is reported only for responded go trials.
    Integration uses vectorized Gauss-Legendre quadrature.
    Logistic variants use shared conditional go accuracy; motor has separate critical/noncritical go-failure rates;
    they ignore RT, response status and deadline, and require use_rt=False.
    """
    def __init__(self, model_type, use_rt=False):
        if model_type not in ['ss_hr_exgau', 'motor_hr_exgau', 'ss_logistic', 'motor_logistic', 'ss_rdex', 'motor_rdex']:
            raise ValueError('Unknown StopSignal model type')
        if model_type.endswith('logistic') and use_rt:
            raise ValueError('Logistic inhibition models are choice-only; use_rt must be False')
        if model_type.endswith('rdex') and use_rt:
            raise ValueError('RDEX here fits correctness only; use_rt must be False')
        self.model_type = model_type
        self.use_rt = use_rt
        self.quadrature_nodes = 512
        self._quadrature_cache = None
        self.input_columns = ['SS_delay', 'trial_condition', 'response_time', 'responded', 'choice', 'block_duration']
        self._default_attrs = ['mu_correct', 'sigma_correct', 'tau_correct', 'mu_error',
                               'sigma_error', 'tau_error', 'mu_stop', 'sigma_stop', 'tau_stop',
                               'p_trigger_failure', 'critical_scale', 'theta', 'scale',
                               'p_go_accuracy', 'p_go_failure', 'p_go_failure_crit', 'p_go_failure_noncrit',
                               'v_correct', 'v_error', 'boundary', 'nondecision', 'critical_boundary_scale']
        for attr in self._default_attrs:
            setattr(self, attr, None)

        race = {'mu_correct': 0, 'sigma_correct': 1, 'tau_correct': 2,
                'mu_error': 3, 'sigma_error': 4, 'tau_error': 5,
                'mu_stop': 6, 'sigma_stop': 7, 'tau_stop': 8, 'p_trigger_failure': 9}
        rdex = {'v_correct': 0, 'v_error': 1, 'boundary': 2, 'nondecision': 3,
                'mu_stop': 4, 'sigma_stop': 5, 'tau_stop': 6,
                'p_go_failure': 7, 'p_trigger_failure': 8}

        self._param_map = {
            'ss_logistic': {'theta': 0, 'scale': 1, 'p_go_accuracy': 2, 'p_go_failure': 3},
            'motor_logistic': {'theta': 0, 'scale': 1, 'p_go_accuracy': 2, 'p_go_failure_crit': 3, 'p_go_failure_noncrit': 4},
            'ss_hr_exgau': race,
            'motor_hr_exgau': dict(race, critical_scale=10),
            'ss_rdex': rdex,
            'motor_rdex': dict(rdex, critical_boundary_scale=9),
        }

        self._function_map = {
            'ss_logistic': self.logistic_function,
            'motor_logistic': self.logistic_function,
            'ss_hr_exgau': self.horse_race_runner3,
            'motor_hr_exgau': self.motor_horse_race_runner3,
            'ss_rdex': self.rdex_function,
            'motor_rdex': self.rdex_function,
        }

    def horse_race_exgau_function(self, scale=1.0):
        # Scale go times only; stopping has the same distribution in both conditions.
        return (_PositiveExGaussian(self.mu_correct, self.sigma_correct, self.tau_correct, scale=scale),
                _PositiveExGaussian(self.mu_error, self.sigma_error, self.tau_error, scale=scale),
                _PositiveExGaussian(self.mu_stop, self.sigma_stop, self.tau_stop))

    def logistic_function(self, params, SS_delay, trial_condition, response_time, responded, choice, block_duration):
        """P(task correct), not P(response). RT, responded and deadline are unused."""
        motor = self.model_type.startswith('motor')
        go_conditions = ['crit_go', 'noncrit_nosignal', 'noncrit_signal'] if motor else ['go']
        stop_condition = 'crit_stop' if motor else 'stop'
        params = np.asarray(params, dtype=float)
        for attr, idx in self._param_map[self.model_type].items():
            setattr(self, attr, params[idx])

        delay = np.asarray(SS_delay, dtype=float)
        condition = np.asarray(trial_condition)
        choice = np.asarray(choice)
        stop = condition == stop_condition

        # Sanity checks for parameter shapes, values, and trial arrays.
        if self.scale <= 0 or np.any((params[2:] <= 0) | (params[2:] >= 1)):
            raise ValueError('Scale must be positive and accuracy/failure probabilities strictly between zero and one')
        if choice.ndim != 1 or not len(choice) or delay.shape != choice.shape or condition.shape != choice.shape:
            raise ValueError('Provide matching nonempty trial arrays')
        if not np.isin(choice, [0, 1]).all() or not np.isin(condition, go_conditions + [stop_condition]).all():
            raise ValueError('Require binary correctness and recognized task conditions')
        if not np.all(np.isfinite(delay[stop]) & (delay[stop] >= 0)):
            raise ValueError('Stop trials require finite nonnegative SSD in seconds')

        p_correct = np.empty(len(choice))
        for name in go_conditions:
            selected = condition == name
            if not motor:
                p_go_failure = self.p_go_failure
            elif name == 'crit_go':
                p_go_failure = self.p_go_failure_crit
            else:
                p_go_failure = self.p_go_failure_noncrit
            # A correct go response needs both go initiation and response accuracy.
            p_correct[selected] = (1 - p_go_failure) * self.p_go_accuracy

        # Go probabilities are bounded away from 0 and 1 by the parameter bounds.
        log_likelihood = np.empty(len(choice))
        log_likelihood[~stop] = np.where(choice[~stop] == 1, np.log(p_correct[~stop]), np.log1p(-p_correct[~stop]))

        # Logistic curve
        z = (self.theta - delay[stop]) / self.scale

        # Calculate log probabilities for correct and incorrect stop trials based on the logistic curve.
        # Note that log_q = log(sigmoid(z)) and log_not_q = log(1 - sigmoid(z)) written in a numerically stable way.
        log_q = -np.logaddexp(0, -z)
        log_not_q = -np.logaddexp(0, z)
        p_go_failure = self.p_go_failure_crit if motor else self.p_go_failure

        # The log probability of a correct stop trial is the log of the sum of two mutually exclusive events:
        # 1. The go runner fails to initiate (p_go_failure)
        # 2. The go runner initiates (1 - p_go_failure) and the stop runner successfully inhibits the response (q)
        # Note that log_success = log(p_go_failure + (1 - p_go_failure) * q) written in a numerically stable way.
        log_success = np.logaddexp(np.log(p_go_failure), np.log(1 - p_go_failure) + log_q)

        # Similarly, log_failure = log((1 - p_go_failure) * (1 - q)) = log(1 - p_go_failure) + log(1 - q)
        log_failure = np.log(1 - p_go_failure) + log_not_q

        # Revert the log probabilities back to the original scale for the stop trials.
        p_correct[stop] = np.exp(log_success)
        log_likelihood[stop] = np.where(choice[stop] == 1, log_success, log_failure)
        # Same observed-probability floor as DD, CCT, and the race models, applied in log space.
        log_likelihood = np.maximum(log_likelihood, np.log(1e-64))
        return log_likelihood, p_correct

    def horse_race_runner3(self, params, SS_delay, trial_condition, response_time, responded, choice, block_duration):
        for attr, idx in self._param_map[self.model_type].items():
            setattr(self, attr, params[idx])
        delay = np.asarray(SS_delay, dtype=float)
        condition = np.asarray(trial_condition)
        rt = np.asarray(response_time, dtype=float)
        responded = np.asarray(responded)
        choice = np.asarray(choice)  # 1 = task correct, including successful stopping.
        deadlines = np.asarray(block_duration, dtype=float)
        stop = condition == 'stop'
        valid_conditions = ['go', 'stop']

        # Sanity checks for parameter shapes, values, and trial arrays.
        if len(params) != len(self._param_map[self.model_type]) or not np.isfinite(params).all() or np.any(np.asarray(params)[[1, 2, 4, 5, 7, 8]] <= 0):
            raise ValueError('Supply the expected finite parameters with positive sigma and tau for each runner. Current parameters: {}'.format(params))
        if not np.isin(condition, valid_conditions).all():
            raise ValueError('Unrecognized trial condition')
        if not np.isin(choice, [0, 1]).all():
            raise ValueError('Correctness must be binary')
        if self.use_rt:
            if not np.isin(responded, [0, 1]).all():
                raise ValueError('responded must be binary')
            response = responded == 1
            if np.any(response & (~np.isfinite(rt) | (rt <= 0) | (rt > deadlines))):
                raise ValueError('Responded trials require 0 < RT <= block_duration in seconds')

        # Cache exact predictor groups and integration nodes
        predictors = np.column_stack([stop, np.where(stop, delay, 0), deadlines])
        cache_key = (self.quadrature_nodes, predictors.shape, predictors.tobytes())
        if self._quadrature_cache is None or self._quadrature_cache[0] != cache_key:
            if not np.isfinite(predictors).all() or np.any(deadlines <= 0) or np.any(delay[stop] < 0):
                raise ValueError('Require positive finite deadlines and nonnegative finite stop SSDs')
            groups, inverse = np.unique(predictors, axis=0, return_inverse=True)
            nodes, weights = roots_legendre(self.quadrature_nodes)
            length = np.where(groups[:, 0] == 1, np.maximum(groups[:, 2] - groups[:, 1], 0), groups[:, 2])
            points = length[:, None] * (nodes[None, :] + 1) / 2
            weights = length[:, None] * weights[None, :] / 2
            self._quadrature_cache = (cache_key, groups, inverse, points, weights)
        _, groups, inverse, points, weights = self._quadrature_cache

        # Generate the ex-Gaussian runners
        correct_go, error_go, stop_runner = self.horse_race_exgau_function()

        stopping = groups[:, 0] == 1
        p_correct_go = np.full(len(groups), np.nan)
        p_go_omission = np.full(len(groups), np.nan)
        p_correct_stop = np.full(len(groups), np.nan)

        # Integrate over correct-runner finishing times before the deadline weighted by the probability that the error runner is still unfinished.
        go_groups = ~stopping
        t = points[go_groups]
        p_correct_go[go_groups] = np.sum(weights[go_groups] * correct_go.pdf(t) * error_go.sf(t), axis=1)

        # Regular-task stop trials.
        stop_groups = stopping
        u = points[stop_groups]
        go_time = u + groups[stop_groups, 1, None] # SSD + stop runner time
        inhibition = np.sum(weights[stop_groups] * stop_runner.pdf(u) * correct_go.sf(go_time) * error_go.sf(go_time), axis=1)
        deadline = groups[stop_groups, 2]
        p_both_go_unfinished = correct_go.sf(deadline) * error_go.sf(deadline)
        finish = np.maximum(deadline - groups[stop_groups, 1], 0)
        p_no_response_given_stop_triggered = inhibition + p_both_go_unfinished * stop_runner.sf(finish)
        p_correct_stop[stop_groups] = (1 - self.p_trigger_failure) * p_no_response_given_stop_triggered + self.p_trigger_failure * p_both_go_unfinished

        if not self.use_rt:
            # Go correctness and stop correctness are different events.
            p_correct_by_group = np.where(stopping, p_correct_stop, p_correct_go)
            p_correct = p_correct_by_group[inverse]
            observed_probability = np.where(choice == 1, p_correct, 1 - p_correct)
            log_likelihood = np.log(np.maximum(observed_probability, 1e-64))
            return log_likelihood, p_correct

        # On stop trials, no response is a correct stop; on go trials, it is an omission.
        # Both go runners remain unfinished at the deadline; responded trials are scored later.
        deadline = groups[go_groups, 2]
        p_go_omission[go_groups] = correct_go.sf(deadline) * error_go.sf(deadline)
        p_no_response_by_group = np.where(stopping, p_correct_stop, p_go_omission)
        likelihood = p_no_response_by_group[inverse].copy()

        # Replace these probability masses with RT densities for responded trials.
        t = rt[response]
        correct_density = correct_go.pdf(t) * error_go.sf(t)
        error_density = error_go.pdf(t) * correct_go.sf(t)
        response_density = correct_density + error_density
        observed_density = np.where(choice[response] == 1, correct_density, error_density)

        # On failed stop trials either key may win, and stopping must not prevent it.
        response_stop = stop[response]
        stop_elapsed = rt[response][response_stop] - delay[response][response_stop] # Elapsed time after the stop signal
        stop_survival = stop_runner.sf(stop_elapsed)
        p_not_stopped = self.p_trigger_failure + (1 - self.p_trigger_failure) * stop_survival
        observed_density[response_stop] = response_density[response_stop] * p_not_stopped
        likelihood[response] = observed_density
        log_likelihood = np.log(np.maximum(likelihood, 1e-64))

        # Among go responses at this RT, what fraction of the density is correct?
        # Compute densities directly in log space; do not floor their ratio.
        log_correct_density = correct_go.logpdf(t) + error_go.logsf(t)
        log_error_density = error_go.logpdf(t) + correct_go.logsf(t)
        # P(correct | RT, go response): division becomes subtraction of logs.
        log_response_density = np.logaddexp(log_correct_density, log_error_density)
        log_correct_probability = log_correct_density[~response_stop] - log_response_density[~response_stop]
        log_error_probability = log_error_density[~response_stop] - log_response_density[~response_stop]
        conditional_accuracy = np.exp(log_correct_probability)
        p_correct = np.full(len(choice), np.nan)
        p_correct[response & ~stop] = conditional_accuracy
        # Select the observed outcome in log space, retaining the existing NLL cap.
        conditional_log_probability = np.where(choice[response][~response_stop] == 1, log_correct_probability, log_error_probability)
        self._conditional_go_loglik = np.maximum(conditional_log_probability, np.log(1e-64))
        return log_likelihood, p_correct

    def motor_horse_race_runner3(self, params, SS_delay, trial_condition, response_time, responded, choice, block_duration):
        for attr, idx in self._param_map[self.model_type].items():
            setattr(self, attr, params[idx])
        delay = np.asarray(SS_delay, dtype=float)
        condition = np.asarray(trial_condition)
        rt = np.asarray(response_time, dtype=float)
        responded = np.asarray(responded)
        choice = np.asarray(choice)  # 1 = task correct, including successful stopping.
        deadlines = np.asarray(block_duration, dtype=float)
        stop = condition == 'crit_stop'
        valid_conditions = ['crit_go', 'crit_stop', 'noncrit_signal', 'noncrit_nosignal']

        # Sanity checks for parameter shapes, values, and trial arrays.
        if len(params) != len(self._param_map[self.model_type]) or not np.isfinite(params).all() or np.any(np.asarray(params)[[1, 2, 4, 5, 7, 8]] <= 0):
            raise ValueError('Supply the expected finite parameters with positive sigma and tau for each runner. Current parameters: {}'.format(params))
        if not np.isin(condition, valid_conditions).all():
            raise ValueError('Unrecognized trial condition')
        if not np.isin(choice, [0, 1]).all():
            raise ValueError('Correctness must be binary')
        if self.use_rt:
            if not np.isin(responded, [0, 1]).all():
                raise ValueError('responded must be binary')
            response = responded == 1
            if np.any(response & (~np.isfinite(rt) | (rt <= 0) | (rt > deadlines))):
                raise ValueError('Responded trials require 0 < RT <= block_duration in seconds')

        # Cache exact predictor groups and integration nodes
        critical = np.isin(condition, ['crit_go', 'crit_stop'])
        predictors = np.column_stack([stop, np.where(stop, delay, 0), deadlines, critical])
        cache_key = (self.quadrature_nodes, predictors.shape, predictors.tobytes())
        if self._quadrature_cache is None or self._quadrature_cache[0] != cache_key:
            if not np.isfinite(predictors).all() or np.any(deadlines <= 0) or np.any(delay[stop] < 0):
                raise ValueError('Require positive finite deadlines and nonnegative finite stop SSDs')
            groups, inverse = np.unique(predictors, axis=0, return_inverse=True)
            nodes, weights = roots_legendre(self.quadrature_nodes)
            length = np.where(groups[:, 0] == 1, np.maximum(groups[:, 2] - groups[:, 1], 0), groups[:, 2])
            points = length[:, None] * (nodes[None, :] + 1) / 2
            weights = length[:, None] * weights[None, :] / 2
            self._quadrature_cache = (cache_key, groups, inverse, points, weights)
        _, groups, inverse, points, weights = self._quadrature_cache

        # Generate the ex-Gaussian runners
        correct_go, error_go, stop_runner = self.horse_race_exgau_function()
        correct_go_critical, error_go_critical, stop_runner_critical = self.horse_race_exgau_function(scale=self.critical_scale)

        stopping = groups[:, 0] == 1
        critical_groups = groups[:, 3] == 1
        p_correct_go = np.full(len(groups), np.nan)
        p_go_omission = np.full(len(groups), np.nan)
        p_correct_stop = np.full(len(groups), np.nan)

        # Noncritical motor go trials.
        noncrit_go_groups = ~stopping & ~critical_groups
        if self.use_rt:
            # Both go runners remain unfinished at the deadline; responded trials are scored later.
            deadline = groups[noncrit_go_groups, 2]
            p_go_omission[noncrit_go_groups] = correct_go.sf(deadline) * error_go.sf(deadline)
        else:
            # Integrate over correct-runner finishing times before the deadline weighted by the probability that the error runner is still unfinished.
            t = points[noncrit_go_groups]
            p_correct_go[noncrit_go_groups] = np.sum(weights[noncrit_go_groups] * correct_go.pdf(t) * error_go.sf(t), axis=1)

        # Critical motor go trials: directly use the scaled distributions.
        critical_go_groups = ~stopping & critical_groups
        if self.use_rt:
            deadline = groups[critical_go_groups, 2]
            p_go_omission[critical_go_groups] = correct_go_critical.sf(deadline) * error_go_critical.sf(deadline)
        else:
            t = points[critical_go_groups]
            p_correct_go[critical_go_groups] = np.sum(weights[critical_go_groups] * correct_go_critical.pdf(t) * error_go_critical.sf(t), axis=1)

        # Critical motor stop trials; stop timing itself is unchanged.
        stop_groups = stopping  # All motor stop trials are critical.
        u = points[stop_groups]
        go_time = u + groups[stop_groups, 1, None]
        inhibition = np.sum(weights[stop_groups] * stop_runner_critical.pdf(u) * correct_go_critical.sf(go_time) * error_go_critical.sf(go_time), axis=1)
        deadline = groups[stop_groups, 2]
        p_both_go_unfinished = correct_go_critical.sf(deadline) * error_go_critical.sf(deadline)
        finish = np.maximum(deadline - groups[stop_groups, 1], 0)
        p_no_response_given_stop_triggered = inhibition + p_both_go_unfinished * stop_runner_critical.sf(finish)
        p_correct_stop[stop_groups] = ((1 - self.p_trigger_failure) * p_no_response_given_stop_triggered + self.p_trigger_failure * p_both_go_unfinished)

        if not self.use_rt:
            # Go correctness and stop correctness are different events.
            p_correct_by_group = np.where(stopping, p_correct_stop, p_correct_go)
            p_correct = p_correct_by_group[inverse]
            observed_probability = np.where(choice == 1, p_correct, 1 - p_correct)
            log_likelihood = np.log(np.maximum(observed_probability, 1e-64))
            return log_likelihood, p_correct

        # On stop trials, no response is a correct stop; on go trials, it is an omission.
        p_no_response_by_group = np.where(stopping, p_correct_stop, p_go_omission)
        likelihood = p_no_response_by_group[inverse].copy()
        # Replace these probability masses with RT densities for responded trials.
        t = rt[response]
        response_critical = critical[response]
        correct_density = np.empty(len(t))
        error_density = np.empty(len(t))
        # Regular/noncritical responses.
        selected = ~response_critical
        correct_density[selected] = correct_go.pdf(t[selected]) * error_go.sf(t[selected])
        error_density[selected] = error_go.pdf(t[selected]) * correct_go.sf(t[selected])
        # Critical responses.
        selected = response_critical
        correct_density[selected] = correct_go_critical.pdf(t[selected]) * error_go_critical.sf(t[selected])
        error_density[selected] = error_go_critical.pdf(t[selected]) * correct_go_critical.sf(t[selected])
        response_density = correct_density + error_density
        observed_density = np.where(choice[response] == 1, correct_density, error_density)

        # On failed stop trials either key may win, and stopping must not prevent it.
        response_stop = stop[response]
        stop_elapsed = rt[response][response_stop] - delay[response][response_stop]
        stop_survival = stop_runner_critical.sf(stop_elapsed)
        p_not_stopped = self.p_trigger_failure + (1 - self.p_trigger_failure) * stop_survival
        observed_density[response_stop] = response_density[response_stop] * p_not_stopped
        likelihood[response] = observed_density
        log_likelihood = np.log(np.maximum(likelihood, 1e-64))

        # Among go responses at this RT, what fraction of the density is correct?
        # Use the same critical/noncritical runners as in the RT likelihood.
        log_correct_density = np.empty(len(t))
        log_error_density = np.empty(len(t))
        selected = ~response_critical
        log_correct_density[selected] = correct_go.logpdf(t[selected]) + error_go.logsf(t[selected])
        log_error_density[selected] = error_go.logpdf(t[selected]) + correct_go.logsf(t[selected])
        selected = response_critical
        log_correct_density[selected] = correct_go_critical.logpdf(t[selected]) + error_go_critical.logsf(t[selected])
        log_error_density[selected] = error_go_critical.logpdf(t[selected]) + correct_go_critical.logsf(t[selected])
        # P(correct | RT, go response): division becomes subtraction of logs.
        log_response_density = np.logaddexp(log_correct_density, log_error_density)
        log_correct_probability = log_correct_density[~response_stop] - log_response_density[~response_stop]
        log_error_probability = log_error_density[~response_stop] - log_response_density[~response_stop]
        conditional_accuracy = np.exp(log_correct_probability)
        p_correct = np.full(len(choice), np.nan)
        p_correct[response & ~stop] = conditional_accuracy
        # Select the observed outcome in log space, retaining the existing NLL cap.
        conditional_log_probability = np.where(choice[response][~response_stop] == 1, log_correct_probability, log_error_probability)
        self._conditional_go_loglik = np.maximum(conditional_log_probability, np.log(1e-64))
        return log_likelihood, p_correct

    def rdex_function(self, params, SS_delay, trial_condition, response_time, responded, choice, block_duration):
        """Correctness-only adaptation of Tanis et al. (2024), doi:10.3758/s13428-023-02295-y.

        Independent shifted Wald go runners (diffusion SD=1, start variability A=0),
        shared boundary/nondecision time, ex-Gaussian stop truncated at zero.
        Zero truncation matches our other race models (paper used 0.05 s).
        PGF means neither go runner launches; PTF means the stop runner fails to launch.
        Motor critical-boundary scaling is our task extension, not the paper's design.
        Uses the existing individual MLE fitter, not the paper's hierarchical RT fit.
        """
        params = np.asarray(params, dtype=float)
        for attr, idx in self._param_map[self.model_type].items():
            setattr(self, attr, params[idx])
        motor = self.model_type.startswith('motor')

        delay = np.asarray(SS_delay, float)
        condition = np.asarray(trial_condition)
        y = np.asarray(choice)
        deadline = np.asarray(block_duration, float)
        conditions = ['crit_go', 'crit_stop', 'noncrit_signal', 'noncrit_nosignal'] if motor else ['go', 'stop']
        stop = condition == ('crit_stop' if motor else 'stop')

        # Sanity checks for parameter shapes, values, and trial arrays.
        if params.shape != (len(self._param_map[self.model_type]),) or not np.isfinite(params).all():
            raise ValueError('Invalid RDEX parameter vector')
        if (y.ndim != 1 or not len(y) or any(x.shape != y.shape for x in [delay, condition, deadline])
                or not np.isin(y, [0, 1]).all() or not np.isin(condition, conditions).all()
                or not np.isfinite(deadline).all() or np.any(deadline <= 0)):
            raise ValueError('Invalid RDEX trial arrays or response deadlines')
        if not np.all(np.isfinite(delay[stop]) & (delay[stop] >= 0)):
            raise ValueError('Stop SSD must be finite and nonnegative')

        # Extract unique trial groups for vectorized integration; cache quadrature nodes.
        critical = np.isin(condition, ['crit_go', 'crit_stop']) if motor else np.zeros(len(y), bool)
        groups, inverse = np.unique(np.column_stack([stop, np.where(stop, delay, 0), deadline, critical]), axis=0, return_inverse=True)
        # Cache only parameter-independent quadrature nodes.
        if not hasattr(self, '_rdex_nodes') or len(self._rdex_nodes[0]) != self.quadrature_nodes:
            self._rdex_nodes = roots_legendre(self.quadrature_nodes)
        nodes, weights = self._rdex_nodes
        probability = np.empty(len(groups))
        stopping = groups[:, 0].astype(bool)
        b = self.boundary * (np.where(groups[:, 3] == 1, self.critical_boundary_scale, 1.0) if motor else np.ones(len(groups)))
        # Wald mean=b/v, shape=b**2 -> scipy mu=1/(b*v), scale=b**2.
        correct = invgauss(1 / (b[:, None] * self.v_correct), loc=self.nondecision, scale=b[:, None]**2)
        error = invgauss(1 / (b[:, None] * self.v_error), loc=self.nondecision, scale=b[:, None]**2)
        group_deadline = groups[:, 2, None]
        omission = (correct.sf(group_deadline) * error.sf(group_deadline))[:, 0]

        # Integrate only above the nondecision shift; below it no response is possible.
        length = np.maximum(groups[:, 2] - self.nondecision, 0)
        t = self.nondecision + length[:, None] * (nodes + 1) / 2
        go_correct = np.sum(length[:, None] / 2 * weights * correct.pdf(t) * error.sf(t), axis=1)
        probability[~stopping] = (1 - self.p_go_failure) * go_correct[~stopping]
        # The helper already truncates and renormalizes the stop runner at zero.
        runner = _PositiveExGaussian(self.mu_stop, self.sigma_stop, self.tau_stop)
        finish = np.maximum(groups[:, 2] - groups[:, 1], 0)

        # Split at t0-SSD: go survival is exactly 1 before that point.
        cut = np.maximum(0, np.minimum(finish, self.nondecision - groups[:, 1]))
        inhibition = np.zeros(len(groups))
        for left, right in [(np.zeros(len(groups)), cut), (cut, finish)]:
            width = np.maximum(right - left, 0)
            u = left[:, None] + width[:, None] * (nodes + 1) / 2
            density = runner.pdf(u)
            inhibition += np.sum(width[:, None] / 2 * weights * density * correct.sf(u + groups[:, 1, None]) * error.sf(u + groups[:, 1, None]), axis=1)
        stop_survival = runner.sf(finish)
        no_response = inhibition + omission * stop_survival
        # Conditional on go initiation: stop is triggered, or fails to trigger.
        p_no_response_given_go = ((1 - self.p_trigger_failure) * no_response[stopping] + self.p_trigger_failure * omission[stopping])
        probability[stopping] = (self.p_go_failure + (1 - self.p_go_failure) * p_no_response_given_go)
        if not np.isfinite(probability).all() or np.any((probability < -1e-7) | (probability > 1+1e-7)):
            raise FloatingPointError('RDEX integration failed; check parameters/quadrature')
        p_correct = probability[inverse]
        observed_probability = np.where(y == 1, p_correct, 1 - p_correct)
        return np.log(np.maximum(observed_probability, 1e-64)), p_correct

    def negative_log_likelihood(self, params, SS_delay, trial_condition, response_time, responded, choice, block_duration):
        log_likelihood, _ = self._function_map[self.model_type](params, SS_delay, trial_condition, response_time, responded, choice, block_duration)
        return float(-log_likelihood.sum())

    def fit(self, data, num_iterations=20, max_workers=None):
        workers = max_workers or os.cpu_count()
        futures = []
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for participant_id, participant_data in data.items():
                future = executor.submit(fit_participant, self, participant_id, participant_data,
                                         self.model_type, num_iterations)
                futures.append(future)
            for future in futures:
                results.append(future.result())
        return pd.DataFrame(results)

    def evaluate(self, params, data):
        results = []

        for participant_id, participant_data in data.items():
            participant_params = np.asarray(params.loc[participant_id], dtype=float)

            # Get this participant's test data. The model function sets their parameters.
            choice = np.asarray(participant_data['choice'], dtype=int)
            log_likelihood, p_correct = self._function_map[self.model_type](
                participant_params, participant_data['SS_delay'], participant_data['trial_condition'],
                participant_data['response_time'], participant_data['responded'],
                choice, participant_data['block_duration'])
            total_nll = float(-log_likelihood.sum())
            mean_nll = float(-log_likelihood.mean())

            if self.use_rt:
                # Joint RT NLL uses all trials; conditional accuracy uses responded go trials only.
                go_response = np.isfinite(p_correct)
                n_go = int(go_response.sum())
                n_correct = int(((p_correct[go_response] > 0.5).astype(int) == choice[go_response]).sum())
                conditional_nll = float(-self._conditional_go_loglik.sum())
                results.append({
                    'participant_id': participant_id,
                    'total_nll': total_nll,
                    'mean_nll': mean_nll,
                    'n_trials': len(choice),
                    'rt_conditional_go_total_nll': conditional_nll if n_go else np.nan,
                    'rt_conditional_go_nll': conditional_nll / n_go if n_go else np.nan,
                    'rt_conditional_go_accuracy': n_correct / n_go if n_go else np.nan,
                    'rt_conditional_go_n_trials': n_go,
                    'rt_conditional_go_n_correct': n_correct
                })
            else:
                # Same participant-level correctness metrics as DD and CCT.
                n_correct = int(((p_correct > 0.5).astype(int) == choice).sum())
                results.append({
                    'participant_id': participant_id,
                    'total_nll': total_nll,
                    'mean_nll': mean_nll,
                    'accuracy': n_correct / len(choice),
                    'n_trials': len(choice),
                    'n_correct': n_correct
                })

        return pd.DataFrame(results)


# Helper function to convert a dataframe into a dictionary of participant data
def dict_generator_cognitive(df, task='dd'):
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
        'dd': {
            'large_amount':   ['large_amount'],
            'small_amount':   ['small_amount'],
            'later_delay':    ['later_delay'],
            'choice':   ['choice'],
        },
        'cct': {
            'gain_amount': ['gain_amount'],
            'loss_amount': ['loss_amount'],
            'gain_probability': ['gain_probability'],
            'loss_probability': ['loss_probability'],
            'choice': ['action'],
        },
        'stop_signal': {
            'SS_delay': ['SS_delay'],
            'trial_condition': ['trial_condition'],
            'response_time': ['response_time'],
            'responded': ['responded'],
            'block_duration': ['block_duration'],
            'choice': ['correct'],
        },
        'motor': {
            'SS_delay': ['SS_delay'],
            'trial_condition': ['trial_condition'],
            'response_time': ['response_time'],
            'responded': ['responded'],
            'block_duration': ['block_duration'],
            'choice': ['correct'],
        },
    }

    if task not in COLUMN_MAP:
        raise ValueError(f"Unsupported task {task!r}")

    # optional: allow different grouping columns too
    group_col = find_col(['worker_id'])

    d = {}
    for subject_id, group in df.groupby(group_col):
        entry = {}
        for key, candidates in COLUMN_MAP[task].items():
            actual_col = find_col(candidates)
            entry[key] = group[actual_col].tolist()
        d[subject_id] = entry

    return d
