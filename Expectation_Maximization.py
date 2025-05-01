import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import seaborn as sns


def log_likelihood(dat, mu1, mu2, sd1, sd2, ppi1, modality='bimodal',
                   mu3=None, sd3=None, ppi2=None, ppi3=None):
    if modality == 'trimodal':
        ll = np.sum(np.log((ppi1) * norm.pdf(dat, mu1, sd1) + ppi2 * norm.pdf(dat, mu2, sd2) +
                           ppi3 * norm.pdf(dat, mu3, sd3)))

        return ll

    else:
        ll = np.sum(np.log((1 - ppi1) * norm.pdf(dat, mu1, sd1) + ppi1 * norm.pdf(dat, mu2, sd2)))

        return ll


def e_step(dat, mu1, mu2, sd1, sd2, ppi1, modality='bimodal', mu3=None, sd3=None, ppi2=None, ppi3=None):
    if modality == 'trimodal':
        # get likelihood of each observation under each component
        p1 = ppi1 * norm.pdf(dat, mu1, sd1)
        p2 = ppi2 * norm.pdf(dat, mu2, sd2)
        p3 = ppi3 * norm.pdf(dat, mu3, sd3)

        # sum_p = np.clip(p1 + p2 + p3, 1e-32, None)
        sum_p = p1 + p2 + p3

        # calculate the responsibility using the current parameter estimates
        # resp1 = np.clip(p1 / sum_p, 1e-32, None)
        # resp2 = np.clip(p2 / sum_p, 1e-32, None)
        # resp3 = np.clip(p3 / sum_p, 1e-32, None)

        resp1 = p1 / sum_p
        resp2 = p2 / sum_p
        resp3 = p3 / sum_p

        return resp1, resp2, resp3

    else:
        # Calculate the responsibility using the current parameter estimates
        resp = ppi1 * norm.pdf(dat, mu2, sd2) / ((1 - ppi1) * norm.pdf(dat, mu1, sd1) + ppi1 * norm.pdf(dat, mu2, sd2))
        return resp


def m_step(dat, resp1, modality='bimodal', resp2=None, resp3=None):
    if modality == 'trimodal':
        ppi1 = np.mean(resp1)
        ppi2 = np.mean(resp2)
        ppi3 = np.mean(resp3)

        mu1 = np.sum(resp1 * dat) / np.sum(resp1)
        mu2 = np.sum(resp2 * dat) / np.sum(resp2)
        mu3 = np.sum(resp3 * dat) / np.sum(resp3)

        sd1 = np.sqrt(np.sum(resp1 * (dat - mu1) ** 2) / np.sum(resp1))
        sd2 = np.sqrt(np.sum(resp2 * (dat - mu2) ** 2) / np.sum(resp2))
        sd3 = np.sqrt(np.sum(resp3 * (dat - mu3) ** 2) / np.sum(resp3))

        # Ensure sd values are positive
        sd1 = max(sd1, 1e-32)
        sd2 = max(sd2, 1e-32)
        sd3 = max(sd3, 1e-32)

        return mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3

    else:
        # Update the parameters using the current responsibilities
        mu1 = np.sum((1 - resp1) * dat) / np.sum(1 - resp1)
        mu2 = np.sum(resp1 * dat) / np.sum(resp1)
        sd1 = np.sqrt(np.sum((1 - resp1) * (dat - mu1) ** 2) / np.sum(1 - resp1))
        sd2 = np.sqrt(np.sum(resp1 * (dat - mu2) ** 2) / np.sum(resp1))
        ppi = np.mean(resp1)
        return mu1, mu2, sd1, sd2, ppi


def initialize_parameters(data, size):
    # Bounded mu initialization
    mu = np.random.uniform(data.min(), data.max(), size)

    # Bounded sd initialization
    sd = np.random.uniform(0.01, (data.max() - data.min()) / 2.0, size)

    return mu, sd


def em_model(data, tolerance=0.0001, random_init=True, return_starting_params=False, modality='bimodal',
             mu1=None, mu2=None, mu3=None, sd1=None, sd2=None, sd3=None, ppi1=None, ppi2=None):
    # set global variables
    change = np.inf

    if modality == 'trimodal':
        if random_init:
            # randomly generate starting mu and sd
            mu1, sd1 = initialize_parameters(data, 1)
            mu2, sd2 = initialize_parameters(data, 1)
            mu3, sd3 = initialize_parameters(data, 1)

            # randomly generate a starting ppi
            ppi1 = np.random.uniform(0.01, 1)
            ppi2 = np.random.uniform(0.01, 1 - ppi1)
            ppi3 = 1 - ppi1 - ppi2

            # record the starting parameters
            (starting_mu1, starting_mu2, starting_mu3, starting_sd1, starting_sd2,
             starting_sd3, starting_ppi1, starting_ppi2, starting_ppi3) = (mu1[0], mu2[0], mu3[0],
                                                                           sd1[0], sd2[0], sd3[0],
                                                                           ppi1, ppi2, ppi3)

        else:
            # Starting parameter estimates
            mu1, sd1 = mu1, sd1
            mu2, sd2 = mu2, sd2
            mu3, sd3 = mu3, sd3
            ppi1 = ppi1
            ppi2 = ppi2
            ppi3 = 1 - ppi1 - ppi2

        # Assuming your data is stored in a list or numpy array named dat
        oldppi1 = 0
        oldppi2 = 0
        oldppi3 = 0

        while change > tolerance:
            # E-Step
            resp1, resp2, resp3 = e_step(data, mu1, mu2, sd1, sd2, ppi1, modality='trimodal', mu3=mu3, sd3=sd3,
                                         ppi2=ppi2, ppi3=ppi3)
            # M-Step
            mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3 = m_step(data, resp1, modality='trimodal', resp2=resp2,
                                                                    resp3=resp3)

            change1 = np.abs(ppi1 - oldppi1)
            change2 = np.abs(ppi2 - oldppi2)
            change3 = np.abs(ppi3 - oldppi3)

            change = max(change1, change2, change3)

            oldppi1 = ppi1
            oldppi2 = ppi2
            oldppi3 = ppi3

        # make sure the larger mean is always mu1
        if mu1 < mu2:
            mu1, mu2 = mu2, mu1
            sd1, sd2 = sd2, sd1
            ppi1, ppi2 = ppi2, ppi1

        if mu1 < mu3:
            mu1, mu3 = mu3, mu1
            sd1, sd3 = sd3, sd1
            ppi1, ppi3 = ppi3, ppi1

        if mu2 < mu3:
            mu2, mu3 = mu3, mu2
            sd2, sd3 = sd3, sd2
            ppi2, ppi3 = ppi3, ppi2

        # Calculate the log likelihood for each observation
        ll = log_likelihood(data, mu1, mu2, sd1, sd2, ppi1, modality='trimodal',
                            mu3=mu3, sd3=sd3, ppi2=ppi2, ppi3=ppi3)

    else:
        if random_init:
            # randomly generate starting mu and sd
            mu1, sd1 = initialize_parameters(data, 1)
            mu2, sd2 = initialize_parameters(data, 1)

            # randomly generate a starting ppi
            ppi = np.random.uniform(0.01, 1)

            # record the starting parameters
            starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi = (mu1[0], mu2[0],
                                                                                    sd1[0], sd2[0], ppi)

        else:
            # Starting parameter estimates
            mu1, sd1 = mu1, sd1
            mu2, sd2 = mu2, sd2
            ppi = ppi1

        # Assuming your data is stored in a list or numpy array named dat
        oldppi = 0

        while change > tolerance:
            # E-Step
            resp1 = e_step(data, mu1, mu2, sd1, sd2, ppi)
            # M-Step
            mu1, mu2, sd1, sd2, ppi = m_step(data, resp1)

            change = np.abs(ppi - oldppi)

            oldppi = ppi

        # make sure the larger mean is always mu1
        if mu1 < mu2:
            mu1, mu2 = mu2, mu1
            sd1, sd2 = sd2, sd1
            ppi = 1 - ppi

        # Calculate the log likelihood for each observation
        ll = log_likelihood(data, mu1, mu2, sd1, sd2, ppi)

    # Calculate the AIC and BIC
    if modality == 'trimodal':
        k = 8
    else:
        k = 5

    n = len(data)
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    # compare with a single normal distribution
    mu_null, sd_null = norm.fit(data)
    ll_null = log_likelihood(data, mu_null, 1, sd_null, 1, 0)
    aic_null = -2 * ll_null + 2 * 2
    bic_null = -2 * ll_null + 2 * np.log(n)

    # calculate the R2
    R2 = 1 - ll / ll_null

    if modality == 'trimodal':
        if return_starting_params:
            return (starting_mu1, starting_mu2, starting_mu3, starting_sd1, starting_sd2,
                    starting_sd3, starting_ppi1, starting_ppi2, starting_ppi3,
                    mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2)

        else:
            return mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2

    else:
        if return_starting_params:
            return (starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi,
                    mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2)

        else:
            return mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2


def pdf_plot_generator(raw_data, result_df, save_path, modality="bimodal", bins=50, density=True, alpha=0.4, color='gray',
                       label="Data", x_label="PropOptimal", y_label="Density", title="EM Fitted Gaussian Mixture Model",
                       legend=True):
    # Use a clear and modern style for the plot
    plt.style.use('seaborn-v0_8-white')

    # # Increase the size of the plot
    # plt.figure(figsize=(10, 6))

    plt.hist(raw_data, bins=bins, density=density, alpha=alpha, color=color, label=label)
    x = np.linspace(min(raw_data), max(raw_data), 1000)

    if modality == 'trimodal':
        # Calculate the Gaussian distributions
        pdf1 = norm.pdf(x, result_df['mu1'].mode().iloc[0], result_df['sd1'].mode().iloc[0])
        pdf2 = norm.pdf(x, result_df['mu2'].mode().iloc[0], result_df['sd2'].mode().iloc[0])
        pdf3 = norm.pdf(x, result_df['mu3'].mode().iloc[0], result_df['sd3'].mode().iloc[0])

        # Weight the pdfs by the mixing coefficients
        ppi1 = result_df['ppi1'].mode().iloc[0]
        ppi2 = result_df['ppi2'].mode().iloc[0]
        ppi3 = result_df['ppi3'].mode().iloc[0]

        weighted_pdf1 = ppi1 * pdf1
        weighted_pdf2 = ppi2 * pdf2
        weighted_pdf3 = ppi3 * pdf3

        # Plot
        plt.plot(x, weighted_pdf3, '#E74C3C', linewidth=2,
                 label=rf"Disadvantageous Learners: $\mu$={result_df['mu3'].mode().iloc[0]:.2f}, "
                       rf"$\sigma$={result_df['sd3'].mode().iloc[0]:.2f}")
        plt.plot(x, weighted_pdf2, '#F39C12', linewidth=2,
                 label=rf"Average Learners: $\mu$={result_df['mu2'].mode().iloc[0]:.2f}, "
                       rf"$\sigma$={result_df['sd2'].mode().iloc[0]:.2f}")
        plt.plot(x, weighted_pdf1, '#2ECC71', linewidth=2,
                 label=rf"Advantageous Learners: $\mu$={result_df['mu1'].mode().iloc[0]:.2f}, "
                       rf"$\sigma$={result_df['sd1'].mode().iloc[0]:.2f}")

    else:
        # Calculate the Gaussian distributions
        pdf1 = norm.pdf(x, result_df['mu1'].mode().iloc[0], result_df['sd1'].mode().iloc[0])
        pdf2 = norm.pdf(x, result_df['mu2'].mode().iloc[0], result_df['sd2'].mode().iloc[0])

        # Weight the pdfs by the mixing coefficients
        ppi = result_df['ppi'].mode().iloc[0]
        weighted_pdf1 = (1 - ppi) * pdf1
        weighted_pdf2 = ppi * pdf2

        # Plot
        plt.plot(x, weighted_pdf2, '#E74C3C', linewidth=2,
                 label=rf"Disdvantageous Learners: $\mu$={result_df['mu2'].mode().iloc[0]:.2f}, "
                       rf"$\sigma$={result_df['sd2'].mode().iloc[0]:.2f}")
        plt.plot(x, weighted_pdf1, '#F39C12', linewidth=2,
                 label=rf"Average Learners: $\mu$={result_df['mu1'].mode().iloc[0]:.2f}, "
                       rf"$\sigma$={result_df['sd1'].mode().iloc[0]:.2f}")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend()

    # Remove background and grid
    plt.gca().set_facecolor('none')
    plt.grid(False)

    # Adjust spines to be less prominent
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_edgecolor('gray')
    plt.gca().spines['bottom'].set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show(dpi=600)


def likelihood_ratio_test(result_df, df, result_df_null=None):
    # if we are comparing two the target model with a simple normal distribution
    if result_df_null is None:
        # get the log likelihood of the target model
        ll = result_df['ll'].mode().iloc[0]

        ll_null = result_df['ll_null'].mode().iloc[0]

    # if we are comparing two target models
    else:
        # get the log likelihood of the target model
        ll = result_df['ll'].mode().iloc[0]

        ll_null = result_df_null['ll'].mode().iloc[0]

    # calculate the likelihood ratio test
    lr = -2 * (ll_null - ll)

    # calculate the p-value
    p_value = chi2.sf(lr, df)

    return p_value


def parameter_extractor(df, modality='bimodal'):
    if modality == 'trimodal':
        mu1 = df['mu1'].mode().iloc[0]
        mu2 = df['mu2'].mode().iloc[0]
        mu3 = df['mu3'].mode().iloc[0]
        sd1 = df['sd1'].mode().iloc[0]
        sd2 = df['sd2'].mode().iloc[0]
        sd3 = df['sd3'].mode().iloc[0]
        ppi1 = df['ppi1'].mode().iloc[0]
        ppi2 = df['ppi2'].mode().iloc[0]
        ppi3 = df['ppi3'].mode().iloc[0]

        return mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3

    else:
        mu1 = df['mu1'].mode().iloc[0]
        mu2 = df['mu2'].mode().iloc[0]
        sd1 = df['sd1'].mode().iloc[0]
        sd2 = df['sd2'].mode().iloc[0]
        ppi = df['ppi'].mode().iloc[0]

        return mu1, mu2, sd1, sd2, ppi


def group_assignment(df, result_df, modality='bimodal'):
    if modality == 'trimodal':
        # extract the parameters from the result
        mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3 = (
            parameter_extractor(result_df, 'trimodal'))

        # generate the probability density function
        prob1 = ppi1 * norm.pdf(df, mu1, sd1)
        prob2 = ppi2 * norm.pdf(df, mu2, sd2)
        prob3 = ppi3 * norm.pdf(df, mu3, sd3)

        # assign participants to each group
        assignments = np.argmax(np.vstack([prob1, prob2, prob3]), axis=0) + 1

        # combine the prob and assignments into a dataframe
        prob_df = pd.DataFrame(np.vstack([prob1, prob2, prob3]).T, columns=['prob1', 'prob2', 'prob3'])
        prob_df['assignments'] = assignments

        return prob_df

    else:
        # extract the parameters from the result
        mu1, mu2, sd1, sd2, ppi = (
            parameter_extractor(df, 'bimodal'))

        # generate the probability density function
        prob1 = (1 - ppi) * norm.pdf(df, mu1, sd1)
        prob2 = ppi * norm.pdf(df, mu2, sd2)

        # assign participants to each group
        assignments = np.argmax(np.vstack([prob1, prob2]), axis=0) + 1

        # combine the prob and assignments into a dataframe
        prob_df = pd.DataFrame(np.vstack([prob1, prob2]).T, columns=['prob1', 'prob2'])
        prob_df['assignments'] = assignments

        return prob_df


def best_fitting_participants(*dfs, keys=None, p_index=None):
    # Filter out None values and extract 'best_nll' column from each dataframe
    cols = [df['AIC'] for df in dfs if df is not None]

    # Use the dataframe names (or you can provide another list of names) as keys
    keys = [f'Model{i + 1}' for i, df in enumerate(dfs) if df is not None]

    # Concatenate the columns side by side
    stacked_df = pd.concat(cols, axis=1, keys=keys)

    # Find the model with the smallest fitting index for each participant
    best_models = stacked_df.idxmin(axis=1)

    # Count the occurrences of each model being the best
    model_counts = best_models.value_counts()

    # Calculate percentages
    total_participants = len(stacked_df)
    percentages = model_counts / total_participants * 100

    print(percentages)

    if p_index is not None:
        # get the participants who are best fitted by the target model
        best = best_models[best_models == keys[p_index]].index + 1
        best = best.tolist()

        return best
