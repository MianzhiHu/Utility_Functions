import sys
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import font_manager as fm
from matplotlib.ticker import FormatStrFormatter


font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)


def plot_predictor_results(df, title='', ylabel='Semantic Loadings', figsize=(12, 6), only_sig=True,
                             abs_ordering=True, save_path=None):
    """
    Plot significant results from PLS analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe from get_pls_results()
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    only_sig : bool
        If True, only plot significant results. If False, plot all results with significant ones colored differently.
    save_path : str, optional
        Path to save the figure. If None, displays the plot instead.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    if only_sig:
        # Filter significant results
        sig_results = df[df['significant'] == True].copy()
        if abs_ordering:
            sig_results = sig_results.sort_values('u1', key=abs, ascending=False)
        else:
            sig_results = sig_results.sort_values('u1', ascending=False)
    else:
        sig_results = df
        sig_results = sig_results.sort_values('u1', ascending=False)

    if sig_results.empty:
        print(f"No significant results found for {title}")
        return None

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars based on positive/negative values
    colors = [sns.color_palette('deep')[0] if val < 0 else sns.color_palette('deep')[3] for val in sig_results['u1']]

    # Use seaborn barplot
    sns.barplot(x=range(len(sig_results)), y=sig_results['u1'], hue=range(len(sig_results)), palette=colors,
                legend=False, ax=ax)

    # Set x-axis labels
    ax.set_xticks(range(len(sig_results)))
    ax.set_xticklabels(sig_results['Variable'], rotation=45, ha='right', fontproperties=prop, fontsize=20)

    # Set y-axis labels
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(20)

    # Labels and title
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=25)
    ax.set_title(title)

    sns.despine()
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_outcome_results(result_dir, boot_ratio_path, method, conditions, LV_Vis=1, plot_option=2, BehavLabels=None,
                         ylabel="Correlation with Semantic Features", title=True, save_path=None):
    # MATLAB is 1-based; Python is 0-based
    lv_idx = LV_Vis - 1

    boot_ratio_path = os.path.join(result_dir, boot_ratio_path)
    result = sio.loadmat(boot_ratio_path)['result'][0, 0]

    boot_result = result['boot_result'][0, 0]
    perm_result = result['perm_result'][0, 0]
    s = result['s']

    # Get the design and confidence intervals based on method
    if method == 1:  # task PLS
        design = np.asarray(boot_result['orig_usc'])[:, lv_idx]
        ll = np.asarray(boot_result['llusc'])[:, lv_idx]
        ul = np.asarray(boot_result['ulusc'])[:, lv_idx]
    else:
        design = np.asarray(boot_result['orig_corr'])[:, lv_idx]
        ll = np.asarray(boot_result['llcorr'])[:, lv_idx]
        ul = np.asarray(boot_result['ulcorr'])[:, lv_idx]

    # Define conditions
    num_conditions = len(conditions)

    # Calculate number of values per condition
    num_values_per_condition = len(design) // num_conditions

    # Permutation p-value and explained covariance
    p_value = np.asarray(perm_result['sprob'])[lv_idx]
    cross_block_cov = (s[lv_idx] ** 2 / np.sum(s ** 2)) * 100

    sns.set_style("white")

    for cond_idx in range(num_conditions):
        idx_start = cond_idx * num_values_per_condition
        idx_end = (cond_idx + 1) * num_values_per_condition

        design_cond = design[idx_start:idx_end]
        ll_cond = ll[idx_start:idx_end]
        ul_cond = ul[idx_start:idx_end]

        # Flip sign if plot_option == 1
        if plot_option == 1:
            y = -design_cond
            # Match MATLAB logic:
            # errorbar(x, -design_cond, -ul_cond + design_cond, -design_cond + ll_cond)
            yerr_lower = -ul_cond + design_cond
            yerr_upper = -design_cond + ll_cond
        else:
            y = design_cond
            # Match MATLAB logic:
            # errorbar(x, design_cond, ll_cond - design_cond, design_cond - ul_cond)
            yerr_lower = ll_cond - design_cond
            yerr_upper = design_cond - ul_cond

        # Error bars must be nonnegative lengths
        yerr_lower = np.abs(yerr_lower)
        yerr_upper = np.abs(yerr_upper)

        x = np.arange(len(y))

        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # seaborn barplot
        # Color bars based on positive/negative values
        colors = [sns.color_palette('deep')[0] if val < 0 else sns.color_palette('deep')[3] for val in y]
        sns.barplot(x=x, y=y, errorbar=None, hue=x, palette=colors, legend=False)

        # add error bars
        plt.errorbar(x=x, y=y, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="black", capsize=5, linewidth=1.5)


        ax.tick_params(axis="both", labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontproperties(prop)
            label.set_fontsize(20)

        # Set x tick labels
        if method == 1:
            ax.set_xticks(x)
            ax.set_xticklabels(conditions)
        elif method == 3 and BehavLabels is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(BehavLabels, rotation=45, ha="right", fontproperties=prop, fontsize=20)

        if title:
            plt.title(
                f"LV {LV_Vis}, {conditions[cond_idx]}, p = {p_value[0]:.4g}, cross-block covariance = {cross_block_cov[0]:.2f}%",
                fontproperties=prop, fontsize=25)
        else:
            plt.title('')
        plt.ylabel(ylabel, fontproperties=prop, fontsize=25)
        plt.xlabel('')

        sns.despine()
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(f'{save_path}PLS_LV{LV_Vis}_{conditions[cond_idx]}.png', dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()