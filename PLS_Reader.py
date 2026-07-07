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
from matplotlib.transforms import ScaledTranslation


font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)
palette = sns.color_palette('deep')
positive_color = palette[3]
negative_color = palette[0]
insignificant_color = palette[7]


def _bar_colors(values, significant=None):
    if significant is None:
        significant = np.ones(len(values), dtype=bool)
    else:
        significant = np.asarray(significant, dtype=bool)

    return [
        insignificant_color if not is_significant else negative_color if val < 0 else positive_color
        for val, is_significant in zip(values, significant)
    ]


def _pad_y_axis(ax, values, lower_errors=None, upper_errors=None):
    values = np.asarray(values, dtype=float)
    if lower_errors is None:
        lower_errors = np.zeros_like(values)
    if upper_errors is None:
        upper_errors = np.zeros_like(values)
    low = np.nanmin(np.r_[values - lower_errors, 0])
    high = np.nanmax(np.r_[values + upper_errors, 0])
    span = max(high - low, 0.05)
    ax.set_ylim(low - span * 0.12, high + span * 0.12)


def _style_bar_axis(ax, ylabel, title='', x_rotation=45, tick_size=18, label_size=25,
                    x_label_dx=0, title_pad=10, title_x=0.5):
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=label_size)
    ax.set_title(title, fontproperties=prop, fontsize=label_size, pad=title_pad, x=title_x)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(axis='both', width=1.6, length=6)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(tick_size)
        lbl.set_rotation(x_rotation)
        lbl.set_ha('right')
        if x_label_dx:
            lbl.set_transform(
                lbl.get_transform()
                + ScaledTranslation(x_label_dx / 72, 0, ax.figure.dpi_scale_trans)
            )
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(tick_size)
    sns.despine(ax=ax)


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

    sns.set_style("white")

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars by sign, with nonsignificant results in gray when all results are shown.
    predictor_significant = None
    raw_only_significant = np.zeros(len(sig_results), dtype=bool)
    if not only_sig and 'significant' in sig_results.columns:
        predictor_significant = sig_results['significant'].fillna(False).to_numpy(dtype=bool)
        if 'p_value' in sig_results.columns:
            raw_significant = sig_results['p_value'].lt(0.05).fillna(False).to_numpy(dtype=bool)
            raw_only_significant = raw_significant & ~predictor_significant
    colors = _bar_colors(sig_results['u1'], predictor_significant)

    # Use seaborn barplot
    sns.barplot(x=range(len(sig_results)), y=sig_results['u1'], hue=range(len(sig_results)), palette=colors,
                legend=False, ax=ax, width=0.72, edgecolor='white', linewidth=0.8)
    for patch, raw_only in zip(ax.patches, raw_only_significant):
        patch.set_alpha(0.94)
        if raw_only:
            patch.set_edgecolor('black')
            patch.set_linewidth(1.8)
            patch.set_linestyle((0, (3, 2)))

    # Set x-axis labels
    ax.set_xticks(range(len(sig_results)))
    ax.set_xticklabels(sig_results['Variable'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    _pad_y_axis(ax, sig_results['u1'])
    _style_bar_axis(ax, ylabel, title=title, x_rotation=45, tick_size=19, label_size=25)
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

    fig, axes = plt.subplots(
        num_conditions,
        1,
        figsize=(11, max(4.8 * num_conditions, 6.8)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for cond_idx, ax in enumerate(axes):
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

        # seaborn barplot
        # Color bars by sign; gray bars have bootstrap intervals crossing zero.
        significant = ((y - yerr_lower) > 0) | ((y + yerr_upper) < 0)
        colors = _bar_colors(y, significant)
        sns.barplot(x=x, y=y, errorbar=None, hue=x, palette=colors, legend=False,
                    width=0.72, edgecolor='white', linewidth=0.8, ax=ax)
        for patch in ax.patches:
            patch.set_alpha(0.94)

        # add error bars
        ax.errorbar(
            x=x,
            y=y,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="black",
            capsize=5.5,
            capthick=2,
            linewidth=2,
            zorder=3,
        )

        # Set x tick labels
        if method == 1:
            ax.set_xticks(x)
            ax.set_xticklabels(conditions)
        elif method == 3 and BehavLabels is not None:
            ax.set_xticks(x)
            labels = list(BehavLabels)
            if len(labels) > len(x):
                labels = labels[:len(x)]
            elif len(labels) < len(x):
                labels = labels + [str(idx + 1) for idx in range(len(labels), len(x))]
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks(x)

        if title:
            plot_title = (
                f"LV {LV_Vis}, {conditions[cond_idx]}, p = {p_value[0]:.4g}, "
                f"cross-block covariance = {cross_block_cov[0]:.2f}%"
            )
        else:
            plot_title = conditions[cond_idx]

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        _pad_y_axis(ax, y, lower_errors=yerr_lower, upper_errors=yerr_upper)
        _style_bar_axis(
            ax,
            ylabel,
            title=plot_title,
            x_rotation=45,
            tick_size=18,
            label_size=25,
            x_label_dx=11,
            title_pad=35,
            title_x=0.46,
        )
        if cond_idx < num_conditions - 1:
            ax.tick_params(axis='x', labelbottom=False)

    fig.tight_layout(h_pad=4.4)
    fig.subplots_adjust(hspace=0.60)

    # Save or show
    if save_path:
        condition_label = '_'.join(str(condition) for condition in conditions)
        fig.savefig(f'{save_path}PLS_LV{LV_Vis}_{condition_label}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
