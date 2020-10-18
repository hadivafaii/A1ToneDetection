import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, BoxStyle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from .generic_utils import reset_df, get_tasks

COLORS = list(sns.color_palette())
COLORMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples",
             "YlOrBr", "PuRd", "Greys", "YlGn", "GnBu"]

sns.set_style('white')


def mk_trajectory_plot(load_dir: str, global_stats: bool = False, name: str = None,
                       save_file=None, display=True, figsize=(12, 14), dpi=600):

    # load data
    results = pd.read_pickle(pjoin(load_dir, 'results.df'))
    extras = np.load(pjoin(load_dir, 'extras.npy'), allow_pickle=True).item()
    fit_metadata = np.load(pjoin(load_dir, 'fit_metadata.npy'), allow_pickle=True).item()

    # sample experiment
    if name is None:
        name = "ken_2016-08-20"

    cond = results.name == name
    if not sum(cond):
        return plt.figure(figsize=figsize, dpi=dpi), None

    # get best t
    if global_stats:
        best_t = results.best_t_global.unique().item()
    else:
        best_t = results.loc[results.name == name, 'best_t'].unique().item()

    # get traj data
    l2i = fit_metadata['lbl2idx']
    i2l = fit_metadata['idx2lbl']

    x_mat = extras[name].X
    lbls = extras[name].Y
    clfs = extras[name].clfs

    proj_mat = clfs[best_t].scalings_[:, :3]
    z = x_mat @ proj_mat

    trajectory_dict = {lbl: z[:, lbls == idx, :] for lbl, idx in l2i.items()}

    nt = len(results.timepoint.unique())
    xticks = range(0, nt + 1, 15)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[1, 4.3])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharex=ax0)

    ax0.axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
    ax1.axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
    ax2.axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
    ax0.axvline(best_t, color='magenta' if global_stats else 'limegreen', ls='--', lw=2)
    ax1.axvline(best_t, color='magenta' if global_stats else 'limegreen', ls='--', lw=2)
    ax2.axvline(best_t, color='magenta' if global_stats else 'limegreen', ls='--', lw=2, label='best t')

    if global_stats:
        df = results
    else:
        df = results.loc[results.name == name]
    sns.lineplot(data=df, y='performance', x='timepoint', color='k', ax=ax0, lw=3)
    sns.lineplot(data=df, y='distance', x='timepoint', color='orangered', ax=ax1, lw=3)
    sns.lineplot(data=df, y='sb', x='timepoint', color='royalblue', ax=ax2, lw=3)

    ax0.set_ylabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax0.set_xlabel('t (s)', fontsize=15)
    ax1.set_xlabel('t (s)', fontsize=15)
    ax2.set_xlabel('t (s)', fontsize=15)
    ax0.set_xticks(xticks)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax0.set_xticklabels([t / 30 for t in xticks])
    ax1.set_xticklabels([t / 30 for t in xticks])
    ax2.set_xticklabels([t / 30 for t in xticks])
    ax0.set_title("performance (mcc)", fontsize=15)
    ax1.set_title("distance", fontsize=15)
    ax2.set_title("scatter matrix between classes", fontsize=15)
    ax2.legend(loc='lower right')

    # 3d scatter
    ax = fig.add_subplot(gs[1, :], projection='3d')

    smin = 10
    smax = 1000
    markers = ['o', 'v', '^', 's']
    legend_elements = []
    for idx, lbl in i2l.items():
        tau = trajectory_dict[lbl]
        mu = tau.mean(1)[:best_t]

        sigma = np.linalg.norm(tau - tau.mean(1, keepdims=True), axis=-1).mean(-1)
        a = (smax - smin) / (max(sigma) - min(sigma))
        b = smin - a * min(sigma)
        scale = a * sigma + b

        ax.plot(mu[:, 0], mu[:, 1], mu[:, 2], color=COLORS[idx], lw=3, alpha=0.5)
        # core
        ax.scatter(
            mu[:, 0], mu[:, 1], mu[:, 2], alpha=1,
            marker=markers[idx], s=50, label=lbl, cmap=COLORMAPS[idx], c=range(best_t))
        # shadow
        ax.scatter(
            mu[:, 0], mu[:, 1], mu[:, 2], alpha=0.2,
            marker=markers[idx], s=scale[:best_t], cmap=COLORMAPS[idx], c=range(best_t))
        # legend
        legend_elements.append(Line2D(
            [0], [0], marker=markers[idx], color='w', label=lbl,
            markerfacecolor=COLORS[idx], markersize=15))

    ax.legend(handles=legend_elements, loc='upper left', fontsize='large')

    if global_stats:
        msg = "representative tarajectory:\n\n name = {}\n t = 0 . . .  {:d} (best t)"
    else:
        msg = "name = {}\n t = 0 . . .  {:d} (best t)"
    msg = msg.format(name, best_t)
    ax.set_title(msg, fontsize=17)

    fig.tight_layout()
    save_fig(fig, None, save_file, display)
    return fig, np.array([ax0, ax1, ax2, ax])


def mk_coeffs_importances_plot(coeffs_filtered, save_file=None, display=True, figsize=(67, 13), dpi=200):
    nc = len(coeffs_filtered.cell_indx.unique())
    nb_seeds = len(coeffs_filtered.seed.unique())
    tasks = get_tasks()

    sns.set_style('white')

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=4, ncols=len(tasks), height_ratios=[0.3, 0.3, 1, 1])

    xticks = range(0, nc + 1, 10)
    axes = []
    percent_nonzeros = {}
    for i, task in enumerate(tasks):
        ax0 = fig.add_subplot(gs[0, i])
        ax1 = fig.add_subplot(gs[1, i])
        ax2 = fig.add_subplot(gs[2, i])
        ax3 = fig.add_subplot(gs[3, i])

        ax0.set_title("{:s}, t = nan".format(task, ), fontsize=15)
        ax0.set_xticks(xticks)
        ax1.set_xticks(xticks)
        ax0.set_yticks([0])
        ax1.set_yticks([0])
        ax0.grid(axis='both', ls=':', lw=2)
        ax1.grid(axis='both', ls=':', lw=2)

        ax2.set_xticks([])
        if i > 0:
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax3.set_yticks([])

        axes.append([ax0, ax1, ax2, ax3])

        cond = coeffs_filtered.task == task
        if not (sum(cond) and nc):
            continue

        selected_df = coeffs_filtered.loc[cond]

        percent_nonzero = selected_df.percent_nonzero.to_numpy()
        percent_nonzero = percent_nonzero.reshape(nb_seeds, nc)
        percent_nonzero = np.unique(percent_nonzero, axis=1)
        percent_nonzeros[task] = percent_nonzero

        best_t = selected_df.timepoint.unique()
        ax0.set_title("{:s}, t = {:d}".format(task, best_t.item()), fontsize=15)

        # 1st and 2nd rows
        sns.lineplot(
            x='cell_indx',
            y='coeffs',
            data=selected_df,
            color=COLORS[i],
            lw=2,
            ax=ax0,
        )
        sns.lineplot(
            x='cell_indx',
            y='importances',
            data=selected_df,
            color=COLORS[i],
            lw=2,
            ax=ax1,
        )
        ax1.set_xlabel('')

        # 3rd and 4th rows
        x = selected_df.x.to_numpy()[:nc]
        y = selected_df.y.to_numpy()[:nc]

        coeffs = selected_df.coeffs.to_numpy()
        coeffs = coeffs.reshape(nb_seeds, nc).mean(0)
        vminmax_coeffs = max(abs(coeffs))

        importances = selected_df.importances.to_numpy()
        importances = importances.reshape(nb_seeds, nc).mean(0)
        vminmax_importances = max(abs(importances))

        _ = ax2.scatter(
            x=x[coeffs == 0],
            y=y[coeffs == 0],
            color='w',
            s=50,
            edgecolors='k',
            linewidths=0.4,
        )
        s = ax2.scatter(
            x=x[coeffs != 0],
            y=y[coeffs != 0],
            c=coeffs[coeffs != 0],
            cmap='seismic',
            s=120,
            vmin=-vminmax_coeffs,
            vmax=vminmax_coeffs,
            edgecolors='k',
            linewidths=0.4,
        )
        plt.colorbar(s, ax=ax2)

        _ = ax3.scatter(
            x=x[importances == 0],
            y=y[importances == 0],
            color='w',
            s=50,
            edgecolors='k',
            linewidths=0.4,
        )
        s = ax3.scatter(
            x=x[importances != 0],
            y=y[importances != 0],
            c=importances[importances != 0],
            cmap=COLORMAPS[i],
            s=120,
            vmin=0,
            vmax=vminmax_importances,
            edgecolors='k',
            linewidths=0.4,
        )
        plt.colorbar(s, ax=ax3)

        if i == 0:
            ax0.set_ylabel("coeffs", fontsize=15)
            ax1.set_ylabel("importances", fontsize=15)
            ax2.set_ylabel("coeffs", fontsize=15)
            ax3.set_ylabel("importances", fontsize=15)

    ax_arr = np.array(axes).T

    msg1 = "classifier coefficients are sparse. averaged results obtained using {} different seeds,   "
    msg1 += "1st & 3rd rows: coeffs  /  2nd and 4th rows: importances\n\n"
    msg1 = msg1.format(nb_seeds)

    msg2 = "avg percent nonzero coeffs for each task (error bars are due to averaging for different random seeds):\n"
    for task, percent_nonzero in percent_nonzeros.items():
        msg2 += "{:s}:  {:.2f} ± {:.2f} {:s},    ".format(task, percent_nonzero.mean(), percent_nonzero.std(), '%')

    sup = fig.suptitle(msg1+msg2, fontsize=16, y=1.04)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr, sup


def mk_reg_selection_plot(performances: pd.DataFrame, criterion: str = 'mcc',
                          save_file=None, display=True, figsize=(50, 8), dpi=200):
    criterion_choices = {'mcc': 0, 'accuracy': 1, 'f1': 2}
    assert criterion in criterion_choices
    metric_indx = criterion_choices[criterion]

    tasks = get_tasks()
    reg_cs = np.unique(performances['reg_C'])

    nb_c = len(reg_cs)
    nb_seeds = len(performances.seed.unique())
    nt = len(performances.timepoint.unique())

    sns.set_style('white')
    fig, ax_arr = plt.subplots(3, len(tasks), sharex='all', sharey='all', figsize=figsize, dpi=dpi)

    xticks = range(0, nt + 1, 15)
    for i, task in enumerate(tasks):
        ax_arr[0, i].set_title(task, fontsize=15)
        cond = performances.task == task
        if not sum(cond):
            continue

        best_reg = performances.loc[cond].best_reg.unique().item()
        best_timepoint = performances.loc[cond].best_timepoint.unique().item()

        a = np.where(reg_cs == best_reg)[0].item()
        b = best_timepoint

        scores_all = performances.score[cond].to_numpy()
        scores_all = scores_all.reshape(nb_c, nb_seeds, 4, nt)

        scores = scores_all[..., metric_indx, :]
        confidences = scores_all[..., -1, :]

        mean_scores = scores.mean(1)
        mean_confidences = confidences.mean(1)

        max_score = np.max(mean_scores[..., 30:])
        threshold = 0.9
        max_score_threshold = threshold * max_score

        above_threshold_bool = mean_scores > max_score_threshold
        above_threshold = mean_scores.copy()
        above_threshold[~above_threshold_bool] = 0

        im_score = ax_arr[0, i].imshow(
            X=mean_scores,
            aspect=nt/nb_c/2,
            cmap='hot',
        )
        plt.colorbar(im_score, ax=ax_arr[0, i], fraction=0.0235, pad=0.04)

        im_confidence = ax_arr[1, i].imshow(
            X=mean_confidences,
            aspect=nt/nb_c/2,
            cmap='Greens',
        )
        msg = "selected avg score: {:.1f} / confidence: {:.1f}"
        msg = msg.format(mean_scores[a, b], mean_confidences[a, b])
        ax_arr[1, i].set_title(msg, fontsize=10)
        plt.colorbar(im_confidence, ax=ax_arr[1, i], fraction=0.0235, pad=0.04)

        im_above = ax_arr[2, i].imshow(
            X=above_threshold,
            aspect=nt/nb_c/2,
            cmap='Purples',
        )
        msg = "selected reg: {} / timepoint: {:d}"
        msg = msg.format(reg_cs[a], b)
        ax_arr[2, i].set_title(msg, fontsize=10)
        plt.colorbar(im_above, ax=ax_arr[2, i], fraction=0.0235, pad=0.04)

        rx = FancyBboxPatch(
            xy=(0, a),
            width=nt - 1,
            height=0,
            boxstyle=BoxStyle("Round", pad=figsize[1] / nb_c * 0.3),
        )
        ry = FancyBboxPatch(
            xy=(b, 0),
            width=0,
            height=nb_c,
            boxstyle=BoxStyle("Round", pad=figsize[0] / nt * 5),
        )
        r = [rx, ry]

        pc = PatchCollection(r, edgecolor='dodgerblue', facecolors='None')
        ax_arr[0, i].add_collection(pc)
        pc = PatchCollection(r, edgecolor='dodgerblue', facecolors='None')
        ax_arr[1, i].add_collection(pc)
        pc = PatchCollection(r, edgecolor='dodgerblue', facecolors='None')
        ax_arr[2, i].add_collection(pc)

        for j in range(3):
            if j == 2:
                ax_arr[j, i].set_xlabel('t (s)', fontsize=15)
                alpha = 0.5
            else:
                alpha = 0.3
            ax_arr[j, i].axvspan(30, 60, facecolor='lightgrey', alpha=alpha)

            ax_arr[j, i].set_xticks(xticks)
            ax_arr[j, i].set_xticklabels([t / 30 for t in xticks])

            ax_arr[j, i].set_yticks(range(nb_c))
            ax_arr[j, i].set_yticklabels(reg_cs)

            if i == 0:
                ax_arr[j, i].set_ylabel('C', fontsize=15)

    msg = "best reg and timepoint selection. Y axis is reg value (smaller = stronger) and X axis is time\n"
    msg += "Top: performance / Mid: confidence / Bottom: points within 90 % threshold of max score"
    sup = fig.suptitle(msg, fontsize=20, y=1.06)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr, sup


def mk_boxplots(df_all, criterion: str = 'mcc', save_file=None, display=True, figsize=(24, 8), dpi=100):
    criterion_choices = ['mcc', 'accuracy', 'f1']
    if criterion not in criterion_choices:
        raise RuntimeError("invalid criterion encountered, allowed options are: {}".format(criterion_choices))

    tasks = get_tasks()
    assert set(tasks) == set(df_all["performances_filtered"].task.unique()), "df must include all the tasks"

    nb_seeds = len(df_all["performances_filtered"].seed.unique().tolist())
    nt = len(df_all["performances"].timepoint.unique().tolist())

    sns.set_style('white')
    fig, ax_arr = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    meanprops = {
        "marker": "o",
        "markerfacecolor": "bisque",
        "markeredgecolor": "black",
        "markersize": "8",
    }

    xticks = range(0, nt + 1, 15)
    _ = ax_arr[0, 0].axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
    sns.boxplot(
        x="best_timepoint",
        y="task",
        data=df_all["performances_filtered"],
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops=meanprops,
        ax=ax_arr[0, 0],
    )
    ax_arr[0, 0].set_xticks(xticks)
    ax_arr[0, 0].set_xticklabels([t / 30 for t in xticks])
    ax_arr[0, 0].set_xlabel("t (s)", fontsize=11)

    xticks = range(30, nt - 15 + 1, 15)
    _ = ax_arr[1, 0].axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
    sns.boxplot(
        x="best_timepoint",
        y="task",
        data=df_all["performances_filtered"],
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops=meanprops,
        ax=ax_arr[1, 0],
    )
    ax_arr[1, 0].set_xlim(30, nt - 15)
    ax_arr[1, 0].set_xticks(xticks)
    ax_arr[1, 0].set_xticklabels([t / 30 for t in xticks])
    ax_arr[1, 0].set_xlabel("t (s)", fontsize=11)
    ax_arr[1, 0].grid(axis='x', ls=':', lw=2)

    sns.boxplot(
        x="score",
        y="task",
        data=df_all["performances_filtered"],
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops=meanprops,
        ax=ax_arr[0, 1],
    )
    ax_arr[0, 1].set_xlim(0, 1)

    sns.boxplot(
        x="score",
        y="task",
        data=df_all["performances_filtered"],
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops=meanprops,
        ax=ax_arr[1, 1],
    )
    ax_arr[1, 1].set_xlim(0.6, 1)
    ax_arr[1, 1].grid(axis='x', ls=':', lw=2)

    sns.boxplot(
        x="percent_nonzero",
        y="task",
        data=df_all["coeffs_filtered"],
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops=meanprops,
        ax=ax_arr[0, 2],
    )
    ax_arr[0, 2].set_xlim(0, 100)
    sns.boxplot(
        x="percent_nonzero",
        y="task",
        data=df_all["coeffs_filtered"],
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops=meanprops,
        ax=ax_arr[1, 2],
    )
    ax_arr[1, 2].set_xlim(0, 20)
    ax_arr[1, 2].grid(axis='x', ls=':', lw=2)

    ax_arr[0, 0].set_title('selected timepoint (used for classification)', fontsize=16)
    ax_arr[0, 1].set_title('best score at selected timepoints', fontsize=16)
    ax_arr[0, 2].set_title('percentage of nonzero coefficients', fontsize=16)

    for i in range(3):
        for j in range(2):
            ax_arr[j, i].get_legend().remove()

            if i == 0:
                ax_arr[j, i].axes.tick_params(axis='y', labelsize=15)
            else:
                ax_arr[j, i].set_yticks([])

    msg = "Results obtained using '{:s}' criterion and {:d} different seeds"
    msg = msg.format(criterion, nb_seeds)
    sup = fig.suptitle(msg, y=1.03, fontsize=25)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr


def mk_gridplot(df, mode="importances", save_file=None, display=True, figsize=(96, 16), dpi=200):
    _allowed_modes = ["importances", "coeffs"]
    if mode not in _allowed_modes:
        raise RuntimeError("invalid mode entered.  allowed options: {}".format(_allowed_modes))

    names = list(df.name.unique())
    tasks = list(df.task.unique())

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(len(tasks), len(names), figsize=figsize, dpi=dpi, sharex='col', sharey='row')

    for i, task in tqdm(enumerate(tasks), total=len(tasks)):
        for j, name in enumerate(names):

            selected_df = df.loc[(df.name == name) & (df.task == task)]
            sns.lineplot(
                x='cell_indx',
                y=mode,
                data=selected_df,
                color=COLORS[i],
                ax=ax_arr[i, j],
            )

            if j == 0:
                ax_arr[i, j].set_ylabel(task, fontsize=9, rotation=75)
            else:
                ax_arr[i, j].set_ylabel('')

            if i == 0:
                ax_arr[i, j].set_title(name, fontsize=10, rotation=0)

            ax_arr[i, j].set_yticks([0])

            try:
                nc = max(selected_df.cell_indx.unique()) + 1
                if i == len(tasks) - 1:
                    ax_arr[i, j].set_xticks(range(0, nc, 10))
            except ValueError:
                continue

    msg = "Average logistic regression '{:s}' across experiments/tasks.\
    Results obtained using {:d} different seeds"
    msg = msg.format(mode, len(df.seed.unique()))
    sup = fig.suptitle(msg, y=1.0, fontsize=20)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr


def mk_gridhist(df, save_file=None, display=True, figsize=(25, 20), dpi=200):
    tasks = list(df.task.unique())

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(len(tasks), len(tasks), figsize=figsize, dpi=dpi, sharex='all', sharey='row')

    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if j == 0:
                ax_arr[i, j].set_ylabel(task1, fontsize=13, rotation=65)
                ax_arr[i, j].set_yticklabels([''] * 5)
            else:
                ax_arr[i, j].set_ylabel('')
            if i == 0:
                ax_arr[i, j].set_title(task2, fontsize=15, rotation=0)
            if i == j:
                legend_elements = [Line2D(
                    [0], [0], marker='o', color='w', label=task1,
                    markerfacecolor=COLORS[i], markersize=12,
                )]
                ax_arr[i, j].legend(handles=legend_elements, loc='center', fontsize=12)
                continue

            selected_tasks = [task1, task2]
            selected_df = df.loc[df['task'].isin(selected_tasks)]

            sns.histplot(
                data=selected_df,
                x="coeffs_normalized",
                hue='task',
                hue_order=selected_tasks,
                palette=[COLORS[i], COLORS[j]],
                bins=10,
                element='step',
                kde=True,
                legend=False,
                ax=ax_arr[i, j],
            )

    ax_arr[-1, -2].get_shared_x_axes().join(ax_arr[-1, -2], ax_arr[-1, -1])
    ax_arr[-1, -1].set_xlabel(ax_arr[-1, -2].get_xlabel())

    msg = "Histogram of 'nonzero' logistic regression coefficients for different tasks.\n\
    Y axis is counts (shared between rows), data is normalized to range [-1, 1] (shared between cols).\n\
    Results obtained using {:d} different seeds."
    msg = msg.format(len(df.seed.unique()))
    sup = fig.suptitle(msg, y=0.97, fontsize=20)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr


def mk_gridscatter(df, mode="importances", save_file=None, display=True, figsize=(140, 20), dpi=300):
    _allowed_modes = ["importances", "coeffs"]
    if mode not in _allowed_modes:
        raise RuntimeError("invalid mode entered.  allowed options: {}".format(_allowed_modes))

    names = list(df.name.unique())
    tasks = list(df.task.unique())

    sns.set_style('white')
    fig, ax_arr = plt.subplots(len(tasks), len(names), figsize=figsize, dpi=dpi, sharex='col', sharey='row')

    for j, name in enumerate(names):
        selected_df = df.loc[df.name == name]
        nc = len(selected_df.cell_indx.unique())

        try:
            z = selected_df[mode].to_numpy().reshape(len(selected_df.seed.unique()), -1, nc)
            best_idxs = np.argmax((z == 0).sum(-1), axis=0)
            best_idxs_dict = dict(zip(selected_df.task.unique(), best_idxs))
        except ValueError:
            best_idxs_dict = {k: 0 for k in selected_df.task.unique()}

        for i, task in enumerate(tasks):
            selected_df = df.loc[(df.name == name) & (df.task == task)]
            nc = len(selected_df.cell_indx.unique())
            if nc == 0:
                continue

            x = selected_df.x_coordinates.to_numpy()[:nc]
            y = selected_df.y_coordinates.to_numpy()[:nc]
            z = selected_df[mode].to_numpy().reshape(len(selected_df.seed.unique()), nc)
            z = z[best_idxs_dict[task]]
            vminmax = max(abs(z))

            _ = ax_arr[i, j].scatter(
                x=x[z == 0],
                y=y[z == 0],
                color='w',
                s=50,
                edgecolors='k',
                linewidths=0.4,
            )
            s = ax_arr[i, j].scatter(
                x=x[z != 0],
                y=y[z != 0],
                c=z[z != 0],
                cmap=COLORMAPS[i] if mode == 'importances' else 'seismic',
                s=120,
                vmin=0 if mode == 'importances' else -vminmax,
                vmax=vminmax,
                edgecolors='k',
                linewidths=0.4,
            )
            plt.colorbar(s, ax=ax_arr[i, j])

            if j == 0:
                ax_arr[i, j].set_ylabel(task, fontsize=9, rotation=75)
            else:
                ax_arr[i, j].set_ylabel('')

            if i == 0:
                ax_arr[i, j].set_title(name, fontsize=10, rotation=0)

            ax_arr[i, j].set_xticks([])
            ax_arr[i, j].set_yticks([])

    msg = "Scatter plot of neurons.  Colormap is scaled according to the average '{:s}',\
    obtained using {:d} different seeds.  Cells with nonzero importance have larger markersize."
    msg = msg.format(mode, len(df.seed.unique()))
    sup = fig.suptitle(msg, y=1.0, fontsize=25)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr


def mk_hist(df, mode="importances", save_file=None, display=True, figsize=(20, 16), dpi=200):
    _allowed_modes = ["importances", "coeffs"]
    if mode not in _allowed_modes:
        raise RuntimeError("invalid mode entered.  allowed options: {}".format(_allowed_modes))

    names = list(df.name.unique())
    tasks = list(df.task.unique())
    nb_seeds = len(df.seed.unique())

    # crealte df to plot
    cols = ['task', 'num_nonzero', 'percent_nonzero']
    df_to_plot = pd.DataFrame(columns=cols)

    for task in tqdm(tasks):
        for name in names:
            selected_df = df.loc[(df.name == name) & (df.task == task)]
            nc = len(selected_df.cell_indx.unique())
            if nc == 0:
                continue
            z = selected_df[mode].to_numpy()
            try:
                z = z.reshape(nb_seeds, nc)
            except ValueError:
                print('missing data, name = {:s}, task = {:s}, moving on . . .'.format(name, task))
                continue
            num_nonzeros = (z != 0).sum(-1)

            assert not sum(num_nonzeros > nc), "num nonzero neurons must be less than total num cells"

            data_dict = {
                'task': [task] * nb_seeds,
                'num_nonzero': num_nonzeros,
                'percent_nonzero': num_nonzeros / nc * 100,
            }
            df_to_plot = df_to_plot.append(pd.DataFrame(data=data_dict))

    df_to_plot = reset_df(df_to_plot)

    nrows, ncols = 4, 4
    assert nrows * ncols >= 2 * len(tasks)

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharey='all')

    tick_spacing1 = 5
    nb_ticks1 = int(np.ceil(max(df_to_plot.num_nonzero.to_numpy()) / tick_spacing1))
    xticks1 = range(0, nb_ticks1 * tick_spacing1 + 1, tick_spacing1)

    tick_spacing2 = 10
    xticks2 = range(0, 100 + 1, tick_spacing2)

    for idx, task in enumerate(tasks):
        i, j = idx // nrows, idx % nrows

        x = df_to_plot.loc[(df_to_plot.task == task)]
        mean = x.num_nonzero.mean()
        ax_arr[i, j].axvline(
            x=mean,
            label='mean ~ {:d}'.format(int(np.rint(mean))),
            color='navy',
            linestyle='dashed',
            linewidth=1,
        )
        sns.histplot(
            data=x,
            x="num_nonzero",
            color=COLORS[idx],
            bins=10,
            element='poly',
            kde=True,
            label=task,
            ax=ax_arr[i, j],
        )
        ax_arr[i, j].set_xticks(xticks1)
        ax_arr[i, j].legend(loc='upper right')

        mean = x.percent_nonzero.mean()
        ax_arr[i + 2, j].axvline(
            x=mean,
            label='mean ~ {:.1f} {:s}'.format(mean, '%'),
            color='darkred',
            linestyle='dashed',
            linewidth=1,
        )
        sns.histplot(
            data=x,
            x="percent_nonzero",
            color=COLORS[idx],
            bins=10,
            element='poly',
            kde=True,
            label=task,
            ax=ax_arr[i + 2, j],
        )
        ax_arr[i + 2, j].set_xticks(xticks2)
        ax_arr[i + 2, j].legend(loc='upper right')

    fig.delaxes(ax_arr[-3, -1])
    fig.delaxes(ax_arr[-1, -1])

    fig.subplots_adjust(hspace=0.4)

    msg = "Hitogram plot of:\n\
    1) num nonzero classifier '{:s}' (top two rows) and\n\
    2) percent nonzero '{:s}' (bottom two rows)\n\
    Dashed lines correspond to averages in each case. Obtained using {:d} different seeds."
    msg = msg.format(mode, mode, nb_seeds)
    sup = fig.suptitle(msg, y=1.0, fontsize=20)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr, df_to_plot


def process_df_mk_bestreg_gridplot(df, save_file=None, display=True, figsize=(400, 30), dpi=100):
    cols = list(list(df.columns) + ['best_t'])
    df_processed = pd.DataFrame(columns=cols)

    names = list(df.name.unique())
    tasks = list(df.task.unique())
    reg_cs = list(df.reg_C.unique())

    nb_c = len(reg_cs)
    nb_seeds = len(df.seed.unique())
    nt = len(df.timepoint.unique())

    sns.set_style('white')
    fig, ax_arr = plt.subplots(len(tasks), len(names), figsize=figsize, dpi=dpi, sharex='all', sharey='all')

    for i, task in tqdm(enumerate(tasks), total=len(tasks)):
        for j, name in enumerate(names):
            # select best reg
            selected_df = df.loc[(df.name == name) & (df.task == task)]
            scores = selected_df.score.to_numpy()
            if not len(scores):
                print('missing data, name = {:s}, task = {:s}, moving on . . .'.format(name, task))
                continue
            else:
                scores = scores.reshape(nb_seeds, nb_c, 3, nt)
            mean_scores = scores.mean(2).mean(0)
            max_score = np.max(mean_scores)

            if np.sum(mean_scores == max_score) == 1:
                a, b = np.unravel_index(np.argmax(mean_scores), mean_scores.shape)
            else:
                only_max_scores = mean_scores.copy()
                only_max_scores[mean_scores < max_score] = 0

                a = max(np.where(only_max_scores)[0])
                b = np.argmax(only_max_scores[a])

            assert mean_scores[a, b] == np.max(mean_scores), "must select max score"

            # mk plot
            im = ax_arr[i, j].imshow(
                X=mean_scores,
                aspect=nt/nb_c/2.5,
                cmap='hot',
            )
            plt.colorbar(im, ax=ax_arr[i, j])
            ax_arr[i, j].set_yticks(range(nb_c))
            ax_arr[i, j].set_yticklabels(reg_cs)

            rx = FancyBboxPatch(
                xy=(0, a),
                width=nt,
                height=0,
                boxstyle=BoxStyle("Round", pad=figsize[1]/nb_c/len(tasks) * 0.5),
            )
            ry = FancyBboxPatch(
                xy=(b, 0),
                width=0,
                height=nb_c,
                boxstyle=BoxStyle("Round", pad=figsize[0]/nt/len(names) * 5),
            )
            r = [rx, ry]

            pc = PatchCollection(r, edgecolor='dodgerblue', facecolors='None')
            ax_arr[i, j].add_collection(pc)

            if j == 0:
                ax_arr[i, j].set_ylabel(task, fontsize=12, rotation=75)
            else:
                ax_arr[i, j].set_ylabel('')

            if i == 0:
                ax_arr[i, j].set_title(name, fontsize=15, rotation=0)

            # save df
            num = nb_seeds * 3 * nt
            assert num == len(selected_df) // nb_c, "otherwise something wrong"

            _seeds = selected_df.seed.to_numpy()
            _seeds = _seeds.reshape(nb_seeds, nb_c, 3, nt)
            _seeds = _seeds[:, a, ...]

            _timepoints = selected_df.timepoint.to_numpy()
            _timepoints = _timepoints.reshape(nb_seeds, nb_c, 3, nt)
            _timepoints = _timepoints[:, a, ...]

            _metric = selected_df.metric.to_numpy()
            _metric = _metric.reshape(nb_seeds, nb_c, 3, nt)
            _metric = _metric[:, a, ...]

            # save processed df
            data_dict = {
                'name': [name] * num,
                'seed': _seeds.flatten(),
                'task': [task] * num,
                'reg_C': [reg_cs[a]] * num,
                'timepoint': _timepoints.flatten(),
                'metric': _metric.flatten(),
                'score': scores[:, a, ...].flatten(),
                'best_t': [b] * num,
            }
            df_processed = df_processed.append(pd.DataFrame(data=data_dict))

    df_processed = reset_df(df_processed)

    msg = "Detecting best regularization hyperparam / time point for each task / experiment,\
        X-axis is time, Y-axis is reg values used in grid search: {},\
        blue boxes indicate selected reg/time pair"
    msg = msg.format(reg_cs)
    sup = fig.suptitle(msg, y=1.0, fontsize=40)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr, df_processed


def mk_performance_plot(df, save_file=None, display=True, figsize=(24, 8), dpi=100):
    tasks = get_tasks()
    assert set(tasks) == set(df.task.unique()), "df must include all the tasks"

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(2, 5, figsize=figsize, dpi=dpi, sharey='all', sharex='all')

    nt = len(df.timepoint.unique().tolist())
    xticks = range(0, nt + 1, 15)
    for idx, task in enumerate(tasks):
        i, j = idx // 5, idx % 5
        selected_df = df.loc[df.task == task]
        sns.lineplot(x="timepoint", y="score", data=selected_df, hue='metric', ax=ax_arr[i, j])
        ax_arr[i, j].axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
        ax_arr[i, j].set_title("{:s}".format(task, fontsize=15))
        if idx > 0:
            ax_arr[i, j].get_legend().remove()
        if i == 1:
            ax_arr[i, j].set_xticks(xticks)
            ax_arr[i, j].set_xticklabels([t / 30 for t in xticks])
            ax_arr[i, j].set_xlabel("t (s)", fontsize=12)

    msg = "Average classification performance for different tasks at different time points"
    sup = fig.suptitle(msg, y=1.0, fontsize=20)

    save_fig(fig, sup, save_file, display)
    return fig, ax_arr


def mk_reg_viz(df_processed, df_stats, save_file=None, display=True, figsize=(20, 8), dpi=200):
    sns.set_style('white')
    f = plt.figure(figsize=figsize, dpi=dpi)
    sns.countplot(x="reg_C", hue="task", data=df_processed, dodge=True)

    msg = "Distribution of best regularization hyperparameter for different tasks\n\
    X axis is inverse regularization strength (i.e. smaller is stronger)"
    sup = f.suptitle(msg, y=1.0, fontsize=20)

    msg1 = "X: best score vs Y: precent nonzero joint distribution.\
    this figure shows for perfect predictions (i.e. score = 1) results are sparser"
    f1, sup1 = _mk_jointplot(df_stats, 'best_score', 'percent_nonzero', sup_msg=msg1, dpi=dpi)

    msg2 = "X: (inverse) regulirization strength vs Y: precent nonzero joint distribution.\
    this figure is a sanity check: stronger regularization should lead to sparser results"
    f2, sup2 = _mk_jointplot(df_stats, 'reg_C', 'percent_nonzero', sup_msg=msg2, dpi=dpi)

    figs = [f, f1, f2]
    sups = [sup, sup1, sup2]
    save_fig(figs, sups, save_file, display, multi=True)
    return figs


def _mk_jointplot(df, x, y, sup_msg='', sup_x=None, position_kws=None,
                  kind='hex', ratio=4, figsize=(30, 10), dpi=100):
    tasks = list(df.task.unique())
    joint_grids = ()
    for idx, task in enumerate(tasks):
        selected_df = df.loc[df.task == task]
        joint_grids += (
            sns.jointplot(
                x=x,
                y=y,
                data=selected_df,
                kind=kind,
                ratio=ratio,
                color=COLORS[idx],
                marginal_kws={'kde': True}),
        )
        plt.close()

    if position_kws is None:
        left = 0.1
        bottom = 0.1
        width_large = 0.3
        height_large = 0.3
        width_small = 0.1
        height_small = 0.15
    else:
        left = position_kws['left']
        bottom = position_kws['bottom']
        width_large = position_kws['width_large']
        height_large = position_kws['height_large']
        width_small = position_kws['width_small']
        height_small = position_kws['height_small']

    f = plt.figure(figsize=figsize, dpi=dpi)
    for jg in joint_grids:
        for a in jg.fig.axes:
            # noinspection PyProtectedMember
            f._axstack.add(f._make_key(a), a)

    for idx, task in enumerate(tasks):
        f.axes[idx * 3 + 1].set_title(task)

        f.axes[idx * 3 + 0].set_position([left, bottom, width_large, height_large])
        f.axes[idx * 3 + 1].set_position([left, height_large + bottom, width_large, height_small])
        f.axes[idx * 3 + 2].set_position([left + width_large, bottom, width_small, height_large])

        if sup_x is None and idx == len(tasks) // 2:
            sup_x = left / 5 + width_large / 10

        left += width_large + 2.5 * width_small

    sup_y = height_large + height_small
    sup = f.suptitle(sup_msg, y=sup_y, x=sup_x, fontsize=20)
    plt.close(f)
    return f, sup


def save_fig(fig, sup, save_file, display, multi=False):
    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
        if not multi:
            fig.savefig(save_file, dpi=fig.dpi, bbox_inches='tight', bbox_extra_artists=[sup])
        else:
            assert len(fig) == len(sup) > 1, "must provide a list of mroe than 1 figures for multi figure saving"
            with PdfPages(save_file) as pages:
                for f, s in zip(fig, sup):
                    canvas = FigureCanvasPdf(f)
                    if s is not None:
                        canvas.print_figure(pages, dpi=f.dpi, bbox_inches='tight', bbox_extra_artists=[s])
                    else:
                        canvas.print_figure(pages, dpi=f.dpi, bbox_inches='tight')

    if display:
        if isinstance(fig, list):
            for f in fig:
                plt.show(f)
        else:
            plt.show(fig)
    else:
        if isinstance(fig, list):
            for f in fig:
                plt.close(f)
        else:
            plt.close(fig)
