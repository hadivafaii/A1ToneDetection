import os
import numpy as np
from tqdm import tqdm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


COLORS = list(sns.color_palette())
COLORMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples",
             "YlOrBr", "PuRd", "Greys", "YlGn", "GnBu"]


def mk_saliency_boxplots(df, criterion, save_file=None, display=True, figsize=(30, 6)):
    tasks = list(df.task.unique())

    sns.set_style('white')
    fig, ax_arr = plt.subplots(1, 3, figsize=figsize, sharey=False)

    _ = ax_arr[0].axvspan(30, 60, facecolor='lightgrey', alpha=0.5, zorder=0)
    sns.boxplot(
        x="best_timepoint",
        y="task",
        data=df,
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops={"marker": "o",
                   "markerfacecolor": "bisque",
                   "markeredgecolor": "black",
                   "markersize": "12"},
        ax=ax_arr[0],
    )
    sns.boxplot(
        x="best_score",
        y="task",
        data=df,
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops={"marker": "o",
                   "markerfacecolor": "bisque",
                   "markeredgecolor": "black",
                   "markersize": "12"},
        ax=ax_arr[1],
    )
    sns.boxplot(
        x="percent_nonzero",
        y="task",
        data=df,
        hue="task",
        palette=list(COLORS),
        order=tasks,
        hue_order=tasks,
        width=0.5,
        whis=1.5,
        dodge=False,
        showmeans=True,
        meanprops={"marker": "o",
                   "markerfacecolor": "bisque",
                   "markeredgecolor": "black",
                   "markersize": "12", },
        ax=ax_arr[2],
    )

    ax_arr[0].set_title('selected timepoint (used for classification)', fontsize=16)
    ax_arr[1].set_title('best score at selected timepoints', fontsize=16)
    ax_arr[2].set_title('percentage of nonzero coefficients', fontsize=16)

    for i in range(3):
        ax_arr[i].get_legend().remove()

        if i == 0:
            ax_arr[i].axes.tick_params(axis='y', labelsize=15)
        else:
            ax_arr[i].set_yticks([])

    msg = "Results obtained using '{:s}' criterion and {:d} different seeds"
    msg = msg.format(criterion, len(df.seed.unique()))
    sup = fig.suptitle(msg, y=1.1, fontsize=25)

    if save_file is not None:
        os.makedirs(save_file, exist_ok=True)
        fig.savefig(save_file, dpi=100, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr


def mk_saliency_gridplot(df, mode="importances", save_file=None, display=True, figsize=(96, 12)):
    _allowed_modes = ["importances", "coeffs"]
    if mode not in _allowed_modes:
        raise RuntimeError("invalid mode entered.  allowed options: {}".format(_allowed_modes))

    names = list(df.name.unique())
    tasks = list(df.task.unique())

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(len(tasks), len(names), figsize=figsize, sharex='col', sharey='row')

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

    if save_file is not None:
        os.makedirs(save_file, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr


def mk_saliency_gridhist(df, save_file=None, display=True, figsize=(25, 20)):
    tasks = list(df.task.unique())

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(len(tasks), len(tasks), figsize=figsize, sharex='all', sharey='row')

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

    if save_file is not None:
        os.makedirs(save_file, exist_ok=True)
        fig.savefig(save_file, dpi=200, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr


def mk_saliency_gridscatter(df, mode="importances", save_file=None, display=True, figsize=(96, 12)):
    _allowed_modes = ["importances", "coeffs"]
    if mode not in _allowed_modes:
        raise RuntimeError("invalid mode entered.  allowed options: {}".format(_allowed_modes))

    names = list(df.name.unique())
    tasks = list(df.task.unique())

    sns.set_style('white')
    fig, ax_arr = plt.subplots(len(tasks), len(names), figsize=figsize, sharex='col', sharey='row')

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

            s = ax_arr[i, j].scatter(
                x=x[z != 0],
                y=y[z != 0],
                c=z[z != 0],
                cmap=COLORMAPS[i],
                s=120,
                vmin=0,
                vmax=vminmax,
                edgecolors='k',
                linewidths=0.4,
            )
            _ = ax_arr[i, j].scatter(
                x=x[z == 0],
                y=y[z == 0],
                color='w',
                s=50,
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

    if save_file is not None:
        os.makedirs(save_file, exist_ok=True)
        fig.savefig(save_file, dpi=600, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr
