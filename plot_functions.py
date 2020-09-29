import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


COLORS = list(sns.color_palette())
COLORMAPS = ["Blues", "Oranges", "Greens", "Reds", "Purples",
             "YlOrBr", "PuRd", "Greys", "YlGn", "GnBu"]


def mk_saliency_boxplots(df, criterion, save_file=None, display=True, figsize=(30, 8)):
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
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
        fig.savefig(save_file, dpi=100, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr


def mk_saliency_gridplot(df, mode="importances", save_file=None, display=True, figsize=(96, 16)):
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
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
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
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
        fig.savefig(save_file, dpi=200, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr


def mk_saliency_gridscatter(df, mode="importances", save_file=None, display=True, figsize=(140, 20)):
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

    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
        fig.savefig(save_file, dpi=300, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr


def mk_saliency_hist(df, mode="importances", save_file=None, display=True, figsize=(20, 16)):
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
                print('some seeds were not accepted, name = {:s}, task = {:s}, moving on . . .'.format(name, task))
                continue
            num_nonzeros = (z != 0).sum(-1)

            assert not sum(num_nonzeros > nc), "num nonzero neurons must be less than total num cells"

            data_dict = {
                'task': [task] * nb_seeds,
                'num_nonzero': num_nonzeros,
                'percent_nonzero': num_nonzeros / nc * 100,
            }
            df_to_plot = df_to_plot.append(pd.DataFrame(data=data_dict))

    df_to_plot = df_to_plot.reset_index(drop=True)
    df_to_plot = df_to_plot.apply(pd.to_numeric, downcast="integer", errors="ignore")

    nrows, ncols = 4, 4
    assert nrows * ncols >= 2 * len(tasks)

    sns.set_style('whitegrid')
    fig, ax_arr = plt.subplots(nrows, ncols, figsize=figsize, sharey='all')

    tick_spacing1 = 5
    nb_ticks1 = int(np.ceil(max(df_to_plot.num_nonzero.to_numpy()) / tick_spacing1))
    xticks1 = range(0, nb_ticks1 * tick_spacing1 + 1, tick_spacing1)

    tick_spacing2 = 10
    xticks2 = range(0, 100 + 1, tick_spacing2)

    for idx, task in enumerate(tasks):
        i, j = idx // 4, idx % 4

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

    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
        fig.savefig(save_file, dpi=100, bbox_inches='tight', bbox_extra_artists=[sup])

    if display:
        plt.show(fig)
    else:
        plt.close(fig)

    return fig, ax_arr, df_to_plot
