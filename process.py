import os
import numpy as np
from os.path import join as pjoin
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def get_bad_trials(data):
    if not isinstance(data, list):
        data = [data]

    bad_trials = []
    for d in data:
        dff = d['dff']
        cells_norm = np.linalg.norm(dff, axis=0, ord=2)
        nan_norm = np.isnan(cells_norm)
        bad_trials.append(np.where(np.all(nan_norm, axis=1))[0])

    return bad_trials


def get_good_indxs(data, nb_std=5):
    if not isinstance(data, list):
        data = [data]

    output = []
    for d in data:
        cells_norm = np.linalg.norm(d['dff'], axis=0, ord=2).mean(0)
        nonnan = ~np.isnan(cells_norm)
        nonnan_bright_cells = np.logical_and(d['bright_cells'], nonnan)
        nonnan_bright_indxs = np.where(nonnan_bright_cells == 1)[0]
        x = cells_norm[nonnan_bright_indxs]
        outlier_indxs = np.where(x - x.mean() > nb_std * x.std())[0]
        good_indxs = np.delete(nonnan_bright_indxs, outlier_indxs)
        output.append((good_indxs, outlier_indxs, nonnan_bright_indxs))

    return output


def plot_outlier_removal(data, nb_std, save_dir="outlier_removal"):
    good_indxs, outlier_indxs, nonnan_bright_indxs = get_good_indxs(data, nb_std=nb_std)[0]
    cells_norm = np.linalg.norm(data[0]['dff'], axis=0, ord=2).mean(0)

    name = "{:s}_{:s}".format(data[0]["name"], data[0]["date"]).lower()
    msg = "--- Removing outliers from experiment '{:s}' ---\n\
    Good cells don't contain nan in any trial, but also:\n\
    \n 1) bright cells:  num bright = {:d}, tot = {:d}, => {:.1f}{:s}\
    \n 2) have norm that is within {:d} std of mean norm:\n\
    Outliers count: {:d}  == > {:.1f}{:s} of cells removed for this reason."
    msg = msg.format(name,
                     sum(data[0]['bright_cells']), len(data[0]['bright_cells']),
                     sum(data[0]['bright_cells']) / len(data[0]['bright_cells']) * 100, '%',
                     nb_std, len(outlier_indxs), len(outlier_indxs) / len(nonnan_bright_indxs) * 100, '%', )

    fig, ax_arr = plt.subplots(1, 3, figsize=(18, 3), sharey='True', dpi=100)
    sup = fig.suptitle(msg, y=1.4, fontsize=15, )

    ax_arr[0].plot(cells_norm)
    ax_arr[0].set_ylabel("Norm")
    ax_arr[0].set_xlabel("All Cells. Count = {:d}".format(len(cells_norm)))

    ax_arr[1].plot(cells_norm[nonnan_bright_indxs])
    ax_arr[1].set_xlabel("Good Cells w/ Outliers. Count = {:d}".format(len(nonnan_bright_indxs)))

    ax_arr[2].plot(cells_norm[good_indxs])
    ax_arr[2].set_xlabel("Good Cells, w/o outliers. Count = {:d}".format(len(good_indxs)))

    save_folder = pjoin(save_dir, "nb_std={:d}".format(nb_std))
    os.makedirs(save_folder, exist_ok=True)
    save_file = pjoin(save_folder, "{:s}.pdf".format(name))
    fig.savefig(save_file, dpi=fig.dpi, bbox_inches='tight', bbox_extra_artists=[sup])
    plt.close()

    return len(outlier_indxs), len(nonnan_bright_indxs)
