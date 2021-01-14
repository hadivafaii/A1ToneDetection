import sys
import argparse
from .lda_analysis import LDA

sys.path.append('..')
from utils.animation import mk_coarse_grained_plot
from utils.generic_utils import *
from utils.plot_functions import *


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cm",
        help="comment describing the fit to use",
        type=str,
    )
    parser.add_argument(
        "--clf_type",
        help="classifier type, choices: {'logreg', 'svm'}",
        type=str,
        choices={'logreg', 'svm'},
        default='svm',
    )
    parser.add_argument(
        "--nb_std",
        help="outlier removal threshold",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--verbose",
        help="verbosity",
        action="store_true",
    )
    parser.add_argument(
        "--base_dir",
        help="base dir where project is saved",
        type=str,
        default='Documents/PROJECTS/Kanold',
    )

    return parser.parse_args()


def main():
    args = _setup_args()

    base_dir = pjoin(os.environ['HOME'], args.base_dir)
    processed_dir = pjoin(base_dir, 'python_processed')
    clf_results_dir = pjoin(base_dir, 'results', args.clf_type, args.cm)
    lda_4way_results_dir = pjoin(base_dir, 'results', 'lda', '4way')
    lda_stimfreq_results_dir = pjoin(base_dir, 'results', 'lda', 'stimfreq')
    h_load_file = pjoin(processed_dir, "organized_nb_std={:d}.h5".format(args.nb_std))
    processed_load_dir = pjoin(processed_dir, "processed_nb_std={:d}".format(args.nb_std))

    if args.verbose:
        print("[INFO] files found:\n{}".format(os.listdir(clf_results_dir)))
    df_all = load_dfs(clf_results_dir)

    names = df_all['performances_filtered'].name.unique().tolist()
    tasks = get_tasks()
    downsample_sizes = [16, 8, 4, 2]

    pbar = tqdm(names, disable=not args.verbose, dynamic_ncols=True)
    for name in pbar:
        pbar.set_description(name)

        # page 0: avg traces
        processed_df = pd.read_pickle(pjoin(processed_load_dir, '{}.df'.format(name)))
        fig0, _, sup0 = mk_data_summary_plot(
            df=processed_df,
            save_file=None,
            display=False,
            figsize=(20, 20),
            dpi=200,
        )

        # page 1: reg selection
        df_p = df_all['performances'].loc[df_all['performances'].name == name]
        fig1, _, sup1 = mk_reg_selection_plot(
            performances=df_p,
            criterion='mcc',
            save_file=None,
            display=False,
            figsize=(50, 8),
            dpi=300,
        )

        # page 2: coeffs and importances
        df_cf = df_all['coeffs_filtered'].loc[df_all['coeffs_filtered'].name == name]
        fig2, _, sup2 = mk_coeffs_importances_plot(
            coeffs_filtered=df_cf,
            save_file=None,
            display=False,
            figsize=(55, 11),
            dpi=300,
        )

        # page 3: trajs (4way)
        fig3, _ = mk_trajectory_plot(
            load_dir=lda_4way_results_dir,
            global_stats=False,
            shuffled=False,
            name=name,
            dim=3,
            save_file=None,
            display=False,
            figsize=(18, 21),
            dpi=400,
        )

        # page 4: trajs (stimfreq)
        fig4, _ = mk_trajectory_plot(
            load_dir=lda_stimfreq_results_dir,
            global_stats=False,
            shuffled=False,
            name=name,
            dim=3,
            save_file=None,
            display=False,
            figsize=(18, 21),
            dpi=400,
        )

        figs = [fig0, fig1, fig2, fig3, fig4]
        sups = [sup0, sup1, sup2, None, None]

        # last 10 pages: coarse-grained for each task
        for task in tqdm(tasks, disable=not args.verbose, leave=False):
            cond = (df_all['performances_filtered'].name == name) & (df_all['performances_filtered'].task == task)
            if not sum(cond):
                continue
            timepoint = df_all['performances_filtered'].loc[cond, 'best_timepoint'].unique().item()
            _fig, _sup, _, _ = mk_coarse_grained_plot(
                df_all=df_all,
                h_load_file=h_load_file,
                name=name,
                task=task,
                timepoint=timepoint,
                downsample_sizes=downsample_sizes,
                save_file=None,
                display=False,
                figsize=(30, 16.5),
                dpi=400,
            )
            figs.append(_fig)
            sups.append(_sup)

        save_dir = pjoin(clf_results_dir, 'individual_results', name)
        save_file = pjoin(save_dir, "summary_{:s}.pdf".format(name))
        save_fig(figs, sups, save_file, display=False, multi=True)

    print("[PROGRESS] done.\n")


if __name__ == "__main__":
    main()
