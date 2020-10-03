import os
import h5py
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List, Tuple, Dict
from datetime import datetime
from os.path import join as pjoin

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, make_scorer


def run_classification_analysis(
        load_file: str,
        tasks: List[str] = None,
        seeds: List[int] = 42,
        save_results_dir: str = None,
        **kwargs: Dict[str, str],) -> Tuple[pd.DataFrame]:

    _allowed_clf_types = ['logreg', 'svm']

    if not isinstance(seeds, list):
        seeds = [seeds]

    if tasks is None:
        tasks = [
            'hit/miss', 'hit/falsealarm', 'hit/correctreject',
            'correctreject/falsealarm', 'miss/falsealarm', 'target/nontarget',
        ]

    classifier_args = {
        'clf_type': 'logreg',
        'penalty': 'l1',
        'C': [1.000, 0.500, 0.100, 0.050, 0.010, 0.005],
        'tol': 1e-4,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'max_iter': 1000,
        'n_jobs': None,
    }
    for k in classifier_args:
        if k in kwargs:
            classifier_args[k] = kwargs[k]

    base_cols = ['name', 'seed', 'task', 'reg_C', 'timepoint']

    cols = base_cols + ['cell_indx', 'coeffs', 'x', 'y']
    df_coeffs = pd.DataFrame(columns=cols)

    cols = base_cols + ['metric', 'score']
    df_performances = pd.DataFrame(columns=cols)

    xv_fold = 5
    msg = "[INFO] running analysis using: {:d}-fold xv, {:d} different seeds.\n[INFO] classifier options: {}"
    msg = msg.format(xv_fold, len(seeds), classifier_args)
    print(msg)

    logger = _setup_logger(classifier_args['clf_type'])

    h5_file = h5py.File(load_file, "r")
    pbar = tqdm(h5_file)
    clf_dict = {}
    for expt in pbar:
        if "rodger" not in expt:
            continue
        behavior = h5_file[expt]["behavior"]
        trial_info_grp = behavior["trial_info"]

        good_cells = np.array(behavior["good_cells"], dtype=int)
        xy = np.array(behavior["xy"], dtype=float)[good_cells]
        dff = np.array(behavior["dff"], dtype=float)[..., good_cells]
        nt, _, nc = dff.shape

        trial_info = {}
        for k, v in trial_info_grp.items():
            trial_info[k] = np.array(v, dtype=int)

        for random_state in seeds:
            random.seed(random_state)
            np.random.seed(random_state)

            for reg_c in classifier_args['C']:
                for task in tasks:
                    try:
                        pos_lbl, neg_lbl = task.split('/')
                        pos = trial_info[pos_lbl]
                        neg = trial_info[neg_lbl]
                    except KeyError:
                        msg = 'missing trials, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, reg_c, task)
                        logger.info(msg)
                        continue

                    include_trials = np.logical_or(pos, neg)
                    pos = pos[include_trials]
                    neg = neg[include_trials]

                    nb_pos_samples = sum(pos)
                    nb_neg_samples = sum(neg)

                    if nb_pos_samples == 0 or nb_neg_samples == 0:
                        msg = 'no samples found, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, reg_c, task)
                        logger.info(msg)
                        continue

                    assert np.all(np.array([pos, neg]).T.sum(-1) == 1), \
                        "pos and neg labels should be mutually exclusive"

                    # get xv indices
                    pos_vld_indxs = random.sample(range(nb_pos_samples), int(np.ceil(nb_pos_samples / xv_fold)))
                    neg_vld_indxs = random.sample(range(nb_neg_samples), int(np.ceil(nb_neg_samples / xv_fold)))

                    pos_vld_indxs = np.where(pos)[0][pos_vld_indxs]
                    neg_vld_indxs = np.where(neg)[0][neg_vld_indxs]

                    vld_indxs = list(pos_vld_indxs) + list(neg_vld_indxs)
                    trn_indxs = list(set(range(nb_pos_samples + nb_neg_samples)).difference(set(vld_indxs)))

                    x = dff[:, include_trials, :]

                    mcc_all = np.zeros(nt)
                    accuracy_all = np.zeros(nt)
                    f1_all = np.zeros(nt)
                    confidence_all = np.zeros(nt)

                    for time_point in range(nt):
                        x_trn, x_vld = x[time_point][trn_indxs], x[time_point][vld_indxs]
                        y_trn, y_vld = pos[trn_indxs], pos[vld_indxs]

                        try:
                            if classifier_args['clf_type'] == 'logreg':
                                clf = LogisticRegression(
                                    penalty=classifier_args['penalty'],
                                    C=reg_c,
                                    tol=classifier_args['tol'],
                                    solver=classifier_args['solver'],
                                    class_weight=classifier_args['class_weight'],
                                    n_jobs=classifier_args['n_jobs'],
                                    max_iter=classifier_args['max_iter'],
                                    random_state=random_state,
                                ).fit(x_trn, y_trn)
                            elif classifier_args['clf_type'] == 'svm':
                                clf = LinearSVC(
                                    penalty=classifier_args['penalty'],
                                    C=reg_c,
                                    tol=classifier_args['tol'],
                                    class_weight=classifier_args['class_weight'],
                                    dual=False,
                                    max_iter=classifier_args['max_iter'],
                                    random_state=random_state,
                                ).fit(x_trn, y_trn)
                            else:
                                msg = "invalid classifier type encountered: {:s}, valid options are: {}"
                                msg = msg.format(classifier_args['clf_type'], _allowed_clf_types)
                                raise ValueError(msg)
                            y_pred = clf.predict(x_vld)
                            k = "{}^{}^{}^{}^{}".format(expt, task, random_state, reg_c, time_point)
                            clf_dict[k] = (clf, x_vld, y_vld)
                        except ValueError:
                            msg = 'num trials too small, name = {:s}, seed = {:d}, C = {}, task = {}'
                            msg = msg.format(expt, random_state, reg_c, task)
                            logger.info(msg)
                            break

                        mcc_all[time_point] = matthews_corrcoef(y_vld, y_pred)
                        accuracy_all[time_point] = accuracy_score(y_vld, y_pred)
                        f1_all[time_point] = f1_score(y_vld, y_pred)

                        confidence = clf.decision_function(x_vld)
                        confidence_all[time_point] = sum(abs(confidence[y_vld == y_pred]))

                        data_dict = {
                            'name': [expt] * nc,
                            'seed': [random_state] * nc,
                            'task': [task] * nc,
                            'reg_C': [reg_c] * nc,
                            'timepoint': [time_point] * nc,
                            'cell_indx': range(nc),
                            'coeffs': clf.coef_.squeeze(),
                            'x': xy[:, 0],
                            'y': xy[:, 1],
                        }
                        df_coeffs = df_coeffs.append(pd.DataFrame(data=data_dict))

                    data_dict = {
                        'name': [expt] * nt * 4,
                        'seed': [random_state] * nt * 4,
                        'task': [task] * nt * 4,
                        'reg_C': [reg_c] * nt * 4,
                        'timepoint': np.tile(range(nt), 4),
                        'metric': ['mcc'] * nt + ['accuracy'] * nt + ['f1'] * nt + ['confidence'] * nt,
                        'score': np.concatenate([mcc_all, accuracy_all, f1_all, confidence_all]),
                    }
                    df_performances = df_performances.append(pd.DataFrame(data=data_dict))

    df_coeffs = reset_df(df_coeffs)
    df_performances = reset_df(df_performances)
    output = _porocess_results(df_performances, df_coeffs)
    df_performances, df_performances_filtered, df_coeffs_filtered = output
    df_coeffs_filtered = compute_feature_importances(df_coeffs_filtered, clf_dict)

    if save_results_dir is not None:
        name_template = "{:s}{:s}_{:s}"
        name_template = name_template.format(
            classifier_args['penalty'],
            classifier_args['solver'],
            datetime.now().strftime("[%Y_%m_%d_%H:%M]"),
        )
        save_dir = pjoin(save_results_dir, classifier_args['clf_type'], name_template)
        os.makedirs(save_dir, exist_ok=True)

        file_name = "coeffs_" + name_template + '.df'
        df_coeffs.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

        file_name = "coeffs_filtered_" + name_template + '.df'
        df_coeffs_filtered.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

        file_name = "performances_" + name_template + '.df'
        df_performances.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

        file_name = "performances_filtered_" + name_template + '.df'
        df_performances_filtered.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

        file_name = "classifiers_" + name_template + '.npy'
        np.save(pjoin(save_dir, file_name), clf_dict)
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

    h5_file.close()
    return df_coeffs, df_coeffs_filtered, df_performances, df_performances_filtered


def compute_feature_importances(df: pd.DataFrame, classifiers: dict) -> pd.DataFrame:
    print('[PROGRESS] computing feature importances')

    empty_df = pd.DataFrame(columns=['importances'], index=df.index)
    new_df = pd.concat([df, empty_df], axis=1)

    for k, (clf, x_vld, y_vld) in classifiers.items():
        name, task, random_state, reg_c, time_point = k.split('^')
        random_state, time_point = int(random_state), int(time_point)
        reg_c = float(reg_c)

        cond = (new_df.name == name) & \
               (new_df.task == task) & \
               (new_df.seed == random_state)

        _c = new_df.loc[cond, 'reg_C'].unique()
        _t = new_df.loc[cond, 'timepoint'].unique()
        assert len(_c) == len(_t) == 1, "filtered df must have only one unique selected timepoint or reg per expt/task"

        if not (_c[0] == reg_c and _t[0] == time_point):
            continue

        importance_result = permutation_importance(
            estimator=clf,
            X=x_vld,
            y=y_vld,
            n_repeats=100,
            n_jobs=-1,
            scoring=make_scorer(matthews_corrcoef),
            random_state=random_state,
        )

        new_df.loc[cond, 'importances'] = pd.Series(
            importance_result['importances_mean'], index=new_df.loc[cond].index)

    return reset_df(new_df)


def reset_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df = df.apply(pd.to_numeric, downcast="integer", errors="ignore")
    return df


def _porocess_results(df_performances: pd.DataFrame, df_coeffs: pd.DataFrame) -> tuple:
    output = ()
    df_performances = _normalize_confidence_score(df_performances)
    df_performances = _detect_best_reg_timepoint(df_performances)
    output += (df_performances,)
    filtered_dfs = _filter_df([df_performances, df_coeffs], df_performances)
    output += tuple(filtered_dfs)
    return output


def _filter_df(dfs: List[pd.DataFrame], augmented_df: pd.DataFrame) -> List[pd.DataFrame]:
    filtered_dfs = []
    for df in dfs:
        filtered_df = deepcopy(df)

        if 'best_reg' in filtered_df.columns:
            print('[PROGRESS] filtering df_performances using best reg / timepoints')
            cond = (filtered_df.reg_C != augmented_df.best_reg) | \
                   (filtered_df.timepoint != augmented_df.best_timepoint)
            filtered_df.drop(filtered_df.loc[cond].index, inplace=True)

        else:
            print('[PROGRESS] filtering df_coeffs using best reg / timepoints')
            names = list(df.name.unique())
            tasks = list(df.task.unique())

            for task in tasks:
                for name in names:
                    selected_df = augmented_df.loc[(augmented_df.name == name) & (augmented_df.task == task)]
                    best_reg = selected_df.best_reg.unique()[0]
                    best_timepoint = selected_df.best_timepoint.unique()[0]

                    cond = (filtered_df.name == name) & \
                           (filtered_df.task == task) & \
                           ((filtered_df.reg_C != best_reg) | (filtered_df.timepoint != best_timepoint))
                    filtered_df.drop(filtered_df.loc[cond].index, inplace=True)

        filtered_dfs.append(reset_df(filtered_df))

    return filtered_dfs


def _detect_best_reg_timepoint(df: pd.DataFrame) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=['best_reg', 'best_timepoint'], index=df.index)
    new_df = pd.concat([df, empty_df], axis=1)

    names = list(df.name.unique())
    tasks = list(df.task.unique())
    reg_cs = list(df.reg_C.unique())

    nb_c = len(reg_cs)
    nb_seeds = len(df.seed.unique())
    nt = len(df.timepoint.unique())

    print('[PROGRESS] detecting best reg / timepoints for df_performances')

    for i, task in tqdm(enumerate(tasks), total=len(tasks)):
        for j, name in enumerate(names):
            # select best reg
            cond = (df.name == name) & (df.task == task)
            selected_df = df.loc[cond]
            if not len(selected_df):
                print('missing data, name = {:s}, task = {:s}, moving on . . .'.format(name, task))
                continue

            scores = selected_df.loc[selected_df.metric.isin(['mcc', 'accuracy', 'f1']), 'score'].to_numpy()
            scores = scores.reshape(nb_seeds, nb_c, 3, nt)
            mean_scores = scores.mean(2).mean(0)

            confidences = selected_df.loc[selected_df.metric == 'confidence', 'score'].to_numpy()
            confidences = confidences.reshape(nb_seeds, nb_c, nt)
            mean_confidences = confidences.mean(0)

            max_score = np.max(mean_scores)

            if np.sum(mean_scores == max_score) == 1:
                a, b = np.unravel_index(np.argmax(mean_scores), mean_scores.shape)
            else:
                only_max_scores = mean_scores.copy()
                only_max_scores[mean_scores < max_score] = 0

                a = max(np.where(only_max_scores)[0])
                max_confidences = mean_confidences * (mean_scores == max_score)
                b = np.argmax(max_confidences[a])

            assert mean_scores[a, b] == np.max(mean_scores), "must select max score"

            new_df.loc[cond, 'best_reg'] = pd.Series([reg_cs[a]] * len(selected_df), index=selected_df.index)
            new_df.loc[cond, 'best_timepoint'] = pd.Series([b] * len(selected_df), index=selected_df.index)

    return reset_df(new_df)


def _normalize_confidence_score(df: pd.DataFrame) -> pd.DataFrame:
    names = list(df.name.unique())
    tasks = list(df.task.unique())
    reg_cs = list(df.reg_C.unique())

    print('[PROGRESS] normalizing confidence scores for df_performance')

    for name in names:
        for task in tasks:
            for reg_c in reg_cs:
                cond = (df.name == name) & \
                       (df.task == task) & \
                       (df.reg_C == reg_c) & \
                       (df.metric == 'confidence')
                selected_df = df.loc[cond]

                confidence_scores = selected_df.score
                max_confidence = max(confidence_scores)
                df.loc[cond, 'score'] = confidence_scores / max_confidence

    return df


def _setup_logger(msg: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    _dir = './log'
    os.makedirs(_dir, exist_ok=True)

    _file = "{:s}_{:s}.log".format(msg, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))
    logger_name = pjoin(_dir, _file)
    file_handler = logging.FileHandler(logger_name)
    print("[PROGRESS] logger '{:s}' created".format(logger_name))

    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
