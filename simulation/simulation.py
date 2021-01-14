import rcca
from utils.generic_utils import *
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm
from sklearn.cross_decomposition import PLSCanonical

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# --------------------------- simulation functions ---------------------------
# ----------------------------------------------------------------------------

def _get_noise(x_list, sigma: float = 0.1, rng: np.random.RandomState = None):
    rng = np.random.RandomState(42) if rng is None else rng
    noise = []
    for x in x_list:
        noise.append(rng.randn(*x.shape) * sigma * np.std(x))
    return noise


def generate_source_signal(
        n_samples: int,
        three_d: bool = False,
        angle_spacing: float = 1.0,
        magnitude_range: range = None,
        rng: np.random.RandomState = None, ):

    rng = np.random.RandomState(42) if rng is None else rng

    def _generate_data():
        phis = rng.choice(np.arange(-180, 180, angle_spacing), size=n_samples, replace=True)
        if three_d:
            thetas = rng.choice(np.arange(0, 180, angle_spacing), size=n_samples, replace=True)
        else:
            thetas = [90] * n_samples
        thetas, phis = tuple(map(np.deg2rad, [thetas, phis]))

        if magnitude_range is not None:
            magnitudes = rng.choice(magnitude_range, size=n_samples, replace=True)
        else:
            magnitudes = np.ones(n_samples, dtype=float).reshape(-1, 1)
        z = np.array([
            np.sin(thetas) * np.cos(phis),
            np.sin(thetas) * np.sin(phis),
            np.cos(thetas),
        ]).T
        z *= magnitudes

        output = {
            'z': z if three_d else z[:, :2],
            'phi': phis,
            'theta': thetas,
            'r': magnitudes,
        }
        return output

    train, test = _generate_data(), _generate_data()
    return train, test


def create_pls_simulation(
        train: dict = None,
        test: dict = None,
        three_d: bool = False,
        angle_spacing: float = 1.0,
        magnitude_range: range = None,
        n_samples: int = int(1e3),
        dim_x: int = 8,
        dim_y: int = 8,
        sigma: float = 0.2,
        orthogonal: bool = False,
        normal: bool = True,
        seed: int = 42, ):

    rng = np.random.RandomState(seed)
    if train is None or test is None:
        train, test = generate_source_signal(
            n_samples=n_samples,
            three_d=three_d,
            angle_spacing=angle_spacing,
            magnitude_range=magnitude_range,
            rng=rng,
        )

    dim_z = 3 if three_d else 2

    p = orthonormalize(u=rng.randn(dim_x, dim_z), orthogonal=orthogonal, normal=normal)
    q = orthonormalize(u=rng.randn(dim_y, dim_z), orthogonal=orthogonal, normal=normal)

    x_train, y_train = train['z'] @ p.T, train['z'] @ q.T
    x_test, y_test = test['z'] @ p.T, test['z'] @ q.T
    e_train, f_train, e_test, f_test = _get_noise([x_train, y_train, x_test, y_test], sigma, rng)
    x_train += e_train
    y_train += f_train
    x_test += e_test
    y_test += f_test

    metadata = {
        'n_samples': n_samples,
        'dim_z': dim_z,
        'dim_x': dim_x,
        'dim_y': dim_y,
        'sigma': sigma,
        'three_d': three_d,
        'angle_spacing': angle_spacing,
        'magnitude_range': magnitude_range,
        'orthogonal': orthogonal,
        'normal': normal,
        'seed': seed,
    }
    output = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'z_train': train['z'],
        'z_test': test['z'],
        'phi_train': train['phi'],
        'phi_test': test['phi'],
        'theta_train': train['theta'],
        'theta_test': test['theta'],
        'r_train': train['r'],
        'r_test': test['r'],
        'P': p,
        'Q': q,
        'e_train': e_train,
        'f_train': f_train,
        'e_test': e_test,
        'f_test': f_test,
        'metadata': metadata,
    }
    return output


def create_cca_simulation(
        train=None,
        test=None,
        num_expts: int = 20,
        min_num_cells: int = 16,
        max_num_cells: int = 128,
        three_d: bool = False,
        angle_spacing: float = 1.0,
        magnitude_range: range = None,
        n_samples: int = int(1e3),
        sigma: float = 0.1,
        normal: bool = True,
        seed: int = 42, ):
    rng = np.random.RandomState(seed)
    if train is None or test is None:
        train, test = generate_source_signal(
            n_samples=n_samples,
            three_d=three_d,
            angle_spacing=angle_spacing,
            magnitude_range=magnitude_range,
            rng=rng,
        )

    dim_z = 3 if three_d else 2
    dims = rng.choice(range(min_num_cells, max_num_cells), size=num_expts, replace=True)
    proj_mats = [rng.randn(d, dim_z) if not normal else normalize(rng.randn(d, dim_z), axis=0) for d in dims]

    x_train = [train['z'] @ p.T for p, d in zip(proj_mats, dims)]
    x_test = [test['z'] @ p.T for p, d in zip(proj_mats, dims)]
    x_train = [x + e for x, e in zip(x_train, _get_noise(x_train, sigma, rng))]
    x_test = [x + f for x, f in zip(x_test, _get_noise(x_test, sigma, rng))]

    metadata = {
        'n_samples': n_samples,
        'num_expts': num_expts,
        'min_num_cells': min_num_cells,
        'max_num_cells': max_num_cells,
        'dim_z': dim_z,
        'sigma': sigma,
        'three_d': three_d,
        'angle_spacing': angle_spacing,
        'magnitude_range': magnitude_range,
        'normal': normal,
        'seed': seed,
    }
    output = {
        'x_train': x_train,
        'x_test': x_test,
        'z_train': train['z'],
        'z_test': test['z'],
        'phi_train': train['phi'],
        'phi_test': test['phi'],
        'theta_train': train['theta'],
        'theta_test': test['theta'],
        'r_train': train['r'],
        'r_test': test['r'],
        'proj_mats': proj_mats,
        'metadata': metadata,
    }
    return output


# --------------------------- plotting functions ---------------------------
# --------------------------------------------------------------------------

def visualize_pls_results(pls, sim, verbose: bool = False):
    _print_sim_info(sim, verbose)
    avg_latent_corrs, pred_r2, pred_r = _plot_latent_retreival(pls, sim, verbose)
    proj_cos_sim = _plot_similarity_matrix(pls, sim, verbose)
    _print_results(pls, sim, verbose)

    results = {
        'latent_corr': np.round(avg_latent_corrs, decimals=3),
        'proj_cos_sim': np.round(proj_cos_sim, decimals=3),
        'pred_r2': np.round(pred_r2, decimals=3),
        'pred_r': np.round(pred_r, decimals=3),
    }
    return results


def visualize_cca_results(cca, sim, verbose: bool = False):
    testcorrs = cca.validate(sim['x_test'])
    pred_r = np.mean([item.mean() for item in testcorrs])

    proj_cos_sim = []
    for expt_id in range(sim['metadata']['num_expts']):
        cos_sim = np.zeros((sim['metadata']['dim_z'], sim['metadata']['dim_z']))
        for i in range(sim['metadata']['dim_z']):
            for j in range(sim['metadata']['dim_z']):
                cos_sim[i, j] = cos_similarity(sim['proj_mats'][expt_id][:, i], cca.ws[expt_id][:, j])
        _corrs = np.max(np.abs(cos_sim), axis=0)
        proj_cos_sim.append(np.mean(_corrs))
    proj_cos_sim = np.mean(proj_cos_sim)

    latent_corr = []
    for expt_id in range(sim['metadata']['num_expts']):
        _corrs = np.zeros((sim['metadata']['dim_z'], sim['metadata']['dim_z']))
        for i in range(sim['metadata']['dim_z']):
            for j in range(sim['metadata']['dim_z']):
                _corrs[i, j] = pearsonr(sim['z_train'][:, i], cca.comps[expt_id][:, j])[0]
        _corrs = np.max(np.abs(_corrs), axis=0)
        latent_corr.append(np.mean(_corrs))
    latent_corr = np.mean(latent_corr)

    results = {
        'latent_corr': np.round(latent_corr, decimals=3),
        'proj_cos_sim': np.round(proj_cos_sim, decimals=3),
        'pred_r': np.round(pred_r, decimals=3),
    }
    return results


def _print_sim_info(sim, verbose):
    if not verbose:
        return

    metadata = sim['metadata']
    msg = 'Simulation info:\n\n'
    msg += 'num samples:\t{:.0e},\nnoise sigma:\t{:.1f},\ndim Z:\t{:d},\ndim X:\t{:d},\ndim Y:\t{:d}'
    msg = msg.format(
        metadata['n_samples'],
        metadata['sigma'],
        metadata['dim_z'],
        metadata['dim_x'],
        metadata['dim_y'], )
    print(msg)
    print('\n\n')


def _plot_latent_retreival(pls, sim, verbose):
    T, U = pls.transform(sim['x_test'], sim['y_test'])
    y_pred = pls.predict(sim['x_test'])
    pred_r2 = r2_score(sim['y_test'], y_pred, multioutput='raw_values') * 100
    pred_r2 = np.maximum(0.0, pred_r2).mean()

    test_corrs = []
    for j in range(y_pred.shape[1]):
        r, p_val = pearsonr(y_pred[:, j], sim['y_test'][:, j])
        test_corrs.append(r)
    pred_r = np.mean(test_corrs)

    dim_z = sim['metadata']['dim_z']
    x_indxs, y_indxs = {}, {}
    for i in range(dim_z):
        current_x_idx = i
        current_y_idx = i
        current_x_corr = 0.0
        current_y_corr = 0.0
        for j in range(dim_z):
            _corr_x = abs(pearsonr(sim['z_test'][:, i], T[:, j])[0])
            _corr_y = abs(pearsonr(sim['z_test'][:, i], U[:, j])[0])

            if _corr_x > current_x_corr:
                current_x_corr = _corr_x
                current_x_idx = j
            if _corr_y > current_y_corr:
                current_y_corr = _corr_y
                current_y_idx = j

        x_indxs[i] = current_x_idx
        y_indxs[i] = current_y_idx

    for i in range(dim_z):
        if pearsonr(T[:, x_indxs[i]], sim['z_test'][:, i])[0] < 0.0:
            T[:, x_indxs[i]] *= -1
        if pearsonr(U[:, y_indxs[i]], sim['z_test'][:, i])[0] < 0.0:
            U[:, y_indxs[i]] *= -1

    corrs_x = [pearsonr(T[:, x_indxs[i]], sim['z_test'][:, i])[0] for i in range(dim_z)]
    corrs_y = [pearsonr(U[:, y_indxs[i]], sim['z_test'][:, i])[0] for i in range(dim_z)]

    if verbose:
        sns.set_style('whitegrid')
        plt.figure(figsize=(6 * dim_z, 6))

        for i in range(dim_z):
            plt.subplot(1, dim_z, i+1)
            plt.plot(T[:30, x_indxs[i]], label='X')
            plt.plot(U[:30, y_indxs[i]], label='Y')
            plt.plot(sim['z_test'][:30, i], 'k--', lw=2, label='true')
            plt.xlabel('samples')
            plt.title('dim # {:d}'.format(i+1))
            plt.legend()

        msg = 'correlation with source signal:\n'
        if dim_z == 2:
            msg += 'X  -->   1st dim:  {:.2f}  ,  2nd dim:  {:.2f}\n'
            msg += 'Y  -->   1st dim:  {:.2f}  ,  2nd dim:  {:.2f}\n\n'
        else:
            msg += 'X  -->   1st dim:  {:.2f}  ,  2nd dim:  {:.2f}  ,  3rd dim:  {:.2f}\n'
            msg += 'Y  -->   1st dim:  {:.2f}  ,  2nd dim:  {:.2f}  ,  3rd dim:  {:.2f}\n\n'
        msg += 'percent variance of Y explained (predicted useing X):   {:.0f} {:s}\n\n'
        msg += 'figure showing latent and retreived signals over a few samples\n'
        print_data = corrs_x + corrs_y + [pred_r2, '%']
        plt.suptitle(msg.format(*print_data), fontsize=15)
        plt.tight_layout()
        plt.show()
        print('\n\n')

    return np.mean(corrs_x + corrs_y), pred_r2, pred_r


def _plot_similarity_matrix(pls, sim, verbose):
    dim_z = sim['metadata']['dim_z']
    cos_sim_x = np.zeros((dim_z, dim_z))
    cos_sim_y = np.zeros((dim_z, dim_z))

    for i in range(dim_z):
        for j in range(dim_z):
            cos_sim_x[i, j] = cos_similarity(sim['P'][:, i], pls.x_loadings_[:, j])
            cos_sim_y[i, j] = cos_similarity(sim['Q'][:, i], pls.y_loadings_[:, j])

    if verbose:
        sns.set_style('white')
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.imshow(cos_sim_x, cmap='bwr', vmin=-1, vmax=1)
        plt.xticks(range(dim_z))
        plt.yticks(range(dim_z))
        plt.title('X')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(cos_sim_y, cmap='bwr', vmin=-1, vmax=1)
        plt.xticks(range(dim_z))
        plt.yticks(range(dim_z))
        plt.title('Y')
        plt.colorbar()

        msg = 'cosine similarity between true and retreived projection vectors\n'
        msg += 'rows correspond to true, and columns correspond to retreived\n'
        msg += 'i.e. element $ij$ means cos sim between $P_i$ and $j$-th retreived loading vector'
        plt.suptitle(msg, fontsize=15)
        plt.tight_layout()
        plt.show()

        print('matrix values:\n\nX:\n{}\n\nY:\n{}'.format(cos_sim_x, cos_sim_y))
        print('\n\n')

    avg_corr_x = np.max(np.abs(cos_sim_x), axis=0).mean()
    avg_corr_y = np.max(np.abs(cos_sim_y), axis=0).mean()
    return np.mean([avg_corr_x, avg_corr_y])


def _print_results(pls, sim, verbose):
    if not verbose:
        return
    msg = 'Angle between true projection vectors:\n'
    msg += 'X:  {:.0f} degrees,   Y:  {:.0f} degrees\n\n'
    msg += 'Angle between retreived loading vectors:\n'
    msg += 'X:  {:.0f} degrees,   Y:  {:.0f} degrees\n\n'
    msg = msg.format(
        np.rad2deg(np.arccos(cos_similarity(sim['P'][:, 0], sim['P'][:, 1]))),
        np.rad2deg(np.arccos(cos_similarity(sim['Q'][:, 0], sim['Q'][:, 1]))),
        np.rad2deg(np.arccos(cos_similarity(pls.x_loadings_[:, 0], pls.x_loadings_[:, 1]))),
        np.rad2deg(np.arccos(cos_similarity(pls.y_loadings_[:, 0], pls.y_loadings_[:, 1]))),
    )
    print(msg)


# --------------------------- loop functions ---------------------------
# --------------------------------------------------------------------------

def run_pls_loop(n_seeds: int = 10, equal_dims: bool = False, **kwargs):
    default_args = {
        'sample_sizes': [int(10 ** i) for i in range(2, 5)],
        'seeds': [int(2 ** i) for i in range(n_seeds)],
        'sigmas': np.linspace(0, 5, num=11),
        'orthogonal': [False],
        'normal': [True],
        'three_d': [False, True],
        'dims_x': np.logspace(1, 4, num=4, base=4, dtype=int),
        'dims_y': np.logspace(1, 4, num=4, base=4, dtype=int),
    }
    for k in default_args:
        if k in kwargs:
            default_args[k] = list(kwargs[k]) if isinstance(kwargs[k], (list, tuple, np.ndarray)) else [kwargs[k]]

    df = pd.DataFrame()
    for n_samples in tqdm(default_args['sample_sizes']):
        for three_d in default_args['three_d']:
            train, test = generate_source_signal(n_samples=n_samples, three_d=three_d)
            for seed in tqdm(default_args['seeds'], leave=False):
                for orthogonal in default_args['orthogonal']:
                    for normal in default_args['normal']:
                        for sigma in default_args['sigmas']:
                            for dim_x in default_args['dims_x']:
                                for dim_y in [dim_x] if equal_dims else default_args['dims_y']:
                                    # create sim
                                    sim = create_pls_simulation(
                                        train=train,
                                        test=test,
                                        n_samples=n_samples,
                                        three_d=three_d,
                                        angle_spacing=1.0,
                                        magnitude_range=None,
                                        dim_x=dim_x,
                                        dim_y=dim_y,
                                        sigma=sigma,
                                        orthogonal=orthogonal,
                                        normal=normal,
                                        seed=seed,
                                    )
                                    # fit PLS
                                    pls = PLSCanonical(
                                        n_components=sim['metadata']['dim_z'],
                                        scale=True,
                                        algorithm='svd',
                                        max_iter=int(1e9),
                                        tol=1e-15,
                                    ).fit(sim['x_train'], sim['y_train'])
                                    # get results
                                    results = visualize_pls_results(pls, sim, verbose=False)
                                    results.update({
                                        'n_samples': n_samples,
                                        'three_d': three_d,
                                        'seed': seed,
                                        'orthogonal': orthogonal,
                                        'normal': normal,
                                        'sigma': sigma,
                                        'dim_x': dim_x,
                                        'dim_y': dim_y,
                                    })
                                    results = {k: [v] for k, v in results.items()}
                                    df = pd.concat([df, pd.DataFrame.from_dict(results)])
    return reset_df(df), default_args


def run_cca_loop(n_seeds: int = 10, reg: float = 0.1, **kwargs):
    default_args = {
        'sample_sizes': [int(10 ** i) for i in range(2, 5)],
        'seeds': [int(2 ** i) for i in range(n_seeds)],
        'sigmas': np.linspace(0, 5, num=11),
        'normal': [True],
        'three_d': [False],
        'num_expts': [10, 20, 30],
        'min_ncs': [int(2**i) for i in range(1, 7)],
        'max_ncs': [int(4**i) for i in range(1, 7)],
    }
    for k in default_args:
        if k in kwargs:
            default_args[k] = list(kwargs[k]) if isinstance(kwargs[k], (list, tuple, np.ndarray)) else [kwargs[k]]

    df = pd.DataFrame()
    for n_samples in tqdm(default_args['sample_sizes']):
        for three_d in default_args['three_d']:
            train, test = generate_source_signal(n_samples=n_samples, three_d=three_d)
            for seed in tqdm(default_args['seeds'], leave=False):
                for normal in default_args['normal']:
                    for sigma in default_args['sigmas']:
                        for num_expts in default_args['num_expts']:
                            for min_nc in default_args['min_ncs']:
                                for max_nc in [item for item in default_args['max_ncs'] if item > min_nc]:
                                    # create sim
                                    sim = create_cca_simulation(
                                        train=train,
                                        test=test,
                                        num_expts=num_expts,
                                        min_num_cells=min_nc,
                                        max_num_cells=max_nc,
                                        n_samples=n_samples,
                                        three_d=three_d,
                                        angle_spacing=1.0,
                                        magnitude_range=None,
                                        sigma=sigma,
                                        normal=normal,
                                        seed=seed,
                                    )
                                    # fit PLS
                                    cca = rcca.CCA(
                                        kernelcca=True,
                                        ktype='linear',
                                        reg=reg,
                                        numCC=sim['metadata']['dim_z'],
                                        verbose=False,
                                    )
                                    cca.train(sim['x_train'])
                                    # get results
                                    results = visualize_cca_results(cca, sim, verbose=False)
                                    results.update({
                                        'n_samples': n_samples,
                                        'three_d': three_d,
                                        'seed': seed,
                                        'normal': normal,
                                        'sigma': sigma,
                                        'num_expts': num_expts,
                                        'min_nc': min_nc,
                                        'max_nc': max_nc,
                                    })
                                    results = {k: [v] for k, v in results.items()}
                                    df = pd.concat([df, pd.DataFrame.from_dict(results)])
    return reset_df(df), default_args
