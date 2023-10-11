from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Data:
    data_matrix: np.array
    energy_matrix: np.array
    dst: np.array
    time: np.array


def load_data(data_dir, beta):
    data_path = os.path.join(data_dir, f'fermi_{beta}')
    data_matrix = np.load(os.path.join(data_path, 'dataMatrix.npy'))
    energy_matrix = np.load(os.path.join(data_path, 'energyMatrix.npy'))
    dst = np.load(os.path.join(data_path, 'dst.npy'))
    time = np.load(os.path.join(data_path, 'time.npy'))
    return Data(data_matrix=data_matrix, energy_matrix=energy_matrix, dst=dst, time=time)


def plot_mode_energies(data, xlim=None, ignore_first_steps=0):
    time_steps, _, _, mode_energies = data

    plt.plot(
        time_steps[ignore_first_steps:],
        mode_energies[0, :][ignore_first_steps:],
        "k-",
        linewidth=1.5,
        label="Mode 1",
        alpha=0.5,
    )
    plt.plot(
        time_steps[ignore_first_steps:],
        mode_energies[1, :][ignore_first_steps:],
        "g-",
        linewidth=1.5,
        label="Mode 2",
        alpha=0.5,
    )
    plt.plot(
        time_steps[ignore_first_steps:],
        mode_energies[2, :][ignore_first_steps:],
        "r-",
        linewidth=1.5,
        label="Mode 3",
        alpha=0.5,
    )
    if xlim:
        plt.xlim(xlim)
    plt.xlabel("$t$ ")
    plt.ylabel("Energy ")
    plt.legend(loc="upper right", shadow=True, fontsize="x-small")
    plt.show()


# def create_input_data_vector(data, sample_manifold=False, n_sample=1000):
#     time_steps, q, p, _ = data
#     X = np.concatenate([q.T, p.T], axis=1)

#     if sample_manifold:
#         idx = np.random.randint(time_steps.shape[0] - n_sample)
#         return X[idx : idx + 1000, :]

#     else:
#         return X


def compute_explained_ratio_q(X, n_components=2, pre_whitening=False, verbose=True):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if pre_whitening:
        cov = np.cov(X.T)
        w, v = LA.eig(cov)
        rank = LA.matrix_rank(cov)
        if verbose == True:
            print(f"Rank of covariance matrix of dim {len(cov)} is {rank}")

        idx_to_keep = np.argwhere(w >= 0.001 * max(w))
        idx_to_keep = np.array(idx_to_keep.flatten())

        X = X[:, :len(idx_to_keep)]
        n_components = X.shape[1]

    model = PCA(n_components=n_components)
    _ = model.fit_transform(X)

    if verbose == True:
        print(
            "Explained variance ratio on all data:", model.explained_variance_ratio_[:5]
        )

    return model.explained_variance_ratio_


def compute_explained_ratio(X, n_components=2, pre_whitening=False, verbose=True):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if pre_whitening:
        cov = np.cov(X.T)
        w, v = LA.eig(cov)
        rank = LA.matrix_rank(cov)
        if verbose == True:
            print(f"Rank of covariance matrix of dim {len(cov)} is {rank}")

        idx_to_keep = np.argwhere(w >= 0.001 * max(w))
        idx_to_keep = np.array(idx_to_keep.flatten())

        X = np.concatenate(
            [
                X[:, : int(len(idx_to_keep) / 2)],
                X[:, 32 : 32 + int(len(idx_to_keep) / 2)],
            ],
            axis=1,
        )
        n_components = X.shape[1]

        X = X[:, :len(idx_to_keep)]
        n_components = X.shape[1]

    model = PCA(n_components=n_components)
    _ = model.fit_transform(X)

    if verbose == True:
        print(
            "Explained variance ratio on all data:", model.explained_variance_ratio_[:5]
        )

    return model.explained_variance_ratio_


def plot_consecutive_distances(data, n_points=None):
    """Plot distances between first point in phase-space and consecutive points
    at time t. Plot first `n_points`.
    """

    times, q, _, _ = data

    dist_vec = [0]
    if n_points is not None:
        n_points = min(len(times), n_points)
    else:
        n_points = len(times)

    for t in range(1, len(times[:n_points])):
        dist_vec.append(LA.norm(q[:, t] - q[:, 0]))

    plt.plot(times[:n_points], dist_vec, ".-")


import matplotlib.ticker as mticker

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)