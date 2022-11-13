import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_mode_energies(data, xlim=None):
    time_steps, _, _, mode_energies = data

    plt.plot(
        time_steps, mode_energies[0, :], "k-", linewidth=1.5, label="Mode 1", alpha=0.5
    )
    plt.plot(
        time_steps, mode_energies[1, :], "g-", linewidth=1.5, label="Mode 2", alpha=0.5
    )
    plt.plot(
        time_steps, mode_energies[2, :], "r-", linewidth=1.5, label="Mode 3", alpha=0.5
    )
    if xlim:
        plt.xlim(xlim)
    plt.xlabel("$t$ ")
    plt.ylabel("Energy ")
    legend = plt.legend(loc="upper right", shadow=True, fontsize="x-small")
    plt.show()


def create_input_data_vector(data, sample_manifold=False, n_sample=1000):
    time_steps, q, p, _ = data
    X = np.concatenate([q.T, p.T], axis=1)

    if sample_manifold:
        idx = np.random.randint(time_steps.shape[0] - n_sample)
        return X[idx : idx + 1000, :]

    else:
        return X


def compute_explained_ratio(X, pre_whitening=False, verbose=True):
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

    model = PCA(n_components=X.shape[1])
    _ = model.fit_transform(X)

    if verbose == True:
        print(
            "Explained variance ratio on all data:", model.explained_variance_ratio_[:5]
        )

    return model.explained_variance_ratio_


def plot_consecutive_distances(data, n_points=None):
    """Plot distances between first point in phase-space and consecutive points
    at time t. Plot first `n_points`
    """
    times, q, p, _ = data
    dist_vec = [0]

    if n_points is not None:
        n_points = min(len(times), n_points)
    else:
        n_points = len(times)

    for t in range(1, len(times[:n_points])):
        dist_vec.append(LA.norm(q[:, t] - q[:, 0]))

    plt.plot(times[:n_points], dist_vec, ".-")
