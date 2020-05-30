from typing import List, Tuple

import argparse

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

Vector = Tuple[float, float]
Covariance = List[List[float]]

PRIORS = [0.3, 0.2, 0.5]
MEANS = [
    (-0.5, -0.6),
    (0.6, -0.4),
    (0.1, 0.3)
]
COVS = [[
    [0.03, 0],
    [0, 0.03]
], [
    [0.04, 0.03],
    [0.01, 0.05]
], [
    [0.07, 0.04],
    [0.01, 0.06]
]]

def plot_data(data: np.ndarray, **kwargs):
    plt.scatter(data[:, 0], data[:, 1], **kwargs)

def generate_data(
    n: int, prior: List[float],
    mean: List[Vector],
    cov: List[Covariance]
) -> np.ndarray:
    cnt = np.random.multinomial(n, prior)
    data = np.vstack([
        np.random.multivariate_normal(
            mean[i], cov[i], cnt[i]
        )
        for i in range(len(prior))
    ])
    np.random.shuffle(data)
    return data

def get_label_by_distance(
    data: np.ndarray,
    p: np.ndarray,
    k: int
) -> np.ndarray:
    dist = np.hstack([
        ((data - p[i])**2).sum(axis=1).reshape(-1,1)
        for i in range(k)
    ])
    label = dist.argmin(axis=1)
    return label

def k_means(
    data: np.ndarray,
    k: int, steps: int,
    debug=False
) -> np.ndarray:
    p = np.random.normal(scale=0.5, size=(k, 2))

    for _ in range(steps):
        label = get_label_by_distance(data, p, k)

        if debug:
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)

            for i in range(k):
                plot_data(data[label == i], alpha=0.25)
            plot_data(p, marker='x')
            plt.show()

        for i in range(k):
            p[i] = np.average(data[label == i], axis=0)

    return p

def get_label_by_guassian(
    data: np.ndarray,
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    ds = [
        sp.stats.multivariate_normal(mu[i], sigma[i])
        for i in range(k)
    ]  # distributions
    prod = np.hstack([
        (pi[i] * ds[i].pdf(data)).reshape(-1, 1)
        for i in range(k)
    ])
    label = prod.argmax(axis=1)
    return prod, label

def expectation_maximize(
    data: np.ndarray,
    k: int, k_steps: int, e_steps: int,
    debug=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # use k-means algorithm to initialize
    mu = k_means(data, k, k_steps, debug=debug)
    label = get_label_by_distance(data, mu, k)
    _, cnt = np.unique(label, return_counts=True)
    pi = cnt / cnt.sum()
    sigma = np.array([
        np.diag(np.var(data[label == i], axis=0))
        for i in range(k)
    ])

    for _ in range(e_steps):
        # evaluate posterior probabilities
        prod, label = get_label_by_guassian(data, pi, mu, sigma, k)
        gamma = prod / prod.sum(axis=1).reshape(-1, 1)

        if debug:
            elbo = (gamma * np.log(prod / gamma)).sum()
            print(f'ELBO = {elbo}')

            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            for i in range(k):
                plot_data(data[label == i], alpha=0.25)
            plot_data(mu, marker='x')
            plt.show()

        # update parameters
        cnt = gamma.sum(axis=0)
        pi = cnt / cnt.sum()
        mu = (gamma.transpose() @ data) / cnt.reshape(-1, 1)

        # ...
        diff = data[np.newaxis, ...] - mu[:, np.newaxis, :]
        diff = diff[..., np.newaxis]
        result = (diff @ diff.transpose(0, 1, 3, 2)).transpose(1, 0, 2, 3)
        sigma = result * gamma[..., np.newaxis, np.newaxis]
        sigma = sigma.sum(axis=0) / cnt[:, np.newaxis, np.newaxis]

    return pi, mu, sigma

def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, help='Number of data points.')
    args = parser.parse_args()

    main(args)