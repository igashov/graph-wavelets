import numpy as np

from scipy.cluster.hierarchy import fcluster, linkage
from src.graphs import Graph
from src.wavelets import compute_chebyshev_polynomial, get_scale2cheb_coeffs
from typing import List


def cut_dendrogram(z: np.ndarray) -> np.ndarray:
    n = z.shape[0] + 1
    levels = np.floor(z[:, 2] * 1000).astype(int)

    aux = 2 * np.ones((n, levels[-1]))
    aux[:, 0] = np.zeros(n)
    com_current = np.zeros((n, 2 * n - 1)).astype(bool)
    com_current[:n, :n] = np.eye(n)
    last = np.ones(n)
    lastvalue = np.zeros((n, 1))

    for i in range(n - 1):
        f1 = np.nonzero(com_current[:, int(z[i, 0])])[0]
        for j in range(len(f1)):
            indices = np.arange(
                n * (last[f1[j]] - 1) + f1[j] - 1,
                n * (levels[i] - 1) + f1[j],
                n
            ).astype(int)
            aux_flatten = aux.T.flatten()
            aux_flatten[indices] = z[i, 2] - lastvalue[f1[j]]
            aux = aux_flatten.reshape(aux.shape).T
        last[f1] = levels[i] + 1
        lastvalue[f1] = z[i, 2]

        f2 = np.nonzero(com_current[:, int(z[i, 1])])[0]
        for j in range(len(f2)):
            indices = np.arange(
                n * (last[f2[j]] - 1) + f2[j] - 1,
                n * (levels[i] - 1) + f2[j],
                n
            ).astype(int)
            aux_flatten = aux.T.flatten()
            aux_flatten[indices] = z[i, 2] - lastvalue[f2[j]]
            aux = aux_flatten.reshape(aux.shape).T
        last[f2] = levels[i] + 1
        lastvalue[f2] = z[i, 2]

        com_current[f1, n + i - 1] = 1
        com_current[f2, n + i - 1] = 1

    ind = np.argmax(np.sum(aux, axis=0))
    partition = fcluster(z, ind / 1000, criterion='distance')
    return partition


def compute_communities(graph: Graph, scales_num: int, r_num: int) -> (List, np.ndarray):
    l1, l2, l3 = sorted(graph.laplacian_eigenvalues)[:3]
    lmax = np.max(graph.laplacian_eigenvalues)
    approximation_interval = [0, lmax]

    scales, s2c = get_scale2cheb_coeffs(scales_num, l2, l3, lmax, approximation_interval)
    signal = np.random.random((graph.N, r_num))
    scale2signal_fwt = compute_chebyshev_polynomial(signal, graph.L, s2c, approximation_interval)
    zg = np.array([
        cut_dendrogram(linkage(signal_fwt, 'average', 'correlation'))
        for signal_fwt in scale2signal_fwt
    ]).T
    return scales, zg
