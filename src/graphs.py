import numpy as np

from collections import Counter
from pyvis.network import Network
from typing import List, Optional

PYVIS_WIDTH_PX = 800
PYVIS_HEIGHT_PX = 800

PYVIS_NODE_SIZE = 6
PYVIS_EDGE_WIDTH = 0.00001
PYVIS_EDGE_COLOR = 'gray'


class Graph:
    N: int
    A: Optional[np.ndarray]
    L: Optional[np.ndarray]
    laplacian_eigenvalues: Optional[np.ndarray]

    def __init__(self, n: int):
        self.N = n
        self.A = None
        self.L = None
        self.laplacian_eigenvalues = None

    @staticmethod
    def _compute_laplacian(A: np.ndarray) -> np.ndarray:
        # if normalize:
        #     D = np.diag(1 / np.sqrt(np.sum(A, axis=0)))
        #     L = np.eye(len(A)) - D @ A @ D
        # else:
        #     L = np.diag(np.sum(A, axis=0)) - A
        # return L

        D = np.diag(1 / np.sqrt(np.sum(A, axis=0)))
        L = np.eye(len(A)) - D @ A @ D
        return L


class SPGraph(Graph):
    rho: float
    k: int
    comm_sizes: List
    communities: Optional[np.ndarray]
    net: Optional[Network]

    def __init__(self, n: int, rho: float, k: int, comm_sizes: List):
        Graph.__init__(self, n)
        if sorted(comm_sizes)[::-1] != comm_sizes:
            raise Exception('Community sizes should decrease')

        self.rho = rho
        self.k = k
        self.comm_sizes = comm_sizes
        self.communities = self.__build_communities(comm_sizes, n)
        self.A = self.__generate_adj_matrix()
        self.L = Graph._compute_laplacian(self.A)
        self.laplacian_eigenvalues, _ = np.linalg.eig(self.L)
        self.net = None

    def get_pyvis_net(self, level: int, notebook: bool, predicted_communities: Optional[np.ndarray] = None) -> Network:
        if self.A is None:
            raise Exception('SP graph is not generated yet')

        nodes = np.arange(self.N).astype(int).tolist()
        idx = np.nonzero(np.triu(self.A.astype(int)))
        edges = np.stack([idx[0], idx[1]]).T.tolist()

        net = Network(
            notebook=notebook,
            heading=f'SPGraph, N={self.N}, comm_sizes={self.comm_sizes}, level={level}',
            height=PYVIS_HEIGHT_PX,
            width=PYVIS_WIDTH_PX)

        node_colors = self.__generate_colors(level, predicted_communities)
        node_sizes = [PYVIS_NODE_SIZE] * self.N

        net.add_nodes(nodes, size=node_sizes, color=node_colors)
        net.add_edges(edges)
        self.net = net
        for e in net.edges:
            e['width'] = PYVIS_EDGE_WIDTH
            e['color'] = PYVIS_EDGE_COLOR

        return net

    def __generate_colors(self, level: int, predicted_communities: np.ndarray) -> List:
        if predicted_communities is None:
            ncolors = self.N // self.comm_sizes[level]
            colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(ncolors)]
            return [colors[i] for i in self.communities[:, level]]
        else:
            ncolors = len(set(predicted_communities[:, level]))
            colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(ncolors)]
            return [colors[i - 1] for i in predicted_communities[:, level]]

    def __generate_adj_matrix(self) -> np.ndarray:
        p, S = self.__compute_probabilities()
        P = np.zeros((self.N, self.N))

        for l in range(self.communities.shape[1]):
            comm_num = max(self.communities[:, l]) + 1
            x = np.repeat(self.communities[:, l].reshape((-1, 1)), comm_num, axis=1)
            y = np.repeat(np.arange(comm_num).reshape(1, -1), self.N, axis=0)
            S = x == y
            for k in range(comm_num):
                x = S[:, k].reshape((-1, 1))
                y = S[:, k].reshape((1, -1))
                mask = (x @ y).astype(np.float64)
                P = P * (1 - mask) + mask * p[l + 1]

        P = np.where(P == 0, p[0], P)
        A = np.ceil(P - np.random.rand(self.N, self.N))
        A = np.triu(A, 1)
        A = A + A.T
        return A

    def __compute_probabilities(self) -> (List, List):
        S = []
        levels = self.communities.shape[1]
        for l in range(levels):
            y, x = np.array(list(Counter(self.communities[:, l]).items())).T
            if np.std(x) != 0:
                raise Exception('Incorrect communities for a SP graph')
            else:
                S.append(x[0])

        S = [S[0] * max(self.communities[:, 0])] + S
        for l in range(1, levels):
            S[l] = S[l] - S[l + 1]
        S[-1] = S[-1] - 1
        if self.k / (self.rho + 1) > S[-1]:
            raise Exception(f'Implicit constraint does not hold: {self.k / (self.rho + 1)} > {S[-1]}')

        p = [(self.rho ** levels) / ((1 + self.rho) ** levels) * self.k / S[0]]
        for l in range(1, levels + 1):
            p_l = (self.rho ** (levels - l)) / ((1 + self.rho) ** (levels - l + 1)) * self.k / S[l]
            p.append(p_l)

        return p, S

    @staticmethod
    def __build_communities(comm_sizes, n: int):
        communities = []
        for comm_size in comm_sizes:
            comm_num = n // comm_size
            mod = n % comm_size
            comm_col = np.concatenate([
                np.ravel([[i] * comm_size for i in range(comm_num)]),
                [comm_num] * mod
            ]).astype(int)
            communities.append(comm_col)
        return np.array(communities).T
