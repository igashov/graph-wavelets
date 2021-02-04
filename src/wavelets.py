import numpy as np

from typing import Callable, List

ALPHA = 2
CHEBYSHEV_ORDER = 50


def wavelet_function(alpha: float, beta: float, t2: float, x: np.ndarray) -> np.ndarray:
    t1 = 1
    g = np.zeros(len(x))

    a1 = (alpha * t2 - beta * t1) / (t1 * t2 * (t1 - t2) ** 2)
    a2 = -(2 * alpha * t2 ** 2 - 2 * beta * t1 ** 2 + alpha * t1 * t2 - beta * t1 * t2) / (t1 * t2 * (t1 - t2) ** 2)
    a3 = (
                 - beta * t1 ** 3
                 - 2 * beta * t1 ** 2 * t2
                 + 2 * alpha * t1 * t2 ** 2
                 + alpha * t2 ** 3
         ) / (t1 * t2 * (t1 - t2) ** 2)
    a4 = (beta * t1 ** 2 - alpha * t2 ** 2 - 2 * t1 * t2 + t1 ** 2 + t2 ** 2) / (t1 ** 2 - 2 * t1 * t2 + t2 ** 2)

    g1 = np.where((x >= 0) & (x < t1))
    g2 = np.where((x >= t1) & (x < t2))
    g3 = np.where(x >= t2)

    g[g1] = x[g1] ** alpha * t1 ** (-alpha)
    x2 = x[g2]
    g[g2] = a4 + a3 * x2 + a2 * x2 ** 2 + a1 * x2 ** 3
    g[g3] = (t2 / x[g3]) ** beta
    return g


def compute_chebyshev_coefficients_internal(g: Callable, m: int, approx: List) -> List:
    n = m + 1
    a1 = (approx[1] - approx[0]) / 2
    a2 = (approx[1] + approx[0]) / 2
    g_res = g(a1 * np.cos((np.pi * (np.arange(1, n + 1) - 0.5)) / n) + a2)
    c = [
        np.sum(g_res * np.cos(np.pi * j * (np.arange(1, n + 1) - .5) / n)) * 2 / n
        for j in range(n)
    ]
    return c


def get_scale2cheb_coeffs(
        scales_num: int,
        l2: float,
        l3: float,
        lmax: float,
        approximation_interval: List
) -> (List, List):
    smin = 1 / l2
    t2 = lmax / l2
    smax = t2 / l2
    scales = np.logspace(np.log10(smin), np.log10(smax), scales_num)
    beta = 1 / np.log10(l3 / l2)

    scale2cheb_coeffs = [
        compute_chebyshev_coefficients_internal(
            lambda x: wavelet_function(ALPHA, beta, t2, s * x),
            CHEBYSHEV_ORDER,
            approximation_interval
        )
        for s in scales
    ]
    return scales, scale2cheb_coeffs


def compute_chebyshev_polynomial(f: np.ndarray, laplacian: np.ndarray, s2c: List, approx: List):
    n_scales = len(s2c)
    m = [
        len(s2c[scale])
        for scale in range(n_scales)
    ]
    max_m = max(m)

    a1 = (approx[1] - approx[0]) / 2
    a2 = (approx[1] + approx[0]) / 2

    twf_old = f
    twf_cur = (laplacian @ f - a2 * f) / a1

    r = [
        0.5 * s2c[scale][0] * twf_old + s2c[scale][1] * twf_cur
        for scale in range(n_scales)
    ]

    for k in range(1, max_m):
        twf_new = (2 / a1) * (laplacian @ twf_cur - a2 * twf_cur) - twf_old
        for scale in range(n_scales):
            if 1 + k < m[scale]:
                r[scale] = r[scale] + s2c[scale][k + 1] * twf_new
        twf_old = twf_cur
        twf_cur = twf_new

    return r
