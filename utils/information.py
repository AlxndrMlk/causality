"""`get_entropy` code comes from https://github.com/paulbrodersen/entropy_estimators/blob/master/entropy_estimators/continuous.py"""


import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma, digamma



def get_entropy(x, k=1, norm='max', min_dist=0., workers=1):
    """
    Code source: https://github.com/paulbrodersen/entropy_estimators/blob/master/entropy_estimators/continuous.py

    Estimates the entropy H of a random variable x (in nats) based on
    the kth-nearest neighbour distances between point samples.
    @reference:
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9–16.
    Arguments:
    ----------
    x: (n, d) ndarray
        n samples from a d-dimensional multivariate distribution
    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution
    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances
    min_dist: float (default 0.)
        minimum distance between data points;
        smaller distances will be capped using this value
	workers: int (default 1)
		number of workers to use for parallel processing in query; 
		-1 uses all CPU threads
    Returns:
    --------
    h: float
        entropy H(X)
    """

    n, d = x.shape

    if norm == 'max': # max norm:
        p = np.inf
        log_c_d = 0 # volume of the d-dimensional unit ball
    elif norm == 'euclidean': # euclidean norm
        p = 2
        log_c_d = (d/2.) * np.log(np.pi) -np.log(gamma(d/2. +1))
    else:
        raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")

    kdtree = KDTree(x)

    # query all points -- k+1 as query point also in initial set
    # distances, _ = kdtree.query(x, k + 1, eps=0, p=norm)
    distances, _ = kdtree.query(x, k + 1, eps=0, p=p, workers=workers)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(np.log(2*distances)) # where did the 2 come from? radius -> diameter
    h = -digamma(k) + digamma(n) + log_c_d + (d / float(n)) * sum_log_dist

    return h