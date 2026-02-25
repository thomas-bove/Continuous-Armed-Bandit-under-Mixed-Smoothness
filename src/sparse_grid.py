import numpy as np
import itertools

def one_dimensional_nodes(level):
    if level == 1:
        return np.array([0.5])
    else:
        n = 2**(level-1)
        return np.linspace(0, 1, n+1)

def smolyak_grid(d, max_level):
    nodes = set()
    for levels in itertools.product(range(1, max_level+1), repeat=d):
        if sum(levels) <= max_level + d - 1:
            grids_1d = [one_dimensional_nodes(l) for l in levels]
            for point in itertools.product(*grids_1d):
                nodes.add(tuple(point))
    return np.array(list(nodes))