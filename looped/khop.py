"""k-hop pointer chasing (copied from StableTransformers_claude/ideas/spectrum_split/toy_tasks.py)."""
import numpy as np


def make_khop(n, S, K, n_query):
    """(x, y) of shape (n, S + n_query); y = -1 on the context, f^K(q) on the queries."""
    T = S + n_query
    x = np.zeros((n, T), dtype=int)
    y = -np.ones((n, T), dtype=int)
    for i in range(n):
        cyc = np.random.permutation(S)              # f(cyc[i]) = cyc[i+1]: one S-cycle
        f = np.empty(S, dtype=int)
        f[cyc] = np.roll(cyc, -1)
        order = np.random.permutation(S)
        x[i, :S] = S + order * S + f[order]
        q = np.random.randint(0, S, n_query)
        x[i, S:] = q
        ans = q.copy()
        for _ in range(K):
            ans = f[ans]
        y[i, S:] = ans
    return x, y
