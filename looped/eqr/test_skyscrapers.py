"""Tests for looped/eqr/skyscrapers.py (numpy only):  python -m pytest -q looped/eqr/test_skyscrapers.py"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skyscrapers as sk  # noqa: E402

NO_CLUES = {k: np.zeros(sk.N, dtype=np.int64) for k in ("top", "bottom", "left", "right")}


def test_visibility_tables():
    ident = np.arange(1, 8)
    assert sk.visible(ident) == 7 and sk.visible(ident[::-1]) == 1 and sk.visible([3, 1, 4, 2, 7, 5, 6]) == 3
    assert len(sk.LINE_PERMS[(0, 0)]) == 5040
    assert len(sk.LINE_PERMS[(7, 0)]) == 1 and len(sk.LINE_PERMS[(0, 7)]) == 1      # only the sorted line
    assert len(sk.LINE_PERMS[(1, 0)]) == 720                                          # tallest first
    assert len(sk.LINE_PERMS[(1, 1)]) == 0                                            # tallest at both ends
    for (a, b), idx in [((2, 3), sk.LINE_PERMS[(2, 3)])]:
        assert all(sk.visible(p) == a and sk.visible(p[::-1]) == b for p in sk.PERMS[idx])


def test_board_round_trip_and_clue_orientation():
    rng = np.random.default_rng(1)
    g = sk.random_solution(rng)
    cl = sk.clues_of(g)
    b = sk.to_board(g, cl)
    g2, cl2 = sk.from_board(b)
    assert (g2 == g).all() and all((cl2[k] == cl[k]).all() for k in cl)
    assert b[0, 0] == b[0, -1] == b[-1, 0] == b[-1, -1] == 0
    assert b[1, 0] == sk.visible(g[0]) and b[1, -1] == sk.visible(g[0, ::-1])         # row 0 from left / right
    assert b[0, 1] == sk.visible(g[:, 0]) and b[-1, 1] == sk.visible(g[::-1, 0])      # column 0 from top / bottom
    assert sk.is_solution(g, cl)


@pytest.mark.parametrize("seed", range(6))
def test_solver_sound_and_complete(seed):
    """Every solution returned is valid, and the planted solution is always among them."""
    rng = np.random.default_rng(seed)
    g = sk.random_solution(rng)
    full = sk.clues_of(g)
    clues = {k: np.where(rng.random(sk.N) < 0.4, v, 0) for k, v in full.items()}
    givens = np.where(rng.random((sk.N, sk.N)) < 0.15, g, 0)
    n, sols, _ = sk.solve(givens, clues, limit=200)
    assert all(sk.is_solution(s, clues, givens) for s in sols)
    assert len({s.tobytes() for s in sols}) == n
    if n < 200:
        assert any((s == g).all() for s in sols)


def test_intercalate_has_exactly_two_solutions():
    rng = np.random.default_rng(3)
    for _ in range(50):
        g = sk.random_solution(rng)
        for r1 in range(7):
            for r2 in range(r1 + 1, 7):
                for c1 in range(7):
                    for c2 in range(c1 + 1, 7):
                        if g[r1, c1] == g[r2, c2] and g[r1, c2] == g[r2, c1]:
                            givens = g.copy()
                            givens[[r1, r1, r2, r2], [c1, c2, c1, c2]] = 0
                            n, sols, _ = sk.solve(givens, NO_CLUES, limit=5)
                            assert n == 2
                            return
    pytest.skip("no intercalate found")


@pytest.mark.parametrize("seed", range(4))
def test_make_puzzle_unique_minimal_and_symmetric(seed):
    rng = np.random.default_rng(100 + seed)
    p, s, rating = sk.make_puzzle(rng)
    g, cl = sk.from_board(p)
    n, sols, r2 = sk.solve(g, cl, limit=2)
    assert n == 1 and (sols[0] == s[1:-1, 1:-1]).all() and r2 == rating
    assert (s[0] == 0).all() and (s[:, 0] == 0).all()                                 # solution board: grid only
    for k in cl:                                                                      # minimal: no clue removable
        for i in np.nonzero(cl[k])[0]:
            c2 = {kk: v.copy() for kk, v in cl.items()}
            c2[k][i] = 0
            assert sk.solve(g, c2, limit=2)[0] == 2
    for r, c in np.argwhere(g > 0):                                                   # ... and no given
        g2 = g.copy()
        g2[r, c] = 0
        assert sk.solve(g2, cl, limit=2)[0] == 2
    key = sk.canonical(p)
    for t in range(8):                                                                # symmetries map puzzles to puzzles
        pt, st = sk.dihedral(p, t), sk.dihedral(s, t)
        gt, ct = sk.from_board(pt)
        n, sols, _ = sk.solve(gt, ct, limit=2)
        assert n == 1 and (sols[0] == st[1:-1, 1:-1]).all()
        assert sk.canonical(np.ascontiguousarray(pt)) == key
