"""Skyscrapers puzzles on Sudoku's 9x9 board (a second task for the EqR experiments; numpy only).

A 7x7 Latin square of building heights 1..7 sits inside a 9x9 board. The 28 border cells hold
visibility clues: how many buildings are seen from that side of the row or column (a building is
seen if it is taller than every building in front of it). Corners are always empty.

    board[0, 1..7]     clue for grid column j seen from the top     board[8, 1..7]  ... from the bottom
    board[1..7, 0]     clue for grid row i seen from the left       board[1..7, 8]  ... from the right
    board[1..7, 1..7]  the grid (puzzle: given heights or 0; solution: every height)

Raw values are 0..7 with 0 = blank / no clue. EqR's dataset format adds 1 (0 = PAD, 1 = blank,
2..8 = 1..7); the labels hold the grid's heights and 0 (= ignore) on the border and corners, so
EqR's exact accuracy is over the 49 grid cells. Whole-board rotations and reflections map puzzles
to puzzles (a clue moves with its line), so the 8 dihedral symmetries are valid augmentations;
relabelling heights is not (visibility depends on their order).

Constraints: each grid row and column is a permutation of 1..7 (as in Sudoku, without boxes), and
each clue fixes the number of left-to-right maxima of its line seen from its side, a constraint on
the order of a whole line rather than on pairs of cells.

The solver propagates each line to generalised arc consistency over the 5040 permutations of 1..7
(only permutations consistent with the line's clues and its cells' candidates survive, and each
cell keeps the heights some surviving permutation puts there), then branches on the cell with the
fewest candidates. `rating` = number of branching nodes needed to prove uniqueness (0: line
propagation alone solves it).
"""
import itertools

import numpy as np

N = 7                      # grid size
B = N + 2                  # board size (grid + clue border)
FULL = (1 << N) - 1

PERMS = np.array(list(itertools.permutations(range(1, N + 1))), dtype=np.int8)        # [5040, 7]
BITS = (1 << (PERMS.astype(np.int16) - 1)).astype(np.uint8)                           # one-hot heights
VIS_FWD = (PERMS == np.maximum.accumulate(PERMS, axis=1)).sum(1)                       # seen from the start
VIS_BWD = (PERMS[:, ::-1] == np.maximum.accumulate(PERMS[:, ::-1], axis=1)).sum(1)    # seen from the end
LINE_PERMS = {(a, b): np.nonzero(((a == 0) | (VIS_FWD == a)) & ((b == 0) | (VIS_BWD == b)))[0]
              for a in range(N + 1) for b in range(N + 1)}
POPCOUNT = np.array([bin(v).count("1") for v in range(256)], dtype=np.int8)
HEIGHT_OF_BIT = np.zeros(256, dtype=np.int8)
HEIGHT_OF_BIT[[1 << v for v in range(N)]] = np.arange(1, N + 1)

# 14 lines over the flat 49-cell grid: rows (read left -> right), then columns (top -> bottom)
LINES = np.array([[r * N + c for c in range(N)] for r in range(N)] +
                 [[r * N + c for r in range(N)] for c in range(N)], dtype=np.int64)


# ------------------------------------------------------------------------------------------- clues
def visible(line):
    line = np.asarray(line)
    return int((line == np.maximum.accumulate(line)).sum())


def clues_of(grid):
    """All 28 clues of a solved 7x7 grid: dict top / bottom / left / right, each of length 7."""
    g = np.asarray(grid)
    return {"top": np.array([visible(g[:, c]) for c in range(N)]),
            "bottom": np.array([visible(g[::-1, c]) for c in range(N)]),
            "left": np.array([visible(g[r, :]) for r in range(N)]),
            "right": np.array([visible(g[r, ::-1]) for r in range(N)])}


def to_board(grid, clues):
    """9x9 raw board (0 = blank) from a 7x7 grid (0 = blank) and clue arrays (0 = no clue)."""
    b = np.zeros((B, B), dtype=np.uint8)
    b[1:-1, 1:-1] = grid
    b[0, 1:-1], b[-1, 1:-1] = clues["top"], clues["bottom"]
    b[1:-1, 0], b[1:-1, -1] = clues["left"], clues["right"]
    return b


def from_board(board):
    b = np.asarray(board)
    return b[1:-1, 1:-1].copy(), {"top": b[0, 1:-1].copy(), "bottom": b[-1, 1:-1].copy(),
                                  "left": b[1:-1, 0].copy(), "right": b[1:-1, -1].copy()}


def _line_clues(clues):
    return [(int(clues["left"][r]), int(clues["right"][r])) for r in range(N)] + \
           [(int(clues["top"][c]), int(clues["bottom"][c])) for c in range(N)]


# ------------------------------------------------------------------------------------------- solver
def _propagate(cand, idx):
    """Line-wise generalised arc consistency, in place. False on contradiction."""
    changed = True
    while changed:
        changed = False
        for l in range(2 * N):
            cells = LINES[l]
            m = cand[cells]
            bits = BITS[idx[l]]
            ok = ((bits & m) != 0).all(axis=1)
            if not ok.all():
                idx[l] = idx[l][ok]                        # replaced, never mutated: callers share lists
                if idx[l].size == 0:
                    return False
                bits = bits[ok]
            new = m & np.bitwise_or.reduce(bits, axis=0)
            if (new != m).any():
                cand[cells] = new
                changed = True
    return True


def _search(cand, idx, limit, rng, sols, stats):
    if not _propagate(cand, idx):
        return 0
    pc = POPCOUNT[cand]
    if (pc == 1).all():
        sols.append(cand.copy())
        return 1
    stats["branches"] += 1
    open_ = np.nonzero(pc > 1)[0]
    cell = open_[np.argmin(pc[open_])]
    vals = [v for v in range(N) if (cand[cell] >> v) & 1]
    if rng is not None:
        rng.shuffle(vals)
    found = 0
    for v in vals:
        c2 = cand.copy()
        c2[cell] = 1 << v
        found += _search(c2, list(idx), limit - found, rng, sols, stats)
        if found >= limit:
            break
    return found


def solve(grid, clues, limit=2, rng=None):
    """Solutions of a puzzle (7x7 grid with 0 = blank, clue dict with 0 = none), up to `limit`.
    Returns (number found, list of 7x7 solutions, rating = branching nodes)."""
    g = np.asarray(grid).reshape(-1).astype(np.int64)
    cand = np.full(N * N, FULL, dtype=np.uint8)
    cand[g > 0] = (1 << (g[g > 0] - 1)).astype(np.uint8)
    idx = [LINE_PERMS[lc] for lc in _line_clues(clues)]
    sols, stats = [], {"branches": 0}
    n = _search(cand, idx, limit, rng, sols, stats)
    return n, [HEIGHT_OF_BIT[s].reshape(N, N) for s in sols], stats["branches"]


def is_solution(grid, clues, givens=None):
    """True iff grid is a Latin square agreeing with every clue (and with the givens, if any)."""
    g = np.asarray(grid)
    target = np.arange(1, N + 1)
    if not all((np.sort(g[r]) == target).all() and (np.sort(g[:, r]) == target).all() for r in range(N)):
        return False
    full = clues_of(g)
    if any(((clues[k] != 0) & (clues[k] != full[k])).any() for k in full):
        return False
    return givens is None or bool(((np.asarray(givens) == 0) | (np.asarray(givens) == g)).all())


# ------------------------------------------------------------------------------------------- generator
def random_solution(rng):
    """A random 7x7 Latin square (randomised search on the empty grid; not exactly uniform)."""
    n, sols, _ = solve(np.zeros((N, N), dtype=np.int8), {k: np.zeros(N, dtype=np.int64) for k in
                                                          ("top", "bottom", "left", "right")}, limit=1, rng=rng)
    assert n == 1
    return sols[0]


def make_puzzle(rng):
    """A unique, minimal puzzle: start from all 28 clues, add givens until the solution is unique, then
    remove clues and givens in random order while it stays unique (no single one can be removed after).
    Returns (raw 9x9 puzzle board, raw 9x9 solution board, rating)."""
    sol = random_solution(rng)
    full = clues_of(sol)
    clues = {k: v.copy() for k, v in full.items()}
    givens = np.zeros((N, N), dtype=np.int8)
    while True:
        n, sols, _ = solve(givens, clues, limit=2)
        if n == 1:
            break
        other = next(s for s in sols if not (s == sol).all())
        diff = np.argwhere(other != sol)
        r, c = diff[rng.integers(len(diff))]
        givens[r, c] = sol[r, c]
    items = [("clue", k, i) for k in clues for i in range(N)] + [("given", int(r), int(c)) for r, c in np.argwhere(givens > 0)]
    order = rng.permutation(len(items))
    for j in order:
        kind, a, b = items[j]
        if kind == "clue":
            old, clues[a][b] = clues[a][b], 0
            if solve(givens, clues, limit=2)[0] != 1:
                clues[a][b] = old
        else:
            old, givens[a, b] = givens[a, b], 0
            if solve(givens, clues, limit=2)[0] != 1:
                givens[a, b] = old
    n, sols, rating = solve(givens, clues, limit=2)
    assert n == 1 and (sols[0] == sol).all()
    full_board = to_board(sol, {k: np.zeros(N, dtype=np.uint8) for k in full})       # labels: grid only
    return to_board(givens, clues), full_board, rating


def dihedral(board, t):
    """The 8 symmetries of the square, applied to a whole board (EqR's dataset.common.dihedral_transform)."""
    return [lambda a: a, lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 3),
            np.fliplr, np.flipud, lambda a: a.T, lambda a: np.fliplr(np.rot90(a, 1))][t](board)


def canonical(board):
    """Byte key shared by all 8 symmetric images of a puzzle (for de-duplication across splits)."""
    return min(np.ascontiguousarray(dihedral(board, t)).tobytes() for t in range(8))
