"""RNA inverse folding (secondary-structure design) as a task for looped reasoners.

Task: given a pseudoknot-free target secondary structure in dot-bracket notation, output a sequence over
{A, C, G, U} whose minimum-free-energy (MFE) structure is exactly the target. The folding engine is ViennaRNA 2
with default settings (Turner 2004 parameters, dangles = 2, 37 C), the "Vienna 2" engine of Eterna100-V2. Many
sequences solve a target, and checking a candidate costs one fold, so a solution is certified exactly
(`verify`), but a sequence that differs from the training label can still be right.

Training pairs are self-consistent by construction (`make_example`):
  1. sample a design-like structure (`sample_structure`: stems, hairpins, bulges, interior loops, multiloops);
  2. give it a sequence (`design_sequence`: Eterna-style composition, GC-rich closing pairs, A-rich loops),
     optionally refined by an adaptive walk toward the sampled structure (`adaptive_walk`);
  3. fold the sequence and use (fold(seq), seq).
The target is therefore always designable and the label is one of its solutions. `mode="random"` replaces
steps 1-2 by a uniformly random sequence (natural-looking but loosely structured folds).

Tokens (shared vocabulary, as EqR expects): PAD 0, '.' 1, '(' 2, ')' 3, bases A C G U = 4..7, pair types
GC CG AU UA GU UG = 8..13 (first letter at the 5' position of the pair). Inputs carry the structure. Labels
carry the base at unpaired positions and the pair type at BOTH positions of a pair (`encode_labels`), and
decoding combines the two positions of each pair (`decode_sequence`), so every predicted pair is
complementary. With one base per position instead, the training labels' coin-flip orientation (GC or CG)
makes the per-position argmax G at both ends (G.G, no pair): a run plateaued at the marginal ceiling and
solved nothing (README). Positions past the target's length are PAD in the input and 0 (ignored) in the label.
"""
from __future__ import annotations

import dataclasses

import numpy as np

try:
    import RNA  # ViennaRNA Python bindings: pip install ViennaRNA
except ImportError:  # the structure utilities work without it; folding does not
    RNA = None

PAD = 0
STRUCT_TOKENS = {".": 1, "(": 2, ")": 3}
NUC_TOKENS = {"A": 4, "C": 5, "G": 6, "U": 7}
NUCS = "ACGU"
PAIRS = ("GC", "CG", "AU", "UA", "GU", "UG")          # canonical + wobble pairs ViennaRNA allows
PAIR_TOKENS = {p: 8 + k for k, p in enumerate(PAIRS)}  # pair type, first letter at the 5' position
VOCAB_SIZE = 14
ID_TO_CHAR = {v: k for k, v in {**STRUCT_TOKENS, **NUC_TOKENS}.items()}
ID_TO_PAIR = {v: k for k, v in PAIR_TOKENS.items()}
MIN_HAIRPIN = 3                                        # ViennaRNA's minimum hairpin loop


# ── structures ───────────────────────────────────────────────────────────────────────────────────────────
def pair_table(db: str) -> np.ndarray:
    """Partner index of every position (-1 if unpaired). Raises ValueError on unbalanced or unknown symbols."""
    pt = np.full(len(db), -1, dtype=np.int64)
    stack = []
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if not stack:
                raise ValueError(f"unbalanced ')' at {i}")
            j = stack.pop()
            pt[i], pt[j] = j, i
        elif c != ".":
            raise ValueError(f"unknown symbol {c!r} at {i}")
    if stack:
        raise ValueError(f"unbalanced '(' at {stack[-1]}")
    return pt


def pairs_of(db: str) -> set:
    pt = pair_table(db)
    return {(i, int(j)) for i, j in enumerate(pt) if j > i}


def is_valid_structure(db: str, min_hairpin: int = MIN_HAIRPIN) -> bool:
    """Balanced, and every pair encloses at least `min_hairpin` positions (ViennaRNA cannot fold tighter)."""
    try:
        pt = pair_table(db)
    except ValueError:
        return False
    return all(j - i - 1 >= min_hairpin for i, j in enumerate(pt) if j > i)


def bp_distance(a: str, b: str) -> int:
    """Base pairs in exactly one of the two structures (ViennaRNA's bp_distance)."""
    if len(a) != len(b):
        raise ValueError("structures differ in length")
    return len(pairs_of(a) ^ pairs_of(b))


def structure_features(db: str) -> dict:
    """Loop decomposition summary, for difficulty analysis (helices = maximal stacks of consecutive pairs)."""
    pt = pair_table(db)
    n = len(db)
    helices, hairpins, interior, bulges, multi = [], 0, 0, 0, 0
    i = 0
    seen = set()
    for i in range(n):
        j = int(pt[i])
        if j <= i or i in seen:
            continue
        # walk the helix starting at (i, j)
        k = 0
        while i + k < j - k and pt[i + k] == j - k:
            seen.add(i + k)
            k += 1
        helices.append(k)
        # classify the loop closed by the innermost pair (i+k-1, j-k+1)
        a, b = i + k - 1, j - k + 1
        branches, p = [], a + 1
        while p < b:
            if pt[p] > p:
                branches.append((p, int(pt[p])))
                p = int(pt[p]) + 1
            else:
                p += 1
        if not branches:
            hairpins += 1
        elif len(branches) == 1:
            left, right = branches[0][0] - a - 1, b - branches[0][1] - 1
            if left and right:
                interior += 1
            else:
                bulges += 1
        else:
            multi += 1
    h = np.array(helices) if helices else np.zeros(1, dtype=int)
    return {"length": n, "pairs": int((pt > np.arange(n)).sum()), "helices": len(helices),
            "min_helix": int(h.min()), "short_helices": int((h <= 2).sum()), "hairpins": hairpins,
            "interior_loops": interior, "bulges": bulges, "multiloops": multi,
            "unpaired_frac": float((pt < 0).mean())}


@dataclasses.dataclass
class StructParams:
    """Design-like structure grammar. A helix's span is the positions it occupies with its interior.

    Every span is allocated with room for `stem_floor` pairs around a minimal hairpin, so a helix shorter
    than `stem_floor` appears only when `stem_probs` asks for it (isolated pairs almost never fold).
    """
    stem_probs: tuple = ((2, .10), (3, .22), (4, .22), (5, .16), (6, .12), (7, .08), (8, .06), (9, .02), (10, .02))
    stem_floor: int = 2
    hairpin_max: int = 10
    loop_side_max: int = 6          # unpaired bases on one side of a bulge / interior loop
    mean_linker: float = 1.5        # unpaired bases between branches (geometric mean)
    p_hairpin: float = 0.3          # close a helix with a hairpin when the interior allows other loops
    p_multi: float = 0.3            # else a multiloop (if it fits), else a bulge / interior loop
    max_depth: int = 12

    @property
    def min_span(self) -> int:
        return 2 * self.stem_floor + MIN_HAIRPIN


def _stem_len(rng, p: StructParams, cap: int) -> int:
    ks, ws = zip(*p.stem_probs)
    w = np.array(ws, dtype=float)
    s = int(rng.choice(ks, p=w / w.sum()))
    return max(1, min(s, cap))


def _helix(rng, out, lo, hi, p, depth):
    """Pairs at the ends of out[lo:hi], then its interior: a hairpin, a bulge/interior loop or a multiloop."""
    span = hi - lo
    s = _stem_len(rng, p, (span - MIN_HAIRPIN) // 2)
    for k in range(s):
        out[lo + k], out[hi - 1 - k] = "(", ")"
    ilo, ihi = lo + s, hi - s
    m = ihi - ilo
    min_span = p.min_span                            # smallest inner helix the allocation allows
    must_hairpin = m < min_span + 1                  # no room for an inner helix plus one unpaired base
    can_hairpin = m <= p.hairpin_max                 # larger interiors always get inner structure
    if must_hairpin or (can_hairpin and (depth >= p.max_depth or rng.random() < p.p_hairpin)):
        return                                        # hairpin loop: interior stays unpaired (m >= 3)
    if m >= 2 * min_span and rng.random() < p.p_multi:
        _multiloop(rng, out, ilo, ihi, p, depth + 1)
        return
    # bulge or interior loop: a + b unpaired bases around one inner helix (a + b >= 1)
    room = m - min_span
    a = int(rng.integers(0, min(p.loop_side_max, room) + 1))
    b = int(rng.integers(0, min(p.loop_side_max, room - a) + 1))
    if a + b == 0:
        a = 1
    _helix(rng, out, ilo + a, ihi - b, p, depth + 1)


def _branches(rng, out, lo, hi, k, p, depth):
    """k helices side by side in out[lo:hi], separated by short unpaired linkers (needs k * min_span room)."""
    m = hi - lo
    linkers = rng.geometric(1.0 / (1.0 + p.mean_linker), size=k + 1) - 1
    while linkers.sum() > m - k * p.min_span:        # shrink linkers until k minimal helices fit
        linkers = np.maximum(linkers - 1, 0)
    free = m - k * p.min_span - int(linkers.sum())
    extra = rng.multinomial(free, rng.dirichlet(np.ones(k)))
    i = lo + int(linkers[0])
    for t in range(k):
        span = p.min_span + int(extra[t])
        _helix(rng, out, i, i + span, p, depth)
        i += span + int(linkers[t + 1])


def _multiloop(rng, out, lo, hi, p, depth):
    """Two to four helices closed by one pair (the caller checks that two minimal helices fit)."""
    k = int(rng.integers(2, max(2, min(4, (hi - lo) // p.min_span)) + 1))
    _branches(rng, out, lo, hi, k, p, depth)


def sample_structure(rng: np.random.Generator, length: int, p: StructParams | None = None) -> str:
    """A random design-like nested structure of exactly `length` positions with at least one pair."""
    p = p or StructParams()
    if length < p.min_span:
        raise ValueError(f"length {length} < {p.min_span}, too short for one helix")
    out = ["."] * length
    # external loop: one to four helices with dangling ends and linkers
    k = int(rng.integers(1, max(1, min(4, length // (3 * p.min_span))) + 1))
    _branches(rng, out, 0, length, k, p, 0)
    db = "".join(out)
    assert is_valid_structure(db), db
    return db


# ── sequences ────────────────────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class SeqParams:
    """Eterna-style composition, drawn per example within these ranges so labels are diverse."""
    gc_pair: tuple = (0.4, 0.85)     # share of GC/CG among inner helix pairs (rest AU/UA, some GU/UG)
    gu_pair: float = 0.05
    closing_gc: float = 0.9          # helix-end pairs (next to a loop or the chain end)
    loop_a: tuple = (0.5, 0.95)      # share of A among unpaired bases (rest uniform over C, G, U)
    hairpin_g_boost: float = 0.5     # G as the first base of a hairpin loop


def _helix_ends(pt: np.ndarray) -> np.ndarray:
    """True at pairs (both positions) adjacent to a loop or chain end, i.e. not stacked on both sides."""
    n = len(pt)
    end = np.zeros(n, dtype=bool)
    for i in range(n):
        j = pt[i]
        if j <= i:
            continue
        inner = i + 1 < j - 1 and pt[i + 1] == j - 1
        outer = i - 1 >= 0 and j + 1 < n and pt[i - 1] == j + 1
        if not (inner and outer):
            end[i] = end[j] = True
    return end


def design_sequence(rng: np.random.Generator, db: str, p: SeqParams | None = None) -> str:
    """A sequence compatible with every pair of `db` (canonical or GU), with Eterna-style composition."""
    p = p or SeqParams()
    pt = pair_table(db)
    n = len(db)
    gc, la = rng.uniform(*p.gc_pair), rng.uniform(*p.loop_a)
    ends = _helix_ends(pt)
    seq = [""] * n
    for i in range(n):
        j = pt[i]
        if j < 0:
            seq[i] = "A" if rng.random() < la else NUCS[1:][rng.integers(3)]
        elif j > i:
            g = p.closing_gc if ends[i] else gc
            r = rng.random()
            if r < g:
                pair = "GC" if rng.random() < 0.5 else "CG"
            elif r < g + p.gu_pair and not ends[i]:
                pair = "GU" if rng.random() < 0.5 else "UG"
            else:
                pair = "AU" if rng.random() < 0.5 else "UA"
            seq[i], seq[j] = pair[0], pair[1]
    # G boost: first base of each hairpin loop
    for i in range(n):
        j = pt[i]
        if j > i and (pt[i + 1:j] < 0).all() and rng.random() < p.hairpin_g_boost:
            seq[i + 1] = "G"
    return "".join(seq)


def random_sequence(rng: np.random.Generator, length: int) -> str:
    gc = rng.uniform(0.3, 0.7)
    probs = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])     # A C G U
    return "".join(NUCS[k] for k in rng.choice(4, size=length, p=probs))


# ── folding ──────────────────────────────────────────────────────────────────────────────────────────────
def _need_rna():
    if RNA is None:
        raise ImportError("ViennaRNA is not installed: pip install ViennaRNA")


def fold(seq: str) -> tuple[str, float]:
    """MFE structure and energy (kcal/mol) under ViennaRNA's default model (the Eterna100-V2 engine)."""
    _need_rna()
    s, e = RNA.fold_compound(seq, RNA.md()).mfe()
    return s, float(e)


def verify(seq: str, target: str) -> dict:
    """Is `seq` a solution of `target`? Plus the base-pair distance of its MFE structure to the target."""
    s, e = fold(seq)
    d = bp_distance(s, target)
    return {"solved": s == target, "bp_distance": d, "mfe": e, "structure": s}


def ensemble_defect(seq: str, target: str) -> float:
    """Expected fraction of positions paired differently from the target in the Boltzmann ensemble."""
    _need_rna()
    fc = RNA.fold_compound(seq, RNA.md())
    _, e = fc.mfe()
    fc.exp_params_rescale(e)                          # avoids overflow in the partition function
    fc.pf()
    return float(fc.ensemble_defect(target))


PAIR_MUTATION_W = np.array([0.35, 0.35, 0.12, 0.12, 0.03, 0.03])     # GC CG AU UA GU UG


def energy_gap(seq: str, target: str) -> tuple[str, float]:
    """MFE structure of `seq` and E(target) - E(MFE) >= 0 (zero iff the target is an MFE structure).

    `seq` must be compatible with every pair of `target`.
    """
    _need_rna()
    fc = RNA.fold_compound(seq, RNA.md())
    s, e = fc.mfe()
    return s, float(fc.eval_structure(target)) - float(e)


def adaptive_walk(rng: np.random.Generator, seq: str, target: str, steps: int) -> tuple[str, bool, int]:
    """Local search toward `target` on the energy gap E(target) - E(MFE): mutate a position the MFE
    structure gets wrong (paired positions as pairs, GC/CG preferred), keep non-worsening moves.

    `seq` must be compatible with `target`. Returns (sequence, solved, folds used); one fold per step.
    """
    pt = pair_table(target)
    cur = list(seq)
    s, gap = energy_gap(seq, target)
    used = 1
    while s != target and used <= steps:
        bad = np.flatnonzero(pair_table(s) != pt)
        i = int(rng.choice(bad))
        j = int(pt[i])
        cand = cur.copy()
        if j >= 0:
            a, b = min(i, j), max(i, j)
            w = np.where([q == cur[a] + cur[b] for q in PAIRS], 0.0, PAIR_MUTATION_W)
            q = PAIRS[rng.choice(len(PAIRS), p=w / w.sum())]
            cand[a], cand[b] = q[0], q[1]
        else:
            cand[i] = [c for c in NUCS if c != cur[i]][rng.integers(3)]
        s2, gap2 = energy_gap("".join(cand), target)
        used += 1
        if gap2 <= gap:
            cur, s, gap = cand, s2, gap2
    return "".join(cur), s == target, used


# ── training examples ────────────────────────────────────────────────────────────────────────────────────
def make_example(rng: np.random.Generator, length: int, mode: str = "design", walk_steps: int = 0,
                 sp: StructParams | None = None, qp: SeqParams | None = None) -> dict:
    """One self-consistent (target, sequence) pair: target = MFE structure of sequence, with >= 1 pair.

    mode "design": sampled structure -> designed sequence (+ adaptive walk) -> fold.
    mode "random": random sequence -> fold (redrawn until it has a pair).
    """
    _need_rna()
    for _ in range(1000):
        if mode == "design":
            sampled = sample_structure(rng, length, sp)
            seq = design_sequence(rng, sampled, qp)
            used = 0
            if walk_steps > 0:
                seq, _, used = adaptive_walk(rng, seq, sampled, walk_steps)
        elif mode == "random":
            sampled, seq, used = None, random_sequence(rng, length), 0
        else:
            raise ValueError(f"unknown mode {mode!r}")
        target, energy = fold(seq)
        if "(" in target:
            return {"target": target, "sequence": seq, "mode": mode, "sampled": sampled,
                    "matched": sampled == target if sampled else None, "walk_folds": used, "mfe": energy}
    raise RuntimeError(f"no structured fold at length {length}")


# ── tokens ───────────────────────────────────────────────────────────────────────────────────────────────
def encode_structure(db: str, seq_len: int) -> np.ndarray:
    if len(db) > seq_len:
        raise ValueError(f"structure of length {len(db)} > seq_len {seq_len}")
    out = np.zeros(seq_len, dtype=np.uint8)
    out[:len(db)] = [STRUCT_TOKENS[c] for c in db]
    return out


def encode_labels(target: str, seq: str, seq_len: int) -> np.ndarray:
    """Label tokens of a solution `seq` of `target`: the base at unpaired positions, the pair type at both
    positions of every pair. Raises ValueError if a pair of `seq` cannot pair."""
    if len(seq) != len(target):
        raise ValueError("sequence and structure differ in length")
    if len(seq) > seq_len:
        raise ValueError(f"sequence of length {len(seq)} > seq_len {seq_len}")
    out = np.zeros(seq_len, dtype=np.uint8)
    for i, j in enumerate(pair_table(target)):
        if j < 0:
            out[i] = NUC_TOKENS[seq[i]]
        else:
            a, b = min(i, j), max(i, j)
            if seq[a] + seq[b] not in PAIR_TOKENS:
                raise ValueError(f"positions {a}, {b} hold {seq[a]}{seq[b]}, which cannot pair")
            out[i] = PAIR_TOKENS[seq[a] + seq[b]]
    return out


def decode_structure(ids) -> str:
    """Dot-bracket from input tokens (stops at the first PAD)."""
    out = []
    for t in np.asarray(ids).tolist():
        if t == PAD:
            break
        out.append(ID_TO_CHAR[t])
    return "".join(out)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def decode_sequence(ids_or_logits, target: str) -> str:
    """Sequence for `target` from label / predicted token ids, or from logits over the vocabulary.

    Logits (seq_len, vocab): unpaired positions take the argmax over the bases; a pair (i, j) takes the pair
    type maximising log p_i(type) + log p_j(type) over the six pair types, so every pair is complementary.
    Ids: the pair token at i (else at j); a pair with no pair token on either side decodes as A.A (no pair),
    a non-base token at an unpaired position as A.
    """
    pt = pair_table(target)
    n = len(target)
    x = np.asarray(ids_or_logits)
    seq = ["A"] * n
    if x.ndim == 2:
        x = x[:n].astype(np.float64)
        pair_lp = _log_softmax(x[:, 8:14])
        for i, j in enumerate(pt):
            if j < 0:
                seq[i] = NUCS[int(x[i, 4:8].argmax())]
            elif j > i:
                p = PAIRS[int((pair_lp[i] + pair_lp[j]).argmax())]
                seq[i], seq[j] = p[0], p[1]
    else:
        ids = [int(t) for t in x[:n]]
        for i, j in enumerate(pt):
            if j < 0:
                seq[i] = ID_TO_CHAR[ids[i]] if 4 <= ids[i] <= 7 else "A"
            elif j > i:
                p = ID_TO_PAIR.get(ids[i]) or ID_TO_PAIR.get(ids[j]) or "AA"
                seq[i], seq[j] = p[0], p[1]
    return "".join(seq)


def gc_a_design(target: str) -> str:
    """Eterna's beginner strategy: every pair GC (G at the 5' position), every unpaired base A. The floor a
    model has to beat: close to what predicting the most common label everywhere gives (GC and CG are about
    equally common; with CG everywhere instead the solve rate is similar)."""
    pt = pair_table(target)
    return "".join("A" if j < 0 else ("G" if j > i else "C") for i, j in enumerate(pt))
