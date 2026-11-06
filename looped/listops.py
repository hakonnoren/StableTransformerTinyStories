"""Synthetic ListOps data generator, in the procedural numpy style of
``data_generators.py`` (no TensorFlow / TFDS / file loading).

ListOps (Nangia & Bowman 2018; LRA) is a nested-list reduction task: an
expression like ``[MAX 2 9 [MIN 4 7 ] 0 ]`` evaluates to a single digit 0-9.
We treat it as LM-style sequence classification — the model reads the
expression and predicts the answer at the final position, exactly like the
parity / count-threshold generators in ``data_generators.py``.

Token vocabulary (integer ids), size = ``LISTOPS_VOCAB_SIZE`` = 16:
    0..9    digit / value tokens  (also the answer-label space)
    10..13  operator-open tokens  [MIN [MAX [MED [SM
    14      close bracket  ]
    15      PAD

All intermediate and final values stay in 0-9 (MIN/MAX/MED of digits are
digits; SM is sum mod 10), so the answer shares the digit token space.

Sequence layout (left-padded so the answer is always the final token, which
lets the existing last-position test funcs read the prediction directly):

    full = [PAD]*(max_length - Le) + expr_tokens + [answer]     # len max_length+1
    x    = full[:-1]                                            # len max_length
    y    = full[1:]  with all but the last position set to -1   # supervise answer only

Note: left-padding means the expression sits at varying absolute positions, so
the model must be somewhat position-tolerant (standard for decoder-LM
classification over variable-length inputs).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
OP_TOKENS = {'[MIN': 10, '[MAX': 11, '[MED': 12, '[SM': 13}
CLOSE_ID = 14
PAD_ID = 15
LISTOPS_VOCAB_SIZE = 16

_OP_FUNCS = {
    '[MIN': lambda xs: int(min(xs)),
    '[MAX': lambda xs: int(max(xs)),
    '[MED': lambda xs: int(np.median(xs)),     # floor of median (values are 0-9)
    '[SM':  lambda xs: int(sum(xs) % 10),
}


# ----------------------------------------------------------------------------
# Expression generation
#
# Faithful port of ``listops_original.generate_tree`` + ``to_value`` (the LRA /
# Nangia-Bowman generator). The original builds nested binary tuples rendered by
# ``to_string`` with ``( )`` scaffolding that the TF input pipeline then strips
# (``rename_close_brackets`` deletes ``(`` / ``)`` and renames ``]``); the
# surviving token stream is exactly ``[OP arg arg ... ]``, which we emit
# directly. ``VALUE_P`` there is the probability a node is an *operator*
# (``r > VALUE_P`` -> leaf, so operator w.p. VALUE_P); we keep that name/meaning
# as ``value_p``.
# ----------------------------------------------------------------------------
def _gen_listops(depth, max_depth, max_args, value_p, ops):
    """One ListOps node, emitting token ids directly.

    Mirrors the original: with prob ``1 - value_p`` a node is a digit leaf; with
    prob ``value_p`` it is an operator with ``randint(2, max_args)`` (inclusive)
    children, each recursively generated. At ``depth >= max_depth`` a node is
    forced to a leaf (original sets ``r = 1 > VALUE_P``).

    ``ops`` is the list of operator tokens to sample from (e.g. ``['[MIN','[MAX']``).

    Returns (tokens, value, length). ``length`` matches the original's count:
    leaf = 1, operator = 2 (``[OP`` + ``]``) + sum(child lengths).
    """
    r = np.random.random() if depth < max_depth else 1.0
    if r > value_p:
        v = int(np.random.randint(0, 10))          # leaf digit (VALUES = range(10))
        return [v], v, 1

    n_args = np.random.randint(2, max_args + 1)     # [2, max_args] incl. (== random.randint)
    op = ops[np.random.randint(len(ops))]
    tokens = [OP_TOKENS[op]]
    vals = []
    length = 2                                      # the [OP and the ] tokens
    for _ in range(n_args):
        sub_tokens, sub_val, sub_len = _gen_listops(depth + 1, max_depth, max_args, value_p, ops)
        tokens.extend(sub_tokens)
        vals.append(sub_val)
        length += sub_len
    tokens.append(CLOSE_ID)
    return tokens, _OP_FUNCS[op](vals), length


def _resolve_ops(ops):
    """Normalize an ``ops`` spec to a validated list of operator tokens.
    ``None`` -> all four. Accepts the bracketed tokens ('[MIN') or bare names
    ('MIN', 'min'). Raises on unknown operators."""
    if ops is None:
        return list(OP_TOKENS.keys())
    resolved = []
    for o in ops:
        key = o if o.startswith('[') else '[' + o.upper()
        if key not in OP_TOKENS:
            raise ValueError(f"Unknown ListOps operator {o!r}; valid: {list(OP_TOKENS)}")
        resolved.append(key)
    if not resolved:
        raise ValueError("ops must contain at least one operator")
    return resolved


def get_xy_listops(max_length, min_length=3, max_depth=10, max_args=10,
                   value_p=0.25, ops=None, max_tries=10000):
    """One ListOps example as an (x, y) pair, both length ``max_length``.

    Generation matches ``listops_original`` (``value_p`` == ``VALUE_P``,
    ``max_depth``/``max_args`` defaults 10/10) with the same length filter
    ``min_length < length <= max_length`` (original used a strict upper bound and
    min_length=500 for the long-range split; here min_length defaults to 3 so
    every example has at least one operator — raise both bounds to reproduce the
    real LRA difficulty).

    Args:
        max_length: padded sequence length / GPT block_size (must be >= 4).
        min_length: reject expressions with ``length <= min_length``.
        max_depth:  max nesting depth (original default 10).
        max_args:   max arguments per operator (original default 10).
        value_p:    probability a node is an operator (original VALUE_P=0.25).
        ops:        which operators to use. ``None`` = all four
                    (MIN/MAX/MED/SM). Pass e.g. ``['MIN','MAX']`` (or the
                    bracketed ``['[MIN','[MAX']``) for an easier, arithmetic-free
                    variant.
        max_tries:  reject-and-retry budget for fitting the length window.

    Returns:
        x: (max_length,) int token ids — left-padded expression.
        y: (max_length,) int targets — all -1 except the final position (answer).
    """
    if max_length < 4:
        raise ValueError("max_length must be >= 4 to fit a minimal ListOps expression")

    ops = _resolve_ops(ops)
    for _ in range(max_tries):
        tokens, val, length = _gen_listops(1, max_depth, max_args, value_p, ops)
        if min_length < length <= max_length:
            break
    else:
        raise RuntimeError(
            f"Could not generate a ListOps expression with "
            f"{min_length} < length <= {max_length} in {max_tries} tries; "
            f"adjust min_length / max_length / value_p / max_depth / max_args."
        )

    Le = len(tokens)
    full = np.array([PAD_ID] * (max_length - Le) + tokens + [val], dtype=int)  # len max_length+1
    x = full[:-1].copy()      # left-padded expression, length max_length
    y = full[1:].copy()       # length max_length
    y[:-1] = -1               # supervise only the final (answer) position
    return x, y


def get_train_test_listops(max_length, samples_per_batch, n_batches_train, n_batches_test,
                           min_length=3, max_depth=10, max_args=10, value_p=0.25, ops=None):
    """Generate train/test ListOps sets in the standard ``data`` dict format.

    Shapes (matching the classification generators in data_generators.py):
        x_train: (n_batches_train, samples_per_batch, max_length)
        y_train: (n_batches_train, samples_per_batch, max_length)  (masked, -1 except answer)
        x_test:  (n_batches_test,  samples_per_batch, max_length)
        y_test:  (n_batches_test,  samples_per_batch, 1)           (answer label only)

    Use ``vocab_size = LISTOPS_VOCAB_SIZE`` and ``block_size = max_length`` when
    building the GPT Config. Evaluate with :func:`test_func_listops`. Generation
    parameters are forwarded to :func:`get_xy_listops` (defaults match the
    original LRA generator except the length window, shrunk for small models).

    ``ops``: which operators to use. ``None`` = all four; pass ``['MIN','MAX']``
    for the easier, arithmetic-free (selection-only) variant.
    """
    x_train = np.zeros((n_batches_train, samples_per_batch, max_length), dtype=int)
    y_train = np.zeros((n_batches_train, samples_per_batch, max_length), dtype=int)
    x_test  = np.zeros((n_batches_test,  samples_per_batch, max_length), dtype=int)
    y_test  = np.zeros((n_batches_test,  samples_per_batch, 1), dtype=int)

    kw = dict(min_length=min_length, max_depth=max_depth, max_args=max_args,
              value_p=value_p, ops=ops)
    for j in range(samples_per_batch):
        for i in range(n_batches_train):
            x, y = get_xy_listops(max_length, **kw)
            x_train[i, j] = x
            y_train[i, j] = y
        for i in range(n_batches_test):
            x, y = get_xy_listops(max_length, **kw)
            x_test[i, j] = x
            y_test[i, j, 0] = y[-1]

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


# ----------------------------------------------------------------------------
# Evaluation + debugging helpers
# ----------------------------------------------------------------------------
def test_func_listops(model, x_test, y_test):
    """Number of wrong answers: read the model's argmax at the final position
    (where the answer is supervised) and compare to the label.

    ``x_test`` is (samples, max_length); ``y_test`` is (samples, 1)."""
    import torch
    x = torch.as_tensor(x_test, dtype=torch.long, device=next(model.parameters()).device)
    y_hat = model(x)[0].argmax(dim=-1)[:, -1].detach().cpu().numpy()
    return int(np.count_nonzero(y_hat - y_test[:, -1]))


def decode_listops(tokens):
    """Render a token-id sequence back to a readable string (for inspection)."""
    inv_op = {v: k for k, v in OP_TOKENS.items()}
    parts = []
    for t in np.asarray(tokens).ravel().tolist():
        t = int(t)
        if t == PAD_ID:
            parts.append('_')
        elif t == CLOSE_ID:
            parts.append(']')
        elif t in inv_op:
            parts.append(inv_op[t])
        else:
            parts.append(str(t))
    return ' '.join(parts)
