"""Tests for looped/eqr/rna (numpy; the folding tests need ViennaRNA: pip install ViennaRNA).

    python -m pytest -q looped/eqr/rna/test_rna_design.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rna_design as rd  # noqa: E402

needs_rna = pytest.mark.skipif(rd.RNA is None, reason="ViennaRNA not installed")


def _helix_lengths(db):
    pt = rd.pair_table(db)
    out, seen = [], set()
    for i, j in enumerate(pt):
        if j <= i or i in seen:
            continue
        k = 0
        while i + k < j - k and pt[i + k] == j - k:
            seen.add(i + k)
            k += 1
        out.append(k)
    return out


def test_pair_table_and_distance():
    pt = rd.pair_table("((..)).")
    assert pt.tolist() == [5, 4, -1, -1, 1, 0, -1]
    for bad in ("((.)", "(.))", "(.x)"):
        with pytest.raises(ValueError):
            rd.pair_table(bad)
    assert rd.bp_distance("((...))", "((...))") == 0
    assert rd.bp_distance("((...))", ".(...).") == 1
    assert rd.bp_distance("((...))", ".......") == 2
    assert rd.is_valid_structure("(...)") and not rd.is_valid_structure("(..)")


def test_structure_features():
    f = rd.structure_features("((((...))))..(((..((....))..)))")
    assert f["helices"] == 3 and f["hairpins"] == 2 and f["interior_loops"] == 1 and f["multiloops"] == 0
    f = rd.structure_features("((((...)).((...))))")
    assert f["multiloops"] == 1 and f["hairpins"] == 2 and f["min_helix"] == 2


def test_sample_structure_valid_and_no_isolated_pairs():
    rng = np.random.default_rng(0)
    for _ in range(300):
        L = int(rng.integers(16, 129))
        db = rd.sample_structure(rng, L)
        assert len(db) == L and rd.is_valid_structure(db) and "(" in db
        assert min(_helix_lengths(db)) >= 2           # stem_probs has no 1-pair stems


def test_design_sequence_pairs_are_compatible():
    rng = np.random.default_rng(1)
    for _ in range(100):
        db = rd.sample_structure(rng, int(rng.integers(16, 129)))
        seq = rd.design_sequence(rng, db)
        assert len(seq) == len(db) and set(seq) <= set("ACGU")
        assert all(seq[i] + seq[j] in rd.PAIRS for i, j in rd.pairs_of(db))


def test_tokens_round_trip():
    db, seq = "((...)).", "GUAAAACA"                  # pairs (0,6) = GC, (1,5) = UA
    x, y = rd.encode_structure(db, 12), rd.encode_labels(db, seq, 12)
    assert x.tolist() == [2, 2, 1, 1, 1, 3, 3, 1, 0, 0, 0, 0]
    GC, UA = rd.PAIR_TOKENS["GC"], rd.PAIR_TOKENS["UA"]
    assert y.tolist()[:8] == [GC, UA, 4, 4, 4, UA, GC, 4] and y[8:].sum() == 0       # pair type at both ends
    assert rd.decode_structure(x) == db and rd.decode_sequence(y, db) == seq
    with pytest.raises(ValueError):
        rd.encode_labels(db, "GGAAAACA", 12)           # (0,6) = G.C fine, (1,5) = G.A cannot pair
    with pytest.raises(ValueError):
        rd.encode_structure("(" * 13, 12)
    assert rd.gc_a_design("((...)).") == "GGAAACCA"


def test_decode_logits_gives_complementary_pairs():
    db = "((...))"
    logits = np.zeros((10, rd.VOCAB_SIZE))
    logits[:, 1] = 9.0                                 # a structure token wins overall: ignored
    logits[:, 6] = 5.0                                 # G is the best base everywhere: only used where unpaired
    k = {p: rd.PAIR_TOKENS[p] for p in rd.PAIRS}
    logits[0, k["GC"]], logits[0, k["CG"]] = 2.0, 1.9  # 5' end prefers GC slightly ...
    logits[6, k["CG"]], logits[6, k["GC"]] = 3.0, 0.0  # ... the 3' end prefers CG strongly: the sum picks CG
    logits[1, k["AU"]], logits[5, k["AU"]] = 1.0, 1.0
    s = rd.decode_sequence(logits, db)
    assert s == "CAGGGUG" and all(s[i] + s[j] in rd.PAIRS for i, j in rd.pairs_of(db))


@needs_rna
def test_fold_and_verify():
    s, e = rd.fold("GGGGAAACCCC")
    assert s == "((((...))))" and e < 0
    v = rd.verify("GGGGAAACCCC", "((((...))))")
    assert v["solved"] and v["bp_distance"] == 0
    assert not rd.verify("AAAAAAAAAAA", "((((...))))")["solved"]
    s, gap = rd.energy_gap("GGGGAAACCCC", "((((...))))")
    assert abs(gap) < 1e-6
    assert rd.energy_gap("GAGAAAAACUC", "(((.....)))")[1] >= 0
    good, bad = rd.ensemble_defect("GGGGAAACCCC", "((((...))))"), rd.ensemble_defect("AAAAAAAAAAA", "((((...))))")
    assert 0.0 <= good < bad <= 1.0                    # the solution's ensemble is closer to the target


@needs_rna
def test_adaptive_walk_solves_hairpin():
    rng = np.random.default_rng(2)
    target = "((((((....))))))"
    seq, solved, used = rd.adaptive_walk(rng, "AUAUAUAAAAAUAUAU", target, 200)
    assert solved and rd.fold(seq)[0] == target and used <= 201


@needs_rna
def test_make_example_is_self_consistent():
    rng = np.random.default_rng(3)
    for mode in ("design", "random"):
        for _ in range(10):
            ex = rd.make_example(rng, int(rng.integers(16, 60)), mode, walk_steps=20)
            assert rd.fold(ex["sequence"])[0] == ex["target"] and "(" in ex["target"]
            assert len(ex["sequence"]) == len(ex["target"])


@needs_rna
def test_builder_smoke(tmp_path):
    tsv = tmp_path / "eterna.tsv"
    cols = ["Puzzle #", "Puzzle Name", "Secondary Structure V2", "Sample Solution (V2/Vienna2)"]
    tsv.write_text("\t".join(cols) + "\n" + "1\tHairpin\t((((...))))\tGGGGAAACCCC\n"
                   + "2\tLong\t" + "(" * 40 + "." * 3 + ")" * 40 + "\t\n")
    cmd = [sys.executable, os.path.join(HERE, "build_rna_dataset.py"), "--out-root", str(tmp_path),
           "--n-train", "12,16", "--n-test", "6", "--len-min", "16", "--len-max", "40", "--walk-steps", "10",
           "--workers", "2", "--eterna-tsv", str(tsv)]
    subprocess.run(cmd, check=True, capture_output=True)
    for name, n_train in (("rna-L40-12", 12), ("rna-L40-16", 16)):
        for split, n in (("train", n_train), ("test", 6)):
            d = tmp_path / name / split
            meta = json.load(open(d / "dataset.json"))
            x, y = np.load(d / "all__inputs.npy"), np.load(d / "all__labels.npy")
            assert x.shape == y.shape == (n, 40) and meta["seq_len"] == 40 and meta["vocab_size"] == rd.VOCAB_SIZE
            assert np.load(d / "all__group_indices.npy").tolist() == list(range(n + 1))
            for xi, yi in zip(x, y):
                t = rd.decode_structure(xi)
                pt = rd.pair_table(t)
                assert all((yi[i] >= 8) == (pt[i] >= 0) for i in range(len(t)))          # pair tokens exactly at pairs
                assert all(yi[i] == yi[pt[i]] for i in range(len(t)) if pt[i] >= 0)      # same token at both ends
                assert rd.fold(rd.decode_sequence(yi, t))[0] == t
    # nested training sets, shared test set
    a = np.load(tmp_path / "rna-L40-12" / "train" / "all__inputs.npy")
    b = np.load(tmp_path / "rna-L40-16" / "train" / "all__inputs.npy")
    assert (a == b[:12]).all()
    assert (np.load(tmp_path / "rna-L40-12" / "test" / "all__inputs.npy")
            == np.load(tmp_path / "rna-L40-16" / "test" / "all__inputs.npy")).all()
    # Eterna: only the puzzle that fits in len-max, with its folding sample solution
    e = tmp_path / "rna-eterna100v2-L40" / "test"
    x, y = np.load(e / "all__inputs.npy"), np.load(e / "all__labels.npy")
    assert x.shape == (1, 40) and rd.decode_structure(x[0]) == "((((...))))"
    assert rd.decode_sequence(y[0], "((((...))))") == "GGGGAAACCCC"
