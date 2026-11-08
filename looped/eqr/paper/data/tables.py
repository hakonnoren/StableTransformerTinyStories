"""Numbers transcribed from summarize_sudoku.py output on IDUN (run root sudoku_25939526, 2026-11-06/07),
from the local inner-loop-length test, and from W&B. One seed (0) in the transcribed blocks below.

Seed-resolved results (rounds three and four, seeds, checkpoint trace, dynamics; 2026-11-08) are read from
runs_*.csv, written by ../export_results.py from the pulled run root; Skyscrapers-7 from sky_wandb.json (W&B, training
curves) and sky_runs_*.csv (export_results.py on the pulled Skyscrapers run roots: final evaluations, test-time scaling,
dynamics). Map analyses (analyze_maps.py) of all rounds are in maps/*.json."""
import csv
import json
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))

# summarize_sudoku.py main table: evaluate.py at training noise 0.01, breadth 1.
# exact16 / cell16: whole test set (423k puzzles), 16 iterations; exact64s / exact256s: first 49,152 puzzles;
# O256 = P(wrong at 256 | exact at 16); dy256 = P(prediction changed 16 -> 256).
MAIN = [
    # arm, step, cell16, exact16, exact64s, exact256s, O256, dy256
    ("eqr", 20000, .893, .695, .732, .758, .0171, .313),
    ("eqr", 50000, .947, .858, .921, .950, .00407, .148),
    ("relax3_leak1", 20000, .873, .639, .654, .664, .0223, .366),
    ("relax3_leak1", 50000, .917, .783, .818, .845, .0128, .235),
    ("kick_pos", 20000, .872, .635, .650, .661, .0188, .358),
    ("kick_pos", 50000, .917, .782, .817, .839, .0136, .232),
    ("relax3_leak4", 20000, .823, .467, .478, .486, .0379, .327),
    ("kick_learned", 20000, .821, .465, .473, .480, .0199, .372),
    ("relax3", 20000, .710, .132, .128, .0351, .803, .973),
    ("kick_fixed", 20000, .726, .135, .147, .0621, .613, .946),
    ("relax3_leak1_injL", 20000, .840, .517, .522, .527, .0268, .386),
    ("relax3_leak1_injL_self", 20000, .867, .617, .635, .649, .0165, .359),
    ("kick_pos_injL", 20000, .851, .562, .567, .578, .0190, .420),
    # round three (2026-11-07 pull; O / dy not computed for these runs)
    ("relax3_leak2", 20000, .881, .658, .680, .699, None, None),
    ("relax3_leak2", 50000, .882, .703, .724, .746, None, None),
    ("relax3_leak3", 20000, .874, .635, .652, .667, None, None),
    ("twinL_leak1", 20000, .871, .628, .656, .675, None, None),
    ("twinL_leak2", 20000, .868, .614, .636, .652, None, None),
    ("twinL_leak2", 50000, .943, .846, .907, .937, None, None),
    ("innerL_self_leak1", 20000, .878, .652, .720, .760, None, None),
    ("innerL_self_leak1", 50000, .931, .810, .900, .940, None, None),
]

# sudoku_scaling.slurm (EqR's evaluate.py, first 2048 test puzzles):
# arm, step, depth, breadth, noise, single, conv1, convk(top-4), vote, any, cell
SCALING_CSV = """
eqr-released,50000,16,1,0.5,0.757,,,,,0.904
eqr-released,50000,16,16,0.5,0.755,0.943,0.896,0.92,0.951,0.903
eqr-released,50000,16,128,0.5,0.753,0.982,0.968,0.975,0.983,0.902
eqr-released,50000,64,1,0.5,0.879,,,,,0.949
eqr-released,50000,64,16,0.5,0.877,0.975,0.953,0.96,0.976,0.948
eqr-released,50000,64,128,0.5,0.879,0.992,0.986,0.987,0.992,0.949
eqr-released,50000,256,1,0.5,0.946,,,,,0.977
eqr-released,50000,256,16,0.5,0.943,0.987,0.979,0.98,0.988,0.975
eqr-released,50000,1024,1,0.5,0.976,,,,,0.989
eqr-released,50000,1024,16,0.5,0.975,0.994,0.988,0.99,0.994,0.989
eqr,20000,16,1,0.5,0.69,,,,,0.887
eqr,20000,16,16,0.5,0.682,0.87,0.809,0.836,0.876,0.882
eqr,20000,16,128,0.5,0.684,0.936,0.909,0.922,0.94,0.883
eqr,20000,64,1,0.5,0.814,,,,,0.927
eqr,20000,64,16,0.1,0.739,0.822,0.791,0.797,0.822,0.903
eqr,20000,64,16,0.25,0.778,0.859,0.831,0.836,0.859,0.915
eqr,20000,64,16,0.5,0.808,0.916,0.883,0.895,0.917,0.924
eqr,20000,64,16,1.0,0.0291,0.0322,0.0295,0.0381,0.0732,0.62
eqr,20000,64,128,0.5,0.808,0.957,0.94,0.943,0.959,0.924
eqr,20000,256,1,0.5,0.874,,,,,0.949
eqr,20000,256,16,0.5,0.876,0.945,0.926,0.934,0.946,0.949
eqr,20000,1024,1,0.5,0.914,,,,,0.964
eqr,20000,1024,16,0.5,0.918,0.969,0.952,0.956,0.969,0.966
eqr,50000,16,1,0.5,0.775,,,,,0.913
eqr,50000,64,1,0.5,0.898,,,,,0.957
eqr,50000,64,16,0.5,0.894,0.969,0.954,0.961,0.971,0.955
eqr,50000,64,128,0.5,0.894,0.987,0.98,0.98,0.988,0.956
eqr,50000,256,1,0.5,0.953,,,,,0.979
eqr,50000,1024,1,0.5,0.971,,,,,0.987
relax3_leak4,20000,16,16,0.5,0.455,0.574,0.533,0.558,0.584,0.81
relax3_leak4,20000,16,128,0.5,0.456,0.628,0.601,0.615,0.639,0.811
relax3_leak4,20000,64,16,0.5,0.538,0.618,0.59,0.598,0.622,0.832
relax3_leak4,20000,64,128,0.5,0.538,0.676,0.651,0.656,0.682,0.831
relax3_leak4,20000,256,16,0.5,0.585,0.655,0.629,0.636,0.658,0.845
relax3_leak1,20000,16,1,0.5,0.599,,,,,0.849
relax3_leak1,20000,16,16,0.5,0.608,0.834,0.752,0.79,0.848,0.854
relax3_leak1,20000,16,128,0.5,0.609,0.93,0.889,0.902,0.935,0.854
relax3_leak1,20000,64,1,0.5,0.747,,,,,0.9
relax3_leak1,20000,64,16,0.1,0.683,0.747,0.728,0.734,0.748,0.883
relax3_leak1,20000,64,16,0.25,0.724,0.813,0.784,0.792,0.813,0.895
relax3_leak1,20000,64,16,0.5,0.749,0.908,0.856,0.868,0.909,0.9
relax3_leak1,20000,64,16,1.0,0.227,0.678,0.488,0.585,0.761,0.694
relax3_leak1,20000,64,128,0.5,0.75,0.959,0.934,0.938,0.96,0.901
relax3_leak1,20000,256,1,0.5,0.851,,,,,0.939
relax3_leak1,20000,256,16,0.5,0.84,0.942,0.911,0.918,0.943,0.934
relax3_leak1,20000,1024,1,0.5,0.904,,,,,0.958
relax3_leak1,20000,1024,16,0.5,0.901,0.963,0.945,0.948,0.963,0.957
relax3_leak1,50000,16,1,0.5,0.625,,,,,0.861
relax3_leak1,50000,64,1,0.5,0.759,,,,,0.907
relax3_leak1,50000,64,16,0.5,0.774,0.927,0.886,0.902,0.928,0.913
relax3_leak1,50000,64,128,0.5,0.772,0.966,0.953,0.956,0.967,0.912
relax3_leak1,50000,256,1,0.5,0.864,,,,,0.946
relax3_leak1,50000,1024,1,0.5,0.925,,,,,0.969
kick_pos,20000,16,1,0.5,0.214,,,,,0.684
kick_pos,20000,16,16,0.5,0.221,0.677,0.451,0.515,0.732,0.686
kick_pos,20000,16,128,0.5,0.218,0.888,0.829,0.925,0.977,0.685
kick_pos,20000,64,1,0.5,0.382,,,,,0.741
kick_pos,20000,64,16,0.1,0.686,0.751,0.73,0.731,0.751,0.883
kick_pos,20000,64,16,0.25,0.693,0.829,0.784,0.794,0.831,0.882
kick_pos,20000,64,16,0.5,0.37,0.795,0.68,0.778,0.897,0.735
kick_pos,20000,64,16,1.0,0.0236,0.0244,0.0248,0.0259,0.0259,0.531
kick_pos,20000,64,128,0.5,0.37,0.863,0.851,0.968,0.991,0.736
kick_pos,20000,256,1,0.5,0.581,,,,,0.814
kick_pos,20000,256,16,0.5,0.572,0.828,0.781,0.9,0.96,0.81
kick_pos,20000,1024,1,0.5,0.715,,,,,0.868
kick_pos,20000,1024,16,0.5,0.716,0.826,0.81,0.938,0.98,0.867
kick_pos,50000,16,1,0.5,0.312,,,,,0.73
kick_pos,50000,64,1,0.5,0.451,,,,,0.775
kick_pos,50000,64,16,0.5,0.44,0.81,0.694,0.717,0.812,0.771
kick_pos,50000,64,128,0.5,0.44,0.937,0.885,0.879,0.938,0.77
kick_pos,50000,256,1,0.5,0.591,,,,,0.828
kick_pos,50000,1024,1,0.5,0.722,,,,,0.877
relax3_leak2,20000,16,1,0.5,0.593,,,,,0.848
relax3_leak2,20000,64,1,0.5,0.725,,,,,0.891
relax3_leak2,20000,64,16,0.5,0.728,0.89,0.84,0.86,0.891,0.892
relax3_leak2,20000,64,128,0.5,0.727,0.95,0.924,0.924,0.95,0.892
relax3_leak2,20000,256,1,0.5,0.822,,,,,0.927
relax3_leak2,20000,1024,1,0.5,0.883,,,,,0.949
relax3_leak2,50000,16,1,0.5,0.0264,,,,,0.691
relax3_leak2,50000,64,1,0.5,0.0259,,,,,0.693
relax3_leak2,50000,64,16,0.5,0.0266,0.0264,0.0269,0.0283,0.0293,0.692
relax3_leak2,50000,256,1,0.5,0.0264,,,,,0.693
relax3_leak2,50000,1024,1,0.5,0.0273,,,,,0.694
relax3_leak3,20000,16,1,0.5,0.494,,,,,0.82
relax3_leak3,20000,64,1,0.5,0.623,,,,,0.858
relax3_leak3,20000,64,16,0.5,0.621,0.786,0.729,0.749,0.788,0.856
relax3_leak3,20000,64,128,0.5,0.622,0.879,0.84,0.844,0.88,0.856
relax3_leak3,20000,256,1,0.5,0.712,,,,,0.886
relax3_leak3,20000,1024,1,0.5,0.784,,,,,0.913
twinL_leak1,20000,16,1,0.5,0.583,,,,,0.849
twinL_leak1,20000,64,1,0.5,0.758,,,,,0.907
twinL_leak1,20000,64,16,0.5,0.767,0.927,0.881,0.896,0.933,0.91
twinL_leak1,20000,64,128,0.5,0.764,0.973,0.955,0.963,0.975,0.909
twinL_leak1,20000,256,1,0.5,0.859,,,,,0.944
twinL_leak1,20000,1024,1,0.5,0.916,,,,,0.965
twinL_leak2,20000,16,1,0.5,0.523,,,,,0.823
twinL_leak2,20000,64,1,0.5,0.796,,,,,0.918
twinL_leak2,20000,64,16,0.5,0.797,0.983,0.953,0.966,0.986,0.919
twinL_leak2,20000,64,128,0.5,0.795,0.996,0.996,0.998,0.999,0.918
twinL_leak2,20000,256,1,0.5,0.926,,,,,0.97
twinL_leak2,20000,1024,1,0.5,0.982,,,,,0.993
twinL_leak2,50000,16,1,0.5,0.68,,,,,0.876
twinL_leak2,50000,64,1,0.5,0.811,,,,,0.922
twinL_leak2,50000,64,16,0.5,0.821,0.96,0.934,0.944,0.961,0.926
twinL_leak2,50000,256,1,0.5,0.906,,,,,0.959
twinL_leak2,50000,1024,1,0.5,0.954,,,,,0.979
innerL_self_leak1,20000,16,1,0.5,0.542,,,,,0.839
innerL_self_leak1,20000,64,1,0.5,0.677,,,,,0.879
innerL_self_leak1,20000,64,16,0.5,0.684,0.825,0.776,0.794,0.83,0.881
innerL_self_leak1,20000,64,128,0.5,0.685,0.887,0.856,0.866,0.894,0.882
innerL_self_leak1,20000,256,1,0.5,0.758,,,,,0.907
innerL_self_leak1,20000,1024,1,0.5,0.818,,,,,0.928
innerL_self_leak1,50000,16,1,0.5,0.7,,,,,0.887
innerL_self_leak1,50000,64,1,0.5,0.838,,,,,0.933
innerL_self_leak1,50000,64,16,0.5,0.831,0.939,0.916,0.926,0.942,0.932
innerL_self_leak1,50000,256,1,0.5,0.903,,,,,0.959
innerL_self_leak1,50000,1024,1,0.5,0.933,,,,,0.971
twinL_leak2,50000,64,128,0.5,0.822,0.986,0.979,0.979,0.987,0.927
innerL_self_leak1,50000,64,128,0.5,0.831,0.956,0.949,0.954,0.962,0.932
"""


def scaling_rows():
    rows = []
    for line in SCALING_CSV.strip().splitlines():
        p = line.split(",")
        num = lambda s: float(s) if s else None
        rows.append(dict(arm=p[0], step=int(p[1]), D=int(p[2]), B=int(p[3]), noise=float(p[4]), single=num(p[5]),
                         conv1=num(p[6]), convk=num(p[7]), vote=num(p[8]), any=num(p[9]), cell=num(p[10])))
    return rows


# Inner-loop length at test time (EqR checkpoints; 128 puzzles, 1 start, 32 iterations; local MPS, float32):
# L_cycles -> (exact at noise 0, exact at noise 0.5)
LCYCLES = {
    "eqr 20k": {4: (.750, .812), 5: (.750, .805), 6: (.711, .797), 7: (.734, .797), 8: (.727, .781)},
    "eqr 50k": {4: (.867, .875), 5: (.914, .891), 6: (.922, .883), 7: (.906, .867), 8: (.906, .867)},
    "eqr released": {4: (.906, .891), 5: (.945, .859), 6: (.945, .844), 7: (.930, .883), 8: (.945, .852)},
}

# Round three (and reference arms) from W&B: in-training eval (whole test set, 16 it., noise 0.01) at 14k,
# training-batch exact averaged over steps 18k-20k, cos(z_H, z_L) and ACT steps averaged likewise.
ROUND3 = [
    # arm, eval14k, train_exact_18_20k, cos_HL, act_steps
    ("eqr", .627, .720, .41, 7.9),
    ("relax3_leak1", .490, .684, 1.00, 8.2),
    ("kick_pos", .535, .648, .58, 8.8),
    ("relax3_leak4", .341, .449, 1.00, 11.1),
    ("relax3_leak2", .563, .742, 1.00, 7.3),
    ("relax3_leak3", .541, .706, 1.00, 7.9),
    ("twinL_leak1", .504, .624, .45, 9.3),
    ("twinL_leak2", .508, .651, .25, 8.9),
    ("innerL_self_leak1", .564, .590, .72, 9.9),
]

# The six arms trained to 50k (W&B, history re-pulled 2026-11-07): in-training eval at 49k (whole test set,
# 16 it., noise 0.01), its maximum over the 7k-spaced evals, training-batch exact / ACT steps / cos(z_H, z_L)
# averaged over steps 48k-50k. Training-batch accuracy is measured on halted sequences of augmented
# training puzzles (Sudoku-Extreme-1k x 1000 augmentations), so it depends on halting.
EXT50 = [
    # arm, eval49k, eval_max (step), train_48_50k, act_48_50k, cos_48_50k
    ("eqr", .859, (.859, 49000), .872, 5.4, .50),
    ("relax3_leak1", .786, (.786, 49000), .933, 4.1, .99),
    ("kick_pos", .782, (.782, 49000), .935, 3.9, .75),
    ("relax3_leak2", .711, (.745, 35000), .968, 2.0, .98),
    ("twinL_leak2", .850, (.850, 49000), .860, 5.6, .41),
    ("innerL_self_leak1", .806, (.806, 49000), .701, 8.3, .71),
    # round four (W&B, pulled 2026-11-08 evening, same definitions). The noise-trained arms' training batches run at
    # their training noise 0.25, so their train column is not comparable with the others'.
    ("twinL_leak3", .834, (.840, 42000), .883, 5.2, .41),
    ("twinL_leak10", .842, (.842, 49000), .878, 5.2, .39),
    ("twinL_leak2_noise25", .821, (.821, 49000), .840, 5.9, .45),
    ("eqr_noise25", .842, (.842, 49000), .842, 5.9, .51),
]

# relax common-mode damping sweep (rate 3, x in both forces): delta -> eval at 14k (W&B)
RELAX_DELTA = [(0.0, .022), (0.25, .341), (1.0, .490), (2.0, .563), (3.0, .541)]


# ------------------------------------------------------------------------------ seed-resolved results (CSV)
def _csv(name):
    path = os.path.join(_HERE, name + ".csv")
    out = []
    for r in csv.DictReader(open(path)):
        d = {}
        for k, v in r.items():
            if k == "arm":
                d[k] = v
            elif v == "":
                d[k] = None
            else:
                d[k] = float(v) if any(c in v for c in ".e") or k in ("N",) else int(v)
        out.append(d)
    return out


RUNS_MAIN = _csv("runs_main")
RUNS_SCALING = _csv("runs_scaling")
RUNS_DYN = _csv("runs_dynamics")
# Skyscrapers-7 run roots (export_results.py --prefix sky_runs --puzzles-from-root): same columns plus `puzzles`.
SKY_MAIN = _csv("sky_runs_main")
SKY_SCALING = _csv("sky_runs_scaling")
SKY_DYN = _csv("sky_runs_dynamics")


def runs(table, arm, step, seeds=None, **match):
    """Rows of one of the RUNS_* tables for an arm and step (all seeds unless given) matching match (e.g. D=1024, B=1, N=0.5)."""
    out = []
    for r in table:
        if r["arm"] != arm or r["step"] != step or (seeds is not None and r["seed"] not in seeds):
            continue
        if all(r.get(k) is not None and abs(r[k] - v) < 1e-9 for k, v in match.items()):
            out.append(r)
    return out


def stat(rows, key):
    """(mean, sd or None, n) of key over rows (seeds); None if no values."""
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return None
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else None), len(v)


# Skyscrapers-7 (W&B project looped-eqr-skyscrapers): in-training eval on the 10k test puzzles at 16 iterations and
# noise 0.01 every 2000 steps, training-batch exact accuracy and ACT steps averaged over steps 18k-20k.
SKY = json.load(open(os.path.join(_HERE, "sky_wandb.json")))
