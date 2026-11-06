# Kick / relax couplings inside EqR (Sudoku-Extreme)

Runs the looped couplings of `looped/model.py` as the recurrent update of
[EqR](https://github.com/locuslab/EqR) (Huang, Geng & Kolter, arXiv:2605.21488), on EqR's own
Sudoku-Extreme pipeline. It reproduces the Sudoku table of `looped/docs/looped_summary.pdf` and adds the
constrained-damping arms of `looped/HANDOFF.md` §5. The October code behind that table was not kept, so
this is a re-implementation.

| file | role |
|---|---|
| `coupled_eqr.py` | `CoupledEqRModel`: EqR's model with the state update replaced by kick / relax (installed as `models/coupled_eqr.py`) |
| `pretrain_rate_groups.patch` | EqR `pretrain.py`: learned rates go in an optimiser group without weight decay (no-op for EqR's models) |
| `pretrain_wandb_resume.patch` | EqR `pretrain.py`: skip setting `wandb.run._step` on resume, which wandb >= 0.31 (IDUN's env) refuses, so every resume crashed |
| `install_overlay.sh` | copies `coupled_eqr.py` (and the tests) into an EqR checkout and applies both patches |
| `setup_eqr.sh` | one-time IDUN setup on the login node: pinned clone, overlay, conda env, adam-atan2 CUDA build, data + released checkpoint |
| `test_coupled_eqr.py` | 24 tests: same parameters/init as EqR, update formulas, exact inverses, log-det, schedules, ACT gradients, `inject_h`, `self_read`, `twin_l` and EqR's order |
| `summarize_sudoku.py` | per-arm tables (mean ± sd over seeds, one row per checkpoint step): accuracy / residual / overthinking from the evaluate.py JSONs, the dynamics below, and the depth × breadth table |
| `dynamics_eqr.py` | per run: learned rates at every checkpoint (+ linear-part radius); noise-free trajectory of 768 test puzzles to 256 iterations (fixed-point and 2-cycle fractions, overthinking, stop-signal rate); spectral radius of one iteration (JVP power iteration, checked against dense eigenvalues) |

Jobs:
- `jobs/looped/eqr_sanity.slurm`: checks, plus the README's released-checkpoint eval.
- `jobs/looped/sudoku_eqr.slurm`: 15 arms (7 from round 1, 4 from round 2, 4 from round 3), one seed by default
  (`SEEDS="0 1 2"` for more). Runs `dynamics_eqr.py` after its evals. Resumes a run from its latest
  checkpoint when there is none at `MAX_STEPS`, which is how runs are extended from 20k to 50k steps.
- `jobs/looped/sudoku_dynamics.slurm`: the dynamics for runs that are already trained.
- `jobs/looped/sudoku_scaling.slurm`: EqR's own test-time scaling evaluation (depth × breadth, noise 0.5,
  convergence top-4 / majority vote / any-correct) on trained runs and on EqR's released checkpoint.
  `GRID` entries are `depth:breadth[:noise]`, so one job can sweep the test-time noise (fixed std, not
  scaled to the state, whose size differs between arms) or go past depth 256 (the batch shrinks with
  depth there, since evaluate.py keeps every iteration's logits when breadth > 1).

## Running on IDUN

```bash
bash looped/eqr/setup_eqr.sh                 # login node; EqR in /cluster/work/$USER/EqR, env in .../conda_envs/eqr
sbatch jobs/looped/eqr_sanity.slurm          # expect ~98.6% top-k exact on the released checkpoint
sbatch --array=0-6 jobs/looped/sudoku_eqr.slurm   # round 1 (seed 0)
python looped/eqr/summarize_sudoku.py /cluster/work/$USER/eqr_runs/sudoku_<array job id>
```

Round 2, into round 1's run root (`JOB_TAG`), so one summary covers both:

```bash
R=/cluster/work/$USER/eqr_runs/sudoku_25939526
sbatch --export=ALL,JOB_TAG=25939526 --array=7-9 jobs/looped/sudoku_eqr.slurm                 # round 2, 20k
sbatch --export=ALL,JOB_TAG=25939526 --array=9,11-14 jobs/looped/sudoku_eqr.slurm             # round 3, 20k
sbatch --time=16:00:00 --export=ALL,JOB_TAG=25939526,MAX_STEPS=50000 --array=0,4,6 \
       jobs/looped/sudoku_eqr.slurm                                                            # extend to 50k
sbatch --export=ALL,RUN_ROOT=$R,CKPT_STEP=20000 jobs/looped/sudoku_scaling.slurm               # depth x breadth
sbatch --export=ALL,RUN_ROOT=$R,ARMS="eqr relax3_leak1 kick_pos",CKPT_STEP=20000,\
       GRID="64:16:0.1 64:16:0.25 64:16:1.0" --array=0-2 jobs/looped/sudoku_scaling.slurm      # noise sweep
sbatch --time=20:00:00 --export=ALL,RUN_ROOT=$R,ARMS="eqr relax3_leak1 kick_pos released",\
       CKPT_STEP=20000,GRID="16:1 64:1 256:1 1024:1 1024:16" --array=0-3 \
       jobs/looped/sudoku_scaling.slurm                                                        # depth to 1024
```

After changing `coupled_eqr.py` or a patch, rerun `install_overlay.sh`. The training job refuses to start if the
copy in the EqR checkout differs from this repo's or `pretrain.py` lacks a patch.

## The port

EqR's update is `h ← (1−λ)h + λ·B(h + u) + noise` with λ = 0.95. B is its two-layer reasoning module, and
its force reads the state itself. The coupled model keeps B (same parameters and initialisation) but feeds
each stream's force only the other stream plus the input injection x:

```
kick :  L ← a (L + B(H + x))                  H ← b H + B(L + x)
relax:  L ← a L + (1−a)(c H + B(H + x))        H ← b H + (1−b)(c L + B(L + x))
```

Here a = e^−γ, b = e^−α and c = e^−δ. Readout is from H, which plays the role of EqR's z_H.

`inject_h=false` (round 2) drops x from H's force, so H ← … B(L) …, as in EqR, whose z_H update has no
input. With x in both forces and one shared B, relax's update is symmetric in (H, L), and in round 1 its
two streams merged (cos(z_H, z_L) = 1.00 in every relax arm). Kick's update is not symmetric, and its
streams stayed apart (cos 0.74).

`self_read=true` is a diagnostic, not an invertible coupling: each force also reads the stream it
updates, as EqR's force does (L ← … B(L + H + x) …, H ← … B(H + L) …). Exact invertibility needs a force
that does not depend on the stream it updates, so with fast rates L becomes a function of H and the
inner L iteration of EqR/HRM disappears. Arm `relax3_leak1_injL_self` differs from `relax3_leak1_injL`
only in this, so the pair measures what the constraint costs.

Round three (invertibility no longer required) puts the inner loop back. `schedule=eqr` runs EqR's order,
L_cycles updates of L and then one of H, instead of the alternation. `twin_l=true` doubles L into twins
L, L′ that read each other where EqR's L reads itself (L ← aL + (1−a)(cL′ + B(L′ + H + x)), and the same
for L′); H reads their mean, without x. If the twins merge, the pair acts as one L that reads itself. The
twins sit in the carry's z_L, stacked along the sequence axis, so EqR's carry, reset and evaluator are
unchanged; `dynamics_eqr.py` reports their cosine (`twin_cos`). Arms: `twinL_leak1`, `twinL_leak2`, the
non-invertible reference `innerL_self_leak1` (EqR's order with self-reading forces, i.e. EqR plus relax's
cross term c · other), and `relax3_leak3` (with `relax3_leak2`) for the damping trend.

**What stays EqR's:**
- the ACT carry, halting and random state reset;
- the H_cycles schedule (all but the last recursion run without gradient);
- the Langevin noise after every block call;
- EMA, the optimiser and the evaluation.

**Deliberate differences, each with its reason in the `coupled_eqr.py` docstring:**
- **Schedule.** A coupling force does not depend on its own stream, so each latent recursion alternates
  streams: H,L,H,L,H,L,H. That is 7 half-steps, the same block-evaluation count as EqR (21 visited, 7
  differentiated per iteration).
- **State precision.** The state is fp32, because bf16 cannot represent e^−0.005.
- **Readout.** Logits come from rms_norm(H).
- **Weight decay.** Learned rates are not weight-decayed. EqR decays every parameter at 1.0 with lr 1e-4,
  which shrinks a raw rate by e^−2 over 20k steps. With no gradient at all, that takes a relax rate from
  3 (keep 5%) to about 0.9 (keep 40%). The PDF reports that learned Sudoku rates drifted to keep 28%, so
  if its port decayed the rates, that drift may be weight decay rather than learning.
