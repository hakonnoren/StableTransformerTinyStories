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
| `install_overlay.sh` | copies the two files above (and the tests) into an EqR checkout |
| `setup_eqr.sh` | one-time IDUN setup on the login node: pinned clone, overlay, conda env, adam-atan2 CUDA build, data + released checkpoint |
| `test_coupled_eqr.py` | 13 tests: same parameters/init as EqR, update formulas, exact inverses, log-det, schedule, ACT gradients |
| `summarize_sudoku.py` | per-arm table (mean ± sd over seeds) from the evaluate.py JSONs, with overthinking O and changed predictions dy |

Jobs: `jobs/looped/eqr_sanity.slurm` (checks, plus the README's released-checkpoint eval) and
`jobs/looped/sudoku_eqr.slurm` (7 arms, one seed by default; `SEEDS="0 1 2"` for more).

## Running on IDUN

```bash
bash looped/eqr/setup_eqr.sh                 # login node; EqR in $USERWORK/EqR, env in $USERWORK/conda_envs/eqr
sbatch jobs/looped/eqr_sanity.slurm          # expect ~98.6% top-k exact on the released checkpoint
sbatch jobs/looped/sudoku_eqr.slurm          # 7 tasks (seed 0)
python looped/eqr/summarize_sudoku.py $USERWORK/eqr_runs/sudoku_<array job id>
```

After changing `coupled_eqr.py`, rerun `install_overlay.sh`. The training job refuses to start if the copy
in the EqR checkout differs from this repo's.

## The port

EqR's update is `h ← (1−λ)h + λ·B(h + u) + noise` with λ = 0.95. B is its two-layer reasoning module, and
its force reads the state itself. The coupled model keeps B (same parameters and initialisation) but feeds
each stream's force only the other stream plus the input injection x:

```
kick :  L ← a (L + B(H + x))                  H ← b H + B(L + x)
relax:  L ← a L + (1−a)(c H + B(H + x))        H ← b H + (1−b)(c L + B(L + x))
```

Here a = e^−γ, b = e^−α and c = e^−δ. Readout is from H, which plays the role of EqR's z_H.

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
