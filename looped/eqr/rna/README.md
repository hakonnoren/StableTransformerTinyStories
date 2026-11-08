# RNA inverse folding for EqR

Given a target secondary structure (dot-bracket, pseudoknot-free), output a sequence whose minimum-free-energy
(MFE) structure is exactly the target. The folding engine is ViennaRNA 2 with default settings (Turner 2004,
dangles 2, 37 °C), the "Vienna 2" engine of Eterna100-V2. Why it suits EqR (see `../task_survey.md`):

- **Many right answers.** A target has many solutions, so the training label is one sample of them. Exact
  agreement with the label is the wrong metric; a prediction is scored by refolding it.
- **An exact check at test time.** One fold (O(L³), about 8 ms at 128 nt) certifies a candidate. Breadth
  (random starts) followed by refolding is a legitimate solver, not an oracle as "any of B correct" is for Sudoku.
- **Variable structure.** Pairing partners change per target, so this needs EqR's attention block
  (`mlp_t: false`, 1D RoPE), not the token-mixing MLP of the Sudoku recipe.
- **A hard external benchmark.** Eterna100-V2, with published results for six design algorithms.

## Status (2026-11-08)

Phase 1 (data) and phase 2 (EqR integration and job) are done and tested on CPU; no GPU run yet. Phases 3–4
below are not started.

| file | role |
|---|---|
| `rna_design.py` | task definition: structure utilities, design-like structure sampler, Eterna-style sequence designer, energy-gap adaptive walk, folding / verification (ViennaRNA), token encoding |
| `build_rna_dataset.py` | EqR-format datasets: nested training sets, one shared test set, optional Eterna100-V2 test set |
| `rna_refold.py` | EqR evaluator `rna_refold@RNARefold`: refolds predictions, reports `rna/solved`, `rna/bp_dist`, `rna/pairs_ok`, `rna/label_agree`, solved by length; per-target CSV |
| `test_rna_design.py` | 9 tests, including a builder smoke test (`python -m pytest -q looped/eqr/rna/test_rna_design.py`; needs `pip install ViennaRNA`) |
| `test_rna_eqr.py` | 5 tests inside an EqR checkout: the evaluator, and EqR / relax / twin with the attention block on RNA batches |

Elsewhere: `../configs/dataset/rna_design.yaml`, `../configs/train/eqr_rna.yaml`, `../layers_sdpa_fallback.patch`
(all installed by `../install_overlay.sh`, which also copies the evaluator and `rna_design.py` into EqR's
`evaluators/`) and the job `jobs/looped/rna_eqr.slurm`.

### Training data

Pairs are self-consistent by construction: sample a structure, design a sequence for it, fold the sequence and
train on (fold(seq), seq). Every target is designable and its label is a verified solution. 80% of examples come
from designed sequences, 20% from random sequences (`--random-frac`).

The sampler (`StructParams`) is calibrated against the 66 Eterna100-V2 targets of length ≤ 128. Per 100 nt
(sampled and folded rows: 60 examples at 128 nt, folded after a 50-step walk):

| | helices | hairpins | interior loops | bulges | multiloops | helices of 2 / 3 / 4 bp |
|---|---|---|---|---|---|---|
| Eterna100-V2 (≤ 128 nt) | 8.6 | 3.9 | 1.7 | 1.7 | 1.2 | 36 / 21 / 19 % |
| sampled structures | 8.1 | 4.0 | 2.0 | 1.0 | 1.1 | 36 / 22 / 16 % |
| training targets (after folding) | 7.0 | 3.2 | 2.0 | 1.1 | 0.7 | 20 / 25 / 21 % |

Eterna also has 11% single-pair helices, which the sampler leaves out: placed at random they almost never fold.
A designed sequence folds back into its sampled structure in 67 / 50 / 28% of cases at 40 / 80 / 128 nt with a
50-step walk (54% overall at the default 100 steps). The rest still give valid pairs, just with a target that
differs from the sampled structure. Generation costs about 110 ms per example on one core (3.2 CPU-hours for
105k examples).

Tokens (shared vocabulary): PAD 0, `.` 1, `(` 2, `)` 3, A 4, C 5, G 6, U 7. Inputs hold the structure, labels the
sequence, with 0 (ignored) past the target's length. Decode predictions with an argmax over A, C, G, U only
(`rna_design.decode_sequence`).

### Baselines: Eterna100-V2, 66 targets of length ≤ 128 (Vienna 2)

From the benchmark repository's `data/results.tsv` (one trial per puzzle):

| DesiRNA | NEMO | LEARNA (retrained V2) | SentRNA | EternaBrain | RNAinverse |
|---|---|---|---|---|---|
| 64 | 59 | 52 | 50 | 35 | 16 |

ViennaRNA 2.8.0 reproduces the benchmark: 99 of 100 V2 sample solutions fold into their targets (puzzle 100 is
the known unsolvable one).

## Building the data (IDUN)

On the login node, with the EqR env:

```bash
module load Anaconda3/2024.02-1; eval "$(conda shell.bash hook)"; conda activate /cluster/work/haakno/conda_envs/eqr
pip install ViennaRNA==2.8.0
curl -L -o /cluster/work/haakno/eterna100_puzzles.tsv \
  https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/data/eterna100_puzzles.tsv
srun --account=share-ie-imf --partition=CPUQ --cpus-per-task=32 --mem=16G --time=01:00:00 \
  python looped/eqr/rna/build_rna_dataset.py --out-root /cluster/work/haakno/EqR/data \
    --n-train 10000,100000 --n-test 5000 --len-max 128 --workers 32 \
    --eterna-tsv /cluster/work/haakno/eterna100_puzzles.tsv
```

This writes `rna-L128-10k`, `rna-L128-100k` and `rna-eterna100v2-L128` under `EqR/data`.

## Training (IDUN)

After a `git pull`, once on the login node: `bash looped/eqr/install_overlay.sh /cluster/work/haakno/EqR`
(ViennaRNA must be in the env, as above). Then the pilot, EqR and the twin (arms 0 and 1) on 100k targets:

```bash
sbatch --array=0-1 jobs/looped/rna_eqr.slurm
```

Arms: `eqr`, `twinL_leak2`, `relax3_leak1`, `kick_pos` (array 0–3; definitions as in `sudoku_eqr.slurm`).
`RNA_DATA=rna-L128-10k` for the small set, `MAX_STEPS`, `SEEDS` and `JOB_TAG` as in the Sudoku job. The job
refuses to start if the overlay files in EqR differ from this checkout, a patch is missing, ViennaRNA is absent or
the data is not built. After training it evaluates the test set and Eterna100-V2 at 16 / 64 / 256 iterations,
prints `rna/solved` for each, and runs `dynamics_eqr.py`.

What phase 2 changed in EqR (overlay):
- **Attention without FlashAttention** (`layers_sdpa_fallback.patch`). EqR's `Attention` refused to run on CUDA
  without `flash_attn`, which our env does not have (the Sudoku model does not use attention). It now falls back
  to PyTorch's `scaled_dot_product_attention`, which runs a flash kernel on Ampere and Hopper anyway.
- **Forward-mode AD through attention.** `dynamics_eqr.py` and `analyze_maps.py` compute Jacobian-vector
  products, which the fused attention kernels do not support; both now use SDPA's math kernel for the JVPs
  (no effect on the Sudoku models, which have no attention).
- **Config.** `train/eqr_rna.yaml`: `mlp_t: false`, `pos_encodings: rope`, `board_height`/`board_width: null`,
  sequence length 128 (+16 puzzle-embedding slots). Otherwise the Sudoku recipe (lr 1e-4, weight decay 1.0),
  which was tuned for 1k training puzzles, not 100k: a starting point.

Tested on CPU (tiny model, 256 training targets): training with in-training refold evaluation, `evaluate.py`
on the test set and on Eterna100-V2 via `dataset_data_path`, the twin arm, `dynamics_eqr.py`, and the job script
end to end (with the training step skipped). Not tested: CUDA, the real model size, GPU memory.

Caveats:
- EqR's halting head is trained on "prediction == label". For RNA that rarely holds even when the prediction
  solves the target, so the head will learn not to halt and runs go to `halt_max_steps`.
- EqR's `accuracy` / `exact_accuracy` (and `dynamics_eqr.py`'s "exact") measure agreement with the label, one
  solution among many. Read `rna/*` instead.
- The 40 GB GPU floor in the job is the Sudoku model's; attention over 144 positions at batch 768 is not measured.

## Next phases

**3. Breadth with verification.** EqR's evaluator only sees the first random start's predictions when
`different_init > 1`, so a separate script: B random starts per target, refold every candidate, and report the
single-start solve rate, any-solved (legitimate here), and the convergence-selected and halting-selected starts.
Run on the generated test set and on Eterna100-V2.

**4. Questions for the dynamics.**
- Are predicted pairs complementary? Per-position argmax of marginals can give G·G where the label distribution
  puts GC and CG at equal odds. Does iteration break this symmetry?
- Do random starts reach different correct sequences (many right attractors) or one? Coverage then means
  something different than on Sudoku.
- Does convergence still pick solved starts when wrong candidates can be stable?

**Open choices.**
- One label per target (now) vs several solutions per target as one group, so that EqR samples a different
  solution each epoch.
- Dot-bracket input (now: the model must match brackets itself) vs an explicit partner channel.
- `--len-max 128` covers 66 of the 100 Eterna100 puzzles; all of them need 400.
