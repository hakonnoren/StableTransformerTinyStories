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

Phase 1 (data) is done; phases 2–4 below are not started.

| file | role |
|---|---|
| `rna_design.py` | task definition: structure utilities, design-like structure sampler, Eterna-style sequence designer, energy-gap adaptive walk, folding / verification (ViennaRNA), token encoding |
| `build_rna_dataset.py` | EqR-format datasets: nested training sets, one shared test set, optional Eterna100-V2 test set |
| `test_rna_design.py` | 9 tests, including a builder smoke test (`python -m pytest -q looped/eqr/rna/test_rna_design.py`; needs `pip install ViennaRNA`) |

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

## Next phases

**2. EqR integration.**
- *Attention without FlashAttention.* EqR's `Attention` calls `flash_attn_func` on CUDA whenever the head
  size is a multiple of 8, and our env has no FlashAttention (the Sudoku model does not use attention). A small
  overlay patch, falling back to `F.scaled_dot_product_attention` when `flash_attn_func is None`, avoids building
  it; PyTorch's SDPA runs a flash kernel on H100 anyway.
- *Configs.* `dataset/rna_design.yaml` and `train/eqr_rna.yaml`: `mlp_t: false`, `pos_encodings: rope`,
  `board_height`/`board_width: null`, sequence length 128. The Sudoku recipe (lr 1e-4, weight decay 1.0) is a
  starting point, but it was tuned for 1k training puzzles, not 100k.
- *Evaluator.* `evaluators/rna_refold.py` (EqR plugin, installed by the overlay): decode, refold, and report the
  solved fraction and mean base-pair distance by length bucket. Needs ViennaRNA in the env.
- *Job.* `TASK=rna` in `jobs/looped/sudoku_eqr.slurm`, as for Skyscrapers.
- *Check* that the coupled arms (`coupled_eqr.py`) run with the attention block; they were only used with
  `mlp_t`.

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
