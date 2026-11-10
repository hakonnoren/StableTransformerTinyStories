# Multi-hop composition: long chains with the Sudoku couplings and methods

**Task.** Kohli et al., *Loop, Think & Generalize* (arXiv 2604.07822; code: OSU-NLP-Group/Loop-Think-Generalize).

- A knowledge graph with 200 entities and 10 relations.
- Query: a head entity and k relations. Answer: the tail entity, predicted in one forward pass.
- Model: a GPT-2 stack of 4 layers, iterated R times with no positional embedding.
- The released dynamic-recurrence model `r_dyn` was trained with R ~ Poisson(4), clipped to [2, 8], and an
  easy-to-hard curriculum.
  - It learned up to 22 hops.
  - With more iterations it extrapolates to about 30.
  - It reaches almost no 40-hop chains (October note: 0.01 at 16 iterations).

The question here is the open limitation of the October note: can the couplings and training methods that worked on
Sudoku-Extreme carry this model to long chains?

**Port.** The released checkpoint is loaded into the authors' own `RecurrentGPT2Block` (imported from their checkout).
`model.Coupled` then runs one of these recurrences with it:

| coupling | update (f = the 4-layer block stack, one iteration = one evaluation of f) | read-out |
|---|---|---|
| `vanilla` | h ← f(h) | h |
| `relax` | L ← aL + (1−a)(f(H) − (1−c)H), then H likewise from L; a = e^−γ, c = e^−δ | stream updated last |
| `twin` | the same two-stream recurrence | (L + H)/2 |

- `relax` with δ = 0 is the October note's conformal relax.
- `twin` is the single-level twin of `looped/dynlab`: the copy difference is invisible to the read-out.
- The port is exact. Checked against the authors' forward:
  - the logits are identical (max |Δ| = 0);
  - `relax` with γ large and δ = 0 equals the released model at every depth.
- Damped couplings start from the released model's function. `--port-warmup` ramps c from 1 to e^−δ, and the twin's
  read-out from the last-updated stream to the mean.

**Training** (`train.py`): the authors' curriculum, starting at 23 hops for every arm.
- Add k + 1 once held-out k-hop accuracy reaches 95%.
- Decisions use a dev third of the held-out facts; evaluation uses the other two thirds.
- The authors' recipe: R ~ clip(Poisson(4), 2, 8).
- Optionally, EqR's recipe from the Sudoku runs:
  - a detached carry: U{1..64} iterations without gradient first, with probability 1/2;
  - relative training noise of 0.01.

**Evaluation** (`evaluate.py`), on 2–100 hops (test.json goes to 100; training stops at 40):
- noise-free accuracy at depths 1–256;
- the authors' adaptive halting (KL + entropy);
- test noise × breadth, with single, vote, most-converged and any selection;
- overthinking (right at 8, wrong at 128);
- training samples against test facts (the gap).

**Running it on IDUN**

```bash
bash looped/multihop/setup_multihop.sh          # once, login node: upstream code, data, checkpoints, env, tests
sbatch jobs/looped/multihop.slurm               # 7 tasks, seed 0 (bash jobs/looped/multihop.slurm list)
python looped/multihop/summarize.py /cluster/work/$USER/multihop_runs/<job>
```

**Local tests:** `LTG_DIR=<checkout> LTG_DATA=<data/multi_hop> python -m pytest -q looped/multihop/test_multihop.py`
(7 tests, a tiny random model).
