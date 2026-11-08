# Tasks for looped reasoners beyond Sudoku and Skyscrapers

Survey of training problems for EqR-style looped reasoners (2026-11-08). The aim is tasks that are useful to
solve, or that test something Sudoku and Skyscrapers cannot, in light of the Sudoku note's main finding
("certification is easy, coverage is hard", `looped/eqr/paper/main.tex` §7).

## What Sudoku and Skyscrapers need

Both are constraint problems with a fixed constraint graph and exactly one solution. Classical solvers do two things:

- **Propagation.** Sudoku has "all different" rules on 27 groups of 9 cells, handled mostly by reasoning about
  pairs of cells. In Skyscrapers each clue constrains a whole line at once (7 cells), which `skyscrapers.py`
  handles by checking all 5040 orderings of a line.
- **Branching and backtracking** on the instances propagation can't finish. Sudoku-Extreme is selected for
  needing many guesses.

EqR maps onto that directly: depth is propagation rounds, noise plus random starts are the randomized restarts
SAT solvers use, and convergence or halting acts as the verifier. With one solution and only hard rules, a wrong
end state breaks some rule and can't stay a stable fixed point. That's why picking the right start is almost free.

Six things neither puzzle has:

1. **An objective.** Wrong answers can obey every rule and just score worse, so they can be truly stable.
2. **Many right answers**, or answers that are right only up to some equivalence.
3. **Noisy inputs.** The best answer is then a probabilistic guess that no check can certify.
4. **Non-local rules** (parity, connectivity) that local propagation provably can't handle.
5. **A theory of hardness** that predicts where local dynamics fail, instead of an empirical "Extreme" rating.
6. **A use outside puzzles.**

**Architecture constraint.** EqR's Sudoku setup mixes tokens with a fixed MLP (`mlp_t: true`) and no position
encoding. The constraint graph is baked into the weights and only the clue values change between instances.
Problems with a fixed structure drop straight into the 81-token pipeline, as Skyscrapers did. Problems whose graph
changes per instance (TSP, SAT, RNA, molecules) need the attention block (`mlp_t: false`, RoPE), which EqR uses for
Maze and TRM uses for Maze and ARC. EqR itself already covers Maze.

## Top three

### 1. Quantum error-correction decoding (surface code, distance 9)

The rotated surface code at distance 9 has 81 data qubits on a 9×9 grid, the same board as Sudoku. The input is
the syndrome (80 parity checks) and the output is a correction for each qubit. It covers five of the six missing
properties:

- **It has a built-in check.** The correction must reproduce the syndrome, much like Sudoku's validity check.
- **Wrong answers can pass that check.** A correction can match the syndrome and still be wrong (wrong logical
  class), so this directly tests whether certification stays free.
- **Many corrections are equivalent.** Exact match is the wrong metric: train on the sampled error and score by
  logical class.
- **Difficulty is a physical knob with theory behind it.** Under bit-flip noise, optimal decoding maps exactly onto
  the 2D random-bond Ising model. The threshold is about 10.9% for optimal decoding and about 10.3% for the
  standard matching decoder, so state noise and temperature mean something precise here.
- **Data is unlimited** (Stim), with strong baselines: PyMatching, BP+OSD, and exact tensor-network decoding at
  small sizes.
- **It's useful.** Recurrent transformer decoders are the current best (AlphaQubit 1 and 2; a 2025 decoder for
  bivariate bicycle codes reports about 5× fewer logical errors than BP-OSD). As far as I found, none of them uses
  random-start breadth or convergence-based selection.
- **Coding theory found EqR's mechanism on its own.** "Noise-aided" bit-flipping decoders for classical LDPC codes
  add noise so the decoder escapes trapping sets, and with enough iterations they approach optimal decoding. Short
  classical codes are an easier variant, with ECCT and its diffusion version as neural baselines.

### 2. An optimization task whose wrong answers are valid

Two options on a fixed grid:

- **2D spin-glass ground states.** These are exactly solvable in polynomial time by matching (with open
  boundaries), so labels are exact at any size. It's the textbook rugged energy landscape. Test-time noise
  schedules turn into simulated annealing. Gaussian couplings give one ground state up to a global flip, while ±1
  couplings give huge degeneracy, so you get a degeneracy knob. It's the same physics as pick 1.
- **Maximum independent set on King's-lattice graphs with random holes.** This is the Rydberg-atom benchmark
  (Ebadi et al. 2022), where hardness tracks degeneracy and the number of local minima. An ILP solver gives exact
  optima at 81–144 nodes.

In both, every rule-satisfying output is a valid answer, so the question becomes whether depth and breadth find
better ones. One caution from the GNN-for-independent-set dispute: always compare against greedy and a well-tuned
simulated annealing.

### 3. Sudoku on a random graph (a cheap controlled ablation)

Keep 81 cells, 9 symbols and 20 neighbours per cell, but swap Sudoku's rows, columns and boxes for a fixed random
graph, and plant a coloring with clues. This isolates what Sudoku's box structure and symmetry augmentation
actually buy. Then sweep the number of neighbours and clues through the regimes of planted coloring. Theory
predicts a hard phase there: the planted solution can be identified, but belief propagation from random starts
gets stuck. A 2024 GNN paper targets this regime with noise-annealed dynamics.

There's also a parity variant: the same graph with XOR rules gives random XORSAT. Gaussian elimination solves it
easily, but local search finds it glassy. If the model solves it, it learned something non-local; if it fails,
that shows its search is local. Most of the Skyscrapers pipeline can be reused.

## The rest, by field

| Field | Problem | Drops into the 81-token setup? | Why it's interesting | Catch |
|---|---|---|---|---|
| Biology | RNA inverse folding: design a sequence that folds into a target structure (Eterna100) | No, needs attention | Many valid answers plus an exact external check (refold with ViennaRNA), which is breadth plus selection. Unlimited training data from folding random sequences. Real design use | Random folds may not look like designed targets. In 2019 the best methods solved about ¼ of the harder half (Eterna100-V2 is the current version) |
| Biology | Protein side-chain packing | No, needs attention | Rosetta's packer uses Monte Carlo annealing, the same mechanism | Heavy engineering |
| Materials | Discrete tomography: a binary image from projection counts on the border | Yes, border clues like Skyscrapers | NP-hard from 3 directions or more. With 2 directions answers aren't unique, so the model has to combine a learned image prior with exact rules. Real use: counting atoms per column in electron tomography | I found no few-projection neural baseline |
| Chemistry | Ground states of lattice alloys | Yes | Materials relevance | It reduces to the spin-glass pipeline |
| Energy / OR | Unit commitment: a generators × hours on/off grid | Yes (e.g. 10×24) | Solved daily by grid operators. A 2026 transformer warm-start paper finds raw network outputs often break the rules, and iterating until consistent is what EqR does | Needs MILP labels; scored by cost |
| OR | Nurse rostering / timetabling | Yes | Sudoku-like row and column rules plus an objective | Labels from CP-SAT |
| Planning | Sokoban (Boxoban, 10×10) | Only as a next-move policy | A recurrent planner solves more levels with extra thinking steps and stores its plan in its activations | EqR's output is a single answer, so it needs a loop that plays out the moves |
| Finance | Winner determination in combinatorial auctions | No, needs attention | One of the four standard ML-for-MILP benchmarks (Gasse et al. 2019) | Most finance problems are either continuous and convex (solvers are already exact) or have no ground truth, so it's a weak fit overall |
| Puzzles | Pencil Puzzle Bench (62k puzzles, 94 types, unique solutions, rule checker) | Type by type | Connectivity puzzles (Slitherlink, Nurikabe, Masyu) test non-local rules. Probabilistic TRM already used some of them | More puzzles, not more use |

## Recommendation

1. **Surface-code decoding first.** It's useful, it's on the same board, and it cleanly breaks the
   "certification is free" assumption.
2. **Then one optimization task.** Spin glasses: exact labels and the same physics.
3. **Sudoku on a random graph** as the cheap ablation.
4. **For biology,** RNA inverse folding, with the attention variant (pipeline: `looped/eqr/rna/`).

## Sources

- [EqR (arXiv 2605.21488)](https://arxiv.org/abs/2605.21488)
- [TRM](https://www.alphaxiv.org/abs/2510.04871) · [Probabilistic TRM](https://arxiv.org/html/2605.19943)
- [Pencil Puzzle Bench](https://arxiv.org/pdf/2603.02119) · [dataset](https://huggingface.co/datasets/bluecoconut/pencil-puzzle-bench)
- [AlphaQubit 2 (arXiv 2512.07737)](https://arxiv.org/abs/2512.07737v1) · [Parallel-window neural decoder (arXiv 2601.09921)](https://www.arxiv.org/abs/2601.09921)
- [ML decoding for bivariate bicycle codes](https://arxiv.org/abs/2504.13043) · [Almost-linear qLDPC decoder](https://arxiv.org/pdf/2409.01440)
- [Noise-aided GDBF decoders](https://hal-mnhn.archives-ouvertes.fr/ETIS-ASTRE/hal-01390467v1) · [Noisy GDBF](https://arxiv.org/pdf/1503.08913)
- [Denoising diffusion ECC](https://pdf.arxiv.org/pdf/2209.13533) · [Foundation decoder, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/093b793f0af498e8fa218eb5445daeb0-Paper-Conference.pdf)
- [Ebadi et al., Rydberg MIS](https://arxiv.org/pdf/2202.09372) · [Classical re-examination (PRR 5, 043277)](https://link.aps.org/doi/10.1103/PhysRevResearch.5.043277)
- [GNN vs greedy MIS dispute, reply](https://arxiv.org/abs/2302.03602)
- [Planted coloring GNN](https://arxiv.org/pdf/2408.01503) · [Statistical physics of inference](https://arxiv.org/pdf/1511.02476) · [Simulated annealing limits on planted coloring](https://arxiv.org/pdf/2206.04760)
- [XORSAT hardness for local search](https://arxiv.org/pdf/cond-mat/0408190) · [Mora–Mézard](https://arxiv.org/pdf/cond-mat/0609099)
- [Exact 2D spin-glass ground states via matching](https://arxiv.org/abs/0801.3143v3)
- [Eterna100-V2](https://daslab.stanford.edu/assets/pdfs/Koodli_et_al_Eterna100V2_bioRxiv_2025.pdf) · [Montparnasse RNA design](https://arxiv.org/pdf/2606.07562)
- [Neural atomic electron tomography review](https://doaj.org/article/290aec270ea642d09cc4a7120cd698ff)
- [Unit commitment warm start (arXiv 2604.21891)](https://arxiv.org/abs/2604.21891)
- [Sokoban recurrent planner](https://arxiv.org/pdf/2407.15421) · [FAR.AI summary](https://far.ai/post/2024-07-learned-planners/)
