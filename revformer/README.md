# RevFormer — reversible-coupling transformer

A reversible architecture set up for an **as-equal-as-possible** comparison
against the vanilla GPT baseline (`GPTModel` in [`../model.py`](../model.py)).

The attention / MLP / LayerNorm primitives are imported **verbatim** from
`../model.py` and run at half width, so the only architectural difference vs.
the baseline is the reversible coupling and the volume scaling.

## The block

The internal state carries two streams `(x, z)`, each of width `d = n_embd`
(state is `2*n_embd` wide). With per-dimension scale vectors `γ`, `α` of length
`d`:

```
x_new = exp(-γ_used) ⊙ (x + Attn(LN₁(z)))
z_new = exp(-α_used) ⊙ z + MLP(LN₂(x_new))
```

Per-position log-volume change of a block is `log|det| = -Σ(γ_used + α_used)`.

## Width matching (no manual doubling)

Attn/MLP run at width `d = n_embd`, i.e. the **same** width as the baseline.
Pass the same `--n_embd` / `--n_head` to both — no doubling needed:

```
# baseline
python train.py --arch baseline   --n_embd 384 --n_head 6  ...

# reversible (attn/MLP also run at width 384)
python train.py --arch reversible --n_embd 384 --n_head 6  --rev_regime vpm_scaling ...
```

The per-block attn+MLP parameter counts then match the baseline **exactly**.
The state is `2*n_embd` wide (two streams), so the embedding / `lm_head` /
final-LayerNorm params are ~2× the baseline's — inherent to the reversible
coupling, which carries two streams.

## Volume regimes (`--rev_regime`)

The only difference between the four is how `γ/α` are centered, which sets each
block's log-volume change. Increasing freedom top to bottom:

| `--rev_regime`  | Volume behavior |
|-----------------|-----------------|
| `vpb_baseline`  | `γ = α = 0`, frozen. Plain identity-residual reversible block, no scaling. |
| `vpb_scaling`   | Each block self-centers `γ/α` ⇒ `log|det| = 0` in **every** block; `γ/α` trainable. |
| `vpm_scaling`   | Global centering across all blocks ⇒ total `log|det|` of the stack `= -rev_lambda` (default; `rev_lambda=0` ⇒ volume preserved across depth, blocks free to expand/contract). |
| `vf_scaling`    | No centering; `γ/α` apply directly. Volume is free to change (deliberately not VP). |

Extra flags: `--rev_lambda` (vpm only), `--rev_epsilon`, `--rev_randn_init`,
`--rev_tanh`.

## Notes

- Initialization, optimizer parameter grouping, data pipeline, and eval loop are
  shared with the baseline through `train.py`, so the comparison isolates the
  architecture.
- This folder is a clean reimplementation wired into this codebase. The sibling
  [`../reversible/`](../reversible/) folder is a reference copy from a different
  (`mingpt`-based) project and is **not** used by `train.py`.
