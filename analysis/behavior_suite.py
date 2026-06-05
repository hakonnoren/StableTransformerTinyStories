"""
Behavioral test suite for trained TinyStories checkpoints — probes the
computational-mechanism differences we found between architectures
(reversible front-loads + monotone-expands; baseline/yurii compress mid-network;
presymp leaks future tokens).

Probes:
  A. causality + perturbation sensitivity
       - leak: does changing input token j move logits at i<j? (causal => 0)
       - context sensitivity: ||Δ next-token dist|| at the last position when an
         earlier token is flipped (how strongly past context steers prediction)
       - embedding-noise robustness: KL(p0 || p_noise) vs Gaussian σ on the embeddings
  B. logit-lens depth readout: per-layer early-exit next-token loss (where does
     the prediction crystallize through depth?)  [models with .blocks]
  C. generation under temperature: distinct-1/2 + rep-4 vs sampling temperature
  D. minimal-pair language probes: P(target) vs P(foil) on hand-built contrasts
     (grammar / attribute recall / induction-copy), single-next-token (leak-safe)

NOTE on presymp: its *teacher-forced eval loss* is invalid (future-token leak),
so it is excluded from B (loss-based). But free-running generation and
single-next-token probes (C, D) only read the last position with no future in
the input, so presymp is included there and flagged.

Run (needs an env with numpy>=2-compatible unpickling; we shim numpy<2):
    python analysis/behavior_suite.py --ckpt_dir fetched/cmp_24693039 --out_dir plots/behavior
"""
import argparse
import os
import sys

# ---- shim so numpy<2 can unpickle checkpoints saved with numpy>=2 ----
import numpy as np
try:
    import numpy.core, numpy.core.multiarray, numpy.core.numeric  # noqa
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
except Exception:
    pass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ModelConfig, GPTModel, YuriiFormerModel, PresympModelETDAB2  # noqa: E402
from revformer import RevFormerModel, RevConfig  # noqa: E402

EOT = 50256  # gpt2


# ---------------------------------------------------------------- loader
def _model_cfg(cfg_dict):
    fields = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd",
              "dropout", "bias", "presymp_mlp_use_attn_vel"}
    return ModelConfig(**{k: v for k, v in cfg_dict.items() if k in fields})


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg, a = _model_cfg(ck["cfg"]), ck.get("args", {})
    arch = a.get("arch", "baseline")
    g = a.get
    if arch == "baseline":
        m = GPTModel(cfg, no_mlp=g("no_mlp", False))
    elif arch == "reversible":
        m = RevFormerModel(cfg, RevConfig(
            regime=g("rev_regime", "vpm_scaling"), lambd=g("rev_lambda", 0.0),
            epsilon=g("rev_epsilon", 1.0), randn_init=g("rev_randn_init", False),
            tanh_scale=g("rev_tanh", False)))
    elif arch == "yurii_lt":
        m = YuriiFormerModel(cfg, use_v0_init=not g("no_v0_init", False),
                             noise_eta=g("yurii_noise_eta", 0.0), noise_gamma=g("yurii_noise_gamma", 0.55),
                             noise_loc=g("yurii_noise_loc", "v"), restart_mode=g("yurii_restart", "none"),
                             restart_min_layer=g("yurii_restart_min_layer", 1), no_mlp=g("no_mlp", False))
    elif arch == "presymp_etd_ab2":
        m = PresympModelETDAB2(cfg, h=g("presymp_h", 1.0), t0=g("presymp_t0", 1.0),
                               eta_mu=g("eta_mu"), eta_log_coef=g("eta_log_coef"), eta_lin_coef=g("eta_lin_coef"),
                               eta_log_init=g("eta_log_init"), eta_lin_init=g("eta_lin_init"),
                               eta_learnable=g("eta_learnable", False), eta_mode=g("eta_mode", "log"),
                               eta_init=g("eta_init"), eta_clip=g("eta_clip", 50.0),
                               presymp_lnp=g("presymp_lnp", "end"), use_v0_init=not g("no_v0_init", False),
                               mlp_use_attn_vel=g("presymp_mlp_use_attn_vel", False),
                               mlp_use_p_vel=g("presymp_mlp_use_p_vel", False),
                               no_mlp=g("no_mlp", False), lookahead=g("presymp_lookahead", False))
    else:
        raise ValueError(f"unknown arch {arch}")
    m.load_state_dict(ck["model"])
    m.eval().to(device)
    return m, {"arch": arch, "regime": g("rev_regime"), "best_val": ck.get("best_val"),
               "is_causal": arch != "presymp_etd_ab2"}


def discover(ckpt_dir):
    out = {}
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.startswith("best_") and f.endswith(".pt"):
                name = os.path.basename(root)
                out[name] = os.path.join(root, f)
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- data
PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom had a big red ball. He liked to play with it every day.",
    "The sun was shining and the birds were singing in the trees.",
    "Once there was a dog. The dog wanted to find a bone.",
    "Anna and Ben went to the park. They saw a cat near the",
    "It was a cold winter day. The snow was white and soft.",
]


def _logits(model, ids, device):
    with torch.no_grad():
        out = model(ids.to(device))
        return (out[0] if isinstance(out, (tuple, list)) else out).float().cpu()


# ---------------------------------------------------------------- Probe A
@torch.no_grad()
def probe_perturbation(models, enc, device):
    print("\n== A. Causality + perturbation sensitivity ==")
    ids = torch.tensor([enc.encode_ordinary(PROMPTS[1])[:24]], dtype=torch.long)
    T = ids.shape[1]; j = T // 2
    print(f"{'model':26s}{'leak(i<j)':>12}{'ctx_sens(last)':>16}")
    leak, ctx = {}, {}
    for name, (m, meta) in models.items():
        l1 = _logits(m, ids, device)
        ids2 = ids.clone(); ids2[0, j] = (int(ids[0, j]) + 7) % m.cfg.vocab_size
        leak[name] = (l1[:, :j] - _logits(m, ids2, device)[:, :j]).abs().max().item()
        ids3 = ids.clone(); ids3[0, 2] = (int(ids[0, 2]) + 11) % m.cfg.vocab_size  # flip an early token
        p0 = F.softmax(l1[0, -1], -1); p3 = F.softmax(_logits(m, ids3, device)[0, -1], -1)
        ctx[name] = 0.5 * (p0 - p3).abs().sum().item()  # TV at last position
        print(f"{name:26s}{leak[name]:12.2e}{ctx[name]:16.4f}")

    # prompt-corruption robustness (leak-safe): TV(last-token dist vs clean) when a
    # fraction of prompt tokens is randomly replaced. Higher = less robust.
    print("\n  prompt-corruption robustness — mean TV(last-token dist vs clean) vs corruption rate:")
    rates = [0.05, 0.1, 0.2, 0.4]; nrep = 5
    L = 12
    seqs = [enc.encode_ordinary(p) for p in PROMPTS]
    base_ids = torch.tensor([s[:L] for s in seqs if len(s) >= L][:4], dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    curves = {}
    for name, (m, meta) in models.items():
        clean = [F.softmax(_logits(m, base_ids[k:k+1], device)[0, -1], -1) for k in range(base_ids.shape[0])]
        row = []
        for r in rates:
            tvs = []
            for _ in range(nrep):
                for k in range(base_ids.shape[0]):
                    seq = base_ids[k:k+1].clone(); Tk = seq.shape[1]; nco = max(1, int(r * Tk))
                    pos = torch.randperm(Tk, generator=gen)[:nco]
                    seq[0, pos] = torch.randint(0, m.cfg.vocab_size, (nco,), generator=gen)
                    pc = F.softmax(_logits(m, seq, device)[0, -1], -1)
                    tvs.append(0.5 * (clean[k] - pc).abs().sum().item())
            row.append(float(np.mean(tvs)))
        curves[name] = (rates, row)
        print(f"  {name:26s} " + "  ".join(f"{int(r*100)}%:{v:.3f}" for r, v in zip(rates, row)))
    return {"leak": leak, "ctx_sens": ctx, "corruption": curves}


# ---------------------------------------------------------------- Probe B
@torch.no_grad()
def probe_logit_lens(models, enc, device):
    print("\n== B. Logit-lens depth readout (early-exit next-token loss per layer) ==")
    print("   (lower = prediction already formed by this layer; presymp skipped: leak)")
    ids = torch.tensor([enc.encode_ordinary(" ".join(PROMPTS))[:128]], dtype=torch.long).to(device)
    x = ids[:, :-1]; y = ids[:, 1:]
    out = {}
    for name, (m, meta) in models.items():
        if not hasattr(m, "blocks") or not meta["is_causal"]:
            continue
        caps = [None] * len(m.blocks)
        hs = []
        for i, blk in enumerate(m.blocks):
            def mk(i):
                def h(_mod, _inp, o):
                    caps[i] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
                return h
            hs.append(blk.register_forward_hook(mk(i)))
        m(x)
        for h in hs:
            h.remove()
        losses = []
        for hcap in caps:
            logits = m.lm_head(m.ln_f(hcap))
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        out[name] = losses
        print(f"  {name:26s} " + " ".join(f"{v:5.2f}" for v in losses) + f"   (min@L{int(np.argmin(losses))})")
    return out


# ---------------------------------------------------------------- Probe C
@torch.no_grad()
def probe_temperature(models, enc, device, temps=(0.3, 0.7, 1.0, 1.3), n_new=120, n_prompts=4):
    print("\n== C. Generation under temperature (distinct-1/2, rep-4) ==")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train import gen_diversity, gen_diversity_mean
    prompts = [torch.tensor([enc.encode_ordinary(p)[:16]], dtype=torch.long) for p in PROMPTS[:n_prompts]]
    out = {}
    for name, (m, meta) in models.items():
        out[name] = {}
        for tmp in temps:
            conts = []
            for pr in prompts:
                g = m.generate(pr.to(device), max_new_tokens=n_new, temperature=tmp,
                               top_k=0, do_sample=True, eos_token_id=EOT)
                conts.append(g[0].tolist()[pr.shape[1]:])
            d = gen_diversity_mean(conts)
            out[name][tmp] = d
        line = "  ".join(f"T{t}:d1={out[name][t].get('distinct_1',0):.2f},rep4={out[name][t].get('rep_4',0):.2f}" for t in temps)
        flag = "" if meta["is_causal"] else "  (presymp: gen ok, eval-loss invalid)"
        print(f"  {name:26s} {line}{flag}")
    return out


# ---------------------------------------------------------------- Probe D
MINIMAL_PAIRS = [
    # (prompt, target_word, foil_word, family)  — single-token target/foil (leak-safe)
    ("She was very happy because she", " was", " were", "grammar"),
    ("The little boy ran fast. He", " was", " were", "grammar"),
    ("The dogs played outside. They", " were", " was", "grammar"),
    ("My friend is nice. We", " are", " is", "grammar"),
    ("Tom had a big red ball. The color of the ball was", " red", " blue", "recall"),
    ("Lily found a small cat. The animal she found was a", " cat", " dog", "recall"),
    ("Sam saw a blue car. The car was", " blue", " red", "recall"),
    ("The girl had a small dog. Her pet was a", " dog", " cat", "recall"),
    ("dog cat dog cat dog", " cat", " dog", "induction"),
    ("red blue red blue red", " blue", " red", "induction"),
    ("one two one two one", " two", " one", "induction"),
    ("big small big small big", " small", " big", "induction"),
    ("The sun is very", " hot", " cold", "world"),
    ("Ice is very", " cold", " hot", "world"),
    ("Fire is very", " hot", " cold", "world"),
    ("Once upon a", " time", " house", "collocation"),
    ("She opened the", " door", " sky", "collocation"),
]


@torch.no_grad()
def probe_minimal_pairs(models, enc, device):
    print("\n== D. Minimal-pair probes: P(target) > P(foil)?  (single next token, leak-safe) ==")
    fams = sorted(set(f for *_, f in MINIMAL_PAIRS))
    print(f"{'model':26s}" + "".join(f"{f[:9]:>11}" for f in fams) + f"{'overall':>11}")
    out = {}
    for name, (m, meta) in models.items():
        per_fam = {f: [] for f in fams}
        margins = []
        for prompt, tgt, foil, fam in MINIMAL_PAIRS:
            pid = enc.encode_ordinary(prompt)
            ti = enc.encode_ordinary(tgt)[0]
            fi = enc.encode_ordinary(foil)[0]
            logits = _logits(m, torch.tensor([pid], dtype=torch.long), device)[0, -1]
            lp = F.log_softmax(logits, -1)
            correct = float(lp[ti] > lp[fi])
            per_fam[fam].append(correct)
            margins.append((lp[ti] - lp[fi]).item())
        fam_acc = {f: float(np.mean(per_fam[f])) for f in fams}
        overall = float(np.mean([c for v in per_fam.values() for c in v]))
        out[name] = {"fam_acc": fam_acc, "overall": overall, "mean_margin": float(np.mean(margins))}
        print(f"{name:26s}" + "".join(f"{fam_acc[f]:>11.2f}" for f in fams) + f"{overall:>11.2f}")
    return out


# ---------------------------------------------------------------- Probe E (E1+E4)
# Subject-verb NUMBER agreement on a TinyStories-vocabulary template grammar.
# E1: at which layer does the agreement decision resolve (logit-lens margin per
#     layer)?  E4: accuracy vs number of opposite-number attractors in between.
SUBJECTS = [("cat", "cats"), ("dog", "dogs"), ("boy", "boys"),
            ("girl", "girls"), ("bird", "birds")]
ATTR_SING = ["tree", "house", "car", "box", "road"]
ATTR_PLUR = ["trees", "houses", "cars", "boxes", "roads"]
VERBS = [(" is", " are"), (" was", " were")]  # (singular, plural)


def agreement_pairs(n_distractors):
    """(prompt, correct_verb, foil_verb, number). Attractors take the OPPOSITE
    number to the head subject (the hard agreement-attractor case)."""
    pairs = []
    for i, (s_sing, s_plur) in enumerate(SUBJECTS):
        for num, subj in (("sing", s_sing), ("plur", s_plur)):
            attrs = ATTR_PLUR if num == "sing" else ATTR_SING  # opposite number
            for vs, vp in VERBS:
                correct, foil = (vs, vp) if num == "sing" else (vp, vs)
                if n_distractors == 0:
                    p = f"The {subj}"
                elif n_distractors == 1:
                    p = f"The {subj} near the {attrs[i % len(attrs)]}"
                else:
                    p = f"The {subj} near the {attrs[i % len(attrs)]} by the {attrs[(i+1) % len(attrs)]}"
                pairs.append((p, correct, foil, num))
    return pairs


@torch.no_grad()
def layer_last_logits(model, ids, device):
    """Per-layer early-exit logits at the LAST position (logit lens). None if the
    model has no .blocks (e.g. presymp)."""
    if not hasattr(model, "blocks"):
        return None
    caps = [None] * len(model.blocks); hs = []
    for i, blk in enumerate(model.blocks):
        def mk(i):
            def h(_m, _inp, o):
                caps[i] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
            return h
        hs.append(blk.register_forward_hook(mk(i)))
    model(ids.to(device))
    for h in hs:
        h.remove()
    return [model.lm_head(model.ln_f(c))[0, -1].float().cpu() for c in caps]


@torch.no_grad()
def probe_agreement(models, enc, device, max_d=2):
    print("\n== E. Subject-verb agreement: accuracy vs #attractors + resolution depth ==")
    accs = {n: {} for n in models}
    for d in range(max_d + 1):
        pairs = agreement_pairs(d)
        for name, (m, meta) in models.items():
            c = 0
            for prompt, cor, foil, _ in pairs:
                lp = F.log_softmax(_logits(m, torch.tensor([enc.encode_ordinary(prompt)]), device)[0, -1], -1)
                c += int(lp[enc.encode_ordinary(cor)[0]] > lp[enc.encode_ordinary(foil)[0]])
            accs[name][d] = c / len(pairs)
    print("  agreement accuracy vs #attractors (E4):")
    print("  " + f"{'model':26s}" + "".join(f"{'d='+str(d):>8}" for d in range(max_d + 1)))
    for name in models:
        print("  " + f"{name:26s}" + "".join(f"{accs[name][d]:>8.2f}" for d in range(max_d + 1)))

    print("\n  resolution depth — first layer agreement margin>0 and stays (d=0, E1):")
    pairs0 = agreement_pairs(0)
    res_depth, margins_by_layer = {}, {}
    for name, (m, meta) in models.items():
        if not hasattr(m, "blocks") or not meta["is_causal"]:
            continue
        NL = len(m.blocks); accum = np.zeros(NL); depths = []
        for prompt, cor, foil, _ in pairs0:
            ci = enc.encode_ordinary(cor)[0]; fi = enc.encode_ordinary(foil)[0]
            layers = layer_last_logits(m, torch.tensor([enc.encode_ordinary(prompt)]), device)
            margins = np.array([(F.log_softmax(lg, -1)[ci] - F.log_softmax(lg, -1)[fi]).item() for lg in layers])
            accum += margins
            depth = NL
            for L in range(NL):
                if np.all(margins[L:] > 0):
                    depth = L; break
            depths.append(depth)
        margins_by_layer[name] = accum / len(pairs0)
        res_depth[name] = float(np.mean(depths))
        print(f"    {name:26s} mean_res_depth={res_depth[name]:.2f}/{NL}  final_margin={margins_by_layer[name][-1]:+.3f}")
    return {"acc": accs, "res_depth": res_depth, "margins_by_layer": margins_by_layer, "max_d": max_d}


# ---------------------------------------------------------------- plots
def make_plots(lens_out, temp_out, pert_out, agr_out, out_dir, fmt="pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    # corruption robustness
    corr = (pert_out or {}).get("corruption")
    if corr:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, (rates, row) in corr.items():
            ax.plot([r * 100 for r in rates], row, marker="o", label=name)
        ax.set_xlabel("% prompt tokens corrupted"); ax.set_ylabel("TV(last-token dist vs clean)")
        ax.set_title("Robustness to prompt corruption"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); p = os.path.join(out_dir, f"corruption.{fmt}"); fig.savefig(p, dpi=140); plt.close(fig)
        print("wrote", p)
    # logit-lens
    if lens_out:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, losses in lens_out.items():
            ax.plot(range(len(losses)), losses, marker="o", label=name)
        ax.set_xlabel("layer (early-exit readout)"); ax.set_ylabel("next-token loss")
        ax.set_title("Logit-lens: where the prediction forms"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); p = os.path.join(out_dir, f"logit_lens.{fmt}"); fig.savefig(p, dpi=140); plt.close(fig)
        print("wrote", p)
    # temperature: rep_4 + distinct_2 vs temp
    if temp_out:
        for metric in ("rep_4", "distinct_2", "distinct_1"):
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for name, byT in temp_out.items():
                ts = sorted(byT); ax.plot(ts, [byT[t].get(metric, float("nan")) for t in ts], marker="o", label=name)
            ax.set_xlabel("temperature"); ax.set_ylabel(metric)
            ax.set_title(f"Generation {metric} vs temperature"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout(); p = os.path.join(out_dir, f"temp_{metric}.{fmt}"); fig.savefig(p, dpi=140); plt.close(fig)
            print("wrote", p)
    # agreement: accuracy vs #attractors (E4) + margin vs layer (E1)
    if agr_out:
        md = agr_out["max_d"]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, byd in agr_out["acc"].items():
            ax.plot(range(md + 1), [byd[d] for d in range(md + 1)], marker="o", label=name)
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xlabel("# opposite-number attractors"); ax.set_ylabel("agreement accuracy")
        ax.set_ylim(0, 1.02); ax.set_xticks(range(md + 1))
        ax.set_title("Subject-verb agreement vs attractors (E4)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); p = os.path.join(out_dir, f"agreement_accuracy.{fmt}"); fig.savefig(p, dpi=140); plt.close(fig)
        print("wrote", p)
        if agr_out["margins_by_layer"]:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for name, mg in agr_out["margins_by_layer"].items():
                ax.plot(range(len(mg)), mg, marker="o", label=name)
            ax.axhline(0.0, ls="--", c="gray", lw=1)
            ax.set_xlabel("layer (early-exit readout)"); ax.set_ylabel("agreement margin  logP(correct)-logP(foil)")
            ax.set_title("Where agreement resolves through depth (E1, 0 attractors)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout(); p = os.path.join(out_dir, f"agreement_resolution_depth.{fmt}"); fig.savefig(p, dpi=140); plt.close(fig)
            print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="fetched/cmp_24693039")
    ap.add_argument("--out_dir", default="plots/behavior")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--format", default="pdf")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    paths = discover(args.ckpt_dir)
    if not paths:
        raise SystemExit(f"no best_*.pt under {args.ckpt_dir}")
    print(f"Loaded checkpoints from {args.ckpt_dir}:")
    models = {}
    for name, p in paths.items():
        m, meta = load_model(p, args.device)
        models[name] = (m, meta)
        bv = meta.get("best_val")
        print(f"  {name:26s} arch={meta['arch']:16s} best_val={bv if bv is None else round(bv,4)} causal={meta['is_causal']}")

    pert_out = probe_perturbation(models, enc, args.device)
    lens_out = probe_logit_lens(models, enc, args.device)
    temp_out = probe_temperature(models, enc, args.device)
    probe_minimal_pairs(models, enc, args.device)
    agr_out = probe_agreement(models, enc, args.device)

    if not args.no_plots:
        make_plots(lens_out, temp_out, pert_out, agr_out, args.out_dir, args.format)
    print("\nDone.")


if __name__ == "__main__":
    main()
