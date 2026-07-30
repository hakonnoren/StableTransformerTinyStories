"""
P1.1-P1.6: the v1 probes, re-run over real held-out TinyStories at scale.

Same measurements as v1, but the unit of observation is a STORY and there are
~2000 of them instead of one prompt, so every number gets a bootstrap interval
over items and every arm-vs-arm claim gets a paired test.

  P1.1  context sensitivity + corruption robustness   (was: 1 and 4 prompts)
  P1.2  causality leak — an assertion, not a measurement
  P1.3  depth curves, raw and tuned lens, per stream  (was: 1 sequence)
  P1.4  per-token loss stratified by linguistic type
  P1.5  in-context-use score (Olsson et al. 2022, adapted)
  P1.6  story-closure calibration

P1.4/P1.5/P1.6 all read the same per-token log-probabilities, so they share ONE
forward pass per arm (`score_corpus`), which also dumps raw per-token logprobs to
.npz so strata can be redefined later without re-running any model.

Strata are computed from token ids alone — no parser required:
  position-in-story bucket, first vs repeated occurrence of that token within the
  story, corpus frequency band, word-initial vs continuation subword, punctuation.
POS/dependency strata (plan P1.4) need spaCy and are additive to this; the
token-only strata already separate "predicting a new entity" from "predicting a
function word", which is where the interesting differences should live.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..corpus import EOT, Story, pad_batch, sorted_batches, token_mask
from ..lenses import lens_logits
from ..loader import LensPolicy, forward_logits, layer_states, supports_policy

# ------------------------------------------------------------------ strata
POS_BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512)]

# Band edges are set to TinyStories' ACTUAL vocabulary, not GPT-2's table. The val
# split uses only 12,615 of the 50,304 embedding rows, and rank<1000 already covers
# 90.5% of all token occurrences. Bands cut at 8k left the top band permanently
# empty, because a 2000-story sample contains ~6.4k distinct tokens at most.
FREQ_BANDS = [("top16", 0, 16), ("top128", 16, 128), ("top1k", 128, 1024),
              ("top4k", 1024, 4096), ("gt4k", 4096, 1 << 30)]


def corpus_frequency_rank(tokens: np.ndarray, vocab_size: int = 50304) -> np.ndarray:
    """rank[token_id] = 0-based frequency rank (0 = most common).

    Pass the WHOLE corpus, not the evaluation sample: frequency is a property of the
    corpus, and ranking within a subsample both shifts the bands and caps the top
    band at the subsample's distinct-token count.

    Tokens that never occur are ranked after every token that does (ties broken
    stably by id), so they land in the last band rather than being silently
    interleaved with observed ones.
    """
    counts = np.bincount(np.asarray(tokens).astype(np.int64), minlength=vocab_size)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(vocab_size, dtype=np.int64)
    rank[order] = np.arange(vocab_size)
    return rank


def _is_word_initial(enc, vocab_size: int = 50304) -> np.ndarray:
    """Mask of tokens whose decoded form starts with a space (word-initial)."""
    out = np.zeros(vocab_size, dtype=bool)
    for t in range(vocab_size):
        try:
            s = enc.decode([t])
        except Exception:
            continue
        if s.startswith(" "):
            out[t] = True
    return out


def _is_punct(enc, vocab_size: int = 50304) -> np.ndarray:
    out = np.zeros(vocab_size, dtype=bool)
    for t in range(vocab_size):
        try:
            s = enc.decode([t]).strip()
        except Exception:
            continue
        if s and all(c in ".,!?;:\"'-()" for c in s):
            out[t] = True
    return out


class Strata:
    """Precomputed token-level property tables, shared across arms.

    `rank_tokens` should be the full corpus token array (see corpus_frequency_rank);
    passing only the evaluation stories biases the frequency bands.
    """

    def __init__(self, rank_tokens: np.ndarray, enc, vocab_size: int = 50304):
        self.rank = corpus_frequency_rank(rank_tokens, vocab_size)
        self.word_initial = _is_word_initial(enc, vocab_size)
        self.punct = _is_punct(enc, vocab_size)

    def freq_band(self, token_ids: np.ndarray) -> np.ndarray:
        r = self.rank[token_ids]
        out = np.full(token_ids.shape, len(FREQ_BANDS) - 1, dtype=np.int8)
        for i, (_, lo, hi) in enumerate(FREQ_BANDS):
            out[(r >= lo) & (r < hi)] = i
        return out


def first_occurrence_mask(tokens: np.ndarray) -> np.ndarray:
    """True where this token id has NOT appeared earlier in the same story.

    Cheap proxy for first-mention vs repeated-mention: predicting a token the story
    has already established is a memory/copying problem, predicting a new one is
    not, and the two should separate the arms differently.
    """
    seen = set()
    out = np.zeros(tokens.size, dtype=bool)
    for i, t in enumerate(tokens):
        ti = int(t)
        if ti not in seen:
            out[i] = True
            seen.add(ti)
    return out


# ------------------------------------------------------------------ P1.4/1.5/1.6 core pass
@torch.no_grad()
def score_corpus(arm, stories: list[Story], strata: Strata, device: str = "cpu",
                 batch_size: int = 8, dump_npz: str | None = None,
                 verbose: bool = True) -> list[dict]:
    """One forward pass per arm over the corpus; returns one record per story.

    Each record carries the story's mean loss plus per-stratum loss sums and counts,
    which is exactly what a paired bootstrap over stories needs. Raw per-token
    logprobs optionally go to `dump_npz` so strata can be redefined later.
    """
    recs: list[dict] = []
    all_lp, all_tok, all_pos, all_story = [], [], [], []
    done = 0
    for batch in sorted_batches(stories, batch_size):
        ids, lengths = pad_batch(batch, device=device)
        T = ids.shape[1]
        mask = token_mask(lengths, T)
        logits = forward_logits(arm.model, ids).float()
        lp_full = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        lp = lp_full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        # P(EOT) at every position, for the closure probe
        p_eot = lp_full[:, :, EOT].exp().cpu().numpy()
        valid = (mask[:, :-1] & mask[:, 1:]).cpu().numpy()

        for b, story in enumerate(batch):
            v = valid[b]
            n = int(v.sum())
            if n == 0:
                continue
            toks = story.tokens[1:1 + n].astype(np.int64)      # targets
            loss = -lp[b, :n]
            pos = np.arange(1, n + 1)
            first = first_occurrence_mask(story.tokens)[1:1 + n]
            band = strata.freq_band(toks)

            rec = {
                "arm": arm.name, "item_id": f"story{story.idx}", "story_idx": story.idx,
                "n_tokens": n, "story_len": len(story),
                "mean_loss": float(loss.mean()),
            }
            # position buckets -> P1.4 + P1.5
            for lo, hi in POS_BUCKETS:
                m = (pos >= lo) & (pos < hi)
                if m.any():
                    rec[f"loss_pos_{lo}_{hi}"] = float(loss[m].mean())
                    rec[f"n_pos_{lo}_{hi}"] = int(m.sum())
            # first vs repeated mention
            for label, m in (("first", first), ("repeat", ~first)):
                if m.any():
                    rec[f"loss_{label}"] = float(loss[m].mean())
                    rec[f"n_{label}"] = int(m.sum())
            # frequency bands
            for i, (name, _, _) in enumerate(FREQ_BANDS):
                m = band == i
                if m.any():
                    rec[f"loss_freq_{name}"] = float(loss[m].mean())
                    rec[f"n_freq_{name}"] = int(m.sum())
            # word-initial vs continuation subword, and punctuation
            wi = strata.word_initial[toks]
            for label, m in (("word_init", wi), ("subword", ~wi),
                             ("punct", strata.punct[toks])):
                if m.any():
                    rec[f"loss_{label}"] = float(loss[m].mean())
                    rec[f"n_{label}"] = int(m.sum())
            # P1.6: closure. P(EOT) just before the real EOT vs mid-story mean.
            eot_pos = n - 1
            rec["p_eot_at_end"] = float(p_eot[b, eot_pos])
            mid = p_eot[b, : max(eot_pos - 1, 1)]
            rec["p_eot_midstory_mean"] = float(mid.mean())
            rec["p_eot_midstory_max"] = float(mid.max())
            recs.append(rec)

            if dump_npz is not None:
                all_lp.append(lp[b, :n].astype(np.float32))
                all_tok.append(toks.astype(np.uint16))
                all_pos.append(pos.astype(np.int32))
                all_story.append(np.full(n, story.idx, dtype=np.int32))

        done += len(batch)
        if verbose and done % (batch_size * 25) < batch_size:
            print(f"    {arm.name}: {done}/{len(stories)} stories", flush=True)

    if dump_npz is not None and all_lp:
        np.savez_compressed(dump_npz, logprob=np.concatenate(all_lp),
                            token=np.concatenate(all_tok), pos=np.concatenate(all_pos),
                            story=np.concatenate(all_story))
    return recs


def in_context_score(recs: list[dict], early=(32, 64), late=(256, 512)) -> list[dict]:
    """P1.5: per-story loss(early bucket) - loss(late bucket).

    Positive = the model predicts better later in the story, i.e. it is exploiting
    accumulated context (Olsson et al. 2022's in-context-learning score, adapted to
    TinyStories' ~200-token documents).
    """
    ek, lk = f"loss_pos_{early[0]}_{early[1]}", f"loss_pos_{late[0]}_{late[1]}"
    out = []
    for r in recs:
        if ek in r and lk in r:
            out.append({"arm": r["arm"], "item_id": r["item_id"],
                        "icl_score": r[ek] - r[lk], "early": r[ek], "late": r[lk]})
    return out


# ------------------------------------------------------------------ P1.1
@torch.no_grad()
def probe_perturbation(arm, stories: list[Story], device: str = "cpu",
                       rates=(0.05, 0.1, 0.2, 0.4), n_rep: int = 3,
                       prefix_len: int = 64, seed: int = 0,
                       batch_size: int = 8) -> list[dict]:
    """P1.1: context sensitivity and corruption robustness, per story.

    Context sensitivity: TV between the clean next-token distribution at the prefix
    end and the distribution after flipping ONE early token.
    Corruption robustness: same TV when a fraction of prefix tokens is replaced.

    v1 computed these from 1 and 4 prompts respectively, which is why its
    corruption curve was non-monotone (baseline 0.288 at 5% but 0.151 at 10%).
    Averaged over ~2000 stories the curve should be monotone in the rate; if it
    still isn't, that is a real property and not sampling noise.
    """
    rng = np.random.default_rng(seed)
    vocab = arm.model.cfg.vocab_size
    usable = [s for s in stories if len(s) > prefix_len + 1]
    recs = []

    for i in range(0, len(usable), batch_size):
        chunk = usable[i:i + batch_size]
        clean_ids = np.stack([s.tokens[:prefix_len].astype(np.int64) for s in chunk])

        def last_dist(arr):
            t = torch.from_numpy(arr).to(device)
            lg = forward_logits(arm.model, t).float()
            return F.softmax(lg[:, -1], dim=-1).cpu().numpy()

        p_clean = last_dist(clean_ids)

        # single-token flip at position 2 (context sensitivity)
        flipped = clean_ids.copy()
        flipped[:, 2] = (flipped[:, 2] + 11) % vocab
        tv_flip = 0.5 * np.abs(p_clean - last_dist(flipped)).sum(axis=1)

        rows = [{"arm": arm.name, "item_id": f"story{s.idx}", "story_idx": s.idx,
                 "ctx_sens": float(tv_flip[j])} for j, s in enumerate(chunk)]

        for rate in rates:
            n_co = max(1, int(rate * prefix_len))
            tvs = np.zeros((len(chunk), n_rep))
            for r in range(n_rep):
                cor = clean_ids.copy()
                for j in range(len(chunk)):
                    pos = rng.choice(prefix_len, size=n_co, replace=False)
                    cor[j, pos] = rng.integers(0, vocab, size=n_co)
                tvs[:, r] = 0.5 * np.abs(p_clean - last_dist(cor)).sum(axis=1)
            for j in range(len(chunk)):
                rows[j][f"tv_rate_{rate}"] = float(tvs[j].mean())
        recs.extend(rows)
    return recs


# ------------------------------------------------------------------ P1.2
@torch.no_grad()
def probe_leak(arm, stories: list[Story], device: str = "cpu", n_trials: int = 200,
               prefix_len: int = 64, seed: int = 0) -> dict:
    """P1.2: causality as a TEST. A causal model's logits at i<j must not move
    when input token j changes. Returns max |Δ| over trials; anything above ~1e-4
    for a causal arch is a genuine leak and should fail the run."""
    rng = np.random.default_rng(seed)
    usable = [s for s in stories if len(s) > prefix_len + 1]
    vocab = arm.model.cfg.vocab_size
    worst = 0.0
    for _ in range(n_trials):
        s = usable[rng.integers(0, len(usable))]
        ids = torch.from_numpy(s.tokens[:prefix_len].astype(np.int64))[None].to(device)
        j = int(rng.integers(prefix_len // 4, prefix_len))
        l1 = forward_logits(arm.model, ids).float()
        ids2 = ids.clone()
        ids2[0, j] = (int(ids2[0, j]) + 7) % vocab
        l2 = forward_logits(arm.model, ids2).float()
        worst = max(worst, float((l1[:, :j] - l2[:, :j]).abs().max()))
    return {"arm": arm.name, "max_abs_delta": worst, "n_trials": n_trials,
            "is_causal": arm.meta["is_causal"],
            "passed": bool(worst < 1e-4) if arm.meta["is_causal"] else True}


# ------------------------------------------------------------------ P1.3
@torch.no_grad()
def probe_lens_depth(arm, stories: list[Story], device: str = "cpu",
                     batch_size: int = 4, tuned=None,
                     policies=(LensPolicy.SUM,), max_pos: int = 64, seed: int = 0,
                     verbose: bool = True) -> list[dict]:
    """P1.3: per-layer early-exit loss per story, for each lens and stream policy.

    One record per (arm, story, policy, lens, layer) — enough to bootstrap a CI on
    every point of every depth curve, which the single-sequence v1 version could
    not do.

    `max_pos` subsamples token positions per story. The lm_head projection to 50304
    classes runs once per layer per lens per policy, so scoring every token is the
    dominant cost; a per-story mean over a random position subsample is an unbiased
    estimate of the story's mean, and the item-level bootstrap absorbs the extra
    sampling noise. Positions are drawn ONCE per batch and shared across all
    layers/lenses/policies, so curves are compared on identical tokens.

    The tuned lens is applied ONLY under policy='sum': train_tuned_lens fits its
    translators to sum-bridged states, so using them on a single stream would read
    an x- or z-state through a map estimated for x+z. Per-stream tuned lenses would
    need their own fits.
    """
    recs = []
    pol = [p for p in policies if supports_policy(arm.model, p)]
    gen = torch.Generator().manual_seed(seed)
    done = 0
    for batch in sorted_batches(stories, batch_size):
        ids, lengths = pad_batch(batch, device=device)
        T = ids.shape[1]
        mask = token_mask(lengths, T)
        valid = (mask[:, :-1] & mask[:, 1:])
        states = layer_states(arm.model, ids)
        tgt = ids[:, 1:]

        # one shared position subsample per story in this batch
        keep = torch.zeros_like(valid)
        for b in range(valid.shape[0]):
            idx = valid[b].nonzero(as_tuple=True)[0]
            if idx.numel() > max_pos:
                sel = torch.randperm(idx.numel(), generator=gen)[:max_pos]
                idx = idx[sel.to(idx.device)]
            keep[b, idx] = True

        for policy in pol:
            for lens_name, lens in (("raw", None), ("tuned", tuned)):
                if lens_name == "tuned" and (lens is None or policy != LensPolicy.SUM):
                    continue
                for li, h in enumerate(states):
                    logits = lens_logits(arm.model, lens, li, h[:, :-1], policy).float()
                    tok_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                        reduction="none").reshape(tgt.shape)
                    for b, story in enumerate(batch):
                        k = keep[b]
                        if not bool(k.any()):
                            continue
                        recs.append({
                            "arm": arm.name, "item_id": f"story{story.idx}",
                            "story_idx": story.idx, "policy": policy,
                            "lens": lens_name, "layer": li,
                            "n_pos": int(k.sum()),
                            "loss": float(tok_loss[b][k].mean()),
                        })
        done += len(batch)
        if verbose and done % (batch_size * 25) < batch_size:
            print(f"    {arm.name} lens: {done}/{len(stories)} stories", flush=True)
    return recs
