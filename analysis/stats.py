"""
Statistics for the behavior suite: intervals, paired tests, bootstrap, FDR.

Every number the suite reports should come with an interval, and every
arm-vs-arm claim with a paired test — see analysis/behavior_suite_v2_plan.md §1
for why (the v1 suite's per-cell CIs were 0.31-0.70 wide, so all of its
"differences" were inside their own noise).

All arms see identical stimuli, so comparisons here are PAIRED throughout:
  - binary outcomes (did the model prefer the target?) -> mcnemar()
  - continuous outcomes (lens margin, per-token loss)  -> paired_bootstrap()
Paired tests are what buys the power: only items where the two arms disagree
carry information, so they need far fewer items than an unpaired comparison
(~500 vs ~2400 for a 5-point difference; see plan §3).

numpy-only on purpose — no scipy dependency, so the suite runs in any env that
can already load a checkpoint. Exact binomial tails are computed in log space
via lgamma, which is stable well past the n we ever pass it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

Z95 = 1.959963984540054


# ------------------------------------------------------------------ intervals
@dataclass
class Interval:
    """A point estimate with a confidence interval."""
    point: float
    lo: float
    hi: float
    n: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    def to_dict(self) -> dict:
        return asdict(self)


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation: it stays inside [0, 1] and keeps
    reasonable coverage at small n and at p near 0 or 1, which is exactly where
    the v1 suite lived (n=4, p=0.25).
    """
    if n <= 0:
        return Interval(float("nan"), 0.0, 1.0, 0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return Interval(p, max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom), n)


def proportion(flags: Sequence[float] | np.ndarray, z: float = Z95) -> Interval:
    """Wilson interval from a vector of 0/1 outcomes."""
    a = np.asarray(flags, dtype=float)
    a = a[~np.isnan(a)]
    return wilson(int(a.sum()), int(a.size), z=z)


def mean_ci(values: Sequence[float] | np.ndarray, z: float = Z95) -> Interval:
    """Normal-approximation CI for a mean (fine for the n>=1000 cells we target)."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return Interval(float(a[0]), float("nan"), float("nan"), 1)
    se = float(a.std(ddof=1)) / math.sqrt(n)
    m = float(a.mean())
    return Interval(m, m - z * se, m + z * se, n)


# ------------------------------------------------------------------ log-space binomial
def _log_binom_pmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    kf = k.astype(float)
    log_coef = (math.lgamma(n + 1) - np.vectorize(math.lgamma)(kf + 1.0)
                - np.vectorize(math.lgamma)(n - kf + 1.0))
    return log_coef + kf * math.log(p) + (n - kf) * math.log1p(-p)


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (the 'sum of outcomes no more likely than
    the observed one' convention, as in scipy's binomtest)."""
    if n == 0:
        return 1.0
    ks = np.arange(n + 1)
    logpmf = _log_binom_pmf(ks, n, p)
    obs = logpmf[k]
    # tolerance guards against ties being dropped by floating-point noise
    keep = logpmf <= obs + 1e-9
    m = logpmf[keep].max()
    return float(min(1.0, np.exp(m) * np.exp(logpmf[keep] - m).sum()))


# ------------------------------------------------------------------ paired tests
@dataclass
class Paired:
    """Result of a paired comparison between two arms on shared items."""
    n_items: int
    delta: float             # arm_a - arm_b (accuracy, or mean difference)
    p_value: float
    n_discordant: int = 0    # McNemar only: items where the arms disagree
    a_wins: int = 0
    b_wins: int = 0
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    test: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        star = "*" if self.p_value < 0.05 else " "
        extra = f" disc={self.n_discordant} ({self.a_wins}/{self.b_wins})" if self.test == "mcnemar" else ""
        return (f"Δ={self.delta:+.3f} [{self.ci_lo:+.3f}, {self.ci_hi:+.3f}] "
                f"p={self.p_value:.4f}{star} n={self.n_items}{extra}")


def mcnemar(a_flags: Sequence[float], b_flags: Sequence[float], exact: bool = True) -> Paired:
    """Paired test for two arms' binary outcomes on the SAME items.

    Only discordant pairs are informative: b = a-right/b-wrong, c = a-wrong/b-right,
    and under the null b ~ Binomial(b + c, 0.5). Items both arms get right (or both
    wrong) carry no signal about which is better, which is why this needs far fewer
    items than comparing two independent accuracies.

    The CI is the Wald interval on the paired difference (Agresti-Min style), which
    is adequate at the n>=1000 the plan targets.
    """
    a = np.asarray(a_flags, dtype=float)
    b = np.asarray(b_flags, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms need matching item counts, got {a.shape} vs {b.shape}")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok] > 0.5, b[ok] > 0.5
    n = int(a.size)
    b_wins_a = int(np.sum(a & ~b))     # a right, b wrong
    a_loses = int(np.sum(~a & b))      # a wrong, b right
    disc = b_wins_a + a_loses
    delta = float(a.mean() - b.mean()) if n else float("nan")

    if disc == 0:
        p = 1.0
    elif exact:
        p = binom_test_two_sided(b_wins_a, disc, 0.5)
    else:
        chi2 = (abs(b_wins_a - a_loses) - 1.0) ** 2 / disc   # continuity-corrected
        p = math.erfc(math.sqrt(chi2 / 2.0))

    # Wald CI on the paired difference of proportions
    if n:
        se = math.sqrt(max(disc - (b_wins_a - a_loses) ** 2 / n, 0.0)) / n
        lo, hi = delta - Z95 * se, delta + Z95 * se
    else:
        lo = hi = float("nan")
    return Paired(n_items=n, delta=delta, p_value=p, n_discordant=disc,
                  a_wins=b_wins_a, b_wins=a_loses, ci_lo=lo, ci_hi=hi, test="mcnemar")


def paired_bootstrap(a_vals: Sequence[float], b_vals: Sequence[float],
                     n_boot: int = 10000, seed: int = 0, stat=np.mean) -> Paired:
    """Bootstrap the paired difference of a continuous metric over items.

    Resamples ITEMS (not observations independently), preserving the pairing —
    the same item index is drawn for both arms, so shared item difficulty cancels.
    The p-value is the two-sided proportion of resamples whose difference crosses
    zero, floored at 1/n_boot since a bootstrap cannot resolve below its own
    resolution.
    """
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms need matching item counts, got {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(a.size)
    if n == 0:
        return Paired(0, float("nan"), 1.0, test="paired_bootstrap")

    d = a - b
    obs = float(stat(d))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat(d[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided: how often does the resampled difference land on the other side of 0
    tail = float(np.mean(boots <= 0.0) if obs > 0 else np.mean(boots >= 0.0))
    p = min(1.0, max(2.0 * tail, 1.0 / n_boot))
    return Paired(n_items=n, delta=obs, p_value=p, ci_lo=float(lo), ci_hi=float(hi),
                  test="paired_bootstrap")


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 stat=np.mean) -> Interval:
    """Bootstrap CI for a single arm's statistic over items."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boots = stat(a[rng.integers(0, n, size=(n_boot, n))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Interval(float(stat(a)), float(lo), float(hi), int(n))


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence,
                         n_boot: int = 10000, seed: int = 0) -> Interval:
    """Bootstrap CI resampling CLUSTERS, not items.

    Use with cluster=seed once multi-seed runs exist: the resulting interval
    includes between-seed variance, which is the only way an architecture-level
    claim is defensible. With one seed per arm this collapses to a point and will
    report a degenerate interval — that is the honest answer, not a bug.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    groups = [v[c == g] for g in np.unique(c)]
    k = len(groups)
    if k == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if k == 1:
        return Interval(float(groups[0].mean()), float("nan"), float("nan"), int(v.size))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        boots[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Interval(float(v.mean()), float(lo), float(hi), int(v.size))


# ------------------------------------------------------------------ multiplicity
def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05):
    """BH-FDR. Returns (rejected mask, q-values) in the input order.

    The suite runs many paradigms x arms; without this the number of 'significant'
    cells is a function of how many tests were run.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty(n)
    q[order] = q_sorted
    return q <= alpha, q


# ------------------------------------------------------------------ power
def n_for_half_width(half_width: float, p: float = 0.5, z: float = Z95) -> int:
    """Items needed for a target CI half-width on a single proportion."""
    return math.ceil(p * (1 - p) * (z / half_width) ** 2)


def mcnemar_power_n(discordant_rate: float, favor_fraction: float = 0.7,
                    power: float = 0.8, alpha: float = 0.05) -> int:
    """Total items needed for `power` at `alpha` in a paired binary comparison.

    discordant_rate: fraction of items where the two arms disagree at all.
    favor_fraction:  of those, the fraction favoring the better arm (0.5 = null).
    """
    z_a, z_b = Z95 if alpha == 0.05 else 1.96, 0.8416
    d = 2.0 * favor_fraction - 1.0
    if d <= 0:
        return math.inf
    n_disc = math.ceil((z_a + z_b) ** 2 / (d * d))
    return math.ceil(n_disc / max(discordant_rate, 1e-9))
