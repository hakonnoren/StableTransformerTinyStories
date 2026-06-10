
from mingpt.model import CausalSelfAttention,NewGELU
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from mingpt.linear_maps import DiagMap, SVDMap, LowRankCayleyMap


class ReversibleBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        n_embd = config.n_embd//2
        

        self.ln_1 = nn.LayerNorm(n_embd) if config.normalize else nn.Identity()
        self.ln_2 = nn.LayerNorm(n_embd) if config.normalize else nn.Identity()

        if config.rev_config.fix_norm:
            #we dont train the layer norm parameters
            for ln in [self.ln_1,self.ln_2]:
                for param in ln.parameters():
                    param.requires_grad = False

        self.mlp_ = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        self.mlp = lambda x: self.mlp_.dropout(self.mlp_.c_proj(self.mlp_.act(self.mlp_.c_fc(x)))) 

        self.attn = CausalSelfAttention(config,n_embd)


        self.epsilon = config.rev_config.epsilon
        self.lambd = config.rev_config.lambd
        self.free_vol = config.rev_config.free_vol
        self.scale_d = config.rev_config.scale_d


        self.rezero = getattr(config, 'rezero', False)
        if self.rezero:
            self.resweight_attn = nn.Parameter(torch.zeros(1))
            self.resweight_mlp = nn.Parameter(torch.zeros(1))
        else:
            self.resweight_attn = 1.0
            self.resweight_mlp = 1.0


        #initialize scalar parameter with name _bias
        init_func = torch.zeros
        if config.rev_config.randn_init:
            init_func = torch.randn

        self.param_scale = nn.Identity()
        self.param_scale_eps = 1.0
        if config.rev_config.tanh_scale:
            self.param_scale = nn.Tanh()
            self.param_scale_eps = self.epsilon

        self.volume_pres = config.rev_config.volume_pres
        self.vol_pres_per_block = config.rev_config.vol_pres_per_block

        if self.volume_pres:
            # Buffers (not parameters) so they follow .to(device) but stay frozen at zero.
            self.register_buffer('gamma_bias', torch.zeros(1))
            self.register_buffer('alpha_bias', torch.zeros(1))
        else:
            if self.scale_d:
                self.register_parameter(name='alpha_bias', param=nn.Parameter(init_func(1,1,n_embd)*self.epsilon,requires_grad=True))
                self.register_parameter(name='gamma_bias', param=nn.Parameter(init_func(1,1,n_embd)*self.epsilon,requires_grad=True))

            else:
                self.register_parameter(name='alpha_bias', param=nn.Parameter(init_func(1)*self.epsilon,requires_grad=True))
                self.register_parameter(name='gamma_bias', param=nn.Parameter(init_func(1)*self.epsilon,requires_grad=True))


    def get_gamma(self):
        return self.param_scale(self.gamma_bias) * self.param_scale_eps
    def get_alpha(self):

        return self.param_scale(self.alpha_bias) * self.param_scale_eps

    def _effective_gamma_alpha(self, avg):
        """Apply the configured centering to get the gamma/alpha actually used
        as exponents inside the block. Returns (gamma_used, alpha_used)."""
        gamma_raw = self.get_gamma()
        alpha_raw = self.get_alpha()
        if self.vol_pres_per_block:
            # Self-center so sum_i (gamma_used + alpha_used) = 0 within this block.
            c = (gamma_raw.mean() + alpha_raw.mean()) / 2
            return gamma_raw - c, alpha_raw - c
        # Default: subtract the external avg correction (or 0 in free_vol mode).
        return gamma_raw - avg, alpha_raw - avg

    def partitioned_step(self,y,avg = 0):


        x,z = torch.split(y,y.shape[-1]//2,dim=-1)

        if self.volume_pres:
            x = x + self.resweight_attn * self.attn(self.ln_1(z))
            z = z + self.resweight_mlp * self.mlp(self.ln_2(x))
        else:
            gamma, alpha = self._effective_gamma_alpha(avg)
            x = torch.exp(-gamma) * (x + self.resweight_attn * self.attn(self.ln_1(z)))
            z = torch.exp(-alpha) * z + self.resweight_mlp * self.mlp(self.ln_2(x))

        return torch.cat([x,z],dim=-1)

    def vanilla_forward(self, x,avg=0):

        if self.volume_pres:
            x = x + self.resweight_attn * self.attn(self.ln_1(x))
            x = x + self.resweight_mlp * self.mlp(self.ln_2(x))
        else:
            gamma, alpha = self._effective_gamma_alpha(avg)
            x = torch.exp(-gamma) * (x + self.resweight_attn * self.attn(self.ln_1(x)))
            x = torch.exp(-alpha) * x + self.resweight_mlp * self.mlp(self.ln_2(x))
        return x

        
    
    def forward(self, x, avg=0,mode='reversible'):
        if mode == 'vanilla':
            return self.vanilla_forward(x,avg)
        else:
            return self.partitioned_step(x,avg)


class _HalfWidthBlock(nn.Module):
    """Pre-LN transformer block at half embedding width, exposing the
    *force* (TransformerBlock(u) - u): attn(ln1(u)) + mlp(ln2(u + attn(ln1(u))))."""

    def __init__(self, config, n_embd, fix_norm, resid_pdrop, rezero):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd) if config.normalize else nn.Identity()
        self.ln_2 = nn.LayerNorm(n_embd) if config.normalize else nn.Identity()
        if fix_norm:
            for ln in (self.ln_1, self.ln_2):
                for p in ln.parameters():
                    p.requires_grad = False

        self.attn = CausalSelfAttention(config, n_embd)
        self.mlp_ = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))

        self.rezero = rezero
        if rezero:
            self.resweight_attn = nn.Parameter(torch.zeros(1))
            self.resweight_mlp  = nn.Parameter(torch.zeros(1))
        else:
            self.resweight_attn = 1.0
            self.resweight_mlp  = 1.0

    def _mlp(self, x):
        m = self.mlp_
        return m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, u):
        a = self.resweight_attn * self.attn(self.ln_1(u))
        m = self.resweight_mlp  * self._mlp(self.ln_2(u + a))
        return a + m


class FullBlockReversibleBlock(nn.Module):
    """Reversible coupling where each force is a *full* pre-LN transformer block:

        x_{l+1} = x_l + F(z_l)
        z_{l+1} = z_l + G(x_{l+1})       (volume-preserving)

    or with the existing gamma/alpha placement (volume_pres=False):

        x_{l+1} = exp(-gamma) * (x_l + F(z_l))
        z_{l+1} = exp(-alpha) *  z_l + G(x_{l+1})

    F and G are independent half-width transformer blocks; each computes the
    *force* TransformerBlock(u) - u so that the outer coupling residual is the
    only identity component on the input — i.e. the continuous limit is a
    divergence-free coupled ODE dx/dt = F(z), dz/dt = G(x).
    """

    def __init__(self, config):
        super().__init__()

        n_embd = config.n_embd // 2

        self.block_F = _HalfWidthBlock(
            config, n_embd,
            fix_norm=config.rev_config.fix_norm,
            resid_pdrop=config.resid_pdrop,
            rezero=getattr(config, 'rezero', False),
        )
        self.block_G = _HalfWidthBlock(
            config, n_embd,
            fix_norm=config.rev_config.fix_norm,
            resid_pdrop=config.resid_pdrop,
            rezero=getattr(config, 'rezero', False),
        )

        self.epsilon = config.rev_config.epsilon
        self.lambd = config.rev_config.lambd
        self.free_vol = config.rev_config.free_vol
        self.scale_d = config.rev_config.scale_d

        init_func = torch.zeros
        if config.rev_config.randn_init:
            init_func = torch.randn

        self.param_scale = nn.Identity()
        self.param_scale_eps = 1.0
        if config.rev_config.tanh_scale:
            self.param_scale = nn.Tanh()
            self.param_scale_eps = self.epsilon

        self.volume_pres = config.rev_config.volume_pres
        self.vol_pres_per_block = config.rev_config.vol_pres_per_block
        if self.volume_pres:
            self.register_buffer('gamma_bias', torch.zeros(1))
            self.register_buffer('alpha_bias', torch.zeros(1))
        else:
            if self.scale_d:
                self.register_parameter('alpha_bias', nn.Parameter(init_func(1, 1, n_embd) * self.epsilon, requires_grad=True))
                self.register_parameter('gamma_bias', nn.Parameter(init_func(1, 1, n_embd) * self.epsilon, requires_grad=True))
            else:
                self.register_parameter('alpha_bias', nn.Parameter(init_func(1) * self.epsilon, requires_grad=True))
                self.register_parameter('gamma_bias', nn.Parameter(init_func(1) * self.epsilon, requires_grad=True))

    def get_gamma(self):
        return self.param_scale(self.gamma_bias) * self.param_scale_eps

    def get_alpha(self):
        return self.param_scale(self.alpha_bias) * self.param_scale_eps

    def _effective_gamma_alpha(self, avg):
        gamma_raw = self.get_gamma()
        alpha_raw = self.get_alpha()
        if self.vol_pres_per_block:
            c = (gamma_raw.mean() + alpha_raw.mean()) / 2
            return gamma_raw - c, alpha_raw - c
        return gamma_raw - avg, alpha_raw - avg

    def partitioned_step(self, y, avg=0):
        x, z = torch.split(y, y.shape[-1] // 2, dim=-1)

        if self.volume_pres:
            x = x + self.block_F(z)
            z = z + self.block_G(x)
        else:
            gamma, alpha = self._effective_gamma_alpha(avg)
            x = torch.exp(-gamma) * (x + self.block_F(z))
            z = torch.exp(-alpha) *  z + self.block_G(x)

        return torch.cat([x, z], dim=-1)

    def forward(self, x, avg=0, mode='reversible'):
        if mode == 'vanilla':
            raise NotImplementedError(
                "FullBlockReversibleBlock does not support vanilla mode; "
                "set rev_config.full_block=False to use vanilla fine-tuning."
            )
        return self.partitioned_step(x, avg)


class LinearMixedReversibleBlock(nn.Module):
    """Reversible block factored as  Y -> S_2 ∘ L ∘ S_1(Y),  generalizing the
    diagonal exp(-γ)/exp(-α) scaling to a controlled-determinant linear map L
    (see ``mingpt.linear_maps``):

        x' = x + attn(ln_1(z))            # S_1  (shear, det = 1)
        [x'', z''] = L([x', z])           # linear mix, log|det L| controlled
        z''' = z'' + mlp(ln_2(x''))       # S_2  (shear, det = 1)

    With ``L = DiagMap`` this reproduces the original ``ReversibleBlock``; with
    ``L = SVDMap`` it contracts/expands learned (rotated) directions. The volume
    correction ``avg`` (subtracted from L's log-scales) is supplied by
    ``GPT.forward`` exactly as for the diagonal model.
    """

    def __init__(self, config):
        super().__init__()
        d = config.n_embd // 2          # half width (x and z each live in R^d)
        m = config.n_embd               # full per-token feature dim (x,z concatenated)

        self.ln_1 = nn.LayerNorm(d) if config.normalize else nn.Identity()
        self.ln_2 = nn.LayerNorm(d) if config.normalize else nn.Identity()
        if config.rev_config.fix_norm:
            for ln in (self.ln_1, self.ln_2):
                for p in ln.parameters():
                    p.requires_grad = False

        self.mlp_ = nn.ModuleDict(dict(
            c_fc=nn.Linear(d, 4 * d), c_proj=nn.Linear(4 * d, d),
            act=NewGELU(), dropout=nn.Dropout(config.resid_pdrop),
        ))
        self.mlp = lambda x: self.mlp_.dropout(self.mlp_.c_proj(self.mlp_.act(self.mlp_.c_fc(x))))
        self.attn = CausalSelfAttention(config, d)

        self.rezero = getattr(config, 'rezero', False)
        if self.rezero:
            self.resweight_attn = nn.Parameter(torch.zeros(1))
            self.resweight_mlp = nn.Parameter(torch.zeros(1))
        else:
            self.resweight_attn = 1.0
            self.resweight_mlp = 1.0

        rc = config.rev_config
        self.volume_pres = rc.volume_pres
        self.vol_pres_per_block = rc.vol_pres_per_block
        self.free_vol = rc.free_vol
        self.lambd = rc.lambd
        self.linear_map = self.build_linear_map(rc, m)

    @staticmethod
    def build_linear_map(rc, m):
        kind = getattr(rc, 'linear_map', 'svd')
        k = getattr(rc, 'n_householder', 0) or None     # 0 => full (k=m)
        frozen = rc.volume_pres                          # volume_pres => orthogonal L (ℓ≡0)
        vpb = rc.vol_pres_per_block
        if kind == 'diag':
            return DiagMap(m, vol_pres_per_block=vpb, frozen=frozen)
        if kind == 'svd':
            return SVDMap(m, k=k, vol_pres_per_block=vpb, frozen=frozen)
        if kind == 'lowrank_cayley':
            return LowRankCayleyMap(m, r=getattr(rc, 'lowrank_r', 4),
                                    h=getattr(rc, 'cayley_h', 1.0),
                                    vol_pres_per_block=vpb, frozen=frozen)
        raise ValueError(f"unknown rev_config.linear_map={kind!r}; "
                         "expected 'diag', 'svd' or 'lowrank_cayley'")

    # --- volume-control interface used by GPT.forward (mirrors get_gamma/alpha) ---
    def mean_logscale(self):
        """Mean raw log-scale of L — averaged across blocks to form the global
        volume correction (the analogue of (mean γ + mean α)/2)."""
        return self.linear_map.mean_logscale()

    def n_scales(self):
        return self.linear_map.m

    def partitioned_step(self, y, avg=0):
        x, z = torch.split(y, y.shape[-1] // 2, dim=-1)
        x = x + self.resweight_attn * self.attn(self.ln_1(z))     # S_1
        x, z = torch.split(self.linear_map(torch.cat([x, z], -1), c=avg),
                           y.shape[-1] // 2, dim=-1)               # L
        z = z + self.resweight_mlp * self.mlp(self.ln_2(x))       # S_2
        return torch.cat([x, z], dim=-1)

    def inverse(self, y, avg=0):
        """Exact inverse of ``partitioned_step`` (algebraic reversibility)."""
        x, z = torch.split(y, y.shape[-1] // 2, dim=-1)
        z = z - self.resweight_mlp * self.mlp(self.ln_2(x))       # undo S_2
        x, z = torch.split(self.linear_map.inverse(torch.cat([x, z], -1), c=avg),
                           y.shape[-1] // 2, dim=-1)               # undo L
        x = x - self.resweight_attn * self.attn(self.ln_1(z))     # undo S_1
        return torch.cat([x, z], dim=-1)

    def forward(self, x, avg=0, mode='reversible'):
        if mode == 'vanilla':
            raise NotImplementedError(
                "LinearMixedReversibleBlock has no vanilla mode (the linear mix "
                "has no single-stream analogue).")
        return self.partitioned_step(x, avg)