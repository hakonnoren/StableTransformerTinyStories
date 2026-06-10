"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from mingpt.utils import CfgNode as CN

from mingpt.grad_tracker import jacobian_metrics


# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config,n_embd):
        super().__init__()

        assert n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = n_embd
        # Optional capture hook for attention-distribution monitoring.
        # When set to a callable, it receives the post-softmax pre-dropout
        # matrix of shape (B, n_head, T, T) and is expected not to mutate it.
        # Default None => no-op.
        self._capture_attn = None


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if self._capture_attn is not None:
            self._capture_attn(att)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd) if config.normalize else nn.Identity()
        self.ln_2 = nn.LayerNorm(config.n_embd) if config.normalize else nn.Identity()

        # ReZero: trainable residual scaling parameters initialized to 0
        self.rezero = getattr(config, 'rezero', False)
        if self.rezero:
            self.resweight_attn = nn.Parameter(torch.zeros(1))
            self.resweight_mlp = nn.Parameter(torch.zeros(1))

        self.attn = CausalSelfAttention(config,config.n_embd)
        self.xP_state = False
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        if self.rezero:
            x0 = x + self.resweight_attn * self.attn(self.ln_1(x))
            x1 = x0 + self.resweight_mlp * self.mlpf(self.ln_2(x0))
        else:
            x0 = x + self.attn(self.ln_1(x))
            x1 = x0 + self.mlpf(self.ln_2(x0))
        return x1

    def store_activations(self,x0,x1):
        #Saving intermediate activations (to be passed on to GPT)
        self.x0,self.x1 = x0,x1




class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.energy = config.energy
        self.reversible = config.reversible
        self.config = config
        self.mode = 'reversible'
        self._fine_tune_vanilla = False
        self._ft_handles = []          # grad hook handles
        self.vanilla_half = "first"    # "first" -> [:d], "second" -> [d:]

        #type_given = config.model_type is not None
        #params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        #assert type_given ^ params_given # exactly one of these (XOR)
        type_given = False
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),

                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        block = Block
        if config.energy:
            from mingpt.energy_model import EnergyBlock
            block = EnergyBlock
        if config.reversible:
            if getattr(config.rev_config, 'linear_map', 'diag') != 'diag':
                from mingpt.reversible_model import LinearMixedReversibleBlock
                block = LinearMixedReversibleBlock
            elif getattr(config.rev_config, 'full_block', False):
                from mingpt.reversible_model import FullBlockReversibleBlock
                block = FullBlockReversibleBlock
            else:
                from mingpt.reversible_model import ReversibleBlock
                block = ReversibleBlock

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd) if config.normalize else nn.Identity(),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #we tie the decoder weights to the input embeddings
        #we also fix KeyError: 'lm_head.weight'

        #self.lm_head.weight = 
        #self.transformer.wte.weight = self.lm_head.weight

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: {}M {}".format(n_params/1e6,n_params))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len([k for k in sd if not k.endswith('.attn.bias')])
        #this is a fix from https://github.com/karpathy/minGPT/issues/120
        #assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self,learning_rate=1e-3, betas=(0.9, 0.95), weight_decay=0.1):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.startswith('resweight'):
                    # ReZero residual scaling parameters should not be decayed
                    no_decay.add(fpn)
                elif 'linear_map' in fpn:
                    # Linear-mix map parameters (log-scales eta, Householder
                    # vectors, low-rank factors) are not weight-decayed — like γ/α.
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        #optim_groups = [
        #    {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": weight_decay},
        #    {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        #]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx, targets=None, jac_metrics=None, return_intermediates=False,return_first_activation=False):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices (batch, seq_len)
            targets: Target token indices for loss computation
            jac_metrics: Optional metrics to compute for Jacobian analysis
            return_intermediates: If True, store intermediate activations for J^{-1} transport.
                                  Returns (logits, loss, intermediates) where intermediates is a dict.
        
        Returns:
            logits, loss (or logits, loss, intermediates if return_intermediates=True)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        if self.reversible and self.mode == "vanilla":
            # We run only on Z (dim=d) and only use half of the embedding params.
            # Choose which half is Z. Here: Z = last half [d:2d].
            d = self.config.n_embd // 2
            assert 2 * d == self.config.n_embd, "n_embd must be even for reversible vanilla-mode"

            tok_emb = F.embedding(idx, self.transformer.wte.weight[:, d:])   # (b,t,d)
            pos_emb = F.embedding(pos, self.transformer.wpe.weight[:, d:])   # (1,t,d)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # Normal path (vanilla GPT or reversible partitioned)
            tok_emb = self.transformer.wte(idx)                              # (b,t,2d) or (b,t,d)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

        if return_first_activation:
            return x



        # Storage for intermediate activations when needed for J^{-1} transport
        intermediates = None
        if return_intermediates and self.reversible:
            intermediates = {
                'activations': [],  # List of (x_l, z_l) tuples at each layer input
                'blocks': [],       # List of block references
                'embedding': None,  # Initial embedding (x_0)
            }

        if self.reversible:
            d = self.config.n_embd // 2
            n = len(self.transformer.h)
            rev_cfg = self.config.rev_config
            T_seq = x.size(1)

            # Decide centering mode: global (default), per-block, free, or trivial.
            if rev_cfg.free_vol or rev_cfg.vol_pres_per_block or rev_cfg.volume_pres:
                # No external centering passed in. Blocks self-handle
                # (vol_pres_per_block) or just ignore avg (volume_pres / free_vol).
                avg_corr = 0.0
            elif getattr(rev_cfg, 'linear_map', 'diag') != 'diag':
                # Linear-mix blocks: center the map's log-scales so the stack's
                # total log|det L| = -lambd (lambd=0 => volume preserving).
                m = self.config.n_embd
                ell_avg = torch.mean(torch.stack([b.mean_logscale() for b in self.transformer.h]))
                avg_corr = ell_avg + rev_cfg.lambd / (n * m * T_seq)
            else:
                # Global centering. Total Jacobian log|det| = -lambd, T-independent.
                gamma_avg = torch.mean(torch.stack([torch.mean(b.get_gamma()) for b in self.transformer.h]), dim=0)
                alpha_avg = torch.mean(torch.stack([torch.mean(b.get_alpha()) for b in self.transformer.h]), dim=0)
                avg = (gamma_avg + alpha_avg) / 2
                avg_corr = avg - rev_cfg.lambd / (2 * n * d * T_seq)
                self.e_gammas = torch.exp(-(torch.stack([b.get_gamma() for b in self.transformer.h]) - avg))
                self.e_alphas = torch.exp(-(torch.stack([b.get_alpha() for b in self.transformer.h]) - avg))

            if return_intermediates:
                intermediates['embedding'] = x.clone()
                x_part, z_part = torch.split(x, d, dim=-1)
                intermediates['activations'].append((x_part.clone(), z_part.clone()))

            for block in self.transformer.h:
                if return_intermediates:
                    intermediates['blocks'].append(block)
                x = block(x, avg_corr, mode=self.mode)
                if return_intermediates:
                    x_part, z_part = torch.split(x, d, dim=-1)
                    intermediates['activations'].append((x_part.clone(), z_part.clone()))

        else:
            for block in self.transformer.h:
                x = block(x)
        
        

        if self.reversible and self.mode == "vanilla":
            d = self.config.n_embd // 2

            # Apply a "half" final LayerNorm (stats over d dims, not 2d)
            if isinstance(self.transformer.ln_f, nn.LayerNorm):
                x = F.layer_norm(
                    x, (d,),
                    self.transformer.ln_f.weight[d:],
                    self.transformer.ln_f.bias[d:],
                    self.transformer.ln_f.eps
                )
            # Unembed with half of lm_head weights
            logits = F.linear(x, self.lm_head.weight[:, d:])  # bias=False in your lm_head
        else:
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
        
        # Store final hidden state in intermediates if requested
        if return_intermediates and intermediates is not None:
            intermediates['final_hidden'] = x.clone()
            intermediates['ln_f'] = self.transformer.ln_f
            intermediates['lm_head'] = self.lm_head
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if return_intermediates:
            return logits, loss, intermediates
        return logits, loss
    


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def compute_layer_derivative(self, L: int, k: int, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian of the output of block L with respect to the input to block L - k.
        
        Arguments:
            L (int): The index of the block whose output we are interested in.
                     (Assumes 0-indexing in self.transformer.h.)
            k (int): The number of blocks back from L at which we take the input.
                     (That is, we compute d(output at block L)/d(input to block L-k).)
            x (torch.Tensor): A tensor that will serve as the input to block L - k.
                              Its shape is (batch_size, sequence_length, n_embd).
                              It should be a reasonably small tensor (e.g. batch_size=1)
                              to avoid excessive memory usage.
        
        Returns:
            torch.Tensor: The Jacobian matrix, i.e. the derivative of the output of block L 
                          with respect to the input of block L - k.
                          Its shape will be (batch_size, sequence_length, n_embd,
                          batch_size, sequence_length, n_embd).
        """
        # We clone and require grad for the input.
        x_in = x.clone().detach().requires_grad_(True)
        
        # Define a function that applies blocks L-k through L sequentially.
        def f(x_input):
            out = x_input
            # Iterate over the selected blocks.
            # Note: the slice [L-k : L+1] applies block indices L-k, L-k+1, ..., L.
            for block in self.transformer.h[L - k : L + 1]:
                out = block(out)
            return out
        
        # Use PyTorch's autograd.functional.jacobian to compute the full derivative.
        jacobian = torch.autograd.functional.jacobian(f, x_in)
        return jacobian


    def block_forward(self, upto=None):
        """Return a function f(x) that applies the block stack with the *same*
        centering / scaling that GPT.forward applies. Used by Jacobian /
        singular-value analysis — see mingpt.extreme_singular_values.

        For reversible models this includes the global avg-lambd/(n*d*T)
        correction; otherwise the SVs would not reflect the actual forward
        map and sum(log sigma) != log|det| of the real operator.

        Args:
            upto (int | None): If given, apply only the first ``upto`` blocks
                (i.e. ``blocks[:upto]``) — used for the *cumulative* Jacobian
                d(activation after block ``upto``)/d(input). ``None`` applies the
                full stack. The global-centering correction always uses the
                full-model block count/averages (``n`` over all blocks), exactly
                as the real forward pass does, so a prefix is a faithful partial
                evaluation of the same map.
        """
        blocks = self.transformer.h
        sel = blocks if upto is None else blocks[:upto]

        if not self.reversible:
            def f(x):
                out = x
                for block in sel:
                    out = block(out)
                return out
            return f

        rev_cfg = self.config.rev_config
        n = len(blocks)
        d = self.config.n_embd // 2
        mode = self.mode

        if rev_cfg.free_vol or rev_cfg.vol_pres_per_block or rev_cfg.volume_pres:
            def f(x):
                out = x
                for block in sel:
                    out = block(out, 0.0, mode=mode)
                return out
            return f

        # Linear-mix blocks: center the map log-scales (analogue of γ/α centering).
        if getattr(rev_cfg, 'linear_map', 'diag') != 'diag':
            m = self.config.n_embd
            def f(x):
                T_seq = x.size(1)
                ell_avg = torch.mean(torch.stack([b.mean_logscale() for b in blocks]))
                avg_corr = ell_avg + rev_cfg.lambd / (n * m * T_seq)
                out = x
                for block in sel:
                    out = block(out, avg_corr, mode=mode)
                return out
            return f

        # Default: global centering. avg_corr is recomputed from current params
        # at every call so this remains differentiable in gamma/alpha.
        def f(x):
            T_seq = x.size(1)
            gamma_avg = torch.mean(torch.stack([torch.mean(b.get_gamma()) for b in blocks]), dim=0)
            alpha_avg = torch.mean(torch.stack([torch.mean(b.get_alpha()) for b in blocks]), dim=0)
            avg = (gamma_avg + alpha_avg) / 2
            avg_corr = avg - rev_cfg.lambd / (2 * n * d * T_seq)
            out = x
            for block in sel:
                out = block(out, avg_corr, mode=mode)
            return out
        return f

    
    def _half_slice(self):
        d = self.config.n_embd // 2
        if self.vanilla_half == "first":
            return slice(0, d)
        elif self.vanilla_half == "second":
            return slice(d, 2*d)
        else:
            raise ValueError("vanilla_half must be 'first' or 'second'")

    def _clear_ft_hooks(self):
        for h in getattr(self, "_ft_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._ft_handles = []

    def _mask_grad_cols(self, start: int, end: int):
        # returns a hook function that zeros gradient columns outside [start:end)
        def hook(grad):
            if grad is None:
                return None
            g = grad
            # keep this cheap: avoid allocating a full mask tensor
            if start > 0:
                g = g.clone()
                g[:, :start] = 0
            if end < g.shape[1]:
                if g is grad:
                    g = g.clone()
                g[:, end:] = 0
            return g
        return hook

    def _apply_fine_tune_vanilla(self, enabled: bool):
        """
        enabled=True:
          - force self.mode='vanilla'
          - freeze everything
          - unfreeze alpha/gamma in reversible blocks
          - unfreeze wte/wpe/lm_head but mask grads to only half columns (first half by default)
        enabled=False:
          - remove grad masks
          - leave requires_grad as-is (you can re-enable full training manually if you want)
        """
        self._clear_ft_hooks()

        self._fine_tune_vanilla = bool(enabled)
        if not self._fine_tune_vanilla:
            return

        assert self.reversible, "fine_tune_vanilla is intended for reversible->vanilla training"
        self.mode = "vanilla"

        # 1) freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # 2) unfreeze alpha/gamma in reversible blocks
        for block in self.transformer.h:
            if hasattr(block, "alpha_bias") and isinstance(block.alpha_bias, torch.nn.Parameter):
                block.alpha_bias.requires_grad = True
            if hasattr(block, "gamma_bias") and isinstance(block.gamma_bias, torch.nn.Parameter):
                block.gamma_bias.requires_grad = True

        # 3) unfreeze embedding/unembedding weights (but only train half columns via hooks)
        self.transformer.wte.weight.requires_grad = True
        self.transformer.wpe.weight.requires_grad = True
        self.lm_head.weight.requires_grad = True

        sl = self._half_slice()
        start, end = sl.start, sl.stop

        self._ft_handles.append(self.transformer.wte.weight.register_hook(self._mask_grad_cols(start, end)))
        self._ft_handles.append(self.transformer.wpe.weight.register_hook(self._mask_grad_cols(start, end)))
        self._ft_handles.append(self.lm_head.weight.register_hook(self._mask_grad_cols(start, end)))

    @property
    def fine_tune_vanilla(self):
        return self._fine_tune_vanilla

    @fine_tune_vanilla.setter
    def fine_tune_vanilla(self, value: bool):
        self._apply_fine_tune_vanilla(bool(value))
