import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import math

class FullSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t): return t.view(B, L, self.h, self.d).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)  # (B,h,L,L)
        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        return self.proj_drop(self.o(y))

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FullSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class InnerClassifier(nn.Module):
    """
    Bidirectional transformer encoder for memo -> category.
    Returns per-example cross-entropy loss: shape (B,).
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 128,
        pad_id: int = 0,
        dropout: float = 0.1,
        tie_cls_to_tok: bool = False,  # keep False; for LMs only
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, 4.0, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) token ids, y: (B,) class ids
        Returns: per-example losses (B,)
        """
        B, L = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"{L=} > max_seq_len {self.max_seq_len}")
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        key_padding_mask = x.eq(self.pad_id)

        for blk in self.blocks:
            h = blk(h, key_padding_mask)

        h = self.norm(h)
        # simple pooling: mean over non-pad tokens
        mask = (~key_padding_mask).float()  # 1 for valid
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, d_model)

        logits = self.head(pooled)  # (B, C)
        # per-example cross-entropy
        loss = F.cross_entropy(logits, y, reduction="none")  # (B,)
        return loss
    # inside class InnerClassifier(nn.Module):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        key_padding_mask = x.eq(self.pad_id)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        h = self.norm(h)
        mask = (~key_padding_mask).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, d_model)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)  # (B, num_classes)
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        key_padding_mask = x.eq(self.pad_id)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        h = self.norm(h)
        mask = (~key_padding_mask).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.head(pooled)
        return torch.softmax(logits, dim=-1)

class DataRater(nn.Module):
    """
    Lightweight scorer ϕ_η: token ids -> scalar score per example.
    Uses pooled token embeddings + a small MLP head.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        max_seq_len: int = 128,
        pad_id: int = 0,
        dropout: float = 0.1,
        temperature: float = 1.0,            # softmax temperature (lower = peakier)
        learnable_temp: bool = False,        # set True to meta-learn temperature
        shared_tok_emb: nn.Embedding = None  # optionally share with inner model
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

        self.tok_emb = shared_tok_emb if shared_tok_emb is not None else nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # small MLP head for scoring
        hidden = max(64, d_model // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) token ids
        returns: scores (B,)
        """
        B, L = x.shape
        if L > self.max_seq_len:
            raise ValueError(f"{L=} > max_seq_len {self.max_seq_len}")

        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        pad_mask = x.eq(self.pad_id)            # (B, L) True at pads
        valid = (~pad_mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B, d_model)

        score = self.head(pooled).squeeze(-1)   # (B,)

        # optional centering improves stability of batch-softmax weights
        score = score - score.mean()

        temp = torch.clamp(self.log_temp.exp(), 1e-2, 100.0)
        return score / temp  # (B,)
import torch
from torch import nn
from torch.nn import functional as F

# assumes EncoderBlock already defined above

class InnerClassifierWithAux(nn.Module):
    def __init__(self, vocab_size, num_classes, aux_dim,
                 d_model=256, n_layers=4, n_heads=4, max_seq_len=128, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, 4.0, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.aux_proj = nn.Sequential(nn.LayerNorm(aux_dim), nn.Linear(aux_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.LayerNorm(2*d_model), nn.Linear(2*d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, num_classes))
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def _encode_text(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        key_padding_mask = x.eq(self.pad_id)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        h = self.norm(h)
        mask = (~key_padding_mask).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, x, y, aux):
        pooled = self._encode_text(x)
        auxp = self.aux_proj(aux)
        logits = self.head(torch.cat([pooled, auxp], dim=-1))
        return F.cross_entropy(logits, y, reduction="none")

    @torch.no_grad()
    def predict_proba(self, x, aux):
        pooled = self._encode_text(x)
        auxp = self.aux_proj[:-2](aux)  # LayerNorm+Linear only
        logits = self.head(torch.cat([pooled, auxp], dim=-1))
        return torch.softmax(logits, dim=-1)


class DataRaterWithAux(nn.Module):
    def __init__(self, vocab_size, aux_dim, d_model=128, n_layers=2, n_heads=4, max_seq_len=128, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, 4.0, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.aux_proj = nn.Sequential(nn.LayerNorm(aux_dim), nn.Linear(aux_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.LayerNorm(2*d_model), nn.Linear(2*d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    def forward(self, x, aux):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        key_padding_mask = x.eq(self.pad_id)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        h = self.norm(h)
        mask = (~key_padding_mask).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        auxp = self.aux_proj(aux)
        score = self.head(torch.cat([pooled, auxp], dim=-1)).squeeze(-1)
        return score - score.mean()
def softmax_batch_weights(scores):
    # σ_B(ϕ(x)) over the current batch
    return torch.softmax(scores, dim=0)

# 3) One inner update unroll step with weighted gradients (eq. (4))
def inner_step_weighted(model, optimizer, x, y, dr_model):
    model.train()
    optimizer.zero_grad()
    # DataRater scores and batch-softmax weights
    with torch.set_grad_enabled(True):
        scores = dr_model(x)                     # (B,)
    weights = softmax_batch_weights(scores)      # sum=1 over batch

    # per-example losses
    per_seq_loss = model(x, y)                   # (B,)
    # weighted mean over examples (sequence-level)
    loss = torch.sum(weights * per_seq_loss)

    # backprop to θ; keep graph so outer step can backprop through unroll
    loss.backward(create_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)  # optional
    optimizer.step()
    return loss.detach()
def inner_step_weighted_with_aux(model, optimizer, x, y, aux, dr_model=None, clip=0.01):
    """
    model: InnerClassifierWithAux
    dr_model: DataRaterWithAux or None
    x:   (B, L) token ids
    aux: (B, F) numeric features
    y:   (B,)   labels
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if dr_model is None:
        per_ex = model(x, y, aux)               # (B,)
        loss = per_ex.mean()
    else:
        scores  = dr_model(x, aux)              # (B,)
        weights = torch.softmax(scores, dim=0)  # sum=1
        per_ex  = model(x, y, aux)              # (B,)
        loss    = (weights * per_ex).sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
    optimizer.step()
    return loss.detach()
# 4) Outer/meta step
def meta_step(dr_model, meta_opt, inner_models, inner_opts, batch_iter_train, batch_iter_val,
              T=2, N_outer=64):
    meta_opt.zero_grad()

    # accumulate meta-gradients across a population of inner models, then average
    total_meta_loss = 0.0
    for m, opt in zip(inner_models, inner_opts):
        # unroll T inner updates on D_train
        # (optionally start from a reinit'd θ periodically)
        for _ in range(T):
            x_tr, y_tr = next(batch_iter_train)
            inner_step_weighted(m, opt, x_tr, y_tr, dr_model)

        # compute outer loss on held-out D_test (same cross-entropy in default setup)
        # IMPORTANT: this must depend on θ_{T}(η), so keep the graph alive
        x_val, y_val = next(batch_iter_val)      # N_outer examples
        with torch.set_grad_enabled(True):
            per_seq_loss_val = m(x_val, y_val)   # (B,)
            outer_loss = per_seq_loss_val.mean() # scalar

        # accumulate; autograd will route ∂outer_loss/∂η through the unroll
        total_meta_loss = total_meta_loss + outer_loss

    # average across the population and take a meta step on η
    total_meta_loss = total_meta_loss / len(inner_models)
    total_meta_loss.backward()   # computes meta-gradients w.r.t. η
    meta_opt.step()
    return total_meta_loss.item()