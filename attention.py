import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=4,
        dropout=0.1,
        attn_mask=None,
        is_causal=False,
        cross_attn: bool = False,
    ):
        
        # TODO fix it so it works even when query & key token length is different

        super().__init__()

        assert (
            dim % num_heads == 0
        ), f"dim={dim} must be divisible by num_heads={num_heads}"

        self.num_heads = num_heads
        self.attn_mask = attn_mask
        self.is_causal = is_causal
        self.cross_attn = cross_attn
        self.dropout = dropout

        if cross_attn:
            self.encoder_kv = nn.Linear(dim, dim * 2)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # only in PyTorch >= 2.0
        self.flash = hasattr(
            F, "scaled_dot_product_attention"
        )  # very memory efficient, can handle longer context

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor = None):

        B, T, D = x.shape
        head_dim = D // self.num_heads

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, T, D/H)

        if encoder_out is not None:  # cross attention
            assert (
                self.cross_attn
            ), "you passed cross_attn as false but also passed encoder_out in the forward"

            Ne = encoder_out.shape[1]
            encoder_kv = (
                self.encoder_kv(encoder_out)
                .reshape(B, Ne, 2, self.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = encoder_kv.unbind(0)

        elif self.cross_attn:
            raise ValueError(
                "you passed cross_attn but didn't pass encoder_out in the forward"
            )

        if self.flash:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                self.attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )

        else:

            attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)
            if self.is_causal:
                assert self.attn_mask is None
                temp_mask = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(q.dtype)

            if self.attn_mask is not None:
                if self.attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(self.attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += self.attn_mask

            scale = head_dim**-0.5
            q = q * scale

            attn_scores = q @ k.transpose(-2, -1)  # (B, H, T, T)
            attn_scores += attn_bias
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)
            out = attn_scores @ v  # (B, H, T, D/H)

        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.proj(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out