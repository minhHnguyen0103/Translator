import torch
import torch.nn as nn

from models.layers import MultiheadAttention, LayerNorm, PositionwiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_out):
        super(EncoderBlock, self).__init__()
        self.attn = MultiheadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)

        self.ffwd = PositionwiseFeedForward(d_model, hidden=d_ff, drop_prob=drop_out)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, inp, src_mask):
        q, k, v = inp.clone(), inp.clone(), inp.clone()
        
        out = self.attn(q,k,v, mask=src_mask)
        out_ = self.norm1(inp + out)

        out = self.ffwd(out_)
        out_ = self.norm1(out_ + out)
        return out_


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_out):
        super(DecoderBlock, self).__init__()
        self.masked_attn = MultiheadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)

        self.attn = MultiheadAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_out)

        self.ffwd = PositionwiseFeedForward(d_model, d_ff, drop_prob=drop_out)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_out)
    
    def forward(self, trg, src, trg_mask, src_mask):
        out_ = trg
        out = self.masked_attn(trg, trg, trg, mask=trg_mask)

        out_ = out_ + out
        out_ = self.norm1(out_)
        out_ = self.dropout1(out_)

        if src is not None:
            out = self.attn(out_, src, src, mask=src_mask)

            out_ = self.norm2(out_ + out)
            out_ = self.dropout2(out_)

        out = self.ffwd(out_)
        out_ = self.norm3(out_ + out)
        out_ = self.dropout3(out_)

        return out_



