import torch
import torch.nn as nn

from models.blocks import EncoderBlock, DecoderBlock
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_ff, n_heads=8, drop_out=0.1, n_layers=24, device='cpu'):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, 
                                              d_model,
                                              max_len,
                                              drop_out,
                                              device=device)

        self.encode = nn.ModuleList([EncoderBlock(d_model, d_ff, n_heads, drop_out) 
                                     for i in range(n_layers)])
    
    def forward(self, src, src_mask):
        em = self.embedding(src)

        for module in self.encode:
            em = module(em, src_mask)

        return em


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_ff, n_heads=8, drop_out=0.1, n_layers=24, device='cpu'):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, 
                                              d_model, 
                                              max_len, 
                                              drop_out, 
                                              device=device)

        self.decode = nn.ModuleList([DecoderBlock(d_model, 
                                                  d_ff, 
                                                  n_heads, 
                                                  drop_out)
                                     for i in range(n_layers)])
        
        self.linear = nn.Linear(d_model, vocab_size)
    

    def forward(self, trg, src, trg_mask, src_mask):
        em = self.embedding(trg)
        # print(em.size(), src.size())

        for module in self.decode:
            em = module(em, src, trg_mask, src_mask)
        # Em: [batch_size, length, vocab_Size]
        # out = nn.Softmax(dim=-1)(self.linear(em))
        out = self.linear(em)
        
        return out


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()

        self.src_pad_idx = opt.src_pad_idx
        self.trg_pad_idx = opt.trg_pad_idx

        self.enc_vocab_size = opt.enc_vocab_size
        self.dec_vocab_size = opt.dec_vocab_size

        self.max_len = opt.max_len

        self.n_layers = opt.n_layers

        self.n_heads = opt.n_heads
        
        self.d_model = opt.d_model
        self.d_ff = opt.d_ff

        self.drop_out = opt.drop_out

        # if opt.gpu == True:
        #     if torch.cuda.is_available():
        #         self.device = 'cuda'
        #     else:
        #         print("No physical device available. Using cpu")
        #         self.device = 'cpu'
        print(opt.device)
        self.device=opt.device

        # self.task = opt.task

        self.encoder = Encoder(vocab_size=self.enc_vocab_size,
                               max_len=self.max_len,
                               d_model=self.d_model,
                               d_ff=self.d_ff,
                               n_heads=self.n_heads,
                               drop_out=self.drop_out,
                               n_layers=self.n_layers,
                               device=self.device
                               )
        self.decoder = Decoder(vocab_size=self.dec_vocab_size,
                               max_len=self.max_len,
                               d_model=self.d_model,
                               d_ff=self.d_ff,
                               n_heads=self.n_heads,
                               drop_out=self.drop_out,
                               n_layers=self.n_layers,
                               device=self.device
                                )
    

    def get_pad_mask(self, src, pad_idx):
        """
        Set pad tokens to False (no learning here)
        """
        return (src != pad_idx).unsqueeze(-2)

    
    def get_subsequent_mask(self, seq):
        """For masking out the subsequent info."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    
    def forward(self, src, trg):
        src_mask = self.get_pad_mask(src, self.src_pad_idx)
        trg_mask = self.get_pad_mask(trg, self.trg_pad_idx) & self.get_subsequent_mask(trg)

        enc_src = self.encoder(src, src_mask)
        dec_trg = self.decoder(trg, enc_src, trg_mask, src_mask)

        return dec_trg

