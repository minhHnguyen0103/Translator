import torch
import torch.nn as nn

class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_len, device='cpu'):
        super(PositionEncoder, self).__init__()
        
        self.encoding = torch.zeros(max_len, d_model, device=device)  # [max_len, d_model]
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device) # [max_len]
        pos = pos.float().unsqueeze(dim=1) # [max_len, 1]

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos/10000**(_2i/d_model))
        self.encoding[:, 1::2] = torch.cos(pos/10000**(_2i/d_model))
    
    def forward(self, tensor):
        batch_len, seq_len = tensor.size()
        return self.encoding[:seq_len, :]