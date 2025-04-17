import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder,EncoderLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.Variate_Embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.encoder = Encoder([EncoderLayer(configs) for _ in range(configs.layers)],configs)
        self.projector = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x_enc,emb=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_embed = self.Variate_Embedding(x_enc.permute(0, 2, 1)) 
        enc_out,vemb,token = self.encoder(x_embed)

        # B N D -> B N F -> B F N 
        pred = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if emb is None: return pred
        else:  return pred,vemb,token
