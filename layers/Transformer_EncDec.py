import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

class EncoderLayer(nn.Module):
    def __init__(self,configs):
        super(EncoderLayer, self).__init__()
        self.attn = CoIn(configs.d_model,configs.n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Dropout(configs.dropout),
            )
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout=nn.Dropout(configs.dropout)
        
    def forward(self, x):
        x_attn= self.attn(x)
        x=self.norm1(x+self.dropout(x_attn))
        
        x_ffn= self.ffn(x)
        x=self.norm2(x+self.dropout(x_ffn))

        return x


class Encoder(nn.Module):
    def __init__(self, attn_layers,configs):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.g_ffn   = nn.Sequential(
            nn.Linear(2*configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Dropout(configs.dropout),
            )
        self.g_token=nn.Parameter(torch.randn(configs.d_model))
    def forward(self, x):
        B,N,D=x.shape
        g_token=self.g_token.unsqueeze(0).unsqueeze(0).repeat(B,1,1)
        x_g=torch.cat([x, g_token], dim=1)
        vemb=x_g
        for attn_layer in self.attn_layers:
            x_g= attn_layer(x_g)
        token=x_g
        x_c=x_g[:,:-1,:]
        g_token=x_g[:,-1,:]
        x=torch.cat((x_c, g_token.unsqueeze(1).repeat(1,N,1)),-1)
        x=self.g_ffn(x)
        return x,vemb,token

# Contextual Weighted Interaction Block
class CoIn(nn.Module):
    def __init__(self, dim,n_heads=8,init=1):
        super(CoIn, self).__init__()
        self.gci = nn.Linear(dim, dim * 3)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        
        self.sigmoid=nn.Sigmoid()

        if init: self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):

        bs, n,dim = input.shape
        gci = self.gci(input).reshape(bs, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        gate, context, interact = gci[0], gci[1], gci[2] # bs head, n, head_dim
        
        gate=gate.reshape(-1,n,self.head_dim)
        context = context.reshape(1,-1,n,self.head_dim) #1,bs*head,n,head_dim
        interact = interact.reshape(1,-1,n,self.head_dim) #1,bs*head,n,head_dim
        
        context_wei = torch.softmax(context, dim=2)  # Shape: (1, bs*head, n, head_dim)
        interact_out = torch.sum(context_wei * interact, dim=2)

        gated_out=self.sigmoid(gate)*(interact_out.permute(1,0,2)) #bs*head,n,dim
        out=gated_out.reshape(bs,self.n_heads,n, self.head_dim).permute(0,2,1,3).reshape(bs,n,-1)
        return out