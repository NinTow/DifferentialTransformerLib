# -*- coding: utf-8 -*-


import torch

class DiT_Attention(torch.nn.Module):
    def __init__(self, dim, num_heads, differential_gain=0.3, mask=None, qkv_bias=False, o_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.differential_gain = torch.nn.Parameter(torch.tensor(differential_gain))
        self.dim = dim
        self.qw1 = torch.nn.ModuleList([torch.nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for a in range(num_heads)])
        self.kw1 = torch.nn.ModuleList([torch.nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for a in range(num_heads)])
        self.qw2 = torch.nn.ModuleList([torch.nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for a in range(num_heads)])
        self.kw2 = torch.nn.ModuleList([torch.nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for a in range(num_heads)])
        self.vw = torch.nn.ModuleList([torch.nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for a in range(num_heads)])
        self.ow = torch.nn.Linear(dim, dim, bias=o_bias)
        self.mask = mask
        self.norm = torch.nn.functional.group_norm
        self.scaling = torch.sqrt(torch.tensor(self.dim//self.num_heads))
    def forward(self, q, k, v):
        #InputShape = [bat, seq, dim]
        xx = []
        if (self.mask == None):
            for a in range(self.num_heads):
                x = self.qw1[a](q[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]) @ torch.transpose(self.kw1[a](k[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]), 1, 2) / self.scaling
                x2 = self.qw2[a](q[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]) @ torch.transpose(self.kw2[a](k[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]), 1, 2) / self.scaling
                x = (torch.nn.functional.softmax(x, dim=-1) - self.differential_gain * torch.nn.functional.softmax(x2, dim=-1)) @ self.vw[a](v[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads])
                x = self.norm(x, v.shape[1])
                xx.append(x)
            x = torch.cat(xx, dim=-1)
            x = self.ow(x)
        else:
            for a in range(self.num_heads):
                x = self.qw1[a](q[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]) @ torch.transpose(self.kw1[a](k[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]), 1, 2) / self.scaling
                x2 = self.qw2[a](q[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]) @ torch.transpose(self.kw2[a](k[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads]), 1, 2) / self.scaling
                x = self.mask * x
                x = (torch.nn.functional.softmax(x, dim=-1) - self.differential_gain * torch.nn.functional.softmax(x2, dim=-1)) @ self.vw[a](v[:, :, a*self.dim//self.num_heads:(a+1)*self.dim//self.num_heads])
                x = self.norm(x, v.shape[1])
                xx.append(x)
            x = torch.cat(xx, dim=-1)
            x = self.ow(x)
        return x