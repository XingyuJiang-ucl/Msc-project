from timeit import default_timer as timer

from torch.sparse import softmax


def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)

	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)

	else:
		raise NotImplementedError

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

import torch
import math
import torch.nn as nn

class SelfAttention(nn.Module):
	def	__init__(self, hidden_dim: torch.tensor = 728) -> None:
		super().__init__()
		self.hidden_dim = hidden_dim
		self.query_proj = nn.Linear(hidden_dim,hidden_dim)
		self.key_proj = nn.Linear(hidden_dim,hidden_dim)
		self.value_proj = nn.Linear(hidden_dim,hidden_dim)

	def forward(self,X):
		B, seq_len, hidden_dim = X.shape
		Q = self.query_proj(X)
		K = self.key_proj(X)
		V = self.value_proj(X)
		# Q,K,V shape = B, seqlen, hidden dim

		# attention value = B, seqlen,seqlen, 所以对K进行转置
		attention_value = torch.matmul(Q, torch.transpose(K,-1,-2))
		attention_weight = torch.softmax(attention_value / torch.sqrt(self.hidden_dim), dim=-1)
		# output shape = B, seqlen, hidden_dim
		output = torch.matmul(attention_weight, V)
		return output

x = torch.randn(2,3,4).to('cuda')
attention = SelfAttention(torch.tensor(4)).to('cuda')

print(attention(x))