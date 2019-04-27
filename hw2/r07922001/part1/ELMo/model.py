import torch
import torch.nn as nn

from .char_embedding import CharEmbedding

class ELMo(nn.Module):
    """
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx, conv_filters, n_highways, word_size):
        super().__init__()
        self.char_embedding = CharEmbedding(
                num_embeddings,
                16,
                padding_idx,
                conv_filters,
                n_highways,
                embedding_dim,
            )
        
        self.forward_lm_1 = nn.LSTM(
                    input_size = embedding_dim,
                    hidden_size = 4*embedding_dim,
                    batch_first = True,
                    num_layers = 1,)
        self.forward_lp_1 = nn.Linear(4*embedding_dim, embedding_dim)
        
        self.forward_lm_2 = nn.LSTM(
                    input_size = embedding_dim,
                    hidden_size = 4*embedding_dim,
                    batch_first = True,
                    num_layers = 1,)
        self.forward_lp_2 = nn.Linear(4*embedding_dim, embedding_dim)
        
        self.backward_lm_1 = nn.LSTM(
                    input_size = embedding_dim,
                    hidden_size = 4*embedding_dim,
                    batch_first = True,
                    num_layers = 1,)
        self.backward_lp_1 = nn.Linear(4*embedding_dim, embedding_dim)
        self.backward_lm_2 = nn.LSTM(
                    input_size = embedding_dim,
                    hidden_size = 4*embedding_dim,
                    batch_first = True,
                    num_layers = 1,)
        
        self.backward_lp_2 = nn.Linear(4*embedding_dim, embedding_dim)
        self.loss = nn.AdaptiveLogSoftmaxWithLoss(embedding_dim, word_size, [100,1000,10000])
    
    def forward(self, f, b, f_y=None, b_y=None, f_mask=None, b_mask=None):
        f1 = self.char_embedding(f)
        b1 = self.char_embedding(b)

        f2 = self.forward_lm_1(f1)[0]
        f2 = self.forward_lp_1(f2)
        b2 = self.backward_lm_1(b1)[0]
        b2 = self.backward_lp_1(b2)
        
        f3 = self.forward_lm_2(f2)[0]
        f3 = self.forward_lp_2(f3)
        b3 = self.backward_lm_2(b2)[0]
        b3 = self.backward_lp_2(b3)
        
        if f_y is None:
            return (f1.data.cpu().numpy(), f2.data.cpu().numpy(), f3.data.cpu().numpy()),\
                    (b1.data.cpu().numpy(), b2.data.cpu().numpy(), b3.data.cpu().numpy())

        dim = f3.size(0)*f3.size(1)
        f = f3.view(dim, -1)
        f_y = f_y.view(-1)
        f_mask = f_mask.view(-1)

        b = b3.view(dim, -1)
        b_y = b_y.view(-1)
        b_mask = b_mask.view(-1)
        
        return (self.loss(f[f_mask],f_y[f_mask]).loss + self.loss(b[b_mask],b_y[b_mask]).loss)
