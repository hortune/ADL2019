import torch
import torch.nn as nn

class CQCQAttnRNNNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product',
                 num_layers=1):
        super(CQCQAttnRNNNet, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size = dim_embeddings, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)

        self.rnn2 = torch.nn.LSTM(
            input_size = 256 * 7, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)
        

        self.CW = torch.nn.Linear(256,256)
        self.softmax = torch.nn.Softmax(dim=2)
        self.similarity = similarity

    def forward(self, context, context_lens, options, option_lens):
        context_attn, context_rnn = self.self_attn(context)
        
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            #option, _ = self.attn(option, False, middle)
            option_attn, option_rnn = self.self_attn(option)
            # DNN Experiment
            option = self.c_attn(option_rnn, option_attn, context_rnn)
            context = self.c_attn(context_rnn, context_attn, option_rnn)
            
            if self.similarity == "weighted_inner_product":
                context = self.CW(context)
            if self.similarity == "inner_product" or "weighted_inner_product":
                logit = (option * context).sum(-1)
            elif self.similarity == "square_error":
                logit = -((context - option) ** 2).sum(-1)
            
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits
    
    def self_attn(self, data):
        data = self.rnn(data)[0]
        data2 = self.CW(data)
        attn_weight = self.softmax(data.matmul(data2.transpose(1,2)))
        rnn2_input =  attn_weight.matmul(data)
        
        return rnn2_input, data
    
    def c_attn(self, rnn_data, attn_data, attn_target):
        att_w = self.softmax(rnn_data.matmul(self.CW(attn_target).transpose(1,2)))
        aux = att_w.matmul(attn_target)
    
        rnn2_input = torch.stack([attn_data, 
                                    rnn_data, 
                                    attn_data - rnn_data, 
                                    attn_data * rnn_data,
                                    aux,
                                    aux - rnn_data,
                                    aux * rnn_data],-1).view(-1,rnn_data.size(1), 256 * 7)
        return self.rnn2(rnn2_input)[0].max(1)[0]
