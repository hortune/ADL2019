import torch
import torch.nn as nn

class CAttnRNNNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product',
                 num_layers=1):
        super(CAttnRNNNet, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size = dim_embeddings, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)

        self.rnn2 = torch.nn.LSTM(
            input_size = 256 * 4, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)
        
        self.rnn3 = torch.nn.LSTM(
            input_size = 256 * 7, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)
        
        #self.W = torch.nn.Linear(256,256)
        self.CW = torch.nn.Linear(256,256)
        self.softmax = torch.nn.Softmax(dim=2)
        self.similarity = similarity

    def forward(self, context, context_lens, options, option_lens):
        context, middle = self.attn(context)
        
        if self.similarity == "weighted_inner_product":
            context = self.CW(context)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option, _ = self.attn(option, False, middle)
            if self.similarity == "inner_product" or "weighted_inner_product":
                logit = (option * context).sum(-1)
            elif self.similarity == "square_error":
                logit = -((context - option) ** 2).sum(-1)
            
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def attn(self, data, context=True, att_target = None):
        data = self.rnn(data)[0]
        data2 = self.CW(data)

        if not context:
            att_w = self.softmax(data.matmul(self.CW(att_target).transpose(1,2)))
            aux = att_w.matmul(att_target)
        
        attn_weight = self.softmax(data.matmul(data2.transpose(1,2)))
        rnn2_input =  attn_weight.matmul(data)
        
        if context:
            rnn2_input = torch.stack([rnn2_input, data, rnn2_input - data, rnn2_input * data],-1).view(-1,data.size(1),1024)
            res = self.rnn2(rnn2_input)[0].max(1)[0]
        else:
            rnn2_input = torch.stack([rnn2_input, 
                                        data, 
                                        rnn2_input - data, 
                                        rnn2_input * data,
                                        aux,
                                        aux - data,
                                        aux * data],-1).view(-1,data.size(1), 256 * 7)
            res = self.rnn3(rnn2_input)[0].max(1)[0]
        
        return res, data
