import torch
import torch.nn as nn

class AttnRNNNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product',
                 num_layers=1):
        super(AttnRNNNet, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size = dim_embeddings, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)

        self.rnn2 = torch.nn.LSTM(
            input_size = 1024, 
            hidden_size = 128, 
            num_layers = num_layers, 
            bias = False, 
            batch_first = True,
            bidirectional = True)
        #self.W = torch.nn.Linear(512,512)
        self.W1 = torch.nn.Linear(256,256)
        self.softmax = torch.nn.Softmax(dim=2)
        self.similarity = similarity

    def forward(self, context, context_lens, options, option_lens):
        context = self.attn(context)
        if self.similarity == "weighted_inner_product":
            context = self.W1(context)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option = self.attn(option)
            if self.similarity == "inner_product" or "weighted_inner_product":
                logit = (option * context).sum(-1)
            elif self.similarity == "square_error":
                logit = -((context - option) ** 2).sum(-1)
            
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def attn(self, data):
        data = self.rnn(data)[0]
        data2 = self.W1(data)
        attn_weight = self.softmax(data.matmul(data2.transpose(1,2)))
        rnn2_input =  attn_weight.matmul(data)
        rnn2_input = torch.stack([rnn2_input, data, rnn2_input - data, rnn2_input * data],-1).view(-1,data.size(1),1024)
        
        res = self.rnn2(rnn2_input)[0].max(1)[0]
        return res
