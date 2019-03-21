import torch


class BasicRNNNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(BasicRNNNet, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size = dim_embeddings, 
            hidden_size = 300, 
            num_layers = 1, 
            bias = False, 
            batch_first = True)
        self.similarity = similarity
        if self.similarity == "weighted_inner_product":
            self.weighted = torch.nn.Linear(300,300)

    def forward(self, context, context_lens, options, option_lens, speaker):
        context = self.rnn(context)[0].max(1)[0]
        
        if self.similarity == "weighted_inner_product":
            context = self.weighted(context)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option = self.rnn(option)[0].max(1)[0]
            if self.similarity == "inner_product" or "weighted_inner_product":
                logit = (option * context).sum(-1)
            elif self.similarity == "square_error":
                logit = -((context - option) ** 2).sum(-1)
            
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits
