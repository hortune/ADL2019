import random
import torch
from torch.utils.data import Dataset


class DialogDataset(Dataset):
    """
    Args:
        data (list): List of samples.
        padding (int): Index used to pad sequences to the same length.
        n_negative (int): Number of false options used as negative samples to
            train. Set to -1 to use all false options.
        n_positive (int): Number of e options used as positive samples to
            train. Set to -1 to use all true options.
        shuffle (bool): Do not shuffle ns when sampling.
            **SHOULD BE FALSE WHEN TESTING**
    """
    def __init__(self, data, padding=0,
                 n_negative=4, n_positive=1,
                 context_padded_len=600, option_padded_len=100, shuffle=True):
        self.data = data
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.context_padded_len = context_padded_len
        self.option_padded_len = option_padded_len
        self.padding = padding
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        positives = data['options'][:data['n_corrects']]
        negatives = data['options'][data['n_corrects']:]
        positive_ids = data['option_ids'][:data['n_corrects']]
        negative_ids = data['option_ids'][data['n_corrects']:]
        
        if self.n_positive == -1:
            positive_indices = list(range(len(positives)))
        else:
            positive_indices = random.sample(list(range(len(positives))), self.n_positive)
        
        if self.n_negative == -1:
            negative_indices = list(range(len(negatives)))
        else:
            negative_indices = random.sample(list(range(len(negatives))), self.n_negative)
        
        # collect sampled options
        data['options'] = (
            [positives[i] for i in positive_indices]
            + [negatives[i] for i in negative_indices]
        )
        data['option_ids'] = (
            [positive_ids[i] for i in positive_indices]
            + [negative_ids[i] for i in negative_indices]
        )
        data['labels'] = [1] * len(positive_indices) + [0] * len(negative_indices)

        # 
        # Manipulate utterance
        #

        data['context'] = [gg for g in data['context'] for gg in g]
        if len(data['context']) > self.context_padded_len:
            data['context'] = data['context'][:self.context_padded_len]

        return data

    def collate_fn(self, datas):
        batch = {}

        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['labels'] = torch.tensor([data['labels'] for data in datas])
        batch['option_ids'] = [data['option_ids'] for data in datas]

        # build tensor of context
        batch['context_lens'] = [len(data['context']) for data in datas]
        padded_len = min(self.context_padded_len, max(batch['context_lens']))
        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas]
        )

        # build tensor of options
        batch['option_lens'] = [
            [min(max(len(opt), 1), self.option_padded_len)
             for opt in data['options']]
            for data in datas]
        padded_len = min(
            self.option_padded_len,
            max(sum(batch['option_lens'], []))
        )
        batch['options'] = torch.tensor(
            [[pad_to_len(opt, padded_len, self.padding)
              for opt in data['options']]
             for data in datas]
        )

        return batch


def pad_to_len(arr, padded_len, padding=0):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    
    if padded_len >= len(arr):
        return arr + [padding]*(padded_len - len(arr))
    else:
        return arr[-padded_len:] #
