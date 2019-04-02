import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

class ELMoDataset(Dataset):
    def __init__(self, data, word_vocab, char_vocab, workers=4):
        self._data = []
        for dd in data:
            tdd = ["<bos>"] + dd + ["<eos>"]
            self._data.extend([(tdd[i:i+64], tdd[i+1:i+65]) for i in range(0,len(tdd)-1,64)])

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def create_collate_fn(word_vocab, char_vocab, max_sent_len, max_word_len):
    word_pad_idx = word_vocab.sp.pad.idx
    char_pad_idx = char_vocab.sp.pad.idx
    def pad(batch, max_len, padding, depth=1):
        for i, b in enumerate(batch):
            if depth == 1:
                batch[i] = b[:max_len]
                batch[i] += [padding for _ in range(max_len - len(b))]
            elif depth == 2:
                for j, bb in enumerate(b):
                    batch[i][j] = bb[:max_len]
                    batch[i][j] += [padding] * (max_len - len(bb))

        return batch

    def collate_fn(batch):
        text_orig = [b[0] for b in batch]
        text_rev  = [b[1] for b in batch]
        
        # Get Forward Target
        text_word = [[word_vocab.vtoi(w) for w in b[1]] for b in batch]

        # Get Forward X
        text_char = [[[char_vocab.vtoi(w)] if w in ['<bos>','<eos>'] else \
                [char_vocab.vtoi(c) for c in w] \
                for w in b[0]] for b in batch]
        
        # Get Backward Target
        text_word_rev = [[word_vocab.vtoi(w) for w in b[0][::-1]] for b in batch]

        # Get Backward X
        text_char_rev = [[[char_vocab.vtoi(w)] if w in ['<bos>','<eos>'] else \
                [char_vocab.vtoi(c) for c in w] \
                for w in b[1][::-1]] for b in batch]

        max_len = min(max(map(len, text_word)), max_sent_len)
        text_word = pad(text_word, max_len, word_pad_idx)
        text_word_rev = pad(text_word_rev, max_len, word_pad_idx)
        
        text_char = pad(text_char, max_len, [char_pad_idx])
        text_char_rev = pad(text_char_rev, max_len, [char_pad_idx])

        max_len = min(np.max([[len(w) for w in s] for s in text_char]), max_word_len)
        text_char = pad(text_char, max_len, char_pad_idx, depth=2)
        text_char_rev = pad(text_char_rev, max_len, char_pad_idx, depth=2)

        text_word = torch.tensor(text_word)
        text_word_rev = torch.tensor(text_word_rev)
        text_char = torch.tensor(text_char)
        text_char_rev = torch.tensor(text_char_rev)
        text_pad_mask = text_word != word_pad_idx
        text_pad_mask_rev = text_word_rev != word_pad_idx
        return {
            'text_orig': text_orig,
            'text_word': text_word,
            'text_char': text_char,
            'text_word_rev': text_word_rev,
            'text_char_rev': text_char_rev,
            'text_pad_mask': text_pad_mask,
            'text_pad_mask_rev': text_pad_mask_rev,
        }

    return collate_fn

def create_data_loader(dataset, word_vocab, char_vocab, max_sent_len, max_word_len,
                       batch_size, n_workers, shuffle=True):
    collate_fn = create_collate_fn(word_vocab, char_vocab, max_sent_len, max_word_len)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
        collate_fn=collate_fn)

    return data_loader
