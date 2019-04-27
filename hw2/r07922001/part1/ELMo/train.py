from common.vocab import Vocab
from .dataset import ELMoDataset, create_data_loader

import pickle
import string
from collections import Counter
from tqdm import tqdm
import ipdb
import sys
from box import Box
from .model import ELMo

from torch.nn import AdaptiveLogSoftmaxWithLoss
from torch.optim import Adam
import torch

def create_vocab(data, cfg):
    print('[*] Creating word vocab')
    words = Counter()
    
    bar = tqdm(
        data, desc='[*] Collecting word tokens',
        dynamic_ncols=True)
    
    for dd in bar:
        words.update([w for w in dd])
    bar.close()
    
    tokens = [w for w, _ in words.most_common(cfg.word.size)]
    word_vocab = Vocab(tokens, **cfg.word, )
    char_vocab = Vocab(list(string.printable), **cfg.char)

    return word_vocab, char_vocab

def main():
    cfg = Box.from_yaml(filename="ELMo/config.yaml")
    with open("data/language_model/corpus_tokenized.txt") as fd:
        data = []
        for i, ele in enumerate(fd):
            if i % 10000 == 0:
                print(i)
            if i == 1250000:
                break
            data.append(ele.strip().split())

    res = create_vocab(data, cfg.vocab)

    with open("dataset/vocab_small_res.pickle", "wb") as fd:
        pickle.dump(res, fd)
    
    dataset = ELMoDataset(data, res[0], res[1])
    data_loader = create_data_loader(
            dataset = dataset,
            word_vocab = res[0],
            char_vocab = res[1],
            **cfg.data_loader)
    
    print("Create Model")
    model = ELMo(num_embeddings = len(res[1]),
            embedding_dim = cfg.elmo_embedder.ctx_emb_dim // 2,
            padding_idx = res[0].sp.pad.idx,
            conv_filters = [(1, 32), (2, 64), (3, 128), (4, 128), (5, 256), (6, 256), (7, 512)], 
            n_highways = 2,
            word_size = len(res[0]),
            ).cuda()
    
    optim = Adam(model.parameters(), lr=1e-3)
    bar = tqdm(
        data_loader, desc='[ Only Epoch ]',
        leave=False, position=1, dynamic_ncols=True)
    losses = []
    for idx, batch in enumerate(bar):
        #if idx % 10000 == 0:
        #    torch.save(model.state_dict(),"dataset/elmo_basic_model_{}.pkl".format(idx))
        if idx == 20000:
            break
        f_y = batch['text_word'].cuda()
        f_x = batch['text_char'].cuda()

        b_y = batch['text_word_rev'].cuda()
        b_x = batch['text_char_rev'].cuda()
        
        f_mask = batch['text_pad_mask'].cuda()
        b_mask = batch['text_pad_mask_rev'].cuda()
        
        optim.zero_grad()
        loss = model(f_x, b_x, f_y, b_y, f_mask, b_mask)
        loss.backward()
        optim.step()
        losses.append((loss.item()))
        bar.set_postfix_str("loss: {}".format(str(loss.item()/2)))
    bar.close()
    torch.save(model.state_dict(),"model/elmo_small_model.pkl")

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main()
