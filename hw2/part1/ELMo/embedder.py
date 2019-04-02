import numpy as np
import pickle
from .model import ELMo
import torch

class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        
        with open("special_files/elmo_best_vocab.pickle","rb") as fd:
            self.word_vocab, self.char_vocab = pickle.load(fd)
        self.model = ELMo(num_embeddings = len(self.char_vocab),
                embedding_dim = ctx_emb_dim // 2,
                padding_idx = self.word_vocab.sp.pad.idx,
                conv_filters = [(1, 32), (2, 64), (3, 128), (4, 128), (5, 256), (6, 256), (7, 512)], 
                n_highways = 2,
                word_size = len(self.word_vocab),
                )
        self.model.load_state_dict(torch.load('special_files/elmo_best_model.pkl'))
        self.model = self.model.eval().cuda()
    
    def evaluate(self, f, b):
        (f1,f2,f3), (b1,b2,b3) = self.model(f, b)
        b1,b2,b3 = b1[:,::-1,:], b2[:,::-1,:], b3[:,::-1,:]
        v1 = np.concatenate([f1,b1] ,axis=2)
        v2 = np.concatenate([f2,b2] ,axis=2)
        v3 = np.concatenate([f3,b3] ,axis=2)

        return np.stack([v1,v2,v3], axis=2)
    
    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
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

        char_pad_idx = self.char_vocab.sp.pad.idx
        max_word_len = 16
        text_char = []
        for b in sentences:
            tbb = ['<bos>'] + b + ['<eos>']
            sent_vocab = []
            for w in tbb:
                if w in ['<bos>','<eos>']:
                    sent_vocab.append([self.char_vocab.vtoi(w)])
                else:
                    char_word = []
                    for c in w:
                        char_word.append(self.char_vocab.vtoi(c))
                    sent_vocab.append(char_word)
            text_char.append(sent_vocab)
        
        max_len = min(max(map(len, sentences)), max_sent_len)
        text_char = pad(text_char, max_len, [char_pad_idx])
        max_len = min(np.max([[len(w) for w in s] for s in text_char]), max_word_len)
        text_char = pad(text_char, max_len, char_pad_idx, depth=2)
        
        
        text_char_rev = [ele[::-1] for ele in text_char]
        text_char = [ele for ele in text_char]

        text_char = torch.tensor(text_char).cuda()
        text_char_rev = torch.tensor(text_char_rev).cuda()
        
        res = self.evaluate(text_char, text_char_rev)
        return res
