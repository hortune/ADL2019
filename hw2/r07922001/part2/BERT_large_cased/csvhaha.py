import argparse
import csv
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path

import ipdb
import spacy
from box import Box
from tqdm import tqdm

from .dataset import Part1Dataset
from common.vocab import Vocab
from pytorch_pretrained_bert import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path, help='Target dataset directory')
    args = parser.parse_args()

    return vars(args)


def load_data(mode, data_path, nlp):
    print('[*] Loading {} data from {}'.format(mode, data_path))
    with data_path.open() as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]
    
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    for d in tqdm(data, desc='[*] Tokenizing', dynamic_ncols=True):
        text = re.sub('-+', ' ', d['text'])
        text = re.sub('\s+', ' ', text)
        d['text'] = tokenizer.tokenize(text)
    print('[-] {} data loaded\n'.format(mode.capitalize()))

    return data


def create_vocab(data, cfg, dataset_dir):
    print('[*] Creating word vocab')
    words = Counter()
    for m, d in data.items():
        bar = tqdm(
            d, desc='[*] Collecting word tokens form {} data'.format(m),
            dynamic_ncols=True)
        for dd in bar:
            words.update([w.lower() for w in dd['text']])
        bar.close()
    tokens = [w for w, _ in words.most_common(cfg.word.size)]
    word_vocab = Vocab(tokens, **cfg.word)
    word_vocab_path = (dataset_dir / 'word.pkl')
    with word_vocab_path.open(mode='wb') as f:
        pickle.dump(word_vocab, f)
    print('[-] Word vocab saved at {}\n'.format(word_vocab_path))

    print('[*] Creating char vocab')
    char_vocab = Vocab(list(string.printable), **cfg.char)
    char_vocab_path = (dataset_dir / 'char.pkl')
    with char_vocab_path.open(mode='wb') as f:
        pickle.dump(char_vocab, f)
    print('[-] Char vocab saved to {}\n'.format(char_vocab_path))

    return word_vocab, char_vocab


def create_dataset(data, word_vocab, char_vocab, dataset_dir):
    for m, d in data.items():
        print('[*] Creating {} dataset'.format(m))
        dataset = Part1Dataset(d, word_vocab, char_vocab)
        dataset_path = (dataset_dir / '{}.pkl'.format(m))
        with dataset_path.open(mode='wb') as f:
            pickle.dump(dataset, f)
        print('[-] {} dataset saved to {}\n'.format(m.capitalize(), dataset_path))


def csv_magic(data_location, dataset_dir=Path("dataset/part2bertcased")):
    try:
        cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Dataset directory({}) must contain config.yaml'.format(dataset_dir))
        exit(1)
    print('[-] Vocabs and datasets will be saved to {}\n'.format(dataset_dir))

    nlp = spacy.load('en')
    nlp.disable_pipes(*nlp.pipe_names)

    data_dir = Path(cfg.data_dir)
    data = {'test': load_data('test', Path(data_location), nlp)}
    # load word_vocab, char_vocab here
    #word_vocab, char_vocab = create_vocab(data, cfg.vocab, dataset_dir)
    word_vocab = pickle.load(open('dataset/part2bertcased/word.pkl','rb'))
    char_vocab = pickle.load(open('dataset/part2bertcased/char.pkl','rb'))
    
    # create_dataset here XD
    create_dataset(data, word_vocab, char_vocab, dataset_dir)
