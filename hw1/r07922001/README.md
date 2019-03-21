# How to train my model
Download the corresponding packed data files into "data/".

[link](https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/data/)

After that, run the following command.

`python src/train.py ./models/{model dir}`

Then, everythng starts.

If you would want to do something with the dataset.
Just replace the make_dataset.py.
I have done some little hack to fulfill the requirement of rnn.sh and attention.sh.


# How to plot the figures
In general, just run the plot.py file.
If you want to use other data, then you add `embed` in `src/modules/cattn_q_c_q.py` to get the weights. I just saved it as tmp.npy when the IPython session pumps out. Then, change the utterance in plot.py. After that, you could get your graph. By the way, I use the package seaborn.


# Things

## RNN
- Bidi
- Attn
- 2, 3
- dropout
- Bilinear
    - co-use with attn
- RMSE
- Inner product
- Speaker Embedding
- Concat everything


## TODO
- [ ] Try speaker encoding
    - [x] Concat 
    - [ ] Speaker embedding?
- [ ] Try other attention
    - [ ] tanh
    - [ ] Cross entropy
- [ ] Attention syncing
- [ ] Clean data or linguistic check
    - [x] url
    - [ ] command
    - [ ] misspelling
    - [x] ing ed es -> original verb
        - https://stackoverflow.com/questions/3753021/using-nltk-and-wordnet-how-do-i-convert-simple-tense-verb-into-its-present-pas
        - http://www.zmonster.me/2016/01/21/lemmatization-survey.html
    - [x] <unk> -> something


## Experiment

- Model: Bidi RNN Clipped
|                    |  Origin  | unk -> sth | oov off | lemmatize |  url  | BOS EOS|
| :----------------: | :------: | :--------: | :-----: | :--------: | :----:| :---:|
| Validation (R@10) |  0.6678 (9.46) |   0.6696 (9.5066)  | 0.682 (9.566) | 0.652 | 0.672 | 0.6912|


| Speaker BOS EOS|       rm dup    |
| :-------------:| :-------------: |
| 0.6714         | 0.6948 (9.4866) |


- Model: Bidi Attn 

|                    |  Origin  | unk -> sth | oov off | lemmatize |
| :----------------: | :------: | :--------: | :-----: | :--------: |
| Validation (R@10)  |          |  0.658 (9.6)    |         |            |


- Attention syncing vs non sync

|              | Non sync | Sync |
|:------------:| :------: | :--: |
| Valid (R@10) |  0.662   |      |


Speaker Embedding



- Change turncate direction with "<begin>" and "<end>"
    - last : 0.6812 (9.52)


- "<unk>" compression


c q c 9.386
