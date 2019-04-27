#! /bin/sh
PYTORCH_PRETRAINED_BERT_CACHE=tmp python -m BERT_large_cased.predict model/bert-large-cased 3 $1 --batch_size 4
cp model/bert-large-cased/predictions/epoch-3.csv $2
