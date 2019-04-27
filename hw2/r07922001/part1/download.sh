#! /bin/sh
mkdir special_files
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part1/elmo_best_model.pkl -O special_files/elmo_best_model.pkl
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part1/elmo_best_vocab.pickle -O special_files/elmo_best_vocab.pickle
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part1/epoch-6.ckpt -O model/submission/ckpts/epoch-6.ckpt

