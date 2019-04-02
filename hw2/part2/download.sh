#! /bin/sh
mkdir -p tmp

wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/tmp/7fb0534b83c42daee7d3ddb0ebaa81387925b71665d6ea195c5447f1077454cd.eea60d9ebb03c75bb36302aa9d241d3b7a04bba39c360cf035e8bf8140816233.json -O tmp/7fb0534b83c42daee7d3ddb0ebaa81387925b71665d6ea195c5447f1077454cd.eea60d9ebb03c75bb36302aa9d241d3b7a04bba39c360cf035e8bf8140816233.json

wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/tmp/7fb0534b83c42daee7d3ddb0ebaa81387925b71665d6ea195c5447f1077454cd.eea60d9ebb03c75bb36302aa9d241d3b7a04bba39c360cf035e8bf8140816233 -O tmp/7fb0534b83c42daee7d3ddb0ebaa81387925b71665d6ea195c5447f1077454cd.eea60d9ebb03c75bb36302aa9d241d3b7a04bba39c360cf035e8bf8140816233

wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/tmp/cee054f6aafe5e2cf816d2228704e326446785f940f5451a5b26033516a4ac3d.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1.json -O tmp/cee054f6aafe5e2cf816d2228704e326446785f940f5451a5b26033516a4ac3d.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1.json

wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/tmp/cee054f6aafe5e2cf816d2228704e326446785f940f5451a5b26033516a4ac3d.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1 -O tmp/cee054f6aafe5e2cf816d2228704e326446785f940f5451a5b26033516a4ac3d.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1


mkdir -p dataset/part2bertcased

wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/dataset/part2bertcased/config.yaml -O dataset/part2bertcased/config.yaml
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/dataset/part2bertcased/word.pkl -O dataset/part2bertcased/word.pkl
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/dataset/part2bertcased/char.pkl -O dataset/part2bertcased/char.pkl
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/dataset/part2bertcased/train.pkl -O dataset/part2bertcased/train.pkl
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw2/part2/dataset/part2bertcased/test.pkl -O dataset/part2bertcased/test.pkl


wget http://cl1.csie.org/~hortune/epoch-3.ckpt -O model/bert-large-cased/ckpts/epoch-3.ckpt
