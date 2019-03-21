echo $1 $2

# preprocess
python3.7 ./src/make_dataset.py ./data --postfix remove-duplicate --input $1

# predict
python3.7 ./src/predict.py ./models/brc_rm_dup/ --sepoch 6 --eepoch 7 --target $2
