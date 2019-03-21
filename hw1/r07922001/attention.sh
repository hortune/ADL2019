# preprocess
python3.7 ./src/make_dataset.py ./data --postfix remove-duplicate --input $1

# predict
python3.7 ./src/predict.py ./models/cattn_q_c_q/ --sepoch 3 --eepoch 5 --target $2
