rm -rf models
rm -rf data
mkdir data
mkdir models
mkdir models/cattn_q_c_q
mkdir models/brc_rm_dup

wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/data/config.json -O data/config.json
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/data/embedding_remove-duplicate.pkl -O data/embedding_remove-duplicate.pkl
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/data/test.json -O data/test.json
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/brc_rm_dup/config.json -O models/brc_rm_dup/config.json
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/brc_rm_dup/model.pkl.6 -O models/brc_rm_dup/model.pkl.6
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/cattn_q_c_q/config.json -O models/cattn_q_c_q/config.json
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/cattn_q_c_q/model.pkl.6 -O models/cattn_q_c_q/model.pkl.6
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/cattn_q_c_q/model.pkl.3 -O models/cattn_q_c_q/model.pkl.3
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/cattn_q_c_q/model.pkl.4 -O models/cattn_q_c_q/model.pkl.4
wget https://www.csie.ntu.edu.tw/~r07922001/adl_hw1/models/cattn_q_c_q/model.pkl.5 -O models/cattn_q_c_q/model.pkl.5
