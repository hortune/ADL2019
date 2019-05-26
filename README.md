# ADL 2019
This the my homework implementation of Advanced Deep Learning 2019 [link](https://www.csie.ntu.edu.tw/~miulab/s107-adl/index.html).
All of the implementation used `pytorch`.

## HW1: Dialogue Modeling
In this homework, we are going to find the best reply among 100 candidates for the query. It's a task of the core of chatbot.


### Implementation
For the baseline, attentions are required. We should apply attention mechanism between each pair of query and response sentences. The reason for applying attention is quite naive, since normal RNN structure cannot remember the information long-time ago. Some people try to deal with this problem with extra memory such like `Memory Network` and `Neural Turing Machine`. However, the extra memory database may cause lots of GPU memory consumption. As a result, I choose to use attention to solve the problem.


## HW2: Contextual Embeddings
In this homework, we are going to train contextual embedding to improve the performance of previous model.

Contextual embeddings are the most important stuff in the studies of NLP in 2018. It could deal with the word with multiple meanings by adding the information of the previous words and latter ones.

### Impementation
As the TA's requirement, I implement the ELMo model and pass the simple baseline. ELMo uses the character embedding to embed words to word vectors. Then, the word vectors will be passed in a bidirectional attention encoder. The output should predict the next word or the previous one. After training, we would take the hidden state of bidirectional attention encoder as the contextual embedding. Then, it could easily pass the simple baseline.

For the strong baseline, I use BERT to train the contextual embedding. However, since BERT is really a huge model, I use the pretrained model on github and fine tune it to the task of HW. BERT embedding could easily pass the strong baseline.

## HW3: Deep Reinforcement Learning
In this homework, we are going to implement REINFORCE and DQN to solve some environemnts in openai gym.


### Implementation
For the environemnt LunarLander, which is a simple baseline in reinforcement learning, the simplest policy gradient method REINFORCE could solve it easily in 5 minutes.

For the environment Assault, normal DQN could solve it easily. By the way, I implement double DQN, which reduces the bias of Q value estimation, and NoisyDQN, which improves the exploration of the agent by applying state dependent noise. Both methods have improvement, however, NoisyDQN really outperforms double DQN and the naive DQN.


## HW4: Conditional GAN
In this homework, we are going to train a conditional GAN to generate images of the dataset `cartoonset`.


### Implementation
In the begining, I just implement DCGAN + ACGAN. The training process is quite eaily to tune, however, the result is not clear enough (FID score: 140). To improve the model, I implement WGAN-GP + ACGAN and leak the information of condition in the input of discriminator. The model works better and the FID score becomes 40.



