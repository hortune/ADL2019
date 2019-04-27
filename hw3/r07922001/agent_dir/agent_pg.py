import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from agent_dir.agent import Agent
from environment import Environment
import json

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        random.seed(87)
        torch.manual_seed(87)
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        state = torch.FloatTensor(state)
        probs = self.model(state.unsqueeze(0))
        if test:
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
        else:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            self.saved_actions.append(-m.log_prob(action))
            action = action.item()
        return action

    def update(self):
        discounted_reward = []
        
        cum = 0
        for ele in reversed(self.rewards):
            cum *= self.gamma
            cum += ele
            discounted_reward.append(cum)
        
        loss = 0

        for reward, prob in zip(reversed(discounted_reward), self.saved_actions):
            loss += prob*reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        step = 0
        record = []
        avg_reward = None # moving average of reward
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                step += 1
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                
                #self.saved_actions.append(action)
                self.rewards.append(reward)

            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            record.append((step, last_reward))
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            
            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
        with open("pg_history.txt","w") as fd:
            json.dump(record, fd)
