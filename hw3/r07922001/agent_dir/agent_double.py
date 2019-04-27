import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import json

from agent_dir.agent import Agent
from environment import Environment

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        
        return torch.stack(obses_t).squeeze(), np.array(actions), np.array(rewards), torch.stack(obses_tp1).squeeze(), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class AgentDQN(Agent):
    def __init__(self, env, args):
        random.seed(87)
        torch.cuda.manual_seed_all(87)
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        self.name = 'DoubleDQN-no-detach'
        self.double = True
        # Initialize your replay buffer
        self.replay_buffer = ReplayBuffer(10000) # Size 10000 replay buffer

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load(self.name)
        
        # discounted reward
        self.GAMMA = 0.99
        
        # training hyperparameters
        self.train_freq = 8 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 1 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        # At first, you decide whether you want to explore the environemnt

        # if explore, you randomly samples one action
        # else, use your model to predict action
        
        threshold = 1 - min(self.steps,1000000) / 1000000 * 0.95
        # decide
        if test:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).cuda()
        
        if not test and random.random() <= threshold:
            action = random.randint(0,self.num_actions-1)
        else:
            action = self.online_net(state).argmax().item()
        
        return action

    def update(self):
        # To update model, we sample some stored experiences as training examples.
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        next_state = next_state.detach()
        action     = torch.LongTensor(action).cuda()
        reward     = torch.FloatTensor(reward).cuda()
        done       = torch.FloatTensor(np.float32(done)).cuda()
        
        
        with torch.no_grad():
            mask = 1 - done.byte()
        

        d_action   = self.online_net(next_state[mask]).max(1)[1]

        target_q_value = (self.GAMMA*self.target_net(next_state[mask]).gather(1, d_action.unsqueeze(1)).squeeze() + reward[mask]).detach()
        current_q_value = self.online_net(state[mask]).gather(1, action[mask].unsqueeze(1)).squeeze()
        
        loss = ((target_q_value - current_q_value)**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        record = []
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                #next_state, reward, done, _ = self.env.step(action[0, 0].data.item())
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state

                # TODO:
                # store the transition in memory
                self.replay_buffer.push(state, action, reward, next_state, done)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save(self.name)

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                record.append((self.steps, total_reward))
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save(self.name)

        with open("{}".format(self.name),"w") as fd:
            json.dump(record, fd)
