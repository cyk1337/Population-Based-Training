#!/usr/bin/env python

# -*- encoding: utf-8

'''
        __   __   _                   ____ _   _    _    ___
        \ \ / /__| | ___   _ _ __    / ___| | | |  / \  |_ _|
         \ V / _ \ |/ / | | | '_ \  | |   | |_| | / _ \  | |
          | |  __/   <| |_| | | | | | |___|  _  |/ ___ \ | |
          |_|\___|_|\_\\__,_|_| |_|  \____|_| |_/_/   \_\___|

 ==========================================================================
@author: Yekun Chai

@license: School of Informatics, Edinburgh

@contact: chaiyekun@gmail.com

@file: ddpg_agent.py

@time: 05/08/2019 17:53 

@desc：       
               
'''
import numpy as np
import random
import copy
from collections import namedtuple, deque

from .model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e-6)  # replay buffer size
BATCH_SIZE = 128  # mini-batch size
GAMMA = .99  # discount factor
TAU = 1e-3  # for soft update of target parameters
lr_ACTOR = 1e-4  # learning rate of the actor
lr_CRITIC = 3e-4  # learning rate of the critic
WEIGHT_DECAY = 1e-4  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    Interacts with envs
    """

    def __init__(self, state_size, action_size, random_seed):
        """
        initialize
        :param state_size: state dim
        :param action_size: action dim
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_ACTOR)

        # Critic network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_CRITIC, weight_decay=WEIGHT_DECAY)

        # noise process
        self.noise = OUNoise(action_size, random_seed)

        # replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """
        save experience in replay memory, and randomly sample from buffer to learn
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        # save experience
        self.memory.add(state, action, reward, next_state, done)

        # learn from sample if the size &lt BATCH_SIZE
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            # learn
            self.learn(experiences, GAMMA)

    def learn(self, experience, gamma):
        """
        update policy and value parameters using given batch of experience tuples:
            Q_target = r + \gamma * critic_target(next_state, actor_target(next_state))
                where:
                    actor_target(next_state) ->  target action
                    critic_target(next_state, action) - > Q-value
        :param experience: (Tuple[torch.tensor]) -> (s,a,r,s', done)
        :param gamma: discount factor
        :return:
        """
        states, actions, rewards, next_states, dones = experience

        # ---------------- update critic  ---------------- #
        deterministic_next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, deterministic_next_actions)
        # compute Q-targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.actor_optimizer.step()

        # ---------------- update actor  ---------------- #
        # compute the actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------- update target networks  ---------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.critic_local)

    def soft_update(self, local_model, target_model, tau):
        """
        soft update model parameters:
            \theta_{target} =  \tau * \theta_{local} + (1 - \tau) * \theta_{target}
        :param local_model: torch model (weights copied from)
        :param target_model: torch model (weights copied to)
        :param tau: interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, state, add_noise=True):
        """
        returns actions for given states as per current policy
        :param state:
        :param add_noise:
        :return:
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


class OUNoise:
    """
    Ornstein-Unlenbeck process
    """

    def __init__(self, size, seed, mu=0, theta=.15, sigma=.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        reset the internal state (noise) to mean(mu)
        :return:
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        update the internal state and return it as a noise sample
        :return:
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    fixed size buffer to store experiences
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        initialize
        :param action_size: action dim
        :param buffer_size: buffer dim
        :param batch_size:  batch size
        :param seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_action", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        append a new experience to the memory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        randomly sample a batch of experience from the memory
        :return:
        """
        experience = random.sample(population=self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype('uint8')).float().to(
            device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        the length of internal memory
        :return:
        """
        return len(self.memory)
