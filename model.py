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

@file: model.py

@time: 05/08/2019 17:22 

@desc：       
               
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    Actor (policy) network
    """

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """
        Initialize
        :param state_size: int, state dim
        :param action_size: int, action dim
        :param seed: int, random seed
        :param fc_units: int, # of hidden layer units
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_params()

    def reset_params(self):
        """
        reset parameters
        :return:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        create the actor network, mapping states to actions
        :param state:
        :return:
        """
        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))


class Critic(nn.Module):
    """
    Critic (value) network
    """

    def __init__(self, state_size, action_size, seed, fc1_units=356, fc2_units=256, fc3_units=128):
        """
        initialize
        :param state_size:
        :param action_size:
        :param seed:
        :param fc1_units:
        :param fc2_units:
        :param fc3_units:
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_params()

    def reset_params(self):
        """
        reset
        :return:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Create the critic (value) network that maps (state, action) pair -> scalar Q value
        :param state:
        :param action:
        :return:
        """
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)