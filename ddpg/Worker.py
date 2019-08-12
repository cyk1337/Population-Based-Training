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

@file: Worker.py

@time: 08/08/2019 19:04 

@desc：       
               
'''


class Worker:
    def __init__(self, worker_id, h, agent, optim, max_t, replay_buffer):
        """
            the entry to PBT
        # ======================== #
        :param h: (namedtuple) hyperparameters
        :param agent: agent in the population
        :param t: (int) maximum time steps
        """
        self.worker_id = worker_id
        self.h = h
        self.agent = agent
        self.optim = optim
        self.max_t = max_t
        self.replay_buffer = replay_buffer

        self.t = 0
        self.performance = None

    def step(self):
        """
        update the weight of the agent with gradient descent
        :return:
        """
        pass

    def eval(self):
        """
        evaluate the performance of the agent
        :return:
        """
        pass

    def ready(self):
        """
        check if the agent is ready for exploitation-and-exploration
        :return:
        """
        pass

    def expoit(self):
        """
        do exploitation
            1. select another member of the population with better performance after ranking
            2. copy the weights and hyperparameters
        :return:
        """
        pass

    def _exploit_by_truncation(self):
        """
        rank all agents in the population by episodic reward. If current agents ranks within the bottom 20%,
        copy the weights from an agent uniformly sampled from the top 20% in the population.
        :return:
        """
        raise NotImplementedError

    def explore(self):
        """
        do exploration
            1. perturb
        :return:
        """
        pass

    def _explore_by_perturbation(self):
        """
        randomly perturb each hyperparameter by \pm 20%
        :return:
        """
        raise NotImplementedError
