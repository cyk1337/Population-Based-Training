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

@file: Hyperparameter.py

@time: 09/08/2019 15:19 

@desc：       
               
'''
import random


class Hyperparameter:
    def __init__(self, worker_id, *hyperparams):
        # BATCH_SIZE, CRITIC_LEARNING_RATE, ACTOR_LEARNING_RATE, ACTION_NOISE_SCALE = hyperparams
        self.worker_id = worker_id
        self.BATCH_SIZE = pow(2, random.randint(5, 8))  # 32 ~ 256
        self.CRITIC_LEARNING_RATE = round(random.uniform(1e-4, 1e-2), 4)  # 0.0001 ~ 0.01
        self.ACTOR_LEARNING_RATE = round(random.uniform(1e-4, 1e-2), 4)  # 0.0001 ~ 0.01
        self.ACTION_NOISE_SCALE = round(random.uniform(0, 1), 1)  # 0~1, with interval 0.1

    def perturb_hyperparams(self):
        """
        explore hyperparameters via perturbation
        :return:
        """
        self.BATCH_SIZE = int(self.BATCH_SIZE * self._mutate_factor)
        self.CRITIC_LEARNING_RATE *= self._mutate_factor
        self.ACTOR_LEARNING_RATE *= self._mutate_factor
        self.ACTION_NOISE_SCALE *= self._mutate_factor

    @property
    def _mutate_factor(self):
        """ generate a constant factor for hyperparameter mutation"""
        return random.choice([.8, 1.2])

    def __repr__(self):
        return "Hyperparams of worker %s: BATCH_SIZE: %d, Critic_LR:%.2e, Actor_LR:%.2e, ACTION_NOISE_SCALE: %.1f" % \
               (self.worker_id, self.BATCH_SIZE, self.CRITIC_LEARNING_RATE, self.ACTOR_LEARNING_RATE,
                self.ACTION_NOISE_SCALE)
