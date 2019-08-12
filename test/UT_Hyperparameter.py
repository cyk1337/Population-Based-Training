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

@file: UT_Hyperparameter.py

@time: 09/08/2019 15:50 

@desc：       
               
'''
from ddpg.Hyperparameter import Hyperparameter
import copy

def UT1():
    population = []
    p2 = []
    for worker_id in range(30):
        h = Hyperparameter(worker_id)
        population.append(copy.deepcopy(h))
        # print(h)
        #     h.perturb_hyperparams()
        #     p2.append(h)
        #     # print(h)
        # for i, (h1, h2) in enumerate(zip(population, p2)):
        #     print(h1)
        #     print(h2)


if __name__ == '__main__':
    UT1()
