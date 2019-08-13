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

@license: CASIA

@contact: chaiyekun@gmail.com

@file: PBT.py

@time: 08/08/2019 19:06 

@desc：       
               
'''
import gym, torch
import random, heapq, os, argparse, logging, copy
from tqdm import tqdm
import numpy as np
import TD3, DDPG, OurDDPG
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--population_size", default=2, type=int)  # population capacity
parser.add_argument("--policy_name", default="OurDDPG")  # policy name
parser.add_argument("--env_name", default="BipedalWalker-v2")  # OpenAI gym environment name Pendulum-v0
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=1e4,
                    type=int)  # How many time steps purely random policy is run for
parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e5, type=float)  # Max time steps to run environment for
parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
args = parser.parse_args()

if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - "%(name)s" - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# env
env = gym.make(args.env_name)

# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


class Hyperparameter:
    def __init__(self, worker_id, *hyperparams):
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
        return random.choice([.8, 1.2])  # for continuous range of hyperparameters

    def __repr__(self):
        return "Hyperparams of worker %s: BATCH_SIZE: %d, Critic_LR:%.2e, Actor_LR:%.2e, ACTION_NOISE_SCALE: %.1f" % \
               (self.worker_id, self.BATCH_SIZE, self.CRITIC_LEARNING_RATE, self.ACTOR_LEARNING_RATE,
                self.ACTION_NOISE_SCALE)


class Worker:
    def __init__(self, worker_id, h, agent, max_t=args.max_timesteps):
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

        self.optim = None
        self.max_t = max_t

        self.generation = None

        self.reset()

    def reset(self):
        self.t = 0
        self.evaluations = None
        self.ready_to_mutate = False  # flag to mutate
        self.performance = None

    def step(self, generation):
        """
        update the weight of the agent with gradient descent
        :return:
        """
        self.generation = generation
        logger.info("Starting to train worker-%d" % self.worker_id)
        logger.debug(self.h)
        replay_buffer = utils.ReplayBuffer()

        # Evaluate untrained policy
        self.evaluations = [self._evaluate_policy()]

        # new add, python2 style -> python3 style
        # ---------------------------------------
        episode_reward = 0
        episode_timesteps = 0
        # ---------------------------------------

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        for total_timesteps in tqdm(range(int(args.max_timesteps))):
            if done:

                if total_timesteps != 0:
                    print(" Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
                        total_timesteps, episode_num, episode_timesteps, episode_reward))
                    if args.policy_name == "TD3":
                        self.agent.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                         args.policy_noise, args.noise_clip, args.policy_freq)
                    else:
                        self.agent.train(replay_buffer, episode_timesteps, self.h.BATCH_SIZE)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    self.evaluations.append(self._evaluate_policy())

                    if args.save_models: self.agent.save(self.__repr__(), directory="./pytorch_models")
                    np.save("./results/%s" % (self.__repr__()), self.evaluations)

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Select action randomly or according to policy
            if total_timesteps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = self.agent.select_action(np.array(obs))
                # if args.expl_noise != 0:
                #     action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                #         env.action_space.low, env.action_space.high)
                action = (action + np.random.normal(0, self.h.ACTION_NOISE_SCALE, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_reward += reward

            # Store data in replay buffer
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            timesteps_since_eval += 1
        self.t += total_timesteps
        self.ready_to_mutate = True
        self.eval()
        np.save("./results/%s" % (self.__repr__()), self.evaluations)

    def eval(self, save_models=True):
        """
        evaluate the performance of the agent
        :return:
        """
        self.performance = self._evaluate_policy()
        self.evaluations.append(self.performance)

        if save_models:
            self.agent.save("%s" % (self.__repr__()), directory="./pytorch_models")

    # Runs policy for X episodes and returns average reward
    def _evaluate_policy(self, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self.agent.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        # print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        # print("---------------------------------------")
        return avg_reward

    def __repr__(self):
        return "G%d-worker-%s-%s_%s_seed=%s" % (
            self.generation, self.worker_id, args.policy_name, args.env_name, str(args.seed))


class PBT:
    def __init__(self, population_size):
        self.size = population_size
        self.population = []
        self.worker_queue = []

        self.P = set()

    def init_population(self):
        if not self.population:
            for worker_id in range(args.population_size):
                h = Hyperparameter(worker_id)
                policy = OurDDPG.DDPG(state_dim, action_dim, max_action, ACTOR_LR=h.ACTOR_LEARNING_RATE,
                                      CRITIC_LR=h.CRITIC_LEARNING_RATE)
                worker = Worker(worker_id=worker_id, h=h, agent=policy)
                self.population.append(worker)
        self.worker_queue = [worker.worker_id for worker in self.population]

    def ready_to_mutate(self):
        return not self.worker_queue

    def expoit_n_explore(self, worker):
        """
        exploit by truncation
            1. select another member of the population with better performance after ranking
            2. If current agents ranks within the bottom 20%,
            copy the weights and hyperparameters from an agent uniformly sampled from the top 20% in the population.
        explore
            1. perturbation
        :return:
        """
        lower_quantiles, upper_quantiles = self._quantiles(quantile_frac=.2)
        if worker.worker_id in lower_quantiles:
            mutate_id = random.choice(upper_quantiles)
            mutate_worker = [worker for worker in self.population if worker.worker_id == mutate_id][0]
            worker.agent.copy_weights(mutate_worker)  # copy weights
            worker.agent.copy_hyperparams(mutate_worker.h)  # copy hyperparams
            worker.h = copy.deepcopy(mutate_worker.h)  # assign attributes worker.h
            # perturb h
            self._explore(worker)

    def _explore(self, worker):
        """
        do exploration
            1. perturb
        :return:
        """
        worker.h.perturb_hyperparams()
        worker.agent.copy_hyperparams(worker.h)

    def _quantiles(self, quantile_frac=.2):
        """
        rank all agents in the population by episodic reward, and return lower and upper quantile workers
        :param quantile_frac:
        :return:
        """
        n = np.ceil(self.size * quantile_frac)
        lower_quantile = heapq.nsmallest(n, self.population, key=lambda x: x.performance)
        upper_quantile = heapq.nlargest(n, self.population, key=lambda x: x.performance)
        return [worker.worker_id for worker in lower_quantile], [worker.worker_id for worker in upper_quantile]

    def add(self, performance, h, generation):
        self.P.add((performance, h, generation))

    def select_optimal(self):
        return sorted(self.P, key=lambda x: x[0])[-1]


def run(n_generation=3):
    """
    entry of PBT
    :return:
    """
    pbt = PBT(population_size=3)
    for g in range(1, n_generation + 1):  # stop criterion
        logger.info("Generation %d ..." % g)
        pbt.init_population()
        while pbt.worker_queue:
            worker_id = pbt.worker_queue.pop(0)
            worker = pbt.population[worker_id]
            worker.step(g)  # train worker individually
            # update Population
            pbt.add(worker.performance, worker.h, g)
        if pbt.ready_to_mutate():
            for worker in pbt.population:
                pbt.expoit_n_explore(worker)
    print(pbt.select_optimal())


if __name__ == '__main__':
    run()
