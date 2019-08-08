import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import warnings
import argparse

sns.set(style='darkgrid')
warnings.filterwarnings('ignore')
np.random.seed(0)


def get_info(filename):
    filename = filename.replace('.npy', '')  # remove .npy
    algo, env, seed = re.split('_', filename)
    seed = int(seed[5:])
    return algo, env, seed


def get_file_name(path='./'):
    file_names = []
    env_names = []
    algo_names = []
    for _, __, file_name in os.walk(path):
        file_names += file_name
    data_name = [f for f in file_names if '.npy' in f]
    for file in data_name:
        algo, env, _ = get_info(file)
        if env not in env_names:
            env_names.append(env)
        if algo not in algo_names:
            algo_names.append(algo)
    return data_name, env_names, algo_names


def exact_data(file_name, steps):
    '''
    exact data from single .npy file
    :param file_name:
    :return: a Dataframe include time, seed, algo_name, avg_reward
    '''
    avg_reward = np.load(file_name).reshape(-1, 1)
    algo, env_name, seed = get_info(file_name)
    df = pd.DataFrame(avg_reward[0:201])
    df.columns = ['Average Return']
    df['Time Steps (1e6)'] = steps
    df['Algorithm'] = algo
    df['env'] = env_name
    df['seed'] = seed
    return df


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--save_path', default='../figures', type=str)
    parse.add_argument('--ci', default=95,
                       help="""ci : int or “sd” or None, optional
                       Size of the confidence interval to draw when aggregating with an estimator. 
                       “sd” means to draw the standard deviation of the data. Setting to None will skip bootstrapping.
                       Click http://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot to seed more
                       """)
    args = parse.parse_args()

    print("=====================================")
    print("Starting Processing Data")
    print("=====================================")

    os.makedirs(args.save_path + '/', exist_ok=True)
    file_names, env_names, algo_names = get_file_name('./')
    df = pd.DataFrame([])
    steps = np.linspace(0, 1, 201)
    for file in file_names:
        data = exact_data(file, steps)
        df = pd.concat([df, data], axis=0)

    print("=====================================")
    print("Starting Visualizing Data")
    print("=====================================")

    for env in env_names:
        sns.lineplot(x='Time Steps (1e6)', y='Average Return', hue_order=algo_names, data=df[df['env'] == env],
                     hue='Algorithm', ci=args.ci)
        plt.title(env)
        plt.savefig(args.save_path + "/" + env + '.svg')
        plt.close()

    print("=====================================")
    print("End, Please check data in " + args.save_path)
    print("=====================================")
