import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from utils import (
    get_env,
    get_fig,
    q_learning,
    exec_policy,
    get_fig,
    logging
)
# Import the open AI gym
import gym
import virl


class MyRandomFunctionApproximator():
    def __init__(self):
        self.model=None
                
    def predict(self, s, a=None):
        q_s_a = np.random.randn(4,1) # defines the probablity for each action 
        return q_s_a
    
    def update(self, s, a, target):
        pass

def train():
    # random agent don't need training.
    pass

def get_eval(savefig=False , is_train=False):
    print('random evaluating...')
    # mkdir
    base_dir = './results/Random/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    results_file = os.path.join(base_dir, 'random.csv')
    log_file = os.path.join(base_dir, 'random.log')
    logger = logging(log_file)
    # evaluate
    if os.path.exists(results_file) and not is_train and not savefig:
        results = pd.read_csv(results_file)
        results = results.sort_values(by=['noisy', 'problem_id'])
        return results
    else:
        if os.path.exists(results_file):
            os.remove(results_file)
        if os.path.exists(log_file):
            os.remove(log_file)
    results = pd.DataFrame([], columns=['problem_id', 'noisy', 'Total_rewards', 'avg_reward_per_action'])
    for problem_id, noisy, env in get_env():
        random_func = MyRandomFunctionApproximator()
        states, rewards, actions = exec_policy(env, random_func,)
        result = {'problem_id':problem_id, 'noisy':noisy, 
                  'Total_rewards':sum(rewards),
                  'avg_reward_per_action':sum(rewards)/len(actions)}
        results = results.append(pd.DataFrame(result, index=[0]), ignore_index=0)
        logger(' '+str(result))
        logger(actions)
        if savefig:
            get_fig(states, rewards)
            pic_name = os.path.join(base_dir, 'problem_id={} noisy={}.jpg'.format(problem_id, noisy))
            plt.savefig(dpi=300, fname=pic_name)
            plt.close()
        env.close()
    results = results.sort_values(by=['noisy', 'problem_id'])
    results.to_csv(results_file, index=0)
    return results

if __name__ == '__main__':
    print(get_eval(savefig=False))