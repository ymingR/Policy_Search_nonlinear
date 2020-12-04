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
    logging
)
# Import the open AI gym
import gym
import virl


class MyDeterministicPolicy():
    def __init__(self, action=0):
        self.model=None
        self.action = action
                
    def predict(self, s, a=None):
        q_s_a = np.zeros((4, 1))
        q_s_a[self.action] += 1 # defines the probablity for each action 
        return q_s_a
    
    def update(self, s, a, target):
        pass
def get_all_problems_fig():
    pic_dir = './results/Deterministic'
    for noisy in [True, False]:
        for act in range(4):
            filename = os.path.join(pic_dir, 'all_problems_noisy={} action={}.png'.format(noisy,act))
            if os.path.exists(filename):
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(10):
                env = virl.Epidemic(problem_id=i, noisy=noisy)
                states = []
                rewards = []
                done = False
                s = env.reset()
                states.append(s)
                while not done:
                    s, r, done, info = env.step(action=act) # deterministic agent
                    states.append(s)
                    rewards.append(r)
                ax.plot(np.array(states)[:,1], label=f'problem_id={i}')
            ax.set_xlabel('weeks since start of epidemic')
            ax.set_ylabel('Number of Infectious persons')
            ax.set_title('Simulation of problem_ids with action {}'.format(act))
            ax.legend()
            plt.savefig(dpi=300, fname=filename)

def get_eval(is_train=False,savefig=False):
    print('deterministic evaluating...')
    # mkdir
    # evaluate
    pic_dir = './results/Deterministic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
 
    log_file = os.path.join(pic_dir, 'Deterministic.log')
    logger = logging(log_file)
    results_file = os.path.join(pic_dir, 'Deterministic_results.csv')
    if os.path.exists(results_file) and not is_train and not savefig:
        results = pd.read_csv(results_file)
        results = results.sort_values(by=['noisy', 'problem_id'])
        return results
    else:
        if os.path.exists(results_file):
            os.remove(results_file)
        if os.path.exists(log_file):
            os.remove(log_file)
    results = pd.DataFrame([], columns=['problem_id', 'noisy', 'action', 'Total_rewards', 'avg_reward_per_action'])
    for problem_id, noisy, env in get_env():
        for act in range(4):
            func = MyDeterministicPolicy(act)
            states, rewards, actions = exec_policy(env, func, verbose=False)
            result = {'problem_id':problem_id, 'noisy':noisy, 'action':act,
                      'Total_rewards':sum(rewards),
                      'avg_reward_per_action':sum(rewards)/len(actions)}
            results = results.append(pd.DataFrame(result, index=[0]), ignore_index=0)
            logger(' '+str(result))
            logger(actions)
            if savefig:
                get_fig(states, rewards)
                pic_name = os.path.join(pic_dir, 'problem_id={} noisy={} action={}.jpg'.format(problem_id, noisy, str(act)))
                plt.savefig(dpi=300, fname=pic_name)
                plt.close()
            env.close()
        results = results.sort_values(by=['noisy', 'problem_id'])
        results.to_csv(results_file, index=0)
    return results

if __name__ == '__main__':
    get_all_problems_fig()
    is_train = False # change to True for training
    results = get_eval(is_train=is_train)