from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import namedtuple
import numpy as np
import time,sys,itertools
import pickle
import os
import sklearn.preprocessing
import virl
from utils import exec_policy, create_policy, logging, get_env, get_fig, plt
import pandas as pd
class NeuralNetwork(nn.Module):
    def __init__(self, env, n_states, n_actions, lr):        
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(n_states, 30)
        self.hidden.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, n_actions)
        self.out.weight.data.normal_(0, 0.1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler().fit(observation_examples)
    def forward(self, x):
        x = self.scaler.transform([x])
        x = Variable(torch.FloatTensor(x))
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return x
    def predict(self, state):
        return self.forward(state).detach().numpy()

    def update(self, s, a, target):
        if len(s) < 200:
            nums = len(s)
        else:
            nums = 200
        for i in range(nums):
            index = np.random.randint(0,len(s))
            state,action,q_next = s[index],a[index],target[index]
            loss = self.loss_func(self.forward(state), torch.FloatTensor(q_next))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def implement(env, func_approximator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    
    Args:
        env: OpenAI environment.
        func_approximator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Exploration strategy; chance the sample a random action. Float between 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    states, rewards, actions,q_values = [],[],[],[]
    for i_episode in range(num_episodes):
        
        # Create a handle to the policy we're following (including exploration)a
        policy = create_policy(
            func_approximator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
                 
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take                        
            action_probs, q_vals = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)
            actions.append(action)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = func_approximator.predict(next_state)
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            q_values_next[0][action] = td_target
            # Update the function approximator using our target
            q_values.append(q_values_next)
            if t % 10 == 0:
                func_approximator.update(states, actions, q_values)
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
                
            state = next_state
    return states, rewards, actions

def train(lr = 0.01, n_episodes=50):
    print('qlearning nn training...')
    env = virl.Epidemic()
    n_actions  = env.action_space.n
    n_states   = env.observation_space.shape[0]
    
    policy_estimator = NeuralNetwork(env, n_states, n_actions, lr=lr)
    stats = implement(env, policy_estimator, n_episodes, discount_factor=0.95)
    results_dir = './results/qlearning_nn'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    pkl_file = os.path.join(results_dir, 'qlearning_nn_lr={}_episodes={}.pkl'.format(lr, n_episodes))
    with open(pkl_file, 'wb')as f:
        pickle.dump(policy_estimator, f)
    return policy_estimator

def get_eval(lr = 0.01, n_episodes=50, is_train=False, savefig=False):
    # mkdir
    print('qlearning_nn evaluating...')
    base_dir = './results/qlearning_nn'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
 
    log_file = os.path.join(base_dir, 'qlearning_nn.log')
    logger = logging(log_file)
    results_file = os.path.join(base_dir, 'qlearning_nn.csv')
    if os.path.exists(results_file) and not is_train and not savefig:
        results = pd.read_csv(results_file)
        results = results.sort_values(by=['noisy', 'problem_id'])
        return results
    else:
        if os.path.exists(results_file):
            os.remove(results_file)
        if os.path.exists(log_file):
            os.remove(log_file)
        pkl_file = os.path.join(base_dir, 'qlearning_nn_lr={}_episodes={}.pkl'.format(lr, n_episodes))
        if os.path.exists(pkl_file):
            q_learning_nn = pickle.load(open(pkl_file,'rb'))
        else:
            q_learning_nn = train(lr=lr, n_episodes=n_episodes)
    # eval
    results = pd.DataFrame([], columns=['problem_id', 'noisy', 'action', 'Total_rewards', 'avg_reward_per_action'])
    for problem_id, noisy, env in get_env():
        states, rewards, actions = implement(env, q_learning_nn, 1, discount_factor=0.95)
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
    train(0.01,50)
    get_eval(0.01,50, is_train=True)