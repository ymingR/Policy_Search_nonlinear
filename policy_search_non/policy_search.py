# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:40:28 2020

@author: pc
"""


import matplotlib.pyplot as plt
import time
import itertools
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

import pandas as pd
import numpy as np

import sys
AIMA_TOOLBOX_ROOT="C:/Users/pc/OneDrive/AI/virl-master"
sys.path.append(AIMA_TOOLBOX_ROOT)
import virl
import os
import random
from collections import namedtuple
import collections
import copy

# Import the open AI gym
import gym
import keras
from keras.models import Sequential
#from keras.layers import Dense
from keras.optimizers import Adam
#from keras import backend as K

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import initializers

import tensorflow as tf

from gym.envs.toy_text import discrete


		
		


# define a couple of helper functions

EpisodeStats = namedtuple("Stats",["episode_num_sus", "episode_num_infec","episode_num_qua","episode_num_rec","episode_rewards","act_values"])

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
 

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)



    return fig2



###############################################
        
        
class NeuralNetworkPolicyEstimator():
    """ 
    A very basic MLP neural network approximator and estimator for poliy search    
    
    The only tricky thing is the traning/loss function and the specific neural network arch
    """
    
    def __init__(self, alpha, n_actions, d_states, nn_config, verbose=False):        
        self.alpha = alpha    
        self.nn_config = nn_config                   
        self.n_actions = n_actions        
        self.d_states = d_states
        self.verbose=verbose # Print debug information        
        self.n_layers = len(nn_config)  # number of hidden layers        
        self.model = []
        self.__build_network(d_states, n_actions)
        self.__build_train_fn()
             

    def __build_network(self, input_dim, output_dim):
        """Create a base network usig the Keras functional API"""
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        for h_dim in nn_config:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)
        net = layers.Dense(output_dim, kernel_initializer=initializers.Zeros())(net)
        net = layers.Activation("softmax")(net)
        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        """ Create a custom train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.        
        Called using self.train_fn([state, action_one_hot, target])`
        which would train the model. 
        
        Hint: you can think of K. as np.
        
        """
        # predefine a few variables
        action_onehot_placeholder   = K.placeholder(shape=(None, self.n_actions),name="action_onehot") # define a variable
        target                      = K.placeholder(shape=(None,), name="target") # define a variable       
        
        # this part defines the loss and is very important!
        action_prob        = self.model.output # the outlout of the neural network        
        action_selected_prob        = K.sum(action_prob * action_onehot_placeholder, axis=1) # probability of the selcted action        
        log_action_prob             = K.log(action_selected_prob) # take the log
        loss = -log_action_prob * target # the loss we are trying to minimise
        loss = K.mean(loss)
        
        # defining the speific optimizer to use
        adam = optimizers.Adam(lr=self.alpha)# clipnorm=1000.0) # let's use a kereas optimiser called Adam
        updates = adam.get_updates(params=self.model.trainable_weights,loss=loss) # what gradient updates to we parse to Adam
            
        # create a handle to the optimiser function    
        self.train_fn = K.function(inputs=[self.model.input,action_onehot_placeholder,target],
                                   outputs=[],
                                   updates=updates) # return a function which, when called, takes a gradient step
      
    
    def predict(self, s, a=None):              
        if a==None:            
            return self._predict_nn(s)
        else:                        
            return self._predict_nn(s)[a]
        
    def _predict_nn(self,state_hat):                          
        """
        Implements a basic MLP with tanh activations except for the final layer (linear)               
        """                
        x = self.model.predict(state_hat)                                                    
        return x
  
    def update(self, states, actions, target):  
        """
            states: a interger number repsenting the discrete state
            actions: a interger number repsenting the discrete action
            target: a real number representing the discount furture reward, reward to go
            
        """
        action_onehot = np_utils.to_categorical(actions, num_classes=self.n_actions) # encodes the state as one-hot
        self.train_fn([states, action_onehot, target]) # call the custom optimiser which takes a gradient step
        return 
        
    def new_episode(self):        
        self.t_episode  = 0.    
        


    
def reinforce(env, estimator_policy, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized         
        num_episodes: Number of episodes to run for
        discount_factor: reward discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    
    Adapted from: https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(
            episode_rewards=np.zeros(num_episodes),
            episode_num_sus=np.zeros(num_episodes),
            episode_num_infec=np.zeros(num_episodes),
            episode_num_qua=np.zeros(num_episodes),
            episode_num_rec=np.zeros(num_episodes),
            act_values=np.zeros(num_episodes))
   #          act_values=[])    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            state_oh = np.zeros((1,env.observation_space.shape[0]))
                                             
            action_probs = estimator_policy.predict(state_oh)
            action_probs = action_probs.squeeze()
            
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            ##
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_num_sus[i_episode] = next_state[:,0]
            stats.episode_num_infec[i_episode]=next_state[:,1]
            stats.episode_num_qua[i_episode]=next_state[:,2]
            stats.episode_num_rec[i_episode]=next_state[:,3]
           
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")            

            if done:
                break
                
            state = next_state
    
        # Go through the episode, step-by-step and make policy updates (note we sometime use j for the individual steps)
        estimator_policy.new_episode()
        new_theta=[]
        for t, transition in enumerate(episode):                 
            state_oh = np.zeros((1,env.env.observation_space.shape[0]))
          
            
            # The return, G_t, after this timestep; this is the target for the PolicyEstimator
            G_t = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
           
            # Update our policy estimator
            estimator_policy.update(state_oh, transition.action,np.array(G_t))            
         
    return stats




env = virl.Epidemic(stochastic=False, noisy=False)

# Instantiate a PolicyEstimator (i.e. the function-based approximation)
alpha      = 0.001  
n_states  = env.observation_space.shape[0]
n_actions  = env.action_space.n
nn_config  = [] # number of neurons in the hidden layers, should be [] as default

policy_estimator = NeuralNetworkPolicyEstimator(alpha, n_actions, n_states, nn_config)

stats = reinforce(env, policy_estimator, 1500, discount_factor=0.99)

plot_episode_stats(stats, smoothing_window=25)


