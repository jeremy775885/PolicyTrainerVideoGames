import numpy as np
from tqdm import tqdm
import gymnasium as gym
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        transitions = random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done
    def size(self):
        return len(self.buffer)
    
def train_on_policy_agent(env,agent,num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' %i) as pbar:
            for i_episodes in range(int(num_episodes/10)) :
                episode_return = 0
                transition_dict = {'states':[],'actions':[],'next_states':[],'rewards':[],'dones':[]}
                state,info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,_=env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if(i_episodes+1)%10==0:
                    pbar.set_postfix({'episode':'%d'%(num_episodes/10*i+i_episodes+1),'return':'%.3f'%np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
def train_on_policy_agent_replay(env,agent,num_episodes,buffer_size,batch_size,minimal_size):
    Replay = ReplayBuffer(buffer_size)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' %i) as pbar:
            for i_episodes in range(int(num_episodes/10)) :
                episode_return = 0
                transition_dict = {'states':[],'actions':[],'next_states':[],'rewards':[],'dones':[]}
                state,info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,_=env.step(action)
                    Replay.add(state,action,reward,next_state,done)
                    state=next_state
                    episode_return += reward
                    if(Replay.size()>minimal_size):
                        b_s,b_a,b_r,b_ns,b_d=Replay.sample(batch_size)
                        transition_dict={'states':b_s,'actions':b_a,'rewards':b_r,'next_states':b_ns,'dones':b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if(i_episodes+1)%10==0:
                    pbar.set_postfix({'episode':'%d'%(num_episodes/10*i+i_episodes+1),'return':'%.3f'%np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
def draw(algo_name,env,return_list):
    episodes_list=list(range(len(return_list)))
    plt.plot(episodes_list,return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('{} on {}'.format(algo_name,env) )
    plt.show()