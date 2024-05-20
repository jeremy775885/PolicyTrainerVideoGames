import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import tqdm
import rl_utils
import __base__
algo_name='Actor_Critic'
class Policy_net(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Policy_net,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)
    
class Value_net(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Value_net,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class Actor_Critic:
    def __init__(self,state_dim,hidden_dim,action_dim,ac_lr,cr_lr,gamma,device):
        self.actor=Policy_net(state_dim,hidden_dim,action_dim).to(device)
        self.critic=Value_net(state_dim,hidden_dim).to(device)
        self.optimizer_actor=torch.optim.Adam(self.actor.parameters(),lr=ac_lr)
        self.optimizer_critic=torch.optim.Adam(self.critic.parameters(),lr=cr_lr)
        self.gamma=gamma
        self.device=device
    def take_action(self,states):
        states = torch.tensor([states], dtype=torch.float).to(self.device)
        probs = self.actor(states)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
    
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)
        log_probs=torch.log(self.actor(states).gather(1,actions))
        actor_loss=torch.mean(-log_probs*td_delta.detach())
        critic_loss=torch.mean(F.mse_loss(self.critic(states),td_target.detach()))
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Actor_Critic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

return_list = __base__.train_on_policy_agent(env, agent, num_episodes)

__base__.draw(algo_name, env_name, return_list)
