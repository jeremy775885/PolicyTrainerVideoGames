import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import __base__

    
class Qnet(torch.nn.Module): 
    def __init__(self,state_dim,hidden_dim,action_dim): ##初始化网络，两个隐藏层
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    def forward(self,x):  ##使用relu作为激活函数 但其实也可以把前向传播函数拆开写 这里只是模仿动手强化学习的码风这么写
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device): ##初始化DQN需要的一系列参数
        self.action_dim=action_dim
        self.q_net=Qnet(state_dim,hidden_dim,action_dim).to(device)
        self.target_net=Qnet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_update=target_update
        self.count=0
        self.device=device

    def take_action(self,state): ##epsilon-greedy策略
        if np.random.random() < self.epsilon:  ##抽随机数 若随机数<epsilon 也就是(0,epsilon)  在所有可能的动作中抽取一个动作 寻找新策略
            action= np.random.choice(self.action_dim)
        else:
            state = torch.tensor(state,dtype=torch.float).to(self.device) ##(epsilon,1),选择现在状态下的最大预期回报动作 使用最优策略
            action = torch.argmax(self.q_net(state)).item()
        return action
        
    def update(self,transition_dict):  
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)   
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
         
        ##view(x,y)语法可以将tensor先拆为一维,再重组为x组y个元素的tensor

        q_values=self.q_net(states).gather(1,actions) ##gather(1,x) 是从x的索引中取出对应的值 1是维度 0是行 1是列
        max_q_values=self.target_net(next_states).max(1)[0].view(-1,1)
        q_targets=rewards+self.gamma*max_q_values*(1-dones) ##1-dones 为了避免终止状态下的下一个状态的预期回报 回合结束时，没有下一个状态
        dqn_loss=F.mse_loss(q_values,q_targets)##使用均方差作为后向传播的损失函数
        self.optimizer.zero_grad()##清零梯度 优化器自带默认梯度
        dqn_loss.backward()##后向传播
        self.optimizer.step()##优化器更新参数


        if self.count % self.target_update==0:##经历target_update次迭代后更新target_net DQN的特点之一就是定期更新目标网络而不是直接使用当前网络进行预测取值
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count+=1
##超参数 直接使用动手强化学习的
lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.8
epsilon=0.01
target_update=10
buffer_size=10000
minimal_size=500
batch_size=64
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##在新版的gymnasium 中 注意这里的seed 已经不能使用env.seed()了
env_name=('CartPole-v1')
env=gym.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)

return_list=[]
return_list=__base__.train_on_policy_agent_replay(env,agent,num_episodes,buffer_size,batch_size,minimal_size)
__base__.draw('DQN',env_name,return_list)