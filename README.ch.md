
## 运行本框架所使用的包版本与所需库
	Python 3.11.8
	gymnasium               0.29.1
	random,tqdm,torch,matplotlib,collections,numpy


## 使用__base__. py 作为基本框架


由以下几个部分组成
	经验回放区 ReplayBuffer 
		调用时使用 __base__.ReplayBuffer(capacity)以设置回放区大小
			__base__.ReplayBuffer.add(state,action,reward,next_state,done)
			使用 __base__. size 查询大小，当来到合适的大小区间时使用. sample 进行抽样，重新进行 agent 的更新
	离线训练 train_on_policy_agent 与 train_on_policy_agent_replay
		调用时使用 __base__.train_on_policy_agent(env,agent,num_episodes)
			将以进度条形式返回训练结果，同时打印出期望
				train_on_policy_agent_replay (env,agent,num_episodes,buffer_size,batch_size,minimal_size)
				将在训练的基础上使用经验回放机制
	画图 draw
		调用时使用 __base__.draw(algo_name,env,return_list)
			return_list 为返回的期望队列
			将画出该算法训练的期望图像

## 目前还未实现的功能/要求
	1.异策略训练
	2.tensor和numpy数组转换时的所需时间太长，需要优化
	3.更多算法的实现
	4.新游戏环境的搭建
