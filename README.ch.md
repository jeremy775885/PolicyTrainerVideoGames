
## 运行本框架所使用的包版本与所需库
	Python 3.11.8
	gymnasium               0.29.1
	random,tqdm,torch,matplotlib,collections,numpy


## 使用__base__. py 作为基本框架



由以下几个部分组成
1. 经验回放区 ReplayBuffer ：
	1. 调用时使用 __base__. ReplayBuffer (capacity) 以设置回放区大小；
	2. 向经验区中添加轨迹__base__. ReplayBuffer.add (state, action, reward, next_state, done)
	3. 使用 __base__. size 查询大小，当来到合适的大小区间时使用. sample 进行抽样，重新进行agent 的更新
2. 离线训练： train_on_policy_agent 与 train_on_policy_agent_replay
	1. 调用时使用 __base__. train_on_policy_agent (env, agent, num_episodes)
	2. 将以进度条形式返回训练结果，同时打印出期望
	3. train_on_policy_agent_replay (env, agent, num_episodes, buffer_size, batch_size, minimal_size) 将在训练的基础上使用经验回放机制
3. 画图 ：draw
	1. 调用时使用 __base__. draw (algo_name, env, return_list)
	# 2. return_list 为返回的期望队列
	3. 将画出该算法训练的期望图像

## 目前还未实现的功能/要求
	1.异策略训练
	2.tensor和numpy数组转换时的所需时间太长，需要优化
	3.更多算法的实现
	4.新游戏环境的搭建
	5.还未完成一键集成式测试训练


## 接下来需要主要完成的部分
将 base 文件进一步修改为由以下结构组成
1. 超参数部分。
2. 经验回放机制等组件。
3. 环境的搭建。完成 gym 库的一键导入，尝试引入 atari 等游戏环境
4. 更多的绘图。使用均值等方法绘制更多的 return-episodes 图像，减小误差，同时体现算法在各个环境中的不同优势


base 作为主文件，最终的流程应该为：输入超参数、选择算法、选择环境→训练→得到图像