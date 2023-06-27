# Report

注：由于文中结果显示用到gif，为了显示效果，采取html形势的报告，若您以原文件名下载解压（../SCMS637004_RL）则不会有问题，若有改动则可能出现图片无法显示的状况，烦请修改一下文件夹名。本文用到所有图片与gif都在自文件夹/Image/中，同时也可以通过运行py文件得到。

[TOC]



## 问题复述

选择：Q3

描述：一个标准的网格世界，有开始 和目标状态，在网格中间有一个向上运行的侧风。动作是标 准的四个── 上，下，右 和 左，但在中间区域，结果的下一个状态向上移 动一个“风”，其强度因列而异。这是一个没有折扣的回合任务，在达到目标状态之前回报 恒定为 −1。

**问题**:自行增加网格规模(如 20*20)，自行增加障碍(必须)，重新设定风向 和强度(可增加随机性)，并设计两种强化学习算法(Q 学习，n 步自举， SARSA(λ)，策略梯度法等)求解该问题，比较不同算法，提供源代码。

### 运动机制

声明本试验中的动作，在遇到边缘或障碍时，动作可以照常采取并一样在达到目标前每次获得-1的回报，但是状态保持原地不变。若在有风区域，则认为风将物体按风向吹到不能运动为止。同时，为了方便及整体运动的一致性，设定风的作用以动作开始时状态所在列的风强度为准。代码如下：

```python
class WindGrid:
	...
	def step(self, state, action):
          i, j = state
          di, dj = self.ACTIONS[action]
          # 计算下一个位置
          next_i = i + di
          next_j = j + dj
          # 如果下一个位置超出了世界边界或是障碍物，则保持在原地
          if next_i < 0 or next_i >= self.world_height or next_j < 0 or next_j >= self.world_width or [next_i, next_j] in self.obstacles:
              next_i = i
              next_j = j
          # 计算风的影响
          wind_strength = self.WIND[j]
          next_i -= wind_strength
          # 如果风吹后的位置是障碍物，则停止移动
          wind_back = 0
          while ([next_i, next_j] in self.obstacles or next_i < 0 or next_i >= self.world_height) and wind_back < wind_strength:
              next_i += 1
              wind_back += 1
          # 计算奖励
          reward = self.REWARD
          if (next_i, next_j) == self.goal:
              reward = 0
          return (next_i, next_j), reward
```



## 算法概述

本试验使用 Q learning 和 SARSA($\lambda$) 两种算法。两种算法的概述：

### Q learning

![q_learning](../SCMS637004_RL/Image/q_learning.png)

### SARSA($\lambda$) 

- SARSA($\lambda$) 算法在SARSA算法的基础上引入了**资格迹（eligibility trace）**，对于经历过的状态有一定的记忆性。
- $\lambda$ 是取值范围为0到1之间的参数，控制权重更新，$\lambda$ 越接近0则算法越考虑即时奖励，越接近1则越关注长期奖励。

<img src="../SCMS637004_RL/Image/sarsa_lambda.png" alt="sarsa_lambda" style="zoom: 67%;" />

- 本试验中，令 $\epsilon$-greedy 方法中的 $\epsilon$ 每次迭代衰减到上一次的 95%（即逐步减少随机探索，这是为了最终收敛到最优解）。

## 试验结果

此次三个试验中，风均为 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] 扩张网格部分没有加入新的风。

### $ 7 \times 10 $ 网格 5 障碍

![Optimal path 7x10 with 5obstacles lambda=0.05 Q_learning](../SCMS637004_RL/Image/Optimal path 7x10 with 5obstacles lambda=0.05 Q_learning.gif)

```
Q_learning
Optimal path:
 [(3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (4, 8), (3, 7)]
Max Reward: -17
```

![Optimal path 7x10 with 5obstacles lambda=0.05 SARSA_Lambda](../SCMS637004_RL/Image/Optimal path 7x10 with 5obstacles lambda=0.05 SARSA_Lambda.gif)

```
SARSA_Lambda
Optimal path:
 [(3, 0), (3, 1), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (4, 8), (3, 7)]
Max Reward: -17
```

同样的步数/回报，两种方法可能产生不同的最佳路径。改变风为WIND = [0, 3, 3, 3, 3, 1, 2, 2, 1, 0]。则最优路径只能贴上墙走。



### $20 \times 20$ 网格 5 障碍

![Optimal path 20x20 with 5obstacles lambda=0.05 Q_learning](../SCMS637004_RL/Image/Optimal path 20x20 with 5obstacles lambda=0.05 Q_learning.gif)

```
Q_learning
Optimal path:
 [(3, 0), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (7, 2), (8, 2), (8, 3), (7, 4), (6, 5), (5, 6), (3, 7)]
Max Reward: -11
```



![Optimal path 20x20 with 5obstacles lambda=0.05 SARSA_Lambda](../SCMS637004_RL/Image/Optimal path 20x20 with 5obstacles lambda=0.05 SARSA_Lambda.gif)

```
SARSA_Lambda
Optimal path:
 [(3, 0), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (7, 2), (8, 2), (8, 3), (7, 4), (6, 5), (5, 6), (3, 7)]
Max Reward: -11
```



### $20 \times 20$ 网格 10 障碍

![Optimal path 20x20 with 10obstacles lambda=0.05 Q_learning](../SCMS637004_RL/Image/Optimal path 20x20 with 10obstacles lambda=0.05 Q_learning.gif)

```
Q_learning
Optimal path:
 [(3, 0), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (9, 2), (9, 3), (9, 4), (8, 5), (6, 5), (5, 6), (3, 7)]
Max Reward: -13
```

![Optimal path 20x20 with 10obstacles lambda=0.05 SARSA_Lambda](../SCMS637004_RL/Image/Optimal path 20x20 with 10obstacles lambda=0.05 SARSA_Lambda.gif)

```
SARSA_Lambda
Optimal path:
 [(3, 0), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (9, 2), (9, 3), (9, 4), (8, 5), (6, 5), (5, 6), (3, 7)]
Max Reward: -13
```

更多的尝试中，更复杂（风、网格规模、障碍数目）的情形中，需要更多迭代收敛，需要注意本身问题的有解与否，比如在风强过大、高度不够的情况下，实际上问题可能不存在解，本文中可以通过设置一次试验的最大步数 来退出运行（参数max_steps，默认np.inf，它也控制了generate_path中的截断，设置其为有限值可以防止你的电脑进入循环手动推出）。

#### 更多障碍

以上任务最终的路径都较为简单（没有回头），为了测试算法的能力，我们首先在无风状况下给出更复杂的障碍：

![Optimal path 20x20 with 44obstacles lambda=0.05 Q_learning](../SCMS637004_RL/Image/Optimal path 20x20 with 44obstacles lambda=0.05 Q_learning.gif)

<center>Q learning</center> 

![Optimal path 20x20 with 44obstacles lambda=0.05 SARSA_Lambda](../SCMS637004_RL/Image/Optimal path 20x20 with 44obstacles lambda=0.05 SARSA_Lambda.gif)

<center>Sarsa lambda</center> 

可以看到两个算法都顺利找到路径。

#### 有风的多障碍

人为加上一个会改变最优策略的风WIND = [0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]。

![Optimal path 20x20 with 44obstacles lambda=0.05 Q_learning](../SCMS637004_RL/Image/Optimal path 20x20 with 44obstacles lambda=0.05 SARSA_Lambda.gif)

<center>Q learning</center> 

![Optimal path 20x20 with 44obstacles lambda=0.05 SARSA_Lambda](../SCMS637004_RL/Image/Optimal path 20x20 with 44obstacles lambda=0.05 SARSA_Lambda.gif)

<center>Sarsa lambda</center> 



## 结果分析

### 步数变化

<img src="../SCMS637004_RL/Image/Steps by Episode 7x10 with 5obstacles lambda=0.05.png" alt="Steps by Episode 7x10 with 5obstacles lambda=0.05" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Steps by Episode 20x20 with 5obstacles lambda=0.05.png" alt="Steps by Episode 20x20 with 5obstacles" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Steps by Episode 20x20 with 10obstacles lambda=0.05.png" alt="Steps by Episode 20x20 with 10obstacles lambda=0.05" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Steps by Episode 20x20 with 44obstacles lambda=0.05.png" alt="Steps by Episode 20x20 with 44obstacles lambda=0.05" style="zoom:25%;" />

上面三幅图分别为在取 $\lambda = 0.05$ 时，两个算法在上一节试验结果中展示的四个情形（$7 \times 10 $ 网格 5 障碍、$20 \times 20$ 网格 5 障碍 $20 \times 20$ 网格 10 障碍和更多障碍带风）中试验过程中步数的变化。可以看出大致上，Q-learning 在早期比 SARSA($\lambda$) 收敛更快，但随着迭代的进行，靠近收敛到最优路线部分时， SARSA($\lambda$) 的稳定性则略优于 Q-learning。为了更好的看到最终的收敛情况，下面给出累计步数的曲线。

### 累计步数变化

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 7x10 with 5obstacles lambda=0.05.png" alt="Cumulative Steps by Episode 7x10 with 5obstacles lambda=0.05" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 20x20 with 5obstacles lambda=0.05.png" alt="Cumulative Steps by Episode 20x20 with 5obstacles lambda=0.05" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.05.png" alt="Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.05" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 20x20 with 44obstacles lambda=0.05.png" alt="Cumulative Steps by Episode 20x20 with 44obstacles lambda=0.05" style="zoom:25%;" />

上面三幅图分别为在取 $\lambda = 0.05$ 时，两个算法在上一节试验结果中展示的三个情形（$7 \times 10 $ 网格 5 障碍、$20 \times 20$ 网格 5 障碍 $20 \times 20$ 网格 10 障碍和更多障碍带风）中试验过程中累计步数的变化，曲线达到直线表示算法收敛。可以看到大部分情况下 SARSA($\lambda$) 都比 Q-learning 更早或以更少的总步数收敛到最优路径。但由于算法中存在随机性，这一现象并不一定存在在每次试验中。

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.1.png" alt="Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.1" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.2.png" alt="Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.2" style="zoom:25%;" />

<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.7000000000000001.png" alt="Cumulative Steps by Episode 20x20 with 10obstacles lambda=0.7000000000000001" style="zoom:25%;" />

上面两张图分别是 $\lambda = 0.1$ 和 $\lambda = 0.2$ 和  $\lambda = 0.7$  时 $20 \times 20$ 网格 10 障碍情形的累积步数曲线。更大的 $\lambda$ 的算法倾向于需要多的迭代达到收敛，也展现出更好的稳定性。但是SARSA $\lambda$ 会有收敛到局部最优的风险。如果lambda值设置得过大，算法可能过于关注长期收益，而忽略了当前的即时奖励，从而陷入局部最优解。（将 $\epsilon$ 的衰减延后，给予初期更大的探索性也有一样的问题。）比较  $\lambda = 0.1, 0.2, \dots, 0.9$，综合收敛速度、稳定性和规避局部最优情形，0.2 是较优的取值。（在这一前提下，学习率 $\alpha$ 取 0.5 是较好的选择）(这里没有进行网格搜索或调包调参，考虑到情形不同最优参数不同，此处只是做了粗略比较)。

### 随机风向

尝试将风强度从给定向量换成随机风（random_wind = True），这里为了算法能出一个结果，是用了比较简单的单向0-1随机风，通过 WindGrid.step() 中的：

```python
if self.random_wind:
            wind_strength = np.random.choice([0, 1])
        else:
            wind_strength = self.WIND[j]
```

实现。在没有障碍物的情形下，得到：

<img src="../SCMS637004_RL/Image/Steps by Episode 7x10 with 0obstacles lambda=0.05 random wind.png" alt="Steps by Episode 7x10 with 0obstacles lambda=0.05 random wind" style="zoom:25%;" />



<img src="../SCMS637004_RL/Image/Cumulative Steps by Episode 7x10 with 0obstacles lambda=0.05 random wind.png" alt="Cumulative Steps by Episode 7x10 with 0obstacles lambda=0.05 random wind" style="zoom:25%;" />

 

理论上随机风在一般情形下应该是不熟练的，在这一试验下也的确如此。

### 小结

 SARSA($\lambda$) 算法是一种在线学习算法，它在每个时间步都进行更新，同时考虑当前状态和动作的奖励，并且还会考虑之前的奖励和动作。在开始阶段， SARSA($\lambda$) 算法可能会探索不同的动作并尝试不同的策略，导致reward较低。

Q-learning算法是一种离线学习算法，在每个时间步只考虑当前状态和动作的奖励，并选择具有最高Q值的动作。因此，Q-learning算法可能会更快地找到一条较优的策略，并在开始阶段获得较高的reward。

然而， SARSA($\lambda$) 算法通过考虑之前的奖励和动作，能够更好地处理动作序列，并且可以在后期的训练中稳定地收敛到较优的策略。这是因为 SARSA($\lambda$) 算法的更新方式允许它通过调整之前的动作选择来纠正之前的错误，并逐渐优化策略。稳定性可能是由于SARSA Lambda算法能够更好地平衡探索和利用之间的权衡，以及对之前动作选择的纠正能力所致。

## 参考资料

- https://zcczhang.github.io/rl/
- Reinforcement Learning: An Introduction, Second edition, in progress, Richard S. Sutton and Andrew G. Barto.
- https://rl.qiwihui.com/zh_CN/latest/



## 主要代码

### WindGrid类

```python
class WindGrid:
    def __init__(self, WORLD_HEIGHT = 7, WORLD_WIDTH = 10, 
                 START = (3, 0), 
                 obstacles = [],
                 GOAL = (3, 7), 
                 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], 
                 EPSILON = 0.1, 
                 ALPHA = 0.1,
                 num_episodes=500,
                 LAMBDA = 0.05,
                 note = "",
                 method="Q_learning",
                 max_steps = np.inf,
                 random_wind = False):
        self.world_height = WORLD_HEIGHT
        self.world_width = WORLD_WIDTH
        self.ACTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        self.ACTION_LABELS = ['R', 'D', 'L', 'U']
        self.start = START 
        self.goal = GOAL
        self.WIND = WIND 
        self.epsilon = EPSILON 
        self.alpha = ALPHA
        self.REWARD = -1
        self.obstacles = obstacles
        self.num_episodes = num_episodes
        self.best_policy = {}
        self.note = note
        self.LAMBDA = LAMBDA
        self.method = method
        self.max_steps = max_steps
        self.random_wind = random_wind
    
    def step(self, state, action):
        """
        由状态和动作产生下一个状态。
        在遇到边缘或障碍时，动作可以照常采取并一样在达到目标前每次获得-1的回报，但是状态保持原地不变。
        若在有风区域，则认为风将物体按风向吹到不能运动为止。

        """
        i, j = state
        di, dj = self.ACTIONS[action]
        # 计算下一个位置
        next_i = i + di
        next_j = j + dj
        # 如果下一个位置超出了世界边界或是障碍物，则保持在原地
        if next_i < 0 or next_i >= self.world_height or next_j < 0 or next_j >= self.world_width or [next_i, next_j] in self.obstacles:
            next_i = i
            next_j = j
        # 计算风的影响
        if self.random_wind:
            wind_strength = np.random.choice([0, 1])
        else:
            wind_strength = self.WIND[j]
        # 如果风吹后的位置是障碍物，则停止移动
        for i in range(wind_strength):
            next_i -= 1
            if (([next_i, next_j] in self.obstacles) or (next_i < 0) or (next_i >= self.world_height)):
                next_i +=1
                break
        
        # 计算奖励
        reward = self.REWARD
        if (next_i, next_j) == self.goal:
            reward = 0
        return (next_i, next_j), reward
    
    def epsilon_greedy(self, Q, state, epsilon):
        """
        epsilon greedy策略选取当前动作

        """
        if np.random.binomial(1, epsilon) == 1:
            return random.randint(0, len(self.ACTIONS) - 1)
        else:
            values = Q[state[0], state[1], :]
            return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
    
    def optimize(self,method="Q_learning"):
        """
        
        强化学习主体，包含 Q learning 和 SARSA Lambda 两种算法。
        迭代更新 Q 函数。

        """
        self.method = method
        # 初始化 Q 表，所有值初始化为 0
        Q = np.zeros((self.world_height, self.world_width, len(self.ACTIONS)))
        
        # # 将障碍物的值设置为负无穷大，以便在选择动作时将其排除
        # for obstacle in self.obstacles:
        #     Q[obstacle[0], obstacle[1], :] = -1e12
        
        # 迭代学习Q函数值
        steps_change = []
        epsilon_episode = self.epsilon
        for episode in range(self.num_episodes):
            for obstacle in self.obstacles:
                Q[obstacle[0], obstacle[1], :] = -1e12
            
            epsilon_episode = epsilon_episode*0.95
            # 使epsilon greedy逐渐衰减随机性
            # 如果探索率epsilon设置过高，智能体可能会过于频繁地选择随机动作，而忽略了已经学到的最佳动作。
            
            if method == "SARSA_Lambda":
                E = np.zeros((self.world_height, self.world_width, len(self.ACTIONS)))
            
            state = self.start
            action = self.epsilon_greedy(Q, state, epsilon_episode)
            steps = 0
            while state != self.goal and steps < self.max_steps:
                next_state, reward = self.step(state, action)
                next_action = self.epsilon_greedy(Q, next_state, epsilon_episode)
                
                # 不同的Q函数迭代方法：Q learning和 SARSA(lambda)
                if method == "Q_learning":
                    Q[state[0], state[1], action] += self.alpha * (reward + np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])
                
                elif method == "SARSA_Lambda":
                    delta = reward + Q[next_state[0],next_state[1], next_action] - Q[state[0],state[1], action]
                    # 更新eligibility traces
                    E[state[0],state[1], action] += 1
                    # 更新Q函数
                    Q += self.alpha * delta * E
                    # 更新eligibility traces
                    E *=  self.LAMBDA
                    
                state = next_state
                # print(state)
                action = next_action
                steps += 1
            if steps == self.max_steps:
                print("检查问题是否有解！")
                raise
            steps_change.append(steps)
            # print(f"===============End episode with {steps} steps!")
    
        return Q,steps_change
    
    def get_best_policy(self, Q):
        """
        

        Parameters
        ----------
        Q : numpy.ndarray
            更新后的 Q 函数值.

        Returns
        -------
        best_policy : dict
            当前 Q 值得到的最优策略.

        """
        
        # 创建一个空的最佳策略字典
        best_policy = {}
        # 遍历所有状态
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                # 获取当前状态的Q值
                state = (i, j)
                # 找到具有最大Q值的动作
                best_action = np.argmax(Q[i, j, :])
                # 将最佳动作添加到最佳策略字典中
                best_policy[state] = best_action
        self.best_policy = best_policy
        return best_policy
    
    def generate_path(self,best_policy={}):
        """
        根据最优策略得到最佳路径。

        """
        # 创建一个空的路径列表
        path = []
        if len(best_policy) == 0:
            best_policy = self.best_policy
        # 将起点添加到路径中
        rewards = 0
        # 当当前状态不是终点时
        state = self.start
        print(state)
        steps = 0
        while state != self.goal:
            path.append(state)
            # 获取当前状态的最佳动作
            action = best_policy[state]
            
            # 根据最佳动作更新当前状态
            next_state,r = self.step(state, action)
            rewards += r
            # 将更新后的状态添加到路径中
            if rewards<-self.max_steps:
                print("检查问题是否有解！")
                raise
            state = next_state
            print(state)
            steps += 1
        path.append(state)
        return path,rewards
    
    def visualize_path(self,path):
        """
        
        绘制路径图，包含网格世界的设定，路径需要作为参数传入。

        """
        
        # 创建一个空的状态空间网格
        grid = np.zeros((self.world_height,self.world_width))
        # 循环逐步更新状态空间网格和路径文本
        for i, state in enumerate(path):
            # 获取状态的坐标
            x, y = state
            
            # 在状态空间网格中将路径上的状态标记为1
            grid[x, y] = 1
            
        for obstacle in self.obstacles:
            x,y = obstacle
            grid[x, y] = 2
            
        # 创建一个图形对象
        fig, ax = plt.subplots()
        
        # 标记起始点和终点
        start_x, start_y = self.start
        ax.text(start_y, start_x, 'S', color='black', ha='center', va='center', fontsize=15)
        end_x, end_y = self.goal
        ax.text(end_y, end_x, 'G', color='black', ha='center', va='center', fontsize=15)
        for obstacle in self.obstacles:
            x,y = obstacle
            ax.text(y, x, 'O', color='black', ha='center', va='center', fontsize=15)
        # 隐藏坐标轴
        ax.axis('off')
        
        # 绘制初始状态空间网格
        ax.imshow(grid, cmap='cool', interpolation='nearest')
        
        # 关闭交互模式
        plt.ioff()
        plt.savefig(PATH+f"/Optimal path in grid {self.note} {self.method}.png",dpi=300)
        # 显示图形
        plt.show()
        
        
    def visualize_path_ion(self, path):
        """
        绘制路径搜索过程的动态图像，并保存为gif文件。
        """

        # 创建一个空的状态空间网格
        grid = np.zeros((self.world_height, self.world_width))
        
        
            
        # 循环逐步更新状态空间网格和路径文本
        images = []
        fig, ax = plt.subplots()
        for i, state in enumerate(path):
            # 获取状态的坐标
            x, y = state
    
            # 在状态空间网格中将路径上的状态标记为1
            grid[x, y] = 1
            
            # 创建一个图形对象
            # 标记起始点和终点
            start_x, start_y = self.start
            ax.text(start_y, start_x, 'S', color='black', ha='center', va='center', fontsize=15)
            end_x, end_y = self.goal
            ax.text(end_y, end_x, 'G', color='black', ha='center', va='center', fontsize=15)
            for obstacle in self.obstacles:
                x, y = obstacle
                grid[x, y] = 2
                ax.text(y, x, 'O', color='black', ha='center', va='center', fontsize=15)
            # 隐藏坐标轴
            ax.axis('off')
    
            # 绘制初始状态空间网格
            ax.imshow(grid, cmap='cool', interpolation='nearest')
    
            # 更新图像和路径文本
            # current_path = path[:i + 1]
            # path_str = ' -> '.join([f'({y}, {x})' for x, y in current_path])
            # ax.text(0, -1, path_str, color='black', ha='left', va='top', fontsize=12)
    
            # 将当前帧的图像转换为ndarray并添加到图像列表中
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
    
            # 恢复状态空间网格中路径上的状态为0，以便在下一帧中更新路径
            grid[x, y] = 0
    
        plt.close(fig)
    
        # 使用imageio库将图像列表保存为gif文件
        imageio.mimsave(PATH + f'/Optimal path {self.note} {self.method}.gif', images,dpi=300)

```

### 总结和绘图

```python
def grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note="",LAMBDA = 0.05,
         ACTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]]),
         ACTION_LABELS = ['R', 'D', 'L', 'U'],
         START = (3, 0), 
         GOAL = (3, 7),
         EPSILON = 0.1, 
         ALPHA = 0.5, 
         REWARD = -1,
         random_wind = False,
         num_episodes = 500):
    
    windgrid1 = WindGrid(WORLD_HEIGHT, WORLD_WIDTH, START, obstacles,
                 GOAL, WIND,
                 EPSILON, ALPHA,
                 num_episodes,
                 LAMBDA,
                 note,
                 random_wind = random_wind)
    
    if random_wind:
        note += " random wind"
        
    # Q learning
    Q, steps_change_ql = windgrid1.optimize(method="Q_learning")
    best_policy = windgrid1.get_best_policy(Q)
    path,rewards_ql = windgrid1.generate_path(best_policy)
    windgrid1.visualize_path(path)
    windgrid1.visualize_path_ion(path)
    print("Q_learning")
    print("Optimal path:\n",path)
    print("Max Reward:",rewards_ql)
    
    # Sarsa lambda
    Q, steps_change_sl = windgrid1.optimize(method="SARSA_Lambda")
    best_policy1 = windgrid1.get_best_policy(Q)
    path1,rewards_sl = windgrid1.generate_path(best_policy1)
    windgrid1.visualize_path(path)
    windgrid1.visualize_path_ion(path)
    print("SARSA_Lambda")
    print("Optimal path:\n",path1)
    print("Max Reward:",rewards_sl)
    
    # 步数对比
    plt.plot(steps_change_sl,label="SARSA Lambda")
    plt.plot(steps_change_ql,label="Q learning")
    plt.grid(linestyle='-', linewidth=0.5, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.title("Steps by Episode")
    plt.savefig(PATH+f"/Steps by Episode {note}.png",dpi=300)
    plt.show()    
    
    #累计步数对比
    cumstep_sl = pd.Series(steps_change_sl).cumsum()
    cumstep_ql = pd.Series(steps_change_ql).cumsum()
    plt.plot(cumstep_sl,label="SARSA Lambda")
    plt.plot(cumstep_ql,label="Q learning")
    plt.grid(linestyle='-', linewidth=0.5, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.title("Cumulative Steps by Episode")
    plt.savefig(PATH+f"/Cumulative Steps by Episode {note}.png",dpi=300)
    plt.show() 
    
```

