import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
PATH = os.path.dirname(__file__)+"/Image"
import imageio
#%%
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

#%%
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
    
#%%
#########################
####### 实验部分
#########################

LAMBDA = 0.05
WORLD_HEIGHT = 7
WORLD_WIDTH = 10
ACTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
ACTION_LABELS = ['R', 'D', 'L', 'U']
START = (3, 0)
GOAL = (3, 7)
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
EPSILON = 0.1
ALPHA = 0.5
REWARD = -1
num_episodes = 500
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
#%%
WORLD_HEIGHT = 7
WORLD_WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
np.random.seed(123)
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA)
#%%
WORLD_HEIGHT = 7
WORLD_WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
obstacles = []
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA,
                  random_wind=True)
#%%
WORLD_HEIGHT = 7
WORLD_WIDTH = 10
WIND = [0, 3, 3, 3, 3, 1, 2, 2, 1, 0]
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
np.random.seed(123)
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA)
#%%
WORLD_HEIGHT = 20
WORLD_WIDTH = 20
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA)

#%%
WORLD_HEIGHT = 20
WORLD_WIDTH = 20
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2],
             [6, 2], [7, 2], [8, 2], [8, 3], [8, 4]]
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
np.random.seed(123)
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA)
#%%
for i in range(0,10):
    WORLD_HEIGHT = 20
    WORLD_WIDTH = 20
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2],
                 [6, 2], [7, 2], [8, 2], [8, 3], [8, 4]]
    LAMBDA = 0.1*i
    note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
    np.random.seed(123)
    num_episodes = 1000
    grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA,
                      num_episodes=num_episodes)
#%%
for i in range(1,6):
    WORLD_HEIGHT = 20
    WORLD_WIDTH = 20
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2],
                 [6, 2], [7, 2], [8, 2], [8, 3], [8, 4]]
    LAMBDA = 0.2
    ALPHA = 0.1*i
    note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles ALPHA={ALPHA}"
    np.random.seed(123)
    num_episodes = 2000
    grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA,
                      num_episodes=num_episodes,
                      ALPHA = ALPHA)
#%%
WORLD_HEIGHT = 20
WORLD_WIDTH = 20
WIND = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
             [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
             [2, 5], [3, 5], [4, 5], [8, 5], [9, 5], [10, 5], [11, 5], [12, 5], 
             [12, 4], [12, 3],
             [8, 3], [8, 4],
             [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
             [5, 11], [5, 12], [5, 13], [5, 14],
             [6, 7], [7, 7], [8, 7], [9, 7], [10, 7],
             [10, 0], [10, 1], [10, 2], [10, 3],
             [0, 2]]
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
np.random.seed(123)
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA)
#%%
WORLD_HEIGHT = 20
WORLD_WIDTH = 20
WIND = [0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
             [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
             [2, 5], [3, 5], [4, 5], [8, 5], [9, 5], [10, 5], [11, 5], [12, 5], 
             [12, 4], [12, 3],
             [8, 3], [8, 4],
             [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
             [5, 11], [5, 12], [5, 13], [5, 14],
             [6, 7], [7, 7], [8, 7], [9, 7], [10, 7],
             [10, 0], [10, 1], [10, 2], [10, 3],
             [0, 2]]
LAMBDA = 0.05
note = f"{WORLD_HEIGHT}x{WORLD_WIDTH} with {len(obstacles)}obstacles lambda={LAMBDA}"
np.random.seed(123)
grid_case_summary(WORLD_HEIGHT, WORLD_WIDTH, WIND, obstacles,note,LAMBDA)

