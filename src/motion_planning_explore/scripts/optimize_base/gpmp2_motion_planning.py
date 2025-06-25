# GPMP2 (Gaussian Process Motion Planner 2) Algorithm Visualization 
"""
当然可以！复现 GPMP2 (Gaussian Process Motion Planner 2) 是一个非常有趣的任务。这是一个将在 CHOMP 基础上，引入更高级概率思想的运动规划器。

下面，我将为你：
1.  详细解释 **GPMP2 的核心思想**，并阐述它与 CHOMP 的关键区别。
2.  在 `gpmp2_motion_planning.py` 文件中，创建一个功能与我们之前案例一致的、简化的 GPMP2 可视化案例。

---

### GPMP2 vs. CHOMP：核心思想的区别

CHOMP 是一个基于**能量最小化**的优化方法，它的目标函数是 \(\mathcal{U} = \mathcal{U}_{\text{smooth}} + \mathcal{U}_{\text{obs}}\)。

GPMP2 则是一个基于**概率推理**的优化方法，它将运动规划问题建模为寻找**最大后验概率（Maximum a Posteriori, MAP）**的轨迹。

根据贝叶斯定理，后验概率正比于先验概率和似然的乘积：

"""
import numpy as np
import matplotlib.pyplot as plt

# --- 配置参数 ---
START = np.array([5, 5])
GOAL = np.array([90, 90])
N_WAYPOINTS = 50      # 轨迹离散点数
LAMBDA_GP = 20.0      # GP先验项权重 (平滑项)
OBSTACLE_WEIGHT = 10.0 # 障碍物似然项权重
ETA = 0.01            # 梯度步长
N_ITER = 2000          # 优化迭代次数
SAFE_DIST = 15.0      # 安全距离

# --- 障碍物 (圆形：中心和半径) ---
OBSTACLES = [
    {'center': np.array([40, 40]), 'radius': 12},
    {'center': np.array([60, 60]), 'radius': 10},
    {'center': np.array([50, 70]), 'radius': 8},
]

# --- 核心函数 ---

def initialize_trajectory(start, goal, n_points):
    """初始化轨迹为直线"""
    return np.linspace(start, goal, n_points)

def gp_prior_gradient(traj):
    """
    计算高斯过程(GP)先验的梯度。
    这对应于最小化速度的平方和，由零均值高斯过程在速度上的先验推导得出。
    """
    grad = np.zeros_like(traj)
    grad[1:-1] = 2 * traj[1:-1] - traj[0:-2] - traj[2:]
    return grad

def obstacle_likelihood_gradient(traj, obstacles):
    """
    计算障碍物似然的梯度。
    这里使用铰链损失(Hinge Loss)作为障碍物代价的近似。
    """
    cost = 0.0
    grad = np.zeros_like(traj)
    for obs in obstacles:
        center = obs['center']
        radius = obs['radius']
        
        vec = traj - center
        dist = np.linalg.norm(vec, axis=1)
        d_to_obs = dist - radius
        
        inside = d_to_obs < SAFE_DIST
        
        # 代价函数 (Hinge Loss)
        cost_term = np.zeros_like(dist)
        cost_term[inside] = OBSTACLE_WEIGHT * (SAFE_DIST - d_to_obs[inside])
        cost += np.sum(cost_term)
        
        # 梯度
        grad_term = np.zeros_like(traj)
        # 梯度大小是恒定的，方向远离障碍物
        grad_term[inside] = -OBSTACLE_WEIGHT * (vec[inside] / (dist[inside][:, None] + 1e-6))
        grad += grad_term
        
    return cost, grad

def gpmp2_optimize(traj, obstacles, n_iter, eta, lambda_gp):
    """GPMP2 主优化循环"""
    traj_hist = [traj.copy()]
    for _ in range(n_iter):
        grad_prior = gp_prior_gradient(traj)
        _, grad_obs = obstacle_likelihood_gradient(traj, obstacles)
        
        # 组合梯度 (对应于最小化负对数后验概率)
        grad = lambda_gp * grad_prior + grad_obs
        
        # 固定起点和终点
        grad[0] = 0
        grad[-1] = 0
        
        traj = traj - eta * grad
        traj_hist.append(traj.copy())
        
    return traj, traj_hist

# --- 可视化 ---

def plot_gpmp2(traj_hist, obstacles, start, goal):
    """绘制优化过程和结果"""
    plt.figure(figsize=(8, 8))
    
    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle(obs['center'], obs['radius'], color='gray', alpha=0.7, zorder=2)
        plt.gca().add_patch(circle)
        
    # 绘制轨迹历史
    for i, traj in enumerate(traj_hist):
        if i == 0:
            plt.plot(traj[:, 0], traj[:, 1], 'b--', alpha=0.6, label='Init Trajectory', zorder=3)
        elif i == len(traj_hist) - 1:
            plt.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=2.5, label='Optimized Trajectory', zorder=5)
        elif i % (len(traj_hist) // 10) == 0: # 绘制更多中间轨迹
            plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.2, zorder=4)
            
    # 绘制起点和终点
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=6)
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal', zorder=6)
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('GPMP2 Trajectory Optimization')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# --- 主函数 ---

if __name__ == '__main__':
    traj_init = initialize_trajectory(START, GOAL, N_WAYPOINTS)
    traj_opt, traj_hist = gpmp2_optimize(traj_init, OBSTACLES, N_ITER, ETA, LAMBDA_GP)
    plot_gpmp2(traj_hist, OBSTACLES, START, GOAL) 