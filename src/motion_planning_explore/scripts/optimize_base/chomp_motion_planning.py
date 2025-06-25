import numpy as np
import matplotlib.pyplot as plt

# 配置参数
START = np.array([5, 5])
GOAL = np.array([90, 90])
N_WAYPOINTS = 50  # 轨迹离散点数
LAMBDA_SMOOTH = 30.0  # 平滑项权重（增大）
ETA = 0.005  # 梯度步长（减小）
N_ITER = 400  # 优化迭代次数（增大）
OBSTACLE_WEIGHT = 80.0  # 障碍物项权重（增大）
SAFE_DIST = 10.0  # 安全距离

DEBUG_FLAG = True

# 障碍物（圆形：中心和半径）
OBSTACLES = [
    {'center': np.array([40, 40]), 'radius': 12},
    {'center': np.array([60, 60]), 'radius': 10},
    {'center': np.array([50, 70]), 'radius': 8},
]

# 轨迹初始化（直线）
def initialize_trajectory(start, goal, n_points):
    return np.linspace(start, goal, n_points)

# 平滑项梯度 - 加速度平滑项梯度
def smoothness_gradient(traj):
    grad = np.zeros_like(traj) # 初始化梯度为0
    """
        对于轨迹上的任意一点，其加速度为：（代码当中使用一阶差分，原文论文里面使用一阶差分）

        除了代码中实现的一阶差分，还可以使用二阶差分，二阶差分可以更精确地计算加速度，但是计算量更大。
        
        二阶差分：
        # 二阶差分 - 加速度项
        a = (traj[i+1] - traj[i]) - (traj[i] - traj[i-1]) = traj[i+1] - 2*traj[i] + traj[i-1]
        # 二阶差分平滑项代价函数为：所有点上这个加速度的平方和 (乘于0.5是为了求导的时候把2的系数消掉)
        J_smooth = 0.5 * sum(a^2)
        # 因此，二阶差分平滑项梯度为：
        grad[i] = 6*traj[i] - 4*traj[i-1] - 4*traj[i+1] + traj[i-2] + traj[i+2]

        一阶差分：
        # 一阶差分 - 速度项
        v = (traj[i+1] - traj[i])
        # 一阶差分平滑项代价函数为：所有点上这个速度的平方和 (乘于0.5是为了求导的时候把2的系数消掉)
        J_smooth = 0.5 * sum(v^2)
        # 因此，一阶差分平滑项梯度为：
        grad[i] = 2*traj[i] - traj[i-1] - traj[i+1]
    """
    # 梯度计算 （一阶差分）
    grad[1:-1] = 2*traj[1:-1] - traj[0:-2] - traj[2:] # 计算平滑项梯度
    return grad

# 增强版障碍物项及其梯度（指数型势场）
def obstacle_cost_and_grad(traj, obstacles):
    cost = 0.0
    grad = np.zeros_like(traj)
    i = 0
    for obs in obstacles:
        if DEBUG_FLAG:
            print("---------------第{}个障碍物---------------".format(i))
            i += 1
        center = obs['center'] # 障碍物中心
        radius = obs['radius'] # 障碍物半径
        vec = traj - center # 轨迹到障碍物中心的向量
        dist = np.linalg.norm(vec, axis=1) # 轨迹到障碍物中心的距离：计算向量的欧几里得范数（向量长度）
        d_to_obs = dist - radius # 轨迹到障碍物中心的距离 - 障碍物半径，得到与障碍物表面的有符号距离
        inside = d_to_obs < SAFE_DIST # 判断距离障碍物表面的有符号距离 是否 小于 安全距离 | inside是bool类型的数组
        """
            CHOMP是局部优化算法。让无穷远处的点也产生梯度是没有意义且浪费计算的。
            我们定义一个“势场”范围 SAFE_DIST 只有当轨迹点进入这个范围 我们才开始计算它的代价和梯度 把它“推出去”
            没进入势场范围的点d_to_obs为正 梯度为0 （没进入势场范围的点，不会产生梯度）
            进入势场范围的点d_to_obs为负 梯度为负 （进入势场范围的点，会产生梯度）
        """
        if DEBUG_FLAG:
            print(f"inside : {inside}")
            print(f"d_to_obs : {d_to_obs}")
            print(f"d_to_obs[inside] : {d_to_obs[inside]}")
        # 指数型势场
        cost_term = np.zeros_like(dist) # 初始化代价函数为0
        cost_term[inside] = 0.5 * OBSTACLE_WEIGHT * (SAFE_DIST - d_to_obs[inside])**2
        cost += np.sum(cost_term)
        if DEBUG_FLAG:
            print(f"cost_term[inside] : {cost_term[inside]}")
            print(f"cost : {cost}")
        # 梯度
        grad_term = np.zeros_like(traj) # 为当前障碍物创建一个临时的代价值数组，默认为0
        # 计算梯度
        """
            公式为：
            左边为梯度的大小: -OBSTACLE_WEIGHT * (SAFE_DIST - d_to_obs[inside])
            右边为计算梯度的方向：距离函数对坐标的导数（梯度方向），在这里是从障碍物中心指向轨迹点的单位向量
            -Wo * (SAFE_DIST - d_to_obs[inside]) * (vec[inside] / (dist[inside][:, None] + 1e-6))
        """
        grad_term[inside] = -OBSTACLE_WEIGHT * (SAFE_DIST - d_to_obs[inside])[:, None] * (vec[inside] / (dist[inside][:, None] + 1e-6))
        if DEBUG_FLAG:
            print(f"grad_term[inside] : {grad_term[inside]}")
            print(f"grad : {grad}")
        grad += grad_term # 将这个障碍物产生的梯度累加到总梯度grad中
    return cost, grad

# CHOMP主循环
def chomp_optimize(traj, obstacles, n_iter, eta, lambda_smooth):
    traj_hist = [traj.copy()]
    if DEBUG_FLAG:
        print(f"初始化轨迹 traj_hist : {traj_hist}")
    for i in range(n_iter):
        grad_smooth = smoothness_gradient(traj)
        cost_obs, grad_obs = obstacle_cost_and_grad(traj, obstacles)
        grad = lambda_smooth * grad_smooth + grad_obs
        # 固定起点和终点
        grad[0] = 0
        grad[-1] = 0
        traj = traj - eta * grad
        traj_hist.append(traj.copy())
    return traj, traj_hist

# 可视化
def plot_chomp(traj_hist, obstacles, start, goal):
    plt.figure(figsize=(7,7))
    for obs in obstacles:
        circle = plt.Circle(obs['center'], obs['radius'], color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
    for i, traj in enumerate(traj_hist):
        if i == 0:
            plt.plot(traj[:,0], traj[:,1], 'b--', alpha=0.5, label='Init Trajectory')
        elif i == len(traj_hist)-1:
            plt.plot(traj[:,0], traj[:,1], 'g-', linewidth=2, label='Optimized Trajectory')
        elif i % (len(traj_hist)//5) == 0:
            plt.plot(traj[:,0], traj[:,1], 'r-', alpha=0.2)
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('CHOMP Trajectory Optimization')
    plt.show()

if __name__ == '__main__':
    # 前端给定初始化轨迹
    traj_init = initialize_trajectory(START, GOAL, N_WAYPOINTS)
    if DEBUG_FLAG:
        # 打印初始化轨迹
        print(f"初始化轨迹 traj_init : {traj_init}")
    # 后端优化轨迹
    traj_opt, traj_hist = chomp_optimize(traj_init, OBSTACLES, N_ITER, ETA, LAMBDA_SMOOTH)
    # 可视化结果
    plot_chomp(traj_hist, OBSTACLES, START, GOAL) 