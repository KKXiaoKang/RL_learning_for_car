# 
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

"""
    这是一个非常棒的观察，它正好揭示了RRT算法的核心机制之一：**碰撞检测**。

您看到的情况——即采样了 `q_rand`，但树没有生长，也没有显示 `q_new`——是完全正常的，它发生在以下情况：

**当从树上最近的节点到新节点 `q_new` 的“生长路径”被障碍物阻挡时，扩展就会失败。**

让我们来分解一下算法在每一步迭代中的完整流程：

1.  **采样 `q_rand`**: 算法在空间中选择一个随机点（您看到的黑色小点）。
2.  **寻找最近邻**: 算法在当前正在扩展的树（比如蓝色树）中，找到离 `q_rand` 最近的已有节点，我们称之为 `q_near`。
3.  **计算潜在新节点 `q_new`**: 算法从 `q_near` 出发，朝着 `q_rand` 的方向，移动一个固定的最大步长 `STEP_SIZE`，计算出新节点 `q_new` 的**预定位置**。
4.  **碰撞检测 (关键步骤)**: 在正式将 `q_new` 添加到树上之前，算法会检查从 `q_near` 到 `q_new` 的**直线路径**。它会沿着这条线段取很多个点，检查这些点是否会碰到任何一个障碍物（灰色矩形）。
5.  **决策**:
    *   **如果路径安全**: `q_new` 被正式采纳，添加到树上，并与 `q_near` 连接。您会看到绿色的“Extend Step”和青色的`q_new`标记。
    *   **如果路径被阻挡 (Trapped)**: 即使 `q_new` 本身不在障碍物内，但只要从 `q_near` 到它的路径穿过了障碍物，这次扩展就会被视为**失败**。`q_new` 会被丢弃，树不会生长。

**所以，您遇到的情况正是第5步中的“路径被阻挡”。** 算法尝试了一次生长，但因为新树枝会“撞墙”，所以它放弃了这次尝试。在可视化中，您就只能看到那一轮被采样的 `q_rand`，而没有看到任何成功的扩展。

这正是RRT（快速探索随机树）算法能够绕过障碍物的根本原因：它不断地进行“试探-检测”的循环，只保留那些不会导致碰撞的安全扩展，从而迫使树“绕着”障碍物生长。
"""

# --- 基本配置 ---
X_DIM = 100
Y_DIM = 100
START_POS = (5, 5)
GOAL_POS = (90, 90)
OBSTACLES = [
    (25, 25, 10, 50),
    (65, 25, 10, 50),
    (40, 10, 20, 10),
    (40, 80, 20, 10),
]
STEP_SIZE = 5.0  # 每步扩展的最大距离
MAX_ITER = 2000 # 最大迭代次数

class Node:
    """
    RRT树的节点
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def distance(node1, node2):
    """计算两个节点之间的欧式距离"""
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def get_random_point():
    """在配置空间中随机采样一个点"""
    # 以较小的概率直接采样目标点，以加速收敛
    if random.randint(0, 100) > 95:
        return Node(GOAL_POS[0], GOAL_POS[1])
    return Node(random.uniform(0, X_DIM), random.uniform(0, Y_DIM))

def get_nearest_node(tree, point_node):
    """在树中找到距离给定点最近的节点"""
    dlist = [distance(node, point_node) for node in tree]
    minind = dlist.index(min(dlist))
    return tree[minind]

def steer(from_node, to_node, extend_length=float("inf")):
    """
    从 from_node 朝 to_node 方向延伸一个步长
    """
    d = distance(from_node, to_node)
    if d < extend_length:
        return to_node
    else:
        # 归一化方向向量并乘以步长
        new_x = from_node.x + extend_length * (to_node.x - from_node.x) / d
        new_y = from_node.y + extend_length * (to_node.y - from_node.y) / d
        return Node(new_x, new_y)

def is_collision(node, obstacles):
    """检查一个点是否与障碍物碰撞"""
    if node is None:
        return True
    for (ox, oy, w, h) in obstacles:
        if ox <= node.x <= ox + w and oy <= node.y <= oy + h:
            return True
    return False

def is_path_collision_free(from_node, to_node, obstacles):
    """
    通过离散化路径段来检查两点之间的路径是否无碰撞
    """
    if to_node is None:
        return False
        
    d = distance(from_node, to_node)
    if d == 0:
        return not is_collision(from_node, obstacles)

    # 检查分辨率为步长的1/5
    n_step = int(d / (STEP_SIZE / 5)) if STEP_SIZE > 0 else 1
    if n_step == 0: n_step = 1

    for i in range(n_step + 1):
        t = i / n_step
        x = from_node.x * (1.0 - t) + to_node.x * t
        y = from_node.y * (1.0 - t) + to_node.y * t
        if is_collision(Node(x, y), obstacles):
            return False
    return True

def extend(tree, target_node, obstacles, step_size):
    """
    从树(tree)向目标点(target_node)扩展一步
    """
    nearest_node = get_nearest_node(tree, target_node)
    new_node = steer(nearest_node, target_node, step_size)

    if is_path_collision_free(nearest_node, new_node, obstacles):
        new_node.parent = nearest_node
        tree.append(new_node)
        
        # 检查是否已到达目标点
        if distance(new_node, target_node) < 1e-6:
            return "Reached", (nearest_node, new_node)
        else:
            return "Advanced", (nearest_node, new_node)
    
    return "Trapped", None

def connect(tree, target_node, obstacles, step_size):
    """
    重复调用extend，尝试完全连接到目标点
    """
    status = "Advanced"
    # 当扩展状态为"Advanced"时，持续向目标点扩展
    while status == "Advanced":
        status, _ = extend(tree, target_node, obstacles, step_size)
    return status

def reconstruct_path(node_a, node_b, tree_a_is_start_tree):
    """从连接点回溯，重建从起点到终点的完整路径"""
    
    # node_a 是 T_a 中新加入的节点 (q_new)
    # node_b 是 T_b 中新加入的、与 q_new 位置几乎重合的节点
    
    path_a = []
    curr = node_a
    while curr is not None:
        path_a.append((curr.x, curr.y))
        curr = curr.parent

    path_b = []
    curr = node_b
    while curr is not None:
        path_b.append((curr.x, curr.y))
        curr = curr.parent

    # 根据 tree_a 是不是起始树来决定哪条路径需要反转
    if tree_a_is_start_tree:
        # T_a 是起点树, T_b 是目标树
        # path_a 是 [连接点, ..., 起点], 需要反转
        path_a.reverse()
        # path_b 是 [连接点, ..., 终点], 不需要反转, 直接拼接
        return path_a + path_b[1:]
    else:
        # T_b 是起点树, T_a 是目标树
        # path_b 是 [连接点, ..., 起点], 需要反转
        path_b.reverse()
        # path_a 是 [连接点, ..., 终点], 不需要反转, 直接拼接
        return path_b + path_a[1:]

# --- PyQt5 GUI 类 ---
class RRTConnectGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RRT-Connect Step-by-Step")
        
        # UI 设置
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Matplotlib 图形和画布
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # 控制按钮
        self.button_layout = QtWidgets.QHBoxLayout()
        self.next_step_button = QtWidgets.QPushButton("Next Step")
        self.next_step_button.clicked.connect(self.next_step)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.button_layout.addWidget(self.next_step_button)
        self.button_layout.addWidget(self.reset_button)
        self.layout.addLayout(self.button_layout)
        
        # RRT 初始化
        self.reset_simulation()
        
    def reset_simulation(self):
        """将RRT模拟重置到初始状态"""
        # 1. 初始化两棵树
        self.T_start = [Node(START_POS[0], START_POS[1])]
        self.T_goal = [Node(GOAL_POS[0], GOAL_POS[1])]
        
        # tree_a 和 tree_b 是指向 T_start 和 T_goal 的可交换引用
        self.tree_a = self.T_start
        self.tree_b = self.T_goal
        
        self.final_path = None
        self.iteration = 0
        
        self.next_step_button.setEnabled(True)
        self.plot_scenario() # 绘制初始状态
        
    def next_step(self):
        """执行RRT-Connect算法的一次迭代"""
        if self.final_path or self.iteration >= MAX_ITER:
            self.next_step_button.setEnabled(False)
            return
            
        self.iteration += 1
        
        # 用于可视化的扩展线段
        extend_line_info = None
        
        # 2a. 随机采样
        q_rand = get_random_point()

        # 2b. 从 T_a 向 q_rand 扩展
        status_extend, extend_info = extend(self.tree_a, q_rand, OBSTACLES, STEP_SIZE)
        
        if extend_info:
            extend_line_info = extend_info
        
        # 2c. 如果扩展成功
        if status_extend != "Trapped":
            q_new = self.tree_a[-1]
            # 2c ii. 尝试让另一棵树 T_b 连接过来
            status_connect = connect(self.tree_b, q_new, OBSTACLES, STEP_SIZE)
            
            # 2c iii. 如果连接成功，路径找到！
            if status_connect == "Reached":
                print(f"路径在 {self.iteration} 次迭代后找到。")
                node_from_b = self.tree_b[-1]
                # 检查当前 tree_a 是不是起始树
                is_a_start_tree = (self.tree_a == self.T_start)
                self.final_path = reconstruct_path(q_new, node_from_b, is_a_start_tree)
                self.next_step_button.setEnabled(False)
                
                # --- 关键修正 ---
                # 找到路径后，立即用最终状态重绘一次，并且不再显示extend_line
                self.plot_scenario(q_rand, extend_line=None)
                return # 结束本次迭代

        # 如果没有找到路径，才交换角色并准备下一次迭代
        # 2d. 交换 T_a 和 T_b 的角色
        self.tree_a, self.tree_b = self.tree_b, self.tree_a

        # 可视化当前状态
        self.plot_scenario(q_rand, extend_line_info)
        
        if not self.final_path and self.iteration >= MAX_ITER:
            print(f"在 {MAX_ITER} 次迭代后未能找到路径。")
            self.next_step_button.setEnabled(False)

    def plot_scenario(self, q_rand=None, extend_line=None):
        """在画布上绘制RRT-Connect算法的当前状态"""
        self.ax.clear()
        
        # 绘制障碍物
        for (ox, oy, w, h) in OBSTACLES:
            self.ax.add_patch(Rectangle((ox, oy), w, h, facecolor='gray'))

        # 绘制两棵树（起始树 T_start 始终为蓝色，目标树 T_goal 始终为红色）
        for node in self.T_start:
            if node.parent:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")
        for node in self.T_goal:
            if node.parent:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-r")

        # 绘制起点和终点
        self.ax.plot(START_POS[0], START_POS[1], "go", markersize=10, label='Start (A)')
        self.ax.plot(GOAL_POS[0], GOAL_POS[1], "yo", markersize=10, label='Goal (B)')

        # 绘制随机采样点 q_rand
        if q_rand:
            self.ax.plot(q_rand.x, q_rand.y, "ko", markersize=3, label='q_rand')
            
        # 绘制EXTEND步长
        if extend_line:
            n_from, n_to = extend_line  # n_to 就是 q_new
            # 绘制步长，绿色粗实线
            self.ax.plot([n_from.x, n_to.x], [n_from.y, n_to.y], 'g-', linewidth=2.5, label='Extend Step')
            # 绘制 q_new，用青色叉号标记
            self.ax.plot(n_to.x, n_to.y, 'cx', markersize=7, label='q_new')

        # 绘制最终路径
        if self.final_path:
            path_x, path_y = zip(*self.final_path)
            self.ax.plot(path_x, path_y, 'g-', linewidth=2, label='Final Path')

        self.ax.set_xlim(0, X_DIM)
        self.ax.set_ylim(0, Y_DIM)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title(f"RRT-Connect Iter: {self.iteration} (Blue: Start Tree, Red: Goal Tree)")
        self.ax.legend(loc='lower right')
        
        # 重绘画布
        self.canvas.draw()


if __name__ == '__main__':
    # 用PyQt5应用启动代替原来的main函数
    app = QtWidgets.QApplication(sys.argv)
    main_window = RRTConnectGUI()
    main_window.show()
    sys.exit(app.exec_()) 