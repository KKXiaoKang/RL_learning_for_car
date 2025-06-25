# RRT* (RRT-Star) Algorithm Visualization 

import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle, Circle
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

"""
这是一个非常好的问题，它精确地指出了RRT\*与我们之前实现的RRT-Connect在**根本目标**上的不同。

您观察到的现象是完全正确的，并且这正是RRT\*算法**有意为之**的行为。

**核心原因：RRT\* 是一个路径优化算法，而不仅仅是路径寻找算法。**

*   **RRT-Connect 的目标**: 尽快找到**任何一条**从起点到终点的可行路径。一旦两棵树连接，任务就完成了，算法立即停止。这是一种“满足即可”（satisficing）的策略。

*   **RRT\* 的目标**: 找到一条**趋近于最优**（最短）的路径。它实现这个目标的过程是持续性的：
    1.  **找到初始路径**: 当树的某个节点第一次到达目标区域时，确实已经找到了一条可行的路径。
    2.  **持续优化**: 但RRT\*不会就此停止。它会继续进行迭代，利用后续的随机采样点，通过我们之前讨论的“**选择最佳父节点**”和“**重连**” (`Rewire`) 机制，不断地对树中已有的路径进行优化。

**为什么到达终点后还要继续？**

想象一下，第一次找到的路径可能是一条绕了很大弯路的曲折路径。在后续的迭代中：
*   可能会有一个新的节点`q_new`产生，它能为通往目标区域的某个旧节点提供一条“捷径”（通过`rewire`）。
*   可能会有另一条完全不同的、但更短的分支也到达了目标区域。

算法需要持续运行，给这些优化过程发生的机会。理论上，随着迭代次数趋向于无穷，RRT\*能够保证找到最优路径。在实际应用中，我们通过设定一个足够大的最大迭代次数 `MAX_ITER` 来让它有充分的时间进行优化。

**在我们的代码中，唯一的停止条件就是达到最大迭代次数 `MAX_ITER`。** 当达到上限后，它才会在所有曾到达过目标区域的节点中，挑选出那条总成本（路径长度）最低的作为最终结果。

---

不过，如果您希望它在找到第一条通往目标的路径后就立即停止（这会让它的行为更像一个标准的RRT，但保留了部分优化特性），我们也可以轻松地添加这个功能。

您想让我为您添加这个“找到即停”的选项吗？
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
STEP_SIZE = 8.0  # RRT* can benefit from slightly larger steps
MAX_ITER = 1000 # Max iterations
RADIUS = 15.0 # Neighborhood search radius
GOAL_SAMPLE_RATE = 5 # 5% chance to sample the goal

class Node:
    """RRT*树的节点"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# --- 核心算法函数 (部分与RRT-Connect共享) ---
def distance(node1, node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def get_random_point():
    if random.randint(0, 100) > (100 - GOAL_SAMPLE_RATE):
        return Node(GOAL_POS[0], GOAL_POS[1])
    return Node(random.uniform(0, X_DIM), random.uniform(0, Y_DIM))

def get_nearest_node(tree, point_node):
    dlist = [distance(node, point_node) for node in tree]
    minind = dlist.index(min(dlist))
    return tree[minind]

def steer(from_node, to_node, extend_length=float("inf")):
    d = distance(from_node, to_node)
    if d < extend_length:
        return to_node
    else:
        new_x = from_node.x + extend_length * (to_node.x - from_node.x) / d
        new_y = from_node.y + extend_length * (to_node.y - from_node.y) / d
        return Node(new_x, new_y)

def is_collision(node, obstacles):
    if node is None: return True
    for (ox, oy, w, h) in obstacles:
        if ox <= node.x <= ox + w and oy <= node.y <= oy + h:
            return True
    return False

def is_path_collision_free(from_node, to_node, obstacles):
    if to_node is None: return False
    d = distance(from_node, to_node)
    if d == 0: return not is_collision(from_node, obstacles)
    n_step = int(d / (STEP_SIZE / 5)) if STEP_SIZE > 0 else 1
    if n_step == 0: n_step = 1
    for i in range(n_step + 1):
        t = i / n_step
        x = from_node.x * (1.0 - t) + to_node.x * t
        y = from_node.y * (1.0 - t) + to_node.y * t
        if is_collision(Node(x, y), obstacles):
            return False
    return True

# --- RRT* 特有函数 ---
def find_near_nodes(tree, new_node, radius):
    """在指定半径内寻找new_node的邻居节点"""
    near_nodes = []
    for node in tree:
        if distance(node, new_node) <= radius:
            near_nodes.append(node)
    return near_nodes

def choose_parent(near_nodes, nearest_node, new_node, obstacles):
    """从邻居中为new_node选择最佳父节点"""
    if not near_nodes:
        return nearest_node

    best_parent = nearest_node
    min_cost = nearest_node.cost + distance(nearest_node, new_node)

    for near_node in near_nodes:
        if is_path_collision_free(near_node, new_node, obstacles):
            new_cost = near_node.cost + distance(near_node, new_node)
            if new_cost < min_cost:
                min_cost = new_cost
                best_parent = near_node
    
    new_node.cost = min_cost
    new_node.parent = best_parent
    return new_node

def rewire(tree, new_node, near_nodes, obstacles):
    """对邻居节点进行重连，检查它们是否能通过new_node获得更短的路径"""
    for near_node in near_nodes:
        if near_node == new_node.parent:
            continue

        if is_path_collision_free(new_node, near_node, obstacles):
            new_cost = new_node.cost + distance(new_node, near_node)
            if new_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_cost
                # In a more complex implementation, you'd recursively update costs of children.
                # For this visualization, changing the parent link is sufficient.
    return tree

def find_final_path(tree, goal_node, goal_search_radius):
    """在所有迭代结束后，寻找通往目标的最佳路径"""
    path_candidates = []
    for node in tree:
        if distance(node, goal_node) <= goal_search_radius:
            path_candidates.append(node)
    
    if not path_candidates:
        return None # No path found

    # Find the candidate with the minimum cost
    best_node = min(path_candidates, key=lambda node: node.cost)
    
    path = []
    curr = best_node
    while curr is not None:
        path.append((curr.x, curr.y))
        curr = curr.parent
    path.reverse()
    return path

# --- PyQt5 GUI 类 ---
class RRTStarGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RRT* Step-by-Step")
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.next_step_button = QtWidgets.QPushButton("Next Step")
        self.next_step_button.clicked.connect(self.next_step)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.auto_run_button = QtWidgets.QPushButton("Auto Run")
        self.auto_run_button.clicked.connect(self.toggle_auto_run)
        
        self.button_layout.addWidget(self.next_step_button)
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addWidget(self.auto_run_button)
        self.layout.addLayout(self.button_layout)
        
        # Timer for auto-run
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_step)
        self.is_auto_running = False
        
        self.reset_simulation()
        
    def reset_simulation(self):
        # Stop the timer if it's running
        if self.is_auto_running:
            self.toggle_auto_run()

        self.start_node = Node(START_POS[0], START_POS[1])
        self.goal_node = Node(GOAL_POS[0], GOAL_POS[1])
        self.tree = [self.start_node]
        self.final_path = None
        self.iteration = 0
        self.next_step_button.setEnabled(True)
        self.auto_run_button.setEnabled(True)
        self.plot_scenario()
        
    def toggle_auto_run(self):
        """Starts or stops the automatic simulation."""
        if not self.is_auto_running:
            self.is_auto_running = True
            self.auto_run_button.setText("Stop")
            self.next_step_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.timer.start(50)  # Fire next_step every 50ms
        else:
            self.is_auto_running = False
            self.timer.stop()
            self.auto_run_button.setText("Auto Run")
            self.next_step_button.setEnabled(True)
            self.reset_button.setEnabled(True)
        
    def next_step(self):
        if self.iteration >= MAX_ITER:
            # If auto-running, stop it
            if self.is_auto_running:
                self.toggle_auto_run()

            self.final_path = find_final_path(self.tree, self.goal_node, STEP_SIZE)
            self.plot_scenario()
            self.next_step_button.setEnabled(False)
            print("Max iterations reached. Displaying best path found.")
            return
            
        self.iteration += 1
        
        q_rand = get_random_point()
        nearest_node = get_nearest_node(self.tree, q_rand)
        q_new = steer(nearest_node, q_rand, STEP_SIZE)
        
        plot_elements = {'q_rand': q_rand}

        if is_path_collision_free(nearest_node, q_new, OBSTACLES):
            near_nodes = find_near_nodes(self.tree, q_new, RADIUS)
            q_new = choose_parent(near_nodes, nearest_node, q_new, OBSTACLES)
            self.tree.append(q_new)
            self.tree = rewire(self.tree, q_new, near_nodes, OBSTACLES)
            plot_elements['q_new'] = q_new
            plot_elements['near_nodes_radius_center'] = q_new
        
        self.plot_scenario(plot_elements)
        
    def plot_scenario(self, elements=None):
        self.ax.clear()
        if elements is None: elements = {}

        # 绘制障碍物
        for (ox, oy, w, h) in OBSTACLES:
            self.ax.add_patch(Rectangle((ox, oy), w, h, facecolor='gray'))

        # 绘制树
        for node in self.tree:
            if node.parent:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b", alpha=0.5)

        # 绘制RRT*特定可视化元素
        if 'near_nodes_radius_center' in elements:
            center = elements['near_nodes_radius_center']
            self.ax.add_patch(Circle((center.x, center.y), RADIUS, color='y', fill=False, linestyle='--', label=f'Search Radius ({RADIUS})'))
        if 'q_rand' in elements:
            self.ax.plot(elements['q_rand'].x, elements['q_rand'].y, "ko", markersize=3, label='q_rand')
        if 'q_new' in elements:
            self.ax.plot(elements['q_new'].x, elements['q_new'].y, 'cx', markersize=7, label='q_new')

        # 绘制最终路径
        if self.final_path:
            path_x, path_y = zip(*self.final_path)
            self.ax.plot(path_x, path_y, 'g-', linewidth=2.5, label='Final Path')

        # 绘制起点和终点
        self.ax.plot(START_POS[0], START_POS[1], "go", markersize=10, label='Start')
        self.ax.plot(GOAL_POS[0], GOAL_POS[1], "ro", markersize=10, label='Goal')

        self.ax.set_xlim(0, X_DIM)
        self.ax.set_ylim(0, Y_DIM)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title(f"RRT* Iteration: {self.iteration}")
        # Re-order legend handles to be more intuitive
        handles, labels = self.ax.get_legend_handles_labels()
        order = ['Start', 'Goal', 'q_rand', 'q_new', 'Search Radius', 'Final Path']
        handles_ordered, labels_ordered = [], []
        label_map = {label: handle for handle, label in zip(handles, labels)}
        for label in order:
            if label in label_map:
                labels_ordered.append(label)
                handles_ordered.append(label_map[label])
        self.ax.legend(handles_ordered, labels_ordered, loc='lower right')
        
        self.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = RRTStarGUI()
    main_window.show()
    sys.exit(app.exec_()) 