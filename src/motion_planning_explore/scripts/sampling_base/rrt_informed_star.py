# Informed RRT* (RRT-Star) Algorithm Visualization 
"""
好的，我们来挑战 RRT\* 的高级变体：**Informed RRT\***。这个算法在 RRT\* 的基础上做出了一个非常聪明的改进，极大地提高了收敛到最优路径的效率。

我将首先为您详细解释 RRT\* 和 Informed RRT\* 的核心区别，然后创建一个新的 `rrt_informed_star.py` 文件，并提供一套功能与我们之前构建的 RRT\* 可视化工具完全一样的完整代码。

---

### RRT\* 与 Informed RRT\* 的核心区别

两者的根本区别在于**采样策略的演进**。

#### RRT\* (我们上一个例子)

*   **探索方式**：**始终在整个地图范围内**（比如我们100x100的矩形）进行随机采样。
*   **优化过程**：它通过“选择最佳父节点”和“重连”来优化路径。虽然它会逐渐找到更短的路径，但它的大量算力被浪费在了那些**不可能**包含更优路径的区域。例如，一旦找到一条成本（长度）为150的路径，它仍然会在地图的某个遥远角落进行采样，而那个角落里的点无论如何也不可能构成一条比150更短的路径。
*   **效率**：在找到初始解后，其收敛到最优解的速度会越来越慢，因为它的“有效”采样（那些真正可能优化路径的采样）在所有采样中的比例会越来越低。

#### Informed RRT\* (我们将要实现的)

*   **探索方式**：这是一个**两阶段**的过程。
    1.  **阶段一 (类似 RRT\*)**: 在找到**第一条**可行路径之前，它的行为和标准的RRT\*完全一样，在整个地图范围内进行采样。
    2.  **阶段二 (核心创新)**: 一旦找到了第一条路径，设其成本为 `c_best`。算法立即**将采样区域从整个地图缩小到一个椭圆内**。
*   **神秘的椭圆**：
    *   **定义**: 这个椭圆的**两个焦点**分别是**起点**和**终点**。
    *   **大小**: 椭圆的大小由当前找到的最佳路径成本 `c_best` 决定。具体来说，椭圆上任意一点到两个焦点（起点和终点）的距离之和正好等于 `c_best`。
    *   **意义**: 根据椭圆的定义，所有能够构成**比 `c_best` 更短路径**的点，**必然**存在于这个椭圆内部。任何在椭圆外的点，其“起点-该点-终点”的路径长度必定大于 `c_best`，因此对寻找更优解毫无意义。
*   **效率**：通过将采样区域限制在这个有意义的椭圆内，Informed RRT\* 将其全部算力都集中在了“精炼”和“优化”已有路径上。每当它在椭圆内找到一条更短的路径时，`c_best` 就会减小，**椭圆区域也会随之收缩**，进一步聚焦搜索。这使得它收敛到最优解的速度比 RRT\* 快得多。

**总结：**
*   **RRT\***：大海捞针，永不放弃整片大海。
*   **Informed RRT\***：先大海捞针找到一根，然后根据这根针的长度画一个圈，以后只在这个圈里捞，而且圈会越捞越小。
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle, Ellipse
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
STEP_SIZE = 8.0
MAX_ITER = 1000
RADIUS = 15.0
GOAL_SAMPLE_RATE = 5

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# --- 核心算法函数 ---
def distance(node1, node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

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

def find_near_nodes(tree, new_node, radius):
    near_nodes = [node for node in tree if distance(node, new_node) <= radius]
    return near_nodes

def choose_parent(near_nodes, nearest_node, new_node, obstacles):
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
    for near_node in near_nodes:
        if near_node != new_node.parent and is_path_collision_free(new_node, near_node, obstacles):
            new_cost = new_node.cost + distance(new_node, near_node)
            if new_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_cost
    return tree

def get_path(node):
    path = []
    curr = node
    while curr is not None:
        path.append((curr.x, curr.y))
        curr = curr.parent
    path.reverse()
    return path

# --- Informed RRT* 特有函数 ---
def sample_from_ellipse(start_node, goal_node, c_best):
    c_min = distance(start_node, goal_node)
    center_x, center_y = (start_node.x + goal_node.x) / 2, (start_node.y + goal_node.y) / 2
    
    # Rotation
    angle = np.arctan2(goal_node.y - start_node.y, goal_node.x - start_node.x)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    C = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Axes
    a = c_best / 2
    b = np.sqrt(c_best**2 - c_min**2) / 2
    
    # Sample from unit circle
    r = np.sqrt(random.random())
    theta = random.uniform(0, 2 * np.pi)
    x, y = r * np.cos(theta), r * np.sin(theta)
    
    # Scale, Rotate, and Translate
    p = np.array([a * x, b * y])
    p = C @ p + np.array([center_x, center_y])
    
    return Node(p[0], p[1])

# --- PyQt5 GUI 类 ---
class InformedRRTStarGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Informed RRT* Step-by-Step")
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.next_step_button = QtWidgets.QPushButton("Next Step")
        self.next_step_button.clicked.connect(self.next_step)
        self.auto_run_button = QtWidgets.QPushButton("Auto Run")
        self.auto_run_button.clicked.connect(self.toggle_auto_run)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.button_layout.addWidget(self.next_step_button)
        self.button_layout.addWidget(self.auto_run_button)
        self.button_layout.addWidget(self.reset_button)
        self.layout.addLayout(self.button_layout)
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_step)
        self.is_auto_running = False
        self.reset_simulation()
        
    def reset_simulation(self):
        if self.is_auto_running: self.toggle_auto_run()
        self.start_node = Node(START_POS[0], START_POS[1])
        self.goal_node = Node(GOAL_POS[0], GOAL_POS[1])
        self.tree = [self.start_node]
        self.final_path = None
        self.best_path_cost = float('inf')
        self.iteration = 0
        self.next_step_button.setEnabled(True)
        self.auto_run_button.setEnabled(True)
        self.plot_scenario()
        
    def toggle_auto_run(self):
        self.is_auto_running = not self.is_auto_running
        if self.is_auto_running:
            self.auto_run_button.setText("Stop")
            self.next_step_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.timer.start(50)
        else:
            self.timer.stop()
            self.auto_run_button.setText("Auto Run")
            self.next_step_button.setEnabled(True)
            self.reset_button.setEnabled(True)

    def next_step(self):
        if self.iteration >= MAX_ITER:
            if self.is_auto_running: self.toggle_auto_run()
            self.next_step_button.setEnabled(False)
            print("Max iterations reached.")
            return

        self.iteration += 1
        
        # --- Informed RRT* Sampling ---
        if self.best_path_cost == float('inf'):
            # Stage 1: Global sampling
            if random.randint(0, 100) > (100 - GOAL_SAMPLE_RATE):
                q_rand = Node(GOAL_POS[0], GOAL_POS[1])
            else:
                q_rand = Node(random.uniform(0, X_DIM), random.uniform(0, Y_DIM))
        else:
            # Stage 2: Ellipsoidal sampling
            q_rand = sample_from_ellipse(self.start_node, self.goal_node, self.best_path_cost)
        
        nearest_node = get_nearest_node(self.tree, q_rand)
        q_new = steer(nearest_node, q_rand, STEP_SIZE)
        
        plot_elements = {'q_rand': q_rand}

        if is_path_collision_free(nearest_node, q_new, OBSTACLES):
            near_nodes = find_near_nodes(self.tree, q_new, RADIUS)
            q_new = choose_parent(near_nodes, nearest_node, q_new, OBSTACLES)
            self.tree.append(q_new)
            self.tree = rewire(self.tree, q_new, near_nodes, OBSTACLES)
            
            # Check for new, better path to goal
            if distance(q_new, self.goal_node) < STEP_SIZE:
                path_cost = q_new.cost + distance(q_new, self.goal_node)
                if path_cost < self.best_path_cost:
                    self.best_path_cost = path_cost
                    self.final_path = get_path(q_new) + [(self.goal_node.x, self.goal_node.y)]

            plot_elements['q_new'] = q_new
            plot_elements['near_nodes_radius_center'] = q_new

        self.plot_scenario(plot_elements)
        
    def plot_scenario(self, elements=None):
        self.ax.clear()
        if elements is None: elements = {}

        for (ox, oy, w, h) in OBSTACLES: self.ax.add_patch(Rectangle((ox, oy), w, h, facecolor='gray'))
        for node in self.tree:
            if node.parent: self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b", alpha=0.5)

        # Draw informed ellipse if a path has been found
        if self.best_path_cost != float('inf'):
            c_min = distance(self.start_node, self.goal_node)
            center_x = (self.start_node.x + self.goal_node.x) / 2
            center_y = (self.start_node.y + self.goal_node.y) / 2
            width = self.best_path_cost
            height = np.sqrt(self.best_path_cost**2 - c_min**2)
            angle_deg = np.rad2deg(np.arctan2(self.goal_node.y - self.start_node.y, self.goal_node.x - self.start_node.x))
            ellipse = Ellipse(xy=(center_x, center_y), width=width, height=height, angle=angle_deg,
                              edgecolor='r', fc='None', lw=1.5, linestyle='--', label="Informed Set")
            self.ax.add_patch(ellipse)

        if 'q_rand' in elements: self.ax.plot(elements['q_rand'].x, elements['q_rand'].y, "ko", markersize=3, label='q_rand')
        if 'q_new' in elements: self.ax.plot(elements['q_new'].x, elements['q_new'].y, 'cx', markersize=7, label='q_new')
        if 'near_nodes_radius_center' in elements:
            center = elements['near_nodes_radius_center']
            self.ax.add_patch(Circle((center.x, center.y), RADIUS, color='y', fill=False, linestyle='--', label=f'Rewire Radius'))

        if self.final_path:
            path_x, path_y = zip(*self.final_path)
            self.ax.plot(path_x, path_y, 'g-', linewidth=2.5, label='Best Path')

        self.ax.plot(START_POS[0], START_POS[1], "go", markersize=10, label='Start')
        self.ax.plot(GOAL_POS[0], GOAL_POS[1], "ro", markersize=10, label='Goal')

        self.ax.set_xlim(0, X_DIM)
        self.ax.set_ylim(0, Y_DIM)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title(f"Informed RRT* Iteration: {self.iteration} | Best Cost: {self.best_path_cost:.2f}")
        
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='lower right', prop={'size': 7})
        self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = InformedRRTStarGUI()
    main_window.show()
    sys.exit(app.exec_()) 