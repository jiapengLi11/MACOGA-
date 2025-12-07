import numpy as np


class ImprovedACO:
    def __init__(self, env, alpha=6, beta=2, rho=0.1, ants=30, max_iter=10):  # 论文表III参数
        self.env = env  # 环境对象
        self.alpha = alpha  # 信息素重要程度因子
        self.beta = beta  # 启发式信息因子
        self.rho = rho  # 信息素蒸发率（信息素随时间减少）
        self.ants = ants  # 蚂蚁数量
        self.max_iter = max_iter  # 最大迭代次数
        self.best_path = None  # 初始化 best_path
        self.best_length = float('inf')  # 初始化 best_length
        env.rows, env.cols = env.size  # 解包元组
        self.pheromone = np.ones((env.rows, env.cols)) * 0.1  # 初始信息素
        # self.pheromone = np.ones(env.size) * 0.1  # 初始信息素(应该是一个二维数组表示网格环境中的信息素分布)

    def find_path(self, start, goal):
        """
        ACO主算法（论文III-B节）

        该函数通过模拟蚂蚁群落的行为来寻找从起点到目标点的最短路径。

        参数:
        - start: 起点
        - goal: 目标点

        返回:
        - self.best_path: 最优路径
        """
        # 初始化最优路径和最优路径长度
        self.best_path = None
        self.best_length = float('inf')

        # 开始迭代寻找最优路径
        for iteration in range(self.max_iter):
            # 每次迭代开始时清空路径列表
            paths = []
            # 每只蚂蚁构建一条路径
            for _ in range(self.ants):
                path = self._construct_path(start, goal)
                # 如果构建的路径有效且到达目标点，则将其添加到路径列表中
                if path and path[-1] == goal:
                    paths.append(path)
                    # 如果该路径更短，则更新最优路径和最优路径长度
                    if len(path) < self.best_length:
                        self.best_length = len(path)
                        self.best_path = path

            # 根据蚂蚁们构建的路径更新信息素
            self._update_pheromone(paths)  # 式(3)
            # 打印每次迭代后的最优路径长度
            print(f"ACO Iteration {iteration + 1}: Best Length={self.best_length}")

        # 返回最优路径
        return self.best_path

    def _construct_path(self, start, goal):
        """单只蚂蚁路径构建（论文式7-11）"""
        path = [start]
        visited = set([start])
        max_steps = self.env.size[0] * 2  # 防无限循环

        while path[-1] != goal and len(path) < max_steps:
            current = path[-1]
            neighbors = [n for n in self.env.get_neighbors(current) if n not in visited]

            if not neighbors:
                break  # 无可用节点

            # 式(7)概率计算
            probs = self._transition_probability(current, neighbors, goal)
            # next_node = neighbors[np.argmax(np.random.multinomial(1, probs))]#可能导致选择偏差
            next_node = neighbors[np.random.choice(len(neighbors), p=probs)]  # 直接按概率选择
            path.append(next_node)
            visited.add(next_node)

        return path if path[-1] == goal else None

    def _transition_probability(self, current, neighbors, goal):
        """论文式7的概率计算-选择下一个节点的概率"""
        prob_values = []
        total = 0.0
        n_pre_total = sum(len(self.env.get_neighbors(n)) for n in neighbors)  # 式9分母

        for n in neighbors:
            tau = self.pheromone[n[0], n[1]]
            heuristic = 1 / (np.linalg.norm(np.array(n) - np.array(goal)) + 1e-6)
            n_pre = len(self.env.get_neighbors(n))
            gamma = n_pre_total / (n_pre + 1e-6)  # 式(9)，分母加1e-6防止除零（me改进）

            prob = (tau ** self.alpha) * (heuristic ** self.beta) * (1 / gamma)
            prob_values.append(prob)
            total += prob

        # 归一化处理
        if total == 0:
            return np.ones(len(neighbors)) / len(neighbors)
        return [p / total for p in prob_values]

    def _update_pheromone(self, valid_paths):
        """信息素更新（论文式3）"""
        # 信息素蒸发
        self.pheromone *= (1 - self.rho)

        # 信息素沉积
        for path in valid_paths:
            if not path:
                continue
            delta = 10.0 / len(path)  # 论文说与路径长度成反比，增强有效路径的信息素1改成10（me改进）
            for node in path:
                self.pheromone[node[0], node[1]] += delta

        self.pheromone = np.clip(self.pheromone, 0.01, 1.0)  # 限制信息素范围（me改进）
