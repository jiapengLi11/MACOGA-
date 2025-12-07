# ================== ga.py ==================
import numpy as np
from aco import ImprovedACO  # 新增：导入 ImprovedACO 类


class EnhancedGA:
    def __init__(self, env, pop_size=30, max_iter=10, k=0.5):
        """
           初始化函数
           参数:
           env: 环境对象，用于执行算法的环境
           pop_size: 种群大小，默认为30，代表种群中个体的数量
           max_iter: 最大迭代次数，默认为10，算法停止的条件之一
           k: 式13参数，算法中的一个重要参数，用于计算适应度或其他关键指标
           """
        self.env = env
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.k = k  # 式13参数

    def optimize(self, initial_paths):
        """GA优化主循环（论文III-C节）"""
        population = self._initialize_population(initial_paths)

        for iter in range(self.max_iter):
            # 计算适应度
            fitness = np.array([self._calculate_fitness(path) for path in population])

            # 选择（式13）
            selected = self._selection(population, fitness)

            # 交叉（式12）
            crossed = self._crossover(selected)

            # 变异（论文III-C.3）
            mutated = self._mutation(crossed)

            population = mutated
            print(f"GA Iteration {iter + 1}: Max Fitness={np.max(fitness):.4f}")

        return max(population, key=lambda x: self._calculate_fitness(x))

    def _initialize_population(self, initial_paths):
        """初始化种群"""
        population = []
        for path in initial_paths:
            if path:
                population.append(path)
            else:
                # 如果路径为空，随机生成一个路径
                population.append(self._random_path())
        return population

    def _random_path(self):  # _initialize_population中若初始路径为空，随机生成路径可能不可达终点添加可达性检查：
        path = [self.env.start]
        current = self.env.start
        visited = set()
        while current != self.env.goal:
            neighbors = [n for n in self.env.get_neighbors(current) if n not in visited]
            if not neighbors:
                break
            current = neighbors[np.random.randint(len(neighbors))]
            visited.add(current)
            path.append(current)
        return path

    def _calculate_fitness(self, path):
        """适应度计算（论文式5）"""
        if not path or path[-1] != self.env.goal:
            return 0.0

        # 路径长度
        length = sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                     for i in range(len(path) - 1))

        # 路径平滑度
        smoothness = 0.0
        for i in range(len(path) - 2):
            a = np.array(path[i])
            b = np.array(path[i + 1])
            c = np.array(path[i + 2])
            angle = np.arccos(
                np.dot(b - a, c - b) /
                (np.linalg.norm(b - a) * np.linalg.norm(c - b) + 1e-6)
            )
            smoothness += angle
        return 1.0 / (length + 1e-6) + 1.0 / (smoothness + 1e-6)  # 适应度计算式5

    def _selection(self, population, fitness):
        """
        执行自适应选择操作（参考论文公式13）。

        本函数的目的是基于个体的适应度进行选择操作，以生成新一代种群。
        选择概率与个体的相对适应度成正比，适应度高的个体有更高的概率被选中。

        参数:
        - population: list, 当前种群的所有个体。
        - fitness: list or numpy array, 种群中每个个体的适应度值。

        返回值:
        - list, 通过自适应选择操作选出的新一代种群个体列表。
        """
        # 计算每个个体的选择概率，基于其适应度值
        exp_fit = np.exp(self.k * (fitness - np.max(fitness)))
        probs = exp_fit / np.sum(exp_fit)

        # 基于计算出的选择概率，随机选择新一代个体
        selected_indices = np.random.choice(
            len(population),
            size=self.pop_size,
            p=probs
        )

        # 返回选中的个体组成的新一代种群
        return [population[i] for i in selected_indices]

    def _crossover(self, population):
        """
        自适应交叉（论文III-C.2）

        对给定种群执行自适应交叉操作，生成新的种群。

        参数:
        population (list of list): 当前种群，每个元素是一个个体（路径）。

        返回:
        list of list: 交叉后的新种群。
        """
        # 初始化新一代种群列表
        new_pop = []
        # 遍历当前种群，每两个个体进行交叉操作
        for i in range(0, len(population) - 1, 2):
            # 选取两个父代个体
            p1, p2 = population[i], population[i + 1]

            # 寻找两个父代个体的共同元素
            common = list(set(p1) & set(p2))
            # 如果共同元素的数量不少于3，进行交叉操作
            if len(common) >= 3:
                # 随机选择一个共同元素作为交叉点
                cp = common[np.random.randint(1, len(common) - 1)]
                # 获取交叉点在两个父代个体中的索引
                idx1 = p1.index(cp)
                idx2 = p2.index(cp)
                # 生成新个体：交叉点前的部分来自第一个父代，后部分来自第二个父代
                child = p1[:idx1] + p2[idx2:]
                # 将新个体加入新一代种群
                new_pop.append(child)
            else:
                # 如果共同元素数量不足3，则不进行交叉，直接将父代个体加入新一代种群
                new_pop.extend([p1, p2])
        # 返回新一代种群
        return new_pop

    def _mutation(self, population):
        """基于ACO的变异（论文III-C.3）

        对种群中的每个个体，以一定概率对其进行变异操作。变异是通过选择个体的一部分路径，
        并使用改进的蚁群算法（ImprovedACO）重新规划这部分路径来实现的。

        参数:
        population (list): 包含多个路径的种群

        返回:
        list: 经过变异操作后的种群
        """
        # 遍历种群中的每个个体
        for i in range(len(population)):
            # 以一定概率进行变异操作
            if np.random.rand() < 0.3 and len(population[i]) > 3:
                # 随机选择变异段
                start = np.random.randint(0, len(population[i]) - 2)
                end = np.random.randint(start + 1, len(population[i]))

                # 使用ACO重新规划
                sub_aco = ImprovedACO(self.env, max_iter=5)
                new_segment = sub_aco.find_path(
                    population[i][start],
                    population[i][end]
                )

                # 如果有更优路径，则替换原路径
                if new_segment:
                    population[i] = (
                            population[i][:start] +
                            new_segment +
                            population[i][end + 1:]
                    )
        return population
