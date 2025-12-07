import numpy as np


class GridEnvironment:
    def __init__(self, size=(15, 15), obstacle_ratio=0.23):  # 初始化函数，设置网格大小和障碍物比例（可改30*30）
        """
        初始化网格世界环境。

        参数:
        size -- 网格的大小，默认为15x15。可考虑修改为30x30以增加复杂度。
        obstacle_ratio -- 障碍物的比例，默认为0.23，即网格中约23%的位置将是障碍物。

        无返回值。
        """
        self.size = size  # 网格大小
        self.grid = np.zeros(size, dtype=int)  # 初始化一个全零的网格，无障碍物
        self.start = (0, 0)  # 设置起点位置
        self.goal = (size[0] - 1, size[1] - 1)  # 设置目标位置，位于网格的右下角
        self._generate_obstacles(obstacle_ratio)  # 根据障碍物比例生成障碍物
        self._ensure_accessibility()  # 确保所有位置都是可到达的，维护环境的可达性

    def _generate_obstacles(self, ratio):
        """
        生成障碍物并确保起点终点畅通

        此方法在网格中随机生成指定比例的障碍物，同时保证起点和终点不会被障碍物阻塞

        参数:
        ratio (float): 障碍物的比例，表示障碍物数量占总网格数量的比例

        返回值:
        无返回值，直接修改网格(self.grid)中的障碍物分布
        """
        # 确保起点和终点不被障碍物占用
        free_cells = [self.start, self.goal]
        # 计算网格中总单元格数
        total_cells = self.size[0] * self.size[1]
        # 随机选择将要设置为障碍物的单元格索引，避开起点和终点
        indices = np.random.choice(
            [i for i in range(total_cells) if i not in
             [self._coord_to_index(p) for p in free_cells]],
            int(total_cells * ratio),
            replace=False
        )
        # 在网格中设置选中的单元格为障碍物
        self.grid.flat[indices] = 1

    def _coord_to_index(self, point):
        """
        坐标转一维索引

        该方法将二维坐标转换为一维索引，适用于将二维空间中的点映射到一维数组或列表中。

        参数:
        point (tuple): 一个元组，表示二维平面上的点，形式为 (x, y)。

        返回:
        int: 计算得到的一维索引值。
        """
        return point[0] * self.size[1] + point[1]

    def _ensure_accessibility(self):
        """BFS确保起点终点可达

        通过广度优先搜索算法确保地图上的起点和终点之间存在至少一条可达路径。
        如果搜索后发现终点不可达，则会重新生成地图并再次检查可达性。
        """
        from collections import deque
        visited = set()
        queue = deque([self.start])

        while queue:
            current = queue.popleft()
            if current == self.goal:
                return
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)

        # 如果不可达则重新生成
        self.grid = np.zeros_like(self.grid)
        self._generate_obstacles(0.23)
        self._ensure_accessibility()

    def get_neighbors(self, node):
        """获取8邻域可行节点

        Args:
            node (tuple): 当前节点的坐标，为一个(x, y)的元组

        Returns:
            list: 返回一个列表，包含所有可行的邻居节点坐标，每个元素也是一个(x, y)的元组
        """
        # 当前节点的坐标
        x, y = node
        # 初始化邻居节点列表
        neighbors = []
        # 遍历所有可能的8邻域方向
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # 跳过节点本身
                if dx == 0 and dy == 0:
                    continue
                # 计算邻居节点的坐标
                nx, ny = x + dx, y + dy
                # 检查邻居节点是否在网格范围内
                if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                    # 检查邻居节点是否为可行节点（无障碍物）
                    if self.grid[nx, ny] == 0:
                        # 将可行的邻居节点添加到列表中
                        neighbors.append((nx, ny))
        # 返回所有可行的邻居节点
        return neighbors

    def is_collision(self, p1, p2):
        """
        Bresenham直线碰撞检测

        此函数用于检测两点之间连线是否与网格中的障碍物碰撞
        参数:
            p1: 起点坐标 (x1, y1)
            p2: 终点坐标 (x2, y2)
        返回值:
            如果两点之间的连线与障碍物碰撞，则返回True，否则返回False
        """
        # 解包起点和终点坐标
        x1, y1 = p1
        x2, y2 = p2

        # 计算x和y方向上的增量
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # 确定x和y方向上的步长
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        # 初始化误差
        err = dx - dy

        # 遍历两点之间的所有点
        while True:
            # 检查当前点是否为障碍物
            if self.grid[x1, y1] == 1:
                return True
            # 检查是否达到终点
            if x1 == x2 and y1 == y2:
                break
            # 更新误差和坐标
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        # 如果没有遇到障碍物，则返回False
        return False
