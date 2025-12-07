class PathSimplifier:
    def __init__(self, env):
        self.env = env

    def simplify(self, path):
        """路径简化（论文III-C.4）"""
        if len(path) < 3:
            return path

        # 步骤1：删除冗余节点
        simplified = self._remove_redundant_nodes(path)

        # 步骤2：路径重连
        optimized = self._path_reconnection(simplified)

        return optimized

    def _remove_redundant_nodes(self, path):
        """
        删除共线冗余节点
        本函数旨在移除路径中不必要的节点，即那些与相邻节点共线的节点，以简化路径
        保持路径的起点和终点不变，仅移除中间的冗余共线节点

        参数:
        path (list of tuples): 代表一系列坐标点的路径，每个元素是一个坐标点（x, y）

        返回:
        list of tuples: 简化后的路径，移除了冗余的共线节点
        """
        # 初始化简化后的路径列表，首先包含路径的起点
        simplified = [path[0]]
        # 遍历路径中的每个节点，除了起点和终点
        for i in range(1, len(path) - 1):
            # 检查当前节点与上一个和下一个节点是否共线
            if not self._is_collinear(simplified[-1], path[i], path[i + 1]):
                # 如果不共线，则将当前节点添加到简化后的路径中
                simplified.append(path[i])
        # 最后添加路径的终点到简化后的路径中
        simplified.append(path[-1])
        # 返回简化后的路径
        return simplified

    def _is_collinear(self, p1, p2, p3):
        """三点共线性检查

        通过计算三点构成的三角形面积，判断三点是否共线。
        共线的三点构成的三角形面积为0，或非常接近0（考虑浮点数精度问题）。

        Parameters:
        p1, p2, p3 (tuple): 三个点的坐标，每个点的坐标为一个二元组，如 (x, y)。

        Returns:
        bool: 如果三点共线，返回 True；否则返回 False。
        """
        # 计算三点构成的三角形面积，使用向量叉乘的方法
        area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        # 判断面积是否接近于0，考虑浮点数精度问题，使用一个非常小的阈值进行比较
        return abs(area) < 1e-6

    def _path_reconnection(self, path):
        """
        优化给定路径，通过尝试在不违反环境约束的条件下删除中间点来减少路径长度。

        此方法创建了输入路径的一个副本，在不引起索引错误的情况下对其进行修改。

        参数:
        - path: 需要优化的原始路径，是一个节点列表。

        返回:
        - optimized: 优化后的路径，中间点尽可能被移除以减少路径长度。
        """
        # 创建路径的副本以避免修改原始路径，直接修改可能会导致索引错乱
        optimized = path.copy()
        i = 0
        # 遍历路径中的点，直到倒数第三个点，因为我们需要检查i和i+2两点之间的连接
        while i < len(optimized) - 2:
            # 如果当前点和下两个点之间的连线没有和障碍物碰撞，则删除中间的点
            if not self.env.is_collision(optimized[i], optimized[i + 2]):
                del optimized[i + 1]
            else:
                # 如果有碰撞，则移动到下一个点
                i += 1
        # 返回优化后的路径
        return optimized
