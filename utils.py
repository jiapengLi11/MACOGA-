import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_environment(env, path=None, pheromone=None, title=None):
    fig = plt.figure(figsize=(8, 8))

    # 绘制网格
    for i in range(env.size[0]):
        for j in range(env.size[1]):
            if env.grid[i, j] == 1:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='black'
                )
                plt.gca().add_patch(rect)

    # 绘制信息素
    if pheromone is not None:
        plt.imshow(pheromone.T, cmap='hot', alpha=0.5,
                   extent=(-0.5, env.size[1] - 0.5, env.size[0] - 0.5, -0.5))
        plt.colorbar(label='Pheromone Intensity')

    # 绘制路径
    if path:
        xs, ys = zip(*path)
        plt.plot(ys, xs, 'b-o', linewidth=2, markersize=8, zorder=3)  # 提高路径图层级

    # 标记起终点
    plt.scatter(env.start[1], env.start[0], c='green', s=200,
                marker='s', edgecolor='black', label='Start')
    plt.scatter(env.goal[1], env.goal[0], c='red', s=200,
                marker='s', edgecolor='black', label='Goal')

    plt.xlim(-0.5, env.size[1] - 0.5)
    plt.ylim(env.size[0] - 0.5, -0.5)
    plt.xticks(range(env.size[1]))
    plt.yticks(range(env.size[0]))
    plt.grid(True)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    return fig
