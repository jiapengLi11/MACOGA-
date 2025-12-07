from matplotlib import pyplot as plt, patches
from environment import GridEnvironment
from aco import ImprovedACO
from ga import EnhancedGA
from path_simplification import PathSimplifier
from utils import plot_environment
import os


def save_plot(figure, filename):
    # 保存图表到results
    if not os.path.exists('results'):
        os.makedirs("results")
    if figure is None:
        print(f"Warning: Figure is None, cannot save plot '{filename}'.")
        return
    figure.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')  # 高分辨率dpi=300
    plt.close(figure)


def main():
    # 初始化环境
    env = GridEnvironment(size=(15, 15), obstacle_ratio=0.23)  # 论文为15*15，30*30，0.23为障碍比例
    print("=== Initial Environment ===")
    fig = plot_environment(env, title="Initial Environment (15x15)")
    if fig is not None:
        save_plot(fig, "01_initial_environment.png")

    # ACO阶段
    print("\n=== ACO Phase ===")
    aco = ImprovedACO(env)
    aco_path = aco.find_path(env.start, env.goal)
    if aco_path:
        print(f"ACO Found Path Length: {len(aco_path)}")
        fig = plot_environment(env, path=aco_path, pheromone=aco.pheromone,
                               title="ACO Result with Pheromone Map")
        if fig is not None:
            save_plot(fig, "02_aco_result.png")
    else:
        print("ACO failed to find path!")
        return

    # GA阶段
    print("\n=== GA Phase ===")
    ga = EnhancedGA(env)
    # 将ACO路径作为初始种群的一部分传递给GA
    ga_path = ga.optimize([aco_path] * 30)  # 用ACO路径初始化种群
    if ga_path:
        print(f"GA Optimized Length: {len(ga_path)}")
        fig = plot_environment(env, path=ga_path, title="GA Optimized Path")
        if fig is not None:
            save_plot(fig, "03_ga_optimized.png")
    else:
        print("GA optimization failed!")
        return

    # 路径简化
    print("\n=== Path Simplification ===")
    simplifier = PathSimplifier(env)
    final_path = simplifier.simplify(ga_path)
    if final_path:
        print(f"Final Path Length: {len(final_path)}")
        fig = plot_environment(env, path=final_path, title="Final Simplified Path")
        if fig is not None:
            save_plot(fig, "04_final_simplified.png")

    # 对比展示
    if aco_path and ga_path and final_path:
        fig = plt.figure(figsize=(10, 6))
        plt.title("Path Comparison")

        # === 手动绘制环境 ===
        # 绘制障碍物
        for i in range(env.size[0]):
            for j in range(env.size[1]):
                if env.grid[i, j] == 1:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor='black',
                        edgecolor='black'
                    )
                    plt.gca().add_patch(rect)

        # 绘制起终点
        plt.scatter(env.start[1], env.start[0], c='green', s=200,
                    marker='s', edgecolor='black', label='Start')
        plt.scatter(env.goal[1], env.goal[0], c='red', s=200,
                    marker='s', edgecolor='black', label='Goal')

        # === 绘制路径 ===
        xs, ys = zip(*aco_path)
        plt.plot(ys, xs, 'b--', linewidth=3, label='ACO Path')
        xs, ys = zip(*ga_path)
        plt.plot(ys, xs, 'g-.', linewidth=2, label='GA Path')
        xs, ys = zip(*final_path)
        plt.plot(ys, xs, 'r-', linewidth=1.5, label='Final Path')

        # === 设置坐标 ===
        plt.xlim(-0.5, env.size[1] - 0.5)
        plt.ylim(env.size[0] - 0.5, -0.5)
        plt.xticks(range(env.size[1]))
        plt.yticks(range(env.size[0]))
        plt.grid(True)
        plt.legend()

        save_plot(fig, "05_path_comparison.png")
        plt.close(fig)
        plt.show()


if __name__ == "__main__":
    main()
