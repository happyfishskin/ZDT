import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

# 設定三個 ZDT 問題
problem_settings = {
    "ZDT1": {"name": "zdt1", "n_var": 30},  # 30 維
    "ZDT4": {"name": "zdt4", "n_var": 10},  # 10 維
    "ZDT6": {"name": "zdt6", "n_var": 10},  # 10 維
}

# 限制參數
max_evaluations = 25000
population_size = 100
generations = max_evaluations // population_size  # 計算最大世代數

# 儲存結果
results = {}

# 解決問題並儲存結果
for problem_name, settings in problem_settings.items():
    print(f"正在執行 {problem_name}...")

    # 加載 ZDT 問題
    problem = get_problem(settings["name"], n_var=settings["n_var"])

    # 設置 NSGA-II 演算法
    algorithm = NSGA2(pop_size=population_size)

    # 執行優化
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=42,  # 固定隨機種子，保證可重現
        save_history=True,
        verbose=True,
    )

    # 儲存結果
    results[problem_name] = res

    # 可視化 Pareto 前沿
    plt.figure(figsize=(8, 6))
    plt.scatter(res.F[:, 0], res.F[:, 1], label="NSGA-II Solutions", color="blue")
    if problem.pareto_front() is not None:
        plt.scatter(
            problem.pareto_front()[:, 0],
            problem.pareto_front()[:, 1],
            label="True Pareto Front",
            edgecolor="red",
            facecolor="none",
        )
    plt.title(f"Pareto Front for {problem_name}")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{problem_name}_pareto_front.png")  # 保存圖表
    plt.show()

# 總結結果
for problem_name, res in results.items():
    print(f"=== {problem_name} 結果總結 ===")
    print(f"目標函數值 (F): {res.F.shape}")
    print(f"決策變數值 (X): {res.X.shape}")
    print(f"總函數評估次數: {res.algorithm.evaluator.n_eval}")
