import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条，方便查看迭代过程

# -------------------------- 1. 核心工具函数（针对Lasso优化） --------------------------
def lasso_objective(X, y, beta, lambda_):
    """
    计算Lasso目标函数值（核心：带L1正则项的最小二乘损失）
    """
    n = X.shape[0]
    loss = 0.5 * np.linalg.norm(X @ beta - y, 2) ** 2 / n
    reg = lambda_ * np.linalg.norm(beta, 1)
    return loss + reg

def soft_threshold(z, t):
    """
    软阈值算子（PGD/AGD/坐标下降/ADMM的核心，求解L1正则项闭式解）
    """
    return np.sign(z) * np.maximum(np.abs(z) - t, 0)

def least_squares_gradient(X, y, beta):
    """
    计算最小二乘损失的梯度（光滑部分，用于各类梯度类算法）
    """
    n = X.shape[0]
    return X.T @ (X @ beta - y) / n

def l1_subgradient(beta):
    """
    计算L1范数的次梯度（次梯度下降的核心，适配非光滑特性）
    """
    subgrad = np.sign(beta)
    subgrad[beta == 0] = 0  # beta=0时次梯度取0，简化计算
    return subgrad

# -------------------------- 2. 8种Lasso有效求解算法实现 --------------------------
def subgradient_descent_lasso(X, y, lambda_, lr=0.01, max_iter=10000, tol=1e-6):
    """
    次梯度下降（Subgradient Descent）：天然适配L1非光滑目标
    """
    n, p = X.shape
    beta = np.zeros(p)
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        # 总次梯度 = 最小二乘梯度 + L1次梯度
        grad_ls = least_squares_gradient(X, y, beta)
        subgrad_l1 = l1_subgradient(beta)
        total_subgrad = grad_ls + lambda_ * subgrad_l1
        # 次梯度更新
        beta = beta - lr * total_subgrad
        # 计算目标函数值并判断收敛
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def proximal_gradient_descent_lasso(X, y, lambda_, lr=0.01, max_iter=10000, tol=1e-6):
    """
    近端梯度下降（PGD）：Lasso核心求解算法，软阈值算子处理L1项
    """
    n, p = X.shape
    beta = np.zeros(p)
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        # 步骤1：光滑部分梯度下降
        grad = least_squares_gradient(X, y, beta)
        beta_temp = beta - lr * grad
        # 步骤2：近端算子（软阈值）处理L1正则项
        beta = soft_threshold(beta_temp, lambda_ * lr)
        # 收敛判断
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def accelerated_gradient_descent_lasso(X, y, lambda_, lr=0.01, max_iter=10000, tol=1e-6):
    """
    加速梯度下降（AGD/Nesterov）：PGD加速版，收敛速度更快
    """
    n, p = X.shape
    beta = np.zeros(p)
    beta_prev = np.zeros(p)
    t = 1.0  # 加速系数
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        # Nesterov加速辅助变量
        beta_aux = beta + ((t - 1) / (t + 1)) * (beta - beta_prev)
        # 光滑部分梯度下降
        grad = least_squares_gradient(X, y, beta_aux)
        beta_temp = beta_aux - lr * grad
        # 软阈值更新
        beta_prev = beta.copy()
        beta = soft_threshold(beta_temp, lambda_ * lr)
        # 更新加速系数
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        t = t_new
        # 收敛判断
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def coordinate_descent_lasso(X, y, lambda_, max_iter=10000, tol=1e-6):
    """
    坐标下降（CD）：逐维度闭式解，高维Lasso（n<p）最优算法之一
    """
    n, p = X.shape
    beta = np.zeros(p)
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        obj_val_prev = obj_vals[-1]
        # 逐坐标更新beta，每个维度求闭式解
        for j in range(p):
            # 排除第j个特征的残差
            res = y - X @ beta + X[:, j] * beta[j]
            x_j = X[:, j]
            sum_xj_res = np.dot(x_j, res) / n
            sum_xj_sq = np.dot(x_j, x_j) / n
            # 软阈值求解第j个维度
            beta[j] = soft_threshold(sum_xj_res, lambda_) / sum_xj_sq if sum_xj_sq != 0 else 0
        # 收敛判断
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_val - obj_val_prev) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def admm_lasso(X, y, lambda_, rho=1.0, alpha=1.0, max_iter=10000, tol=1e-6):
    """
    交替方向乘子法（ADMM）：分布式友好，Lasso求解稳定性强
    """
    n, p = X.shape
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    # 预处理：计算闭式解所需的逆矩阵
    XTX = X.T @ X
    inv_mat = np.linalg.inv(XTX + rho * np.eye(p))
    XTy = X.T @ y

    for _ in range(max_iter):
        # Step1: 更新beta（最小二乘闭式解）
        beta = inv_mat @ (XTy + rho * (z - u))
        # Step2: 更新z（L1正则项，软阈值）
        z = soft_threshold(alpha * beta + u, lambda_ / rho)
        # Step3: 更新u（对偶变量）
        u = u + alpha * beta - z
        # 收敛判断
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def stochastic_gradient_descent_lasso(X, y, lambda_, lr=0.001, batch_size=1, max_iter=10000, tol=1e-6):
    """
    随机梯度下降（SGD）：大规模数据Lasso求解，降低单次迭代计算量
    """
    n, p = X.shape
    beta = np.zeros(p)
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        # 随机选取批次样本
        idx = np.random.choice(n, batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        # 批次梯度计算
        grad = X_batch.T @ (X_batch @ beta - y_batch) / batch_size
        # SGD更新（处理L1正则项）
        beta = beta - lr * (grad + lambda_ * np.sign(beta))
        # 收敛判断（用全量样本评估目标函数）
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def svrg_lasso(X, y, lambda_, lr=0.001, m=100, max_iter=10000, tol=1e-6):
    """
    随机方差缩减（SVRG）：降方差方法，解决SGD波动大的问题
    """
    n, p = X.shape
    beta = np.zeros(p)
    beta_hat = np.zeros(p)
    obj_vals = [lasso_objective(X, y, beta, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        # 计算全局梯度（方差缩减核心）
        grad_hat = least_squares_gradient(X, y, beta_hat)
        # 内层m次随机迭代
        for _ in range(m):
            idx = np.random.choice(n)
            X_i = X[idx:idx+1]
            y_i = y[idx:idx+1]
            # 个体梯度与全局梯度结合，降低方差
            grad_i = X_i.T @ (X_i @ beta - y_i)
            grad_i_hat = X_i.T @ (X_i @ beta_hat - y_i)
            beta = beta - lr * (grad_i - grad_i_hat + grad_hat + lambda_ * np.sign(beta))
        # 更新beta_hat
        beta_hat = beta.copy()
        # 收敛判断
        obj_val = lasso_objective(X, y, beta, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta, obj_vals, total_time, len(obj_vals)-1

def adam_lasso(X, y, lambda_, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=10000, tol=1e-6):
    """
    Adam算法：动量+自适应学习率，Lasso求解鲁棒性强，无需手动调参
    """
    n, p = X.shape
    beta_coef = np.zeros(p)  # 避免与Adam参数beta1/beta2重名
    m_t = np.zeros(p)  # 一阶动量
    v_t = np.zeros(p)  # 二阶动量
    t = 0  # 时间步
    obj_vals = [lasso_objective(X, y, beta_coef, lambda_)]
    start_time = time.time()

    for _ in range(max_iter):
        t += 1
        # 计算总梯度
        grad = least_squares_gradient(X, y, beta_coef)
        total_grad = grad + lambda_ * np.sign(beta_coef)
        # 更新一阶/二阶动量
        m_t = beta1 * m_t + (1 - beta1) * total_grad
        v_t = beta2 * v_t + (1 - beta2) * (total_grad ** 2)
        # 偏差修正
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        # Adam更新
        beta_coef = beta_coef - lr * m_t_hat / (np.sqrt(v_t_hat) + eps)
        # 收敛判断
        obj_val = lasso_objective(X, y, beta_coef, lambda_)
        obj_vals.append(obj_val)
        if np.abs(obj_vals[-1] - obj_vals[-2]) < tol:
            break

    end_time = time.time()
    total_time = end_time - start_time
    return beta_coef, obj_vals, total_time, len(obj_vals)-1

# -------------------------- 3. 实验配置与数据生成 --------------------------
def generate_data(n, p, seed=42):
    """
    生成Lasso模拟数据（真实beta稀疏，贴合Lasso应用场景）
    """
    np.random.seed(seed)
    X = np.random.randn(n, p)  # 特征矩阵
    true_beta = np.zeros(p)
    true_beta[:int(0.1 * p)] = np.random.randn(int(0.1 * p))  # 10%非零系数（稀疏）
    y = X @ true_beta + 0.1 * np.random.randn(n)  # 带噪声标签
    return X, y, true_beta

# 算法配置（仅8种有效算法）
algorithm_names = [
    "Subgradient Descent", "Proximal Gradient", "Accelerated Gradient",
    "Coordinate Descent", "ADMM", "SGD", "SVRG", "Adam"
]
algorithm_functions = [
    subgradient_descent_lasso, proximal_gradient_descent_lasso, accelerated_gradient_descent_lasso,
    coordinate_descent_lasso, admm_lasso, stochastic_gradient_descent_lasso, svrg_lasso, adam_lasso
]

# (n,p)组合（覆盖Lasso典型应用场景）
np_combinations = [(100, 500), (500, 500), (1000, 100)]  # n<p, n=p, n>p
lambda_ = 0.1  # 正则化参数（固定，保证对比公平）

# -------------------------- 4. 实验运行与结果记录 --------------------------
def run_experiment():
    experiment_results = []
    convergence_curves = {}  # 存储收敛曲线：键=(n,p,algo_name)

    # 遍历每个(n,p)组合
    for n, p in tqdm(np_combinations, desc="实验整体进度"):
        X, y, true_beta = generate_data(n, p)
        # 遍历每个有效算法
        for algo_name, algo_func in zip(algorithm_names, algorithm_functions):
            print(f"\n运行算法：{algo_name} | 数据规模：(n={n}, p={p})")
            # 运行算法
            beta, obj_vals, total_time, iter_num = algo_func(X, y, lambda_)
            # 记录结果
            result_dict = {
                "n": n,
                "p": p,
                "algorithm": algo_name,
                "total_time": total_time,
                "iter_num": iter_num,
                "final_obj_val": obj_vals[-1],
                "sparsity": np.sum(beta == 0) / p  # 稀疏性（零系数占比）
            }
            experiment_results.append(result_dict)
            # 存储收敛曲线
            convergence_curves[(n, p, algo_name)] = obj_vals

    # 转换为DataFrame
    results_df = pd.DataFrame(experiment_results)
    return results_df, convergence_curves

# -------------------------- 5. 图表绘制（仅8种算法） --------------------------
def plot_results(results_df, convergence_curves):
    # 支持中文和负号
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 各(n,p)组合的算法耗时对比柱状图
    for n, p in np_combinations:
        sub_df = results_df[(results_df["n"] == n) & (results_df["p"] == p)]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(algorithm_names))
        plt.bar(x, sub_df["total_time"], width=0.6, color=plt.cm.Set3(np.linspace(0, 1, len(algorithm_names))))
        plt.xticks(x, algorithm_names, rotation=45, ha="right")
        plt.xlabel("Lasso有效求解算法")
        plt.ylabel("总耗时（秒）")
        plt.title(f"Lasso算法耗时对比 (n={n}, p={p})")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"lasso_effective_time_n{n}_p{p}.png", dpi=300)
        plt.show()

    # 2. 收敛曲线对比图（以(n=100,p=500)为例，高维小样本典型场景）
    n, p = np_combinations[0]
    plt.figure(figsize=(10, 6))
    for algo_name in algorithm_names:
        key = (n, p, algo_name)
        obj_vals = convergence_curves[key]
        # 截断过长迭代，方便可视化
        max_iter_plot = min(1000, len(obj_vals))
        plt.plot(range(max_iter_plot), obj_vals[:max_iter_plot], label=algo_name, linewidth=2)
    plt.xlabel("迭代次数")
    plt.ylabel("目标函数值")
    plt.title(f"Lasso有效算法收敛曲线对比 (n={n}, p={p})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"lasso_effective_convergence_n{n}_p{p}.png", dpi=300)
    plt.show()

    # 3. 系数稀疏性对比图（Lasso核心特性验证）
    for n, p in np_combinations:
        sub_df = results_df[(results_df["n"] == n) & (results_df["p"] == p)]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(algorithm_names))
        plt.bar(x, sub_df["sparsity"], width=0.6, color=plt.cm.Set2(np.linspace(0, 1, len(algorithm_names))))
        plt.xticks(x, algorithm_names, rotation=45, ha="right")
        plt.xlabel("Lasso有效求解算法")
        plt.ylabel("系数稀疏性（零系数占比）")
        plt.title(f"Lasso算法稀疏性对比 (n={n}, p={p})")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"lasso_effective_sparsity_n{n}_p{p}.png", dpi=300)
        plt.show()

# -------------------------- 6. 主函数（运行实验） --------------------------
if __name__ == "__main__":
    # 运行实验
    results_df, convergence_curves = run_experiment()
    # 保存结果
    results_df.to_csv("lasso_effective_algorithm_results.csv", index=False, encoding="utf-8-sig")
    print("\n实验结果已保存至 lasso_effective_algorithm_results.csv")
    # 绘制图表
    plot_results(results_df, convergence_curves)
    # 打印结果概览
    print("\n8种Lasso有效算法结果概览：")
    print(results_df.groupby(["n", "p", "algorithm"])[["total_time", "iter_num", "final_obj_val"]].mean())

