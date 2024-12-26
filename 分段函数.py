import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def get_spline_expression(spline):
    """
    获取样条曲线的分段多项式表达式

    参数:
    spline: LSQUnivariateSpline对象

    返回:
    expression_str: 包含所有分段多项式表达式的字符串
    """
    # 获取节点（包括端点）
    knots = spline.get_knots()
    # 获取系数
    coefficients = spline.get_coeffs()
    # 样条阶数固定为3（三次样条）
    k = 3

    expressions = []
    n = len(coefficients)

    # 对每一段区间构建多项式表达式
    for i in range(len(knots) - 1):
        # 确定当前区间的系数
        if i < n - k:
            # 计算在当前区间的多项式系数
            x_mid = (knots[i] + knots[i + 1]) / 2
            # 使用多项式拟合得到这一段的系数
            x_local = np.linspace(knots[i], knots[i + 1], 4)
            y_local = spline(x_local)
            poly_coeffs = np.polyfit(x_local - x_mid, y_local, k)

            # 将系数转换为多项式表达式
            terms = []
            for j, coef in enumerate(poly_coeffs):
                if abs(coef) < 1e-10:  # 忽略接近0的系数
                    continue

                if j == 0:  # 三次项
                    if abs(coef) > 1e-10:
                        terms.append(f"{coef:.6f}*x^3")
                elif j == 1:  # 二次项
                    if abs(coef) > 1e-10:
                        terms.append(f"{coef:.6f}*x^2")
                elif j == 2:  # 一次项
                    if abs(coef) > 1e-10:
                        terms.append(f"{coef:.6f}*x")
                else:  # 常数项
                    if abs(coef) > 1e-10:
                        terms.append(f"{coef:.6f}")

            # 组合多项式表达式
            if terms:
                expr = " + ".join(terms)
                interval = f"{knots[i]:.6f} ≤ x < {knots[i+1]:.6f}"
                expressions.append(f"区间 {interval}:\ny = {expr}\n")

    return "\n".join(expressions)


def fit_smooth_curve_improved(filename):
    """
    使用最小二乘B样条曲线拟合数据，特别优化了峰值区域的拟合效果
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    peak_threshold = np.mean(y) + 2 * np.std(y)
    peak_mask = y > peak_threshold

    num_knots = len(x) // 10

    peak_x = x[peak_mask]
    if len(peak_x) > 0:
        peak_knots = np.linspace(peak_x[0], peak_x[-1], num_knots // 2)
        non_peak_knots = np.linspace(x[0], x[-1], num_knots // 2)
        knots = np.unique(np.concatenate([peak_knots, non_peak_knots]))
        knots = knots[1:-1]
    else:
        knots = np.linspace(x[1], x[-2], num_knots)[1:-1]

    try:
        spline = LSQUnivariateSpline(x, y, knots, k=3)
    except:
        reduced_knots = knots[::2]
        spline = LSQUnivariateSpline(x, y, reduced_knots, k=3)

    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = spline(x_new)

    return x_new, y_new, spline


def plot_fitting_result(x, y, x_new, y_new, output_dir):
    """绘制原始数据点和拟合曲线"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="原始数据点", alpha=0.5)
    plt.plot(x_new, y_new, color="red", label="拟合曲线", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("数据拟合结果")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "分段函数拟合.png"))
    plt.show()


if __name__ == "__main__":
    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    filename = "./data/1_TDM.txt"
    # 读取数据并拟合
    x_new, y_new, spline = fit_smooth_curve_improved(filename)

    # 读取原始数据用于绘图
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]

    output_dir = "./output/分段函数拟合"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "分段函数拟合.txt")

    # 绘制结果
    plot_fitting_result(x, y, x_new, y_new, output_dir)

    # 获取并打印拟合曲线表达式
    # print("\n拟合曲线的分段多项式表达式：")
    expression = get_spline_expression(spline)
    # print(expression)

    # 保存表达式到文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(expression)
    print("\n表达式已保存到:分段函数拟合.txt")
