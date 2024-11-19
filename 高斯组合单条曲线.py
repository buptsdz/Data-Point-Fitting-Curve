import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl


def triple_gaussian_with_background(
    x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, c0, c1, c2
):
    """
    三个高斯函数的组合（一个主峰和两个负峰）加多项式背景
    a1, mu1, sigma1: 主峰参数
    a2, mu2, sigma2: 左侧凹谷参数
    a3, mu3, sigma3: 右侧凹谷参数
    c0-c2: 多项式系数
    """
    gaussian1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))  # 主峰
    gaussian2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))  # 左凹谷
    gaussian3 = a3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3**2))  # 右凹谷
    background = c0 + c1 * x + c2 * x**2
    return gaussian1 + gaussian2 + gaussian3 + background


def get_function_expression(params):
    """生成函数表达式的字符串形式"""
    a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, c0, c1, c2 = params

    # 格式化每个高斯函数和多项式项
    expr_parts = []

    # 主峰
    gauss1 = f"{a1:.6f} * exp(-(x - {mu1:.6f})^2 / (2 * {sigma1:.6f}^2))"
    expr_parts.append(gauss1)

    # 左凹谷
    gauss2 = f"{a2:.6f} * exp(-(x - {mu2:.6f})^2 / (2 * {sigma2:.6f}^2))"
    expr_parts.append(gauss2)

    # 右凹谷
    gauss3 = f"{a3:.6f} * exp(-(x - {mu3:.6f})^2 / (2 * {sigma3:.6f}^2))"
    expr_parts.append(gauss3)

    # 多项式背景
    if c0 != 0:
        expr_parts.append(f"{c0:.6f}")
    if c1 != 0:
        expr_parts.append(f"{c1:.6f}*x")
    if c2 != 0:
        expr_parts.append(f"{c2:.6f}*x^2")

    # 组合所有部分
    expression = " + ".join(f"({part})" for part in expr_parts)
    # 替换一些过于复杂的加减
    expression = expression.replace("+ (-", "- ")

    return f"f(x) = {expression}"


def load_data(filename):
    """从文本文件加载数据"""
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def fit_spectrum(x, y):
    """拟合频谱数据"""
    # 初始参数估计
    max_y = np.max(y)
    min_y = np.min(y)
    peak_idx = np.argmax(y)
    mu_init = x[peak_idx]

    # 估计凹谷位置
    left_valley_x = -0.08
    right_valley_x = 0.08

    # 初始参数
    p0 = [
        max_y - min_y,  # a1: 主峰高度
        mu_init,  # mu1: 主峰位置
        0.02,  # sigma1: 主峰宽度
        -0.5,  # a2: 左凹谷深度
        left_valley_x,  # mu2: 左凹谷位置
        0.03,  # sigma2: 左凹谷宽度
        -0.5,  # a3: 右凹谷深度
        right_valley_x,  # mu3: 右凹谷位置
        0.03,  # sigma3: 右凹谷宽度
        min_y,  # c0: 常数项
        0,  # c1: 一次项
        0,  # c2: 二次项
    ]

    # 设置参数边界
    bounds = (
        [
            0,
            -0.1,
            0,
            -np.inf,
            -0.2,
            0,
            -np.inf,
            0,
            0,
            -np.inf,
            -np.inf,
            -np.inf,
        ],  # 下界
        [np.inf, 0.1, 0.1, 0, 0, 0.1, 0, 0.2, 0.1, np.inf, np.inf, np.inf],  # 上界
    )

    # 拟合
    popt, _ = curve_fit(triple_gaussian_with_background, x, y, p0=p0, bounds=bounds)

    return lambda x: triple_gaussian_with_background(x, *popt), popt


def plot_fit_result(x, y, fit_func):
    """绘制拟合结果"""
    plt.figure(figsize=(12, 8))

    # 生成平滑的x点以画出连续曲线
    x_smooth = np.linspace(min(x), max(x), 1000)

    # 绘制原始数据和拟合曲线
    plt.scatter(x, y, color="black", label="原始数据", s=20)
    # plt.plot(x_smooth, fit_func(x_smooth), "r-", label="拟合曲线", linewidth=2)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.title("TDM数据拟合")
    plt.show()


def main(filename):
    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 加载数据
    x, y = load_data(filename)

    # 拟合数据
    fit_func, params = fit_spectrum(x, y)

    # 绘制结果
    plot_fit_result(x, y, fit_func)

    # 输出拟合参数
    param_names = [
        "主峰幅值(a1)",
        "主峰位置(μ1)",
        "主峰宽度(σ1)",
        "左凹谷幅值(a2)",
        "左凹谷位置(μ2)",
        "左凹谷宽度(σ2)",
        "右凹谷幅值(a3)",
        "右凹谷位置(μ3)",
        "右凹谷宽度(σ3)",
        "常数项(c0)",
        "一次项(c1)",
        "二次项(c2)",
    ]

    print("\n拟合参数:")
    for name, value in zip(param_names, params):
        print(f"{name}: {value:.6f}")

    # 输出函数表达式
    print("\n拟合函数表达式:")
    print(get_function_expression(params))

    return fit_func, params


if __name__ == "__main__":
    filename = "TDM.txt"  # 替换为你的数据文件名
    fit_func, params = main(filename)
