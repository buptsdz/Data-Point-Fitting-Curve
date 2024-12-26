import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
import os


def five_gaussian_with_background(
    x,
    a1,
    mu1,
    sigma1,
    a2,
    mu2,
    sigma2,
    a3,
    mu3,
    sigma3,
    a4,
    mu4,
    sigma4,
    a5,
    mu5,
    sigma5,
    c0,
    c1,
    c2,
):
    """
    五个高斯函数的组合（一个主峰、两个负峰和两个侧峰）加多项式背景
    a1, mu1, sigma1: 主峰参数
    a2, mu2, sigma2: 左侧凹谷参数
    a3, mu3, sigma3: 右侧凹谷参数
    a4, mu4, sigma4: 左侧凸峰参数
    a5, mu5, sigma5: 右侧凸峰参数
    c0-c2: 多项式系数
    """
    gaussian1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))  # 主峰
    gaussian2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))  # 左凹谷
    gaussian3 = a3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3**2))  # 右凹谷
    gaussian4 = a4 * np.exp(-((x - mu4) ** 2) / (2 * sigma4**2))  # 左侧凸峰
    gaussian5 = a5 * np.exp(-((x - mu5) ** 2) / (2 * sigma5**2))  # 右侧凸峰
    background = c0 + c1 * x + c2 * x**2
    return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + background


def get_function_expression(params):
    """生成函数表达式的字符串形式"""
    (
        a1,
        mu1,
        sigma1,
        a2,
        mu2,
        sigma2,
        a3,
        mu3,
        sigma3,
        a4,
        mu4,
        sigma4,
        a5,
        mu5,
        sigma5,
        c0,
        c1,
        c2,
    ) = params

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

    # 左侧凸峰
    gauss4 = f"{a4:.6f} * exp(-(x - {mu4:.6f})^2 / (2 * {sigma4:.6f}^2))"
    expr_parts.append(gauss4)

    # 右侧凸峰
    gauss5 = f"{a5:.6f} * exp(-(x - {mu5:.6f})^2 / (2 * {sigma5:.6f}^2))"
    expr_parts.append(gauss5)

    # 多项式背景
    if c0 != 0:
        expr_parts.append(f"{c0:.6f}")
    if c1 != 0:
        expr_parts.append(f"{c1:.6f}*x")
    if c2 != 0:
        expr_parts.append(f"{c2:.6f}*x^2")

    expression = " + ".join(f"({part})" for part in expr_parts)
    expression = expression.replace("+ (-", "- ")

    return f"f(x) = {expression}"


def load_data(filename):
    """从文本文件加载数据"""
    data = np.loadtxt(filename)
    return data


def fit_spectrum(x, y):
    """拟合频谱数据"""
    # 初始参数估计
    max_y = np.max(y)
    min_y = np.min(y)
    peak_idx = np.argmax(y)
    mu_init = x[peak_idx]

    # 估计凹谷和凸峰位置
    left_valley_x = -0.1
    right_valley_x = 0.1
    left_peak_x = -0.25
    right_peak_x = 0.25

    # 初始参数
    p0 = [
        min_y - max_y,  # a1: 主峰高度
        mu_init,  # mu1: 主峰位置
        0.02,  # sigma1: 主峰宽度
        0.5,  # a2: 左凹谷深度
        left_valley_x,  # mu2: 左凹谷位置
        0.03,  # sigma2: 左凹谷宽度
        0.5,  # a3: 右凹谷深度
        right_valley_x,  # mu3: 右凹谷位置
        0.03,  # sigma3: 右凹谷宽度
        -0.8,  # a4: 左侧凸峰高度
        left_peak_x,  # mu4: 左侧凸峰位置
        0.15,  # sigma4: 左侧凸峰宽度
        -0.8,  # a5: 右侧凸峰高度
        right_peak_x,  # mu5: 右侧凸峰位置
        0.15,  # sigma5: 右侧凸峰宽度
        min_y,  # c0: 常数项
        0,  # c1: 一次项
        0,  # c2: 二次项
    ]

    # 设置参数边界
    bounds = (
        [
            -np.inf,
            -0.1,
            0,  # 主峰
            0,
            -0.2,
            0,  # 左凹谷
            0,
            0,
            0,  # 右凹谷
            -np.inf,
            -0.4,
            0,  # 左侧凸峰
            -np.inf,
            0.2,
            0,  # 右侧凸峰
            -np.inf,
            -np.inf,
            -np.inf,  # 背景多项式
        ],  # 下界
        [
            0,
            0.1,
            0.1,  # 主峰
            np.inf,
            0,
            0.1,  # 左凹谷
            np.inf,
            0.2,
            0.1,  # 右凹谷
            0,
            -0.2,
            1.3,  # 左侧凸峰
            0,
            0.4,
            1.3,  # 右侧凸峰
            np.inf,
            np.inf,
            np.inf,  # 背景多项式
        ],  # 上界
    )

    # 拟合
    popt, _ = curve_fit(five_gaussian_with_background, x, y, p0=p0, bounds=bounds)

    return lambda x: five_gaussian_with_background(x, *popt), popt


def plot_fit_result(x, y, fit_func, output_dir):
    """绘制拟合结果"""
    plt.figure(figsize=(12, 8))

    x_smooth = np.linspace(min(x), max(x), 1000)
    plt.plot(x_smooth, fit_func(x_smooth), "r-", label="拟合曲线", linewidth=2)

    plt.scatter(x, y, color="black", label="原始数据", s=20)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.title("CB1_interp_a数据拟合")
    plt.savefig(os.path.join(output_dir, "图2_五高斯组合拟合.png"))
    plt.show()


def main(filex, filey, output_dir):
    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 加载数据
    x = load_data(filex)
    y = load_data(filey)

    # 拟合数据
    fit_func, params = fit_spectrum(x, y)

    # 绘制结果
    plot_fit_result(x, y, fit_func, output_dir)

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
        "左侧凸峰幅值(a4)",
        "左侧凸峰位置(μ4)",
        "左侧凸峰宽度(σ4)",
        "右侧凸峰幅值(a5)",
        "右侧凸峰位置(μ5)",
        "右侧凸峰宽度(σ5)",
        "常数项(c0)",
        "一次项(c1)",
        "二次项(c2)",
    ]

    print("\n拟合参数:")
    for name, value in zip(param_names, params):
        print(f"{name}: {value:.6f}")

    # 输出函数表达式
    print("\n拟合函数表达式:")
    with open(
        os.path.join(output_dir, "图2_五高斯组合拟合.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("五高斯组合拟合函数表达式:\n")
        f.write(get_function_expression(params))

    print(get_function_expression(params))

    return fit_func, params


if __name__ == "__main__":
    output_dir = "./output/高斯组合拟合"
    os.makedirs(output_dir, exist_ok=True)
    filex = "./data/2_x.txt"  # 替换为你的数据文件名
    filey = "./data/2_CB1_interp_a.txt"
    fit_func, params = main(filex, filey, output_dir)
