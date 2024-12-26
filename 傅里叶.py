import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
import os


# 数据加载和排序
def load_and_sort_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    return x, y


# 处理第二张图的数据
def load_and_sort_data_split(filename_x, filename_y):
    x = np.loadtxt(filename_x)
    y = np.loadtxt(filename_y)
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    return x, y


# 傅里叶级数的定义
# N 为级数阶数
def fourier_series(x, *coeffs):
    a0 = coeffs[0]
    series = a0
    n_terms = (len(coeffs) - 1) // 2
    for n in range(1, n_terms + 1):
        an = coeffs[2 * n - 1]
        bn = coeffs[2 * n]
        series += an * np.cos(n * x) + bn * np.sin(n * x)
    return series


# 拟合傅里叶级数
def fit_fourier(x, y, n_terms):
    # 初始系数猜测（a0, a1, b1, a2, b2, ...）
    initial_guess = [0] * (2 * n_terms + 1)

    # 曲线拟合
    coeffs, _ = curve_fit(lambda x, *c: fourier_series(x, *c), x, y, p0=initial_guess)
    return coeffs


def fit_fourier_weighted(x, y, n_terms, target_x=None, weight=10):
    # 如果指定了需要重点关注的点
    if target_x is not None:
        weights = np.ones_like(y)
        # 找到最接近 target_x 的点，并赋予更高的权重
        closest_idx = np.argmin(np.abs(x - target_x))
        weights[closest_idx] *= weight
    else:
        weights = np.ones_like(y)

    # 初始系数猜测（a0, a1, b1, a2, b2, ...）
    initial_guess = [0] * (2 * n_terms + 1)

    # 曲线拟合
    def weighted_residuals(c, x, y, w):
        return w * (y - fourier_series(x, *c))

    result = curve_fit(
        lambda x, *c: fourier_series(x, *c), x, y, p0=initial_guess, sigma=1 / weights
    )
    coeffs = result[0]
    return coeffs


# 导出傅里叶级数公式
def export_fourier_formula(coeffs, output_filename, output_dir):
    n_terms = (len(coeffs) - 1) // 2
    formula = f"y(x) = {coeffs[0]:.10f}"

    for n in range(1, n_terms + 1):
        an = coeffs[2 * n - 1]
        bn = coeffs[2 * n]
        if abs(an) >= 0.000000001:
            formula += f" + {an:.10f} * cos({n}x)"
        if abs(bn) >= 0.000000001:
            formula += f" + {bn:.10f} * sin({n}x)"

    output_txt = os.path.join(output_dir, output_filename + ".txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("傅里叶级数公式\n")
        f.write(formula)


# 绘图函数
def plot_fit(x, y, coeffs, output_filename, output_dir):
    plt.figure(figsize=(10, 6))

    # 原始数据
    plt.scatter(x, y, color="black", s=10, label="原始数据")

    # 拟合曲线
    x_fit = np.linspace(x.min(), x.max(), 1000)
    y_fit = fourier_series(x_fit, *coeffs)
    plt.plot(x_fit, y_fit, color="red", label="傅里叶拟合")

    plt.title("傅里叶级数拟合")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    output_png = os.path.join(output_dir, output_filename + ".png")
    plt.savefig(output_png, dpi=300)
    plt.show()


if __name__ == "__main__":
    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    output_dir = "./output/傅里叶级数拟合"
    os.makedirs(output_dir, exist_ok=True)

    # 数据读取1
    # data = "./data/1_TDM.txt"
    # x, y = load_and_sort_data(data)
    # output_filename = "fourier_series1"
    # n_terms = 140  # 设置傅里叶级数的阶数

    # 数据读取2
    data_x = "./data/2_x.txt"
    data_y = "./data/2_CB1_interp_a.txt"
    x, y = load_and_sort_data_split(data_x, data_y)
    output_filename = "fourier_series2"
    n_terms = 80  # 设置傅里叶级数的阶数

    # coeffs = fit_fourier(x, y, n_terms)
    coeffs = fit_fourier_weighted(x, y, n_terms, target_x=0)

    plot_fit(x, y, coeffs, output_filename, output_dir)
    export_fourier_formula(coeffs, output_filename, output_dir)
