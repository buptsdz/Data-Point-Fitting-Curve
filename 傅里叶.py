import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl


# 数据加载和排序
def load_and_sort_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    return x, y


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


# 导出傅里叶级数公式
def export_fourier_formula(coeffs, filename="fourier_series.txt"):
    n_terms = (len(coeffs) - 1) // 2
    formula = f"y(x) = {coeffs[0]:.6f}"

    for n in range(1, n_terms + 1):
        an = coeffs[2 * n - 1]
        bn = coeffs[2 * n]
        if abs(an) >= 0.00001:
            formula += f" + {an:.6f} * cos({n}x)"
        if abs(bn) >= 0.00001:
            formula += f" + {bn:.6f} * sin({n}x)"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("傅里叶级数公式\n")
        f.write(formula)


# 绘图函数
def plot_fit(x, y, coeffs):
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
    plt.show()


if __name__ == "__main__":
    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    # filename = "TDM.txt"
    # x, y = load_and_sort_data(filename)
    # output_file = "fourier_series.txt"

    filename_x = "x.txt"
    filename_y = "CB1_interp_a.txt"
    x, y = load_and_sort_data_split(filename_x, filename_y)
    output_file = "fourier_series2.txt"

    n_terms = 80  # 设置傅里叶级数的阶数（可以调整）
    coeffs = fit_fourier(x, y, n_terms)

    plot_fit(x, y, coeffs)
    export_fourier_formula(coeffs, output_file)
