import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def calculate_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    return mse, rmse, mae, r2


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
    gaussian1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
    gaussian2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
    gaussian3 = a3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3**2))
    gaussian4 = a4 * np.exp(-((x - mu4) ** 2) / (2 * sigma4**2))
    gaussian5 = a5 * np.exp(-((x - mu5) ** 2) / (2 * sigma5**2))
    background = c0 + c1 * x + c2 * x**2
    return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + background


def get_function_expression(params):
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

    expr_parts = [
        f"{a1:.6f} * exp(-(x - {mu1:.6f})^2 / (2 * {sigma1:.6f}^2))",
        f"{a2:.6f} * exp(-(x - {mu2:.6f})^2 / (2 * {sigma2:.6f}^2))",
        f"{a3:.6f} * exp(-(x - {mu3:.6f})^2 / (2 * {sigma3:.6f}^2))",
        f"{a4:.6f} * exp(-(x - {mu4:.6f})^2 / (2 * {sigma4:.6f}^2))",
        f"{a5:.6f} * exp(-(x - {mu5:.6f})^2 / (2 * {sigma5:.6f}^2))",
    ]

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
    return np.loadtxt(filename)


def fit_spectrum(x, y):
    max_y = np.max(y)
    min_y = np.min(y)
    peak_idx = np.argmax(y)
    mu_init = x[peak_idx]

    left_valley_x = -0.1
    right_valley_x = 0.1
    left_peak_x = -0.25
    right_peak_x = 0.25

    p0 = [
        min_y - max_y,
        mu_init,
        0.02,
        0.5,
        left_valley_x,
        0.03,
        0.5,
        right_valley_x,
        0.03,
        -0.8,
        left_peak_x,
        0.15,
        -0.8,
        right_peak_x,
        0.15,
        min_y,
        0,
        0,
    ]

    bounds = (
        [
            -np.inf,
            -0.1,
            0,
            0,
            -0.2,
            0,
            0,
            0,
            0,
            -np.inf,
            -0.4,
            0,
            -np.inf,
            0.2,
            0,
            -np.inf,
            -np.inf,
            -np.inf,
        ],
        [
            0,
            0.1,
            0.1,
            np.inf,
            0,
            0.1,
            np.inf,
            0.2,
            0.1,
            0,
            -0.2,
            1.3,
            0,
            0.4,
            1.3,
            np.inf,
            np.inf,
            np.inf,
        ],
    )

    popt, _ = curve_fit(five_gaussian_with_background, x, y, p0=p0, bounds=bounds)
    return lambda x_val: five_gaussian_with_background(x_val, *popt), popt


def plot_fit_result(x, y, fit_func, output_dir):
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
    plt.close()


def main(filex, filey, output_dir):
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    x = load_data(filex)
    y = load_data(filey)

    fit_func, params = fit_spectrum(x, y)
    plot_fit_result(x, y, fit_func, output_dir)

    param_names = [
        "主峰幅值(a1)",
        "主峰位置(mu1)",
        "主峰宽度(sigma1)",
        "左凹谷幅值(a2)",
        "左凹谷位置(mu2)",
        "左凹谷宽度(sigma2)",
        "右凹谷幅值(a3)",
        "右凹谷位置(mu3)",
        "右凹谷宽度(sigma3)",
        "左侧凸峰幅值(a4)",
        "左侧凸峰位置(mu4)",
        "左侧凸峰宽度(sigma4)",
        "右侧凸峰幅值(a5)",
        "右侧凸峰位置(mu5)",
        "右侧凸峰宽度(sigma5)",
        "常数项(c0)",
        "一次项(c1)",
        "二次项(c2)",
    ]

    print("\n拟合参数:")
    for name, value in zip(param_names, params):
        print(f"{name}: {value:.6f}")

    metrics = calculate_metrics(y, fit_func(x))
    formula = get_function_expression(params)

    print("\n拟合函数表达式:")
    with open(
        os.path.join(output_dir, "图2_五高斯组合拟合.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("五高斯组合拟合函数表达式:\n")
        f.write(formula)
        f.write("\n\n拟合精度（误差）:\n")
        f.write(f"MSE = {metrics[0]:.10e}\n")
        f.write(f"RMSE = {metrics[1]:.10e}\n")
        f.write(f"MAE = {metrics[2]:.10e}\n")
        f.write(f"R^2 = {metrics[3]:.10f}\n")

    print(formula)

    return fit_func, params


if __name__ == "__main__":
    output_dir = "./output/高斯组合拟合"
    os.makedirs(output_dir, exist_ok=True)
    filex = "./data/2_x.txt"
    filey = "./data/2_CB1_interp_a.txt"
    fit_func, params = main(filex, filey, output_dir)
