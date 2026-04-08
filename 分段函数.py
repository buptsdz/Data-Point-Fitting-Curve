import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def calculate_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    return mse, rmse, mae, r2


def get_spline_expression(spline):
    knots = spline.get_knots()
    coefficients = spline.get_coeffs()
    k = 3

    expressions = []
    n = len(coefficients)

    for i in range(len(knots) - 1):
        if i < n - k:
            x_mid = (knots[i] + knots[i + 1]) / 2
            x_local = np.linspace(knots[i], knots[i + 1], 4)
            y_local = spline(x_local)
            poly_coeffs = np.polyfit(x_local - x_mid, y_local, k)

            terms = []
            for j, coef in enumerate(poly_coeffs):
                if abs(coef) < 1e-10:
                    continue

                if j == 0:
                    terms.append(f"{coef:.6f}*x^3")
                elif j == 1:
                    terms.append(f"{coef:.6f}*x^2")
                elif j == 2:
                    terms.append(f"{coef:.6f}*x")
                else:
                    terms.append(f"{coef:.6f}")

            if terms:
                expr = " + ".join(terms)
                interval = f"{knots[i]:.6f} <= x < {knots[i + 1]:.6f}"
                expressions.append(f"区间 {interval}:\ny = {expr}\n")

    return "\n".join(expressions)


def fit_smooth_curve_improved(filename):
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
    except Exception:
        reduced_knots = knots[::2]
        spline = LSQUnivariateSpline(x, y, reduced_knots, k=3)

    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = spline(x_new)

    return x, y, x_new, y_new, spline


def plot_fitting_result(x, y, x_new, y_new, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="原始数据点", alpha=0.5)
    plt.plot(x_new, y_new, color="red", label="拟合曲线", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("数据拟合结果")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "分段函数拟合.png"))
    plt.close()


if __name__ == "__main__":
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    filename = "./data/1_TDM.txt"
    x, y, x_new, y_new, spline = fit_smooth_curve_improved(filename)

    output_dir = "./output/分段函数拟合"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "分段函数拟合.txt")

    plot_fitting_result(x, y, x_new, y_new, output_dir)

    expression = get_spline_expression(spline)
    y_fit = spline(x)
    mse, rmse, mae, r2 = calculate_metrics(y, y_fit)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(expression)
        f.write("\n拟合精度（误差）:\n")
        f.write(f"MSE = {mse:.10e}\n")
        f.write(f"RMSE = {rmse:.10e}\n")
        f.write(f"MAE = {mae:.10e}\n")
        f.write(f"R^2 = {r2:.10f}\n")

    print("\n表达式已保存到: 分段函数拟合.txt")
