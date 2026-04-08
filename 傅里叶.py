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


def fourier_series(x, *coeffs):
    a0 = coeffs[0]
    series = a0
    n_terms = (len(coeffs) - 1) // 2
    for n in range(1, n_terms + 1):
        an = coeffs[2 * n - 1]
        bn = coeffs[2 * n]
        series += an * np.cos(n * x) + bn * np.sin(n * x)
    return series


def fit_fourier(x, y, n_terms):
    initial_guess = [0] * (2 * n_terms + 1)
    coeffs, _ = curve_fit(lambda x_val, *c: fourier_series(x_val, *c), x, y, p0=initial_guess)
    return coeffs


def fit_fourier_weighted(x, y, n_terms, target_x=None, weight=10):
    if target_x is not None:
        weights = np.ones_like(y)
        closest_idx = np.argmin(np.abs(x - target_x))
        weights[closest_idx] *= weight
    else:
        weights = np.ones_like(y)

    initial_guess = [0] * (2 * n_terms + 1)

    result = curve_fit(
        lambda x_val, *c: fourier_series(x_val, *c),
        x,
        y,
        p0=initial_guess,
        sigma=1 / weights,
    )
    coeffs = result[0]
    return coeffs


def export_fourier_formula(coeffs, metrics, output_filename, output_dir):
    n_terms = (len(coeffs) - 1) // 2
    formula = f"y(x) = {coeffs[0]:.10f}"

    for n in range(1, n_terms + 1):
        an = coeffs[2 * n - 1]
        bn = coeffs[2 * n]
        if abs(an) >= 1e-9:
            formula += f" + {an:.10f} * cos({n}x)"
        if abs(bn) >= 1e-9:
            formula += f" + {bn:.10f} * sin({n}x)"

    output_txt = os.path.join(output_dir, output_filename + ".txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("傅里叶级数公式\n")
        f.write(formula)
        f.write("\n\n拟合精度（误差）:\n")
        f.write(f"MSE = {metrics[0]:.10e}\n")
        f.write(f"RMSE = {metrics[1]:.10e}\n")
        f.write(f"MAE = {metrics[2]:.10e}\n")
        f.write(f"R^2 = {metrics[3]:.10f}\n")


def plot_fit(x, y, coeffs, output_filename, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="black", s=10, label="原始数据")

    x_fit = np.linspace(x.min(), x.max(), 1000)
    y_fit = fourier_series(x_fit, *coeffs)
    plt.plot(x_fit, y_fit, color="red", label="傅里叶拟合")

    plt.title(f"傅里叶级数拟合n={n_terms}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    output_png = os.path.join(output_dir, output_filename + ".png")
    plt.savefig(output_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    output_dir = "./output/傅里叶级数拟合"
    os.makedirs(output_dir, exist_ok=True)

    data_x = "./data/2_x.txt"
    data_y = "./data/2_CB1_interp_a.txt"
    x, y = load_and_sort_data_split(data_x, data_y)
    output_filename = "fourier_series"
    n_terms = 60

    coeffs = fit_fourier_weighted(x, y, n_terms, target_x=0)
    y_fit = fourier_series(x, *coeffs)
    metrics = calculate_metrics(y, y_fit)

    plot_fit(x, y, coeffs, output_filename, output_dir)
    export_fourier_formula(coeffs, metrics, output_filename, output_dir)
