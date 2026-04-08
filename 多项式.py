import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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


mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

filename = "./data/1_TDM.txt"
x_data, y_data = load_and_sort_data(filename)

print("X 数据范围:", x_data.min(), x_data.max())
print("Y 数据范围:", y_data.min(), y_data.max())

n = 45

weights = np.ones_like(y_data)
focus_indices1 = (x_data > -0.01) & (x_data < 0.01)
focus_indices2 = (x_data > 0.075) & (x_data < 0.095)
focus_indices3 = (x_data > -0.095) & (x_data < -0.075)
focus_indices4 = (x_data > 0.11) & (x_data < 0.17)
focus_indices5 = (x_data > -0.17) & (x_data < -0.11)
focus_indices6 = (x_data > 0.18) & (x_data < 0.25)
focus_indices7 = (x_data > -0.25) & (x_data < -0.18)

weights[focus_indices1] = 10
weights[focus_indices2] = 5
weights[focus_indices3] = 5
weights[focus_indices4] = 3
weights[focus_indices5] = 3
weights[focus_indices6] = 2
weights[focus_indices7] = 2

coeffs = np.polyfit(x_data, y_data, n, w=weights)
poly_func = np.poly1d(coeffs)

poly_expression = " + ".join(
    [f"{coeff:.10f}*x^{i}" for i, coeff in enumerate(coeffs[::-1])]
)

output_folder = "./output/多项式拟合"
os.makedirs(output_folder, exist_ok=True)
output_txt = os.path.join(output_folder, "多项式拟合.txt")

y_fit = poly_func(x_data)
mse, rmse, mae, r2 = calculate_metrics(y_data, y_fit)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(f"{n} 次多项式表达式\n")
    f.write(f"y = {poly_expression}\n")
    f.write("\n拟合精度（误差）:\n")
    f.write(f"MSE = {mse:.10e}\n")
    f.write(f"RMSE = {rmse:.10e}\n")
    f.write(f"MAE = {mae:.10e}\n")
    f.write(f"R^2 = {r2:.10f}\n")

print(f"多项式表达式已保存到文件: {output_txt}")

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=10, label="原始数据", color="black")
plt.plot(x_data, y_fit, label=f"{n} 次多项式拟合", color="red", linewidth=2)
plt.title(f"{n} 次多项式拟合")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
output_png = os.path.join(output_folder, "多项式拟合.png")
plt.savefig(output_png, dpi=300)
plt.close()
