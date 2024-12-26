import numpy as np
import matplotlib.pyplot as plt
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


# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 数据读取
filename = "./data/1_TDM.txt"
x_data, y_data = load_and_sort_data(filename)

# 检查数据范围
print("X 数据范围:", x_data.min(), x_data.max())
print("Y 数据范围:", y_data.min(), y_data.max())

# 定义多项式阶数
n = 45  # 假设是四次多项式，你可以调整 n

# 设置权重
# 对于关注的点（如 x 在某个区间内），赋予更大的权重
weights = np.ones_like(y_data)  # 默认权重为 1
focus_indices1 = (x_data > -0.01) & (x_data < 0.01)  # 设定关注点条件
focus_indices2 = (x_data > 0.075) & (x_data < 0.095)
focus_indices3 = (x_data > -0.095) & (x_data < -0.075)
focus_indices4 = (x_data > 0.11) & (x_data < 0.17)
focus_indices5 = (x_data > -0.17) & (x_data < -0.11)
focus_indices6 = (x_data > 0.18) & (x_data < 0.25)
focus_indices7 = (x_data > -0.25) & (x_data < -0.18)

weights[focus_indices1] = 10  # 设置关注点的权重
weights[focus_indices2] = 5
weights[focus_indices3] = 5
weights[focus_indices4] = 3
weights[focus_indices5] = 3
weights[focus_indices6] = 2
weights[focus_indices7] = 2

# 使用 numpy.polyfit 拟合多项式（加权）
coeffs = np.polyfit(x_data, y_data, n, w=weights)
poly_func = np.poly1d(coeffs)

# 输出多项式表达式
poly_expression = " + ".join(
    [f"{coeff:.10f}*x^{i}" for i, coeff in enumerate(coeffs[::-1])]
)
# print(f"{n} 次多项式表达式: y = {poly_expression}")

# 将表达式写入文件
output_folder = "./output/多项式拟合"
os.makedirs(output_folder, exist_ok=True)
output_txt = os.path.join(output_folder, "多项式拟合.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(f"{n} 次多项式表达式:\n")
    f.write(f"y = {poly_expression}\n")
print(f"多项式表达式已保存到文件: {output_txt}")

# 使用拟合函数计算 y 值
y_fit = poly_func(x_data)

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=10, label="原始数据", color="black")
plt.plot(x_data, y_fit, label=f"{n} 次多项式拟合", color="red", linewidth=2)
plt.title(f"{n} 次多项式拟合")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
output_png = os.path.join(output_folder, "多项式拟合.png")
plt.savefig(output_png, dpi=300)
plt.show()
