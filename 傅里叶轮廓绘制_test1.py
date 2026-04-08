import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def heart_contour(num_points=800):
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )
    return x, y


def fourier_descriptors(x, y):
    z = x + 1j * y
    return np.fft.fft(z) / len(z)


def reconstruct_contour(coeffs, n_terms):
    n = len(coeffs)
    t = np.arange(n)
    reconstructed = np.zeros(n, dtype=complex)
    reconstructed += coeffs[0]

    for k in range(1, n_terms + 1):
        reconstructed += coeffs[k] * np.exp(2j * np.pi * k * t / n)
        reconstructed += coeffs[-k] * np.exp(-2j * np.pi * k * t / n)

    return reconstructed.real, reconstructed.imag


def plot_reconstruction(x, y, coeffs, terms_list, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, n_terms in zip(axes, terms_list):
        x_rec, y_rec = reconstruct_contour(coeffs, n_terms)
        ax.plot(x, y, color="lightgray", linewidth=2, label="原始轮廓")
        ax.plot(x_rec, y_rec, color="crimson", linewidth=2, label=f"重建轮廓 ({n_terms} 项)")
        ax.set_title(f"傅里叶轮廓重建: {n_terms} 项")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend()

    plt.suptitle("用傅里叶级数画任意轮廓: 心形示例", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def export_notes(output_txt, terms_list):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("傅里叶轮廓绘制示例\n")
        f.write("示例轮廓: 心形闭合曲线\n\n")
        f.write("核心思路:\n")
        f.write("1. 将二维轮廓写成复数序列 z = x + iy\n")
        f.write("2. 对 z 做离散傅里叶变换，得到傅里叶描述子\n")
        f.write("3. 保留有限项频率分量后反变换，就能逐步重建轮廓\n\n")
        f.write("本图展示的重建项数:\n")
        for n_terms in terms_list:
            f.write(f"- {n_terms} 项\n")


if __name__ == "__main__":
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    output_dir = "./output/傅里叶轮廓绘制_test1"
    os.makedirs(output_dir, exist_ok=True)

    x, y = heart_contour()
    coeffs = fourier_descriptors(x, y)
    terms_list = [3, 8, 20, 80]

    output_png = os.path.join(output_dir, "heart_fourier_contour.png")
    output_txt = os.path.join(output_dir, "heart_fourier_contour.txt")

    plot_reconstruction(x, y, coeffs, terms_list, output_png)
    export_notes(output_txt, terms_list)
