import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def heart_contour(num_points=600):
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )
    return x, y


def compute_epicycles(x, y, n_terms):
    z = x + 1j * y
    coeffs = np.fft.fft(z) / len(z)
    n = len(z)

    freqs = [0]
    for k in range(1, n_terms + 1):
        freqs.extend([k, -k])

    components = []
    for freq in freqs:
        coeff = coeffs[freq % n]
        components.append(
            {
                "freq": freq,
                "coeff": coeff,
                "amp": np.abs(coeff),
                "phase": np.angle(coeff),
            }
        )

    components.sort(key=lambda item: item["amp"], reverse=True)
    return components


def endpoint_at_time(components, t):
    point = 0j
    chain = [point]

    for comp in components:
        point += comp["coeff"] * np.exp(1j * comp["freq"] * t)
        chain.append(point)

    return chain


def create_animation(x, y, components, output_gif, output_png):
    fig, ax = plt.subplots(figsize=(8, 8))
    max_radius = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.4
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("傅里叶圆轮绘制爱心 (n=8)")

    original_line, = ax.plot(x, y, color="lightgray", linewidth=1.5, label="目标轮廓")
    draw_line, = ax.plot([], [], color="crimson", linewidth=2.5, label="已绘制轨迹")
    chain_line, = ax.plot([], [], color="#1f77b4", linewidth=1.2, label="向量链")
    tip_point, = ax.plot([], [], "o", color="black", markersize=4)
    circle_lines = []

    for _ in components:
        circle_line, = ax.plot([], [], color="#4c78a8", alpha=0.35, linewidth=1)
        circle_lines.append(circle_line)

    ax.legend(loc="upper right")

    drawn_points = []
    frames = len(x)

    def init():
        draw_line.set_data([], [])
        chain_line.set_data([], [])
        tip_point.set_data([], [])
        for circle_line in circle_lines:
            circle_line.set_data([], [])
        return [original_line, draw_line, chain_line, tip_point, *circle_lines]

    def update(frame):
        t = 2 * np.pi * frame / frames
        chain = endpoint_at_time(components, t)

        xs = [p.real for p in chain]
        ys = [p.imag for p in chain]
        chain_line.set_data(xs, ys)
        tip_point.set_data([xs[-1]], [ys[-1]])

        drawn_points.append(chain[-1])
        draw_line.set_data(
            [p.real for p in drawn_points],
            [p.imag for p in drawn_points],
        )

        theta = np.linspace(0, 2 * np.pi, 120)
        for i, comp in enumerate(components):
            center = chain[i]
            radius = comp["amp"]
            circle_x = center.real + radius * np.cos(theta)
            circle_y = center.imag + radius * np.sin(theta)
            circle_lines[i].set_data(circle_x, circle_y)

        return [original_line, draw_line, chain_line, tip_point, *circle_lines]

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=30,
        blit=True,
        repeat=True,
    )

    animation.save(output_gif, writer=PillowWriter(fps=30))
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def export_notes(output_txt, n_terms):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("傅里叶圆轮动画示例\n")
        f.write("目标轮廓: 爱心\n")
        f.write(f"保留频率项数: n = {n_terms}\n\n")
        f.write("动画内容:\n")
        f.write("1. 多个旋转向量首尾相接\n")
        f.write("2. 每个向量对应一个频率分量\n")
        f.write("3. 最末端轨迹逐步描出爱心轮廓\n\n")
        f.write("输出文件:\n")
        f.write("- heart_epicycles_n8.gif\n")
        f.write("- heart_epicycles_n8_preview.png\n")


if __name__ == "__main__":
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    output_dir = "./output/傅里叶轮廓绘制_test2"
    os.makedirs(output_dir, exist_ok=True)

    x, y = heart_contour()
    n_terms = 8
    components = compute_epicycles(x, y, n_terms)

    output_gif = os.path.join(output_dir, "heart_epicycles_n8.gif")
    output_png = os.path.join(output_dir, "heart_epicycles_n8_preview.png")
    output_txt = os.path.join(output_dir, "heart_epicycles_n8.txt")

    create_animation(x, y, components, output_gif, output_png)
    export_notes(output_txt, n_terms)
