#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №9
Геометричні афінні та перспективні перетворення цифрових зображень

Варіант 2: текстовий документ на столі
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, OpenCV, SciPy (scipy.ndimage), Matplotlib
"""

# %% Блок 1. Підготовка середовища, імпорт та генерація сцени з документом
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# каталоги для графіків і даних
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

# параметри варіанта 2
THETA, SCALE, TX, TY = -20.0, 0.90, -30, 40       # афінне перетворення
SRC_QUAD = np.float32([[180, 80], [620, 130], [710, 510], [70, 470]])   # P1..P4
W_OUT, H_OUT = 500, 350                          # розмір виправленого документа
FRAME_H, FRAME_W = 600, 800                      # розмір вихідного кадру
ZOOM = 2                                         # збільшення при порівнянні інтерполяцій


def generate_document_canvas(width=W_OUT, height=H_OUT):
    """Плоский документ: заголовок, рядки тексту, таблиця, логотип і рамка."""
    doc = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(doc, (0, 0), (width - 1, 52), (120, 60, 30), -1)
    cv2.putText(doc, "REPORT 2024", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.circle(doc, (width - 40, 26), 16, (0, 200, 255), -1)
    for i in range(5):                                   # рядки тексту
        y = 90 + i * 24
        cv2.putText(doc, f"Line {i + 1}: signal and image", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    for i in range(4):                                   # коротші "абзаци"
        cv2.rectangle(doc, (20, 225 + i * 18), (20 + 220 - 35 * i, 233 + i * 18), (90, 90, 90), -1)
    for r in range(4):                                   # таблиця праворуч
        for c in range(3):
            cv2.rectangle(doc, (290 + c * 62, 215 + r * 28), (352 + c * 62, 243 + r * 28), (60, 60, 60), 1)
    cv2.putText(doc, "p. 1", (width - 70, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.rectangle(doc, (0, 0), (width - 1, height - 1), (40, 40, 40), 3)
    return doc


def generate_perspective_scene():
    """Стіл із документом, спотвореним перспективою. Повертає кадр, документ і матрицю H0."""
    doc = generate_document_canvas()
    # фон: теплий градієнт з сіткою
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    desk = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
    desk[..., 0] = 70 + 25 * xx / FRAME_W
    desk[..., 1] = 100 + 25 * yy / FRAME_H
    desk[..., 2] = 130 + 20 * xx / FRAME_W
    desk = desk.astype(np.uint8)
    for x in range(0, FRAME_W, 50):
        cv2.line(desk, (x, 0), (x, FRAME_H - 1), (55, 80, 105), 1)
    for y in range(0, FRAME_H, 50):
        cv2.line(desk, (0, y), (FRAME_W - 1, y), (55, 80, 105), 1)

    doc_corners = np.float32([[0, 0], [W_OUT - 1, 0], [W_OUT - 1, H_OUT - 1], [0, H_OUT - 1]])
    h0 = cv2.getPerspectiveTransform(doc_corners, SRC_QUAD)
    warped = cv2.warpPerspective(doc, h0, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((H_OUT, W_OUT), 255, np.uint8), h0, (FRAME_W, FRAME_H),
                               flags=cv2.INTER_NEAREST)
    scene = np.where(mask[..., None] > 0, warped, desk)
    return scene, doc, h0


def mark_corners(ax, pts):
    """Підписує кути P1..P4 на осях Matplotlib (зображення залишається чистим)."""
    ax.scatter(pts[:, 0], pts[:, 1], s=30, color='red', zorder=3)
    for k, (px, py) in enumerate(pts, start=1):
        ax.text(px + 8, py - 8, f"P{k}", color='red', fontsize=11, fontweight='bold')


img_scene, doc_flat, H_forward = generate_perspective_scene()
cv2.imwrite("data/perspective_scene.png", img_scene)
print("Сцену збережено: data/perspective_scene.png")
print(f"Кадр: {img_scene.shape[1]}x{img_scene.shape[0]} пікс. (ширина x висота), "
      f"документ: {W_OUT}x{H_OUT} пікс.")


# %% Блок 2. Афінне перетворення: масштаб, поворот і зсув
scene = cv2.imread("data/perspective_scene.png")
h_img, w_img = scene.shape[:2]
centre = (w_img / 2, h_img / 2)

# матриця 2x3: поворот навколо центру кадру зі зміною масштабу, потім зсув
M = cv2.getRotationMatrix2D(centre, THETA, SCALE)
M[0, 2] += TX
M[1, 2] += TY

# те саме за формулою з теорії
alpha = SCALE * np.cos(np.deg2rad(THETA))
beta = SCALE * np.sin(np.deg2rad(THETA))
M_formula = np.array([[alpha, beta, (1 - alpha) * centre[0] - beta * centre[1] + TX],
                      [-beta, alpha, beta * centre[0] + (1 - alpha) * centre[1] + TY]])

scene_affine = cv2.warpAffine(scene, M, (w_img, h_img), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))

# власна реалізація оберненого відображення: для кожного вихідного пікселя
# знаходимо джерело T^-1(x', y') і беремо білінійну інтерполяцію
M3 = np.vstack([M, [0, 0, 1]])
M3_inv = np.linalg.inv(M3)
xs_out, ys_out = np.meshgrid(np.arange(w_img), np.arange(h_img))
x_src = M3_inv[0, 0] * xs_out + M3_inv[0, 1] * ys_out + M3_inv[0, 2]
y_src = M3_inv[1, 0] * xs_out + M3_inv[1, 1] * ys_out + M3_inv[1, 2]
manual_gray = ndimage.map_coordinates(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY).astype(np.float64),
                                      [y_src, x_src], order=1, mode='constant', cval=128)
cv_gray = cv2.cvtColor(scene_affine, cv2.COLOR_BGR2GRAY).astype(np.float64)
inside = (x_src >= 0) & (x_src <= w_img - 1) & (y_src >= 0) & (y_src <= h_img - 1)
diff_inv = np.abs(manual_gray - cv_gray)[inside].max()

# куди переходять кути документа
quad_h = np.hstack([SRC_QUAD, np.ones((4, 1), dtype=np.float32)])
quad_new = quad_h @ M.T
in_frame = [(0 <= x < w_img) and (0 <= y < h_img) for x, y in quad_new]

np.set_printoptions(precision=4, suppress=True)
print("\n--- АФІННЕ ПЕРЕТВОРЕННЯ ---")
print(f"Параметри: theta = {THETA} град, s = {SCALE}, tx = {TX}, ty = {TY}; центр обертання {centre}")
print(f"alpha = s*cos(theta) = {alpha:.4f}, beta = s*sin(theta) = {beta:.4f}")
print("Матриця M (2x3) з cv2.getRotationMatrix2D + зсув:")
print(M)
print(f"Збігається з формулою теорії: max|M - M_formula| = {np.abs(M - M_formula).max():.2e}")
print(f"Визначник лінійної частини det = {np.linalg.det(M[:, :2]):.4f} (дорівнює s^2 = {SCALE ** 2:.4f})")
print("Нові координати кутів P1..P4:")
for k, ((x, y), ok) in enumerate(zip(quad_new, in_frame), start=1):
    print(f"  P{k}: ({x:7.1f}, {y:7.1f})  {'у кадрі' if ok else 'за межами кадру'}")
print(f"Власне обернене відображення (білінійне) проти cv2.warpAffine: max різниця = {diff_inv:.2f} рівня")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
axes[0].add_patch(plt.Polygon(SRC_QUAD, closed=True, fill=False, edgecolor='lime', linewidth=2))
mark_corners(axes[0], SRC_QUAD)
axes[0].set_title("Вихідний кадр 800x600")
axes[1].imshow(cv2.cvtColor(scene_affine, cv2.COLOR_BGR2RGB))
axes[1].add_patch(plt.Polygon(quad_new, closed=True, fill=False, edgecolor='lime', linewidth=2))
mark_corners(axes[1], quad_new)
axes[1].set_title(f"Афінне перетворення: θ = {THETA}°, s = {SCALE}, tx = {TX}, ty = {TY}")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/affine_transform_results.png", dpi=300)
plt.close()


# %% Блок 3. Порівняння методів інтерполяції при збільшенні
# фрагмент із текстом документа: 70 рядків x 90 стовпців
y0, x0 = 135, 235
crop = scene[y0:y0 + 70, x0:x0 + 90]
methods = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}

# для об'єктивної оцінки: зменшуємо фрагмент вдвічі, повертаємо до початкового розміру
# кожним методом і порівнюємо з оригіналом
small = cv2.resize(crop, None, fx=1 / ZOOM, fy=1 / ZOOM, interpolation=cv2.INTER_AREA)
crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

print("\n--- ПОРІВНЯННЯ ІНТЕРПОЛЯЦІЙ (збільшення у {} рази) ---".format(ZOOM))
print(f"Фрагмент {crop.shape[1]}x{crop.shape[0]} пікс. -> {crop.shape[1] * ZOOM}x{crop.shape[0] * ZOOM} пікс.")
print(f"{'Метод':<16}{'Лапласіан, дисп.':>18}{'Макс. градієнт':>16}{'PSNR відновл., дБ':>20}{'Час, мкс':>11}")
zoomed = {}
for name, flag in methods.items():
    z = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=flag)
    zoomed[name] = z
    zg = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(zg, cv2.CV_64F).var()
    gmax = np.hypot(cv2.Sobel(zg, cv2.CV_64F, 1, 0), cv2.Sobel(zg, cv2.CV_64F, 0, 1)).max()
    back = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=flag)
    mse = np.mean((cv2.cvtColor(back, cv2.COLOR_BGR2GRAY).astype(float) - crop_gray.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    t0 = time.perf_counter()
    for _ in range(300):
        cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=flag)
    t_us = (time.perf_counter() - t0) / 300 * 1e6
    print(f"{name:<16}{lap_var:>18.1f}{gmax:>16.0f}{psnr:>20.2f}{t_us:>11.1f}")

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (name, z) in zip(axes, zoomed.items()):
    ax.imshow(cv2.cvtColor(z, cv2.COLOR_BGR2RGB), interpolation='nearest')
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/interpolation_comparison.png", dpi=300)
plt.close()


# %% Блок 4. Перспективне виправлення документа за гомографією
dst_quad = np.float32([[0, 0], [W_OUT - 1, 0], [W_OUT - 1, H_OUT - 1], [0, H_OUT - 1]])
H_mat = cv2.getPerspectiveTransform(SRC_QUAD, dst_quad)
rectified = cv2.warpPerspective(scene, H_mat, (W_OUT, H_OUT), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def homography_from_points(src, dst):
    """Розв'язує систему 8x8 для h00..h21 (h22 = 1) за чотирма парами точок."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    h = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


H_manual = homography_from_points(SRC_QUAD, dst_quad)

# перевірка: образи кутів та порівняння з відомим документом
mapped = cv2.perspectiveTransform(SRC_QUAD.reshape(-1, 1, 2), H_mat).reshape(-1, 2)
corner_err = np.abs(mapped - dst_quad).max()
H_true = np.linalg.inv(H_forward)
H_true = H_true / H_true[2, 2]
mse_rect = np.mean((rectified.astype(float) - doc_flat.astype(float)) ** 2)
psnr_rect = 10 * np.log10(255 ** 2 / mse_rect)

print("\n--- ПЕРСПЕКТИВНЕ ВИПРАВЛЕННЯ (ГОМОГРАФІЯ) ---")
print(f"Цільовий розмір: {W_OUT}x{H_OUT} пікс.; вихідні кути P1..P4: {SRC_QUAD.astype(int).tolist()}")
print("Матриця гомографії H (3x3):")
print(H_mat)
print(f"det(H) = {np.linalg.det(H_mat):.4e}")
print(f"h20 = {H_mat[2, 0]:.3e}, h21 = {H_mat[2, 1]:.3e} (ненульові: перспективне, а не афінне перетворення)")
print(f"Власне розв'язання системи 8x8 проти cv2: max|H - H_manual| = {np.abs(H_mat - H_manual).max():.2e}")
print(f"Образи кутів H*P_k проти цільових вершин: max похибка = {corner_err:.2e} пікс.")
print(f"Відхилення від істинної оберненої матриці: max|H - H_true| = {np.abs(H_mat - H_true).max():.2e}")
print(f"Виправлений документ проти оригінального плаского документа: MSE = {mse_rect:.2f}, "
      f"PSNR = {psnr_rect:.2f} дБ")
print(f"Зведення: M (2x3) і H (3x3) наведені вище; det(H) = {np.linalg.det(H_mat):.4e}")

roi_vis = scene.copy()
cv2.polylines(roi_vis, [SRC_QUAD.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].imshow(cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB))
mark_corners(axes[0], SRC_QUAD)
axes[0].set_title("Кадр з виділеною областю документа (ROI)")
axes[1].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Виправлений документ {W_OUT}x{H_OUT} (INTER_CUBIC)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/perspective_rectified.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
