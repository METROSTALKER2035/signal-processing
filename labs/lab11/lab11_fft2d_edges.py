#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №11
2D перетворення Фур'є та детектування контурів

Варіант 2: дефектоскопія друкованих плат (межі мідних провідників)
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy (numpy.fft), OpenCV, Matplotlib
"""

# %% Блок 1. Підготовка середовища, імпорт та генерація тестової сцени
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
D0 = 50.0                      # радіус зрізу ідеального фільтра, пікс.
SIGMA_F = 35.0                 # параметр гаусових фільтрів, пікс.
K_SOBEL = 3                    # розмір ядра Собеля
CANNY_T_LOW, CANNY_T_HIGH = 60, 180   # пороги Canny варіанта


def generate_fourier_edge_scene(size=512):
    """
    Півтонова сцена друкованої плати: підкладка, мідні доріжки (горизонтальна,
    вертикальна та діагональна під 45 градусів), контактні площадки з отворами,
    решітка виводів мікросхеми, концентричні кільця та напис.
    """
    img = np.full((size, size), 45, dtype=np.uint8)
    cv2.rectangle(img, (40, 82), (300, 96), 190, -1)                # горизонтальна доріжка
    cv2.rectangle(img, (413, 40), (427, 250), 190, -1)              # вертикальна доріжка
    cv2.rectangle(img, (300, 82), (314, 200), 190, -1)              # коліно
    cv2.line(img, (60, 200), (250, 390), 190, 12)                   # діагональна доріжка (45 градусів)
    for cx, cy in [(40, 89), (420, 40), (60, 200), (250, 390), (314, 200)]:
        cv2.circle(img, (cx, cy), 22, 215, -1)                      # контактні площадки
        cv2.circle(img, (cx, cy), 9, 40, -1)                        # отвори
    for i in range(16):                                              # решітка виводів мікросхеми
        x = 300 + i * 10
        cv2.rectangle(img, (x, 360), (x + 4, 430), 205, -1)
    for r, col in [(60, 200), (46, 55), (32, 200), (18, 55)]:       # концентричні кільця
        cv2.circle(img, (110, 440), r, col, -1)
    cv2.putText(img, "PCB-02", (330, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 230, 2)
    # слабкоконтрастні подряпини: контраст відносно підкладки 12, 25, 45 та 80 рівнів
    for k, contrast in enumerate([12, 25, 45, 80]):
        y = 120 + k * 17
        cv2.rectangle(img, (90, y), (260, y + 5), 45 + contrast, -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8).astype(np.float64)
    rng = np.random.default_rng(42)
    return np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)


scene = generate_fourier_edge_scene()
cv2.imwrite("data/test_edge_scene.png", scene)
print("Тестову сцену збережено: data/test_edge_scene.png")


# %% Блок 2. 2D ШПФ: амплітудний та фазовий спектри
img = cv2.imread("data/test_edge_scene.png", cv2.IMREAD_GRAYSCALE)
f = img.astype(np.float64)
M_, N_ = f.shape

F = np.fft.fft2(f)
F_shift = np.fft.fftshift(F)            # низькі частоти переміщено в центр
magnitude = np.abs(F_shift)
log_spec = 20 * np.log10(magnitude + 1)
phase = np.angle(F_shift)

# перевірки: теорема Парсеваля та зворотне перетворення
parseval = np.sum(f ** 2), np.sum(magnitude ** 2) / (M_ * N_)
recon_err = np.abs(np.fft.ifft2(F).real - f).max()
dc_ok = np.isclose(F[0, 0].real, f.sum())

# кутовий розподіл енергії спектра (без нульової частоти та самих низьких частот)
cy, cx = M_ // 2, N_ // 2
vv, uu = np.mgrid[0:M_, 0:N_]
rad = np.hypot(uu - cx, vv - cy)
ang = np.degrees(np.arctan2(-(vv - cy), uu - cx)) % 180
energy = magnitude ** 2
valid = rad > 3
e_total = energy[valid].sum()


def sector(center, half=8):
    d = np.abs(((ang - center + 90) % 180) - 90)
    return 100 * energy[valid & (d <= half)].sum() / e_total


print("--- 2D ШПФ ---")
print(f"Форма спектра: {F.shape}")
print(f"|F|: від {magnitude.min():.2e} до {magnitude.max():.2e}; максимум логарифмічного спектра {log_spec.max():.2f} дБ")
print(f"Нульова частота F(0,0) = сума яскравостей: {dc_ok}; частка потужності в DC {100 * magnitude[cy, cx] ** 2 / energy.sum():.2f} %")
print(f"Парсеваль: sum|f|^2 = {parseval[0]:.6e}, sum|F|^2/(MN) = {parseval[1]:.6e}")
print(f"Похибка зворотного перетворення: max|ifft2(F) - f| = {recon_err:.2e}")
print("Частка потужності спектра (без r <= 3) у кутових секторах +-8 градусів:")
for name, c in [("0 градусів (частоти вздовж u: вертикальні структури)", 0),
                ("90 градусів (частоти вздовж v: горизонтальні структури)", 90),
                ("45 градусів", 45), ("135 градусів", 135)]:
    print(f"  {name:<58}{sector(c):6.2f} %")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Вхідна сцена 512x512")
im1 = axes[1].imshow(log_spec, cmap='magma')
axes[1].set_title("Амплітудний спектр 20·lg(|F| + 1), дБ")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
im2 = axes[2].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[2].set_title("Фазовий спектр, рад")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/fft2d_spectrum_analysis.png", dpi=300)
plt.close()


# %% Блок 3. Частотна фільтрація: ідеальний та гаусові фільтри
u_axis = np.arange(N_) - N_ / 2
v_axis = np.arange(M_) - M_ / 2
U, V = np.meshgrid(u_axis, v_axis)
D_matrix = np.sqrt(U ** 2 + V ** 2)

H_ilpf = (D_matrix <= D0).astype(np.float64)                      # ідеальний ФНЧ, радіус D0
H_glpf = np.exp(-D_matrix ** 2 / (2 * SIGMA_F ** 2))              # гаусів ФНЧ, sigma_f
H_ghpf = 1.0 - H_glpf                                             # гаусів ФВЧ


def apply_frequency_filter(F_shift_mat, H_mask):
    """G = F * H, зворотний зсув та ШПФ; повертає дійсну частину та кадр uint8."""
    g = np.fft.ifft2(np.fft.ifftshift(F_shift_mat * H_mask)).real
    return g, np.clip(g, 0, 255).astype(np.uint8)


g_ilpf, img_ilpf = apply_frequency_filter(F_shift, H_ilpf)
g_glpf, img_glpf = apply_frequency_filter(F_shift, H_glpf)
g_ghpf, img_ghpf = apply_frequency_filter(F_shift, H_ghpf)

lo_, hi_ = float(f.min()), float(f.max())
print("\n--- ЧАСТОТНА ФІЛЬТРАЦІЯ (D0 = {:.0f}, sigma_f = {:.0f}) ---".format(D0, SIGMA_F))
print(f"{'Фільтр':<22}{'частка потужності, %':>22}{'min виходу':>12}{'max виходу':>12}{'поза [{:.0f}; {:.0f}], %'.format(lo_, hi_):>20}{'PSNR, дБ':>11}")
for name, g, H in [("ILPF (ідеальний)", g_ilpf, H_ilpf), ("GLPF (Гаусів)", g_glpf, H_glpf)]:
    power = 100 * np.sum((magnitude * H) ** 2) / energy.sum()
    out = 100 * np.mean((g < lo_ - 0.5) | (g > hi_ + 0.5))
    mse = np.mean((g - f) ** 2)
    print(f"{name:<22}{power:>22.2f}{g.min():>12.1f}{g.max():>12.1f}{out:>20.2f}{10 * np.log10(255 ** 2 / mse):>11.2f}")
power_h = 100 * np.sum((magnitude * H_ghpf) ** 2) / energy.sum()
print(f"GHPF (Гаусів ФВЧ): частка потужності {power_h:.2f} %, СКВ виходу {g_ghpf.std():.2f}, середнє {g_ghpf.mean():.2f}")
print(f"Розмір головної пелюстки Гаусового ФНЧ за рівнем 0.5: D = {SIGMA_F * np.sqrt(2 * np.log(2)):.1f} пікс.")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, H, title in [(axes[0, 0], H_ilpf, f"Маска ILPF, D0 = {D0:.0f}"),
                     (axes[0, 1], H_glpf, f"Маска GLPF, σf = {SIGMA_F:.0f}"),
                     (axes[0, 2], H_ghpf, f"Маска GHPF, σf = {SIGMA_F:.0f}")]:
    ax.imshow(H, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
for ax, a, title in [(axes[1, 0], img_ilpf, "ILPF (кільця Гіббса біля меж)"),
                     (axes[1, 1], img_glpf, "GLPF (без кілець)"),
                     (axes[1, 2], np.clip(g_ghpf + 128, 0, 255).astype(np.uint8), "GHPF (зсув +128): межі")]:
    ax.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/frequency_filtering_results.png", dpi=300)
plt.close()


# %% Блок 4. Градієнтні оператори Собеля та Шарра
Gx_s = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=K_SOBEL)
Gy_s = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=K_SOBEL)
G_sobel = np.sqrt(Gx_s ** 2 + Gy_s ** 2)
Gx_c = cv2.Scharr(f, cv2.CV_64F, 1, 0)
Gy_c = cv2.Scharr(f, cv2.CV_64F, 0, 1)
G_scharr = np.sqrt(Gx_c ** 2 + Gy_c ** 2) / 4.0      # на плавному перепаді Шарр дає 32, Собель 8: ділимо на 4

# кути градієнта для перевірки кутової точності
theta_s = np.degrees(np.arctan2(Gy_s, Gx_s))
theta_c = np.degrees(np.arctan2(Gy_c, Gx_c))

print("\n--- ГРАДІЄНТИ (Собель k = {}, Шарр) ---".format(K_SOBEL))
print(f"{'Оператор':<10}{'max |Gx|':>12}{'max |Gy|':>12}{'max G':>12}{'середнє G':>12}{'G > T_high, пікс.':>20}")
for name, gx, gy, g in [("Собель", Gx_s, Gy_s, G_sobel), ("Шарр", Gx_c, Gy_c, G_scharr)]:
    print(f"{name:<10}{np.abs(gx).max():>12.0f}{np.abs(gy).max():>12.0f}{g.max():>12.0f}{g.mean():>12.2f}"
          f"{int(np.count_nonzero(g > CANNY_T_HIGH)):>20}")
print(f"Середня абсолютна різниця G Собеля та Шарра (Шарр поділено на 4): {np.mean(np.abs(G_sobel - G_scharr)):.2f}")

# перевірка кутової точності на ділянці діагональної доріжки: край має нормаль під 45 градусів
patch = (slice(260, 290), slice(100, 130))
sel = G_sobel[patch] > 100
print(f"Діагональна доріжка: кут градієнта (по модулю): Собель {np.median(np.abs(theta_s[patch][sel])):.1f} град, "
      f"Шарр {np.median(np.abs(theta_c[patch][sel])):.1f} град (істинний напрям нормалі 45 град)")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for row, (name, gx, gy, g) in enumerate([("Собель", Gx_s, Gy_s, G_sobel), ("Шарр", Gx_c, Gy_c, G_scharr)]):
    axes[row, 0].imshow(np.abs(gx), cmap='gray')
    axes[row, 0].set_title(f"{name}: |Gx| (вертикальні межі)")
    axes[row, 1].imshow(np.abs(gy), cmap='gray')
    axes[row, 1].set_title(f"{name}: |Gy| (горизонтальні межі)")
    axes[row, 2].imshow(g, cmap='gray')
    axes[row, 2].set_title(f"{name}: модуль градієнта G")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/sobel_scharr_gradients.png", dpi=300)
plt.close()


# %% Блок 5. Детектор Canny: вплив порогів
# перший крок Canny: гаусове згладжування (cv2.Canny його не виконує)
smoothed = cv2.GaussianBlur(img, (5, 5), 1.4)
threshold_sets = {
    f"Низькі [{CANNY_T_LOW // 2}, {CANNY_T_HIGH // 2}]": (CANNY_T_LOW // 2, CANNY_T_HIGH // 2),
    f"Варіант 2 [{CANNY_T_LOW}, {CANNY_T_HIGH}]": (CANNY_T_LOW, CANNY_T_HIGH),
    f"Високі [{CANNY_T_LOW * 2}, {CANNY_T_HIGH * 2}]": (CANNY_T_LOW * 2, CANNY_T_HIGH * 2),
}

grad_l1 = np.abs(cv2.Sobel(smoothed.astype(np.float64), cv2.CV_64F, 1, 0, ksize=K_SOBEL)) + \
    np.abs(cv2.Sobel(smoothed.astype(np.float64), cv2.CV_64F, 0, 1, ksize=K_SOBEL))
print("\n--- ДЕТЕКТОР CANNY (aperture = {}) ---".format(K_SOBEL))
print(f"{'Пороги':<24}{'Пікселів країв':>16}{'Компонент (8-зв.)':>20}{'G > T_low (Собель), пікс.':>26}{'Товщина, разів':>16}")
canny_results = {}
for name, (tl, th) in threshold_sets.items():
    edges = cv2.Canny(smoothed, tl, th, apertureSize=K_SOBEL)
    n_cc = cv2.connectedComponents(edges, connectivity=8)[0] - 1
    thick = int(np.count_nonzero(grad_l1 > tl))
    canny_results[name] = edges
    print(f"{name:<24}{cv2.countNonZero(edges):>16}{n_cc:>20}{thick:>26}{thick / max(cv2.countNonZero(edges), 1):>16.2f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
for ax, (name, edges) in zip(axes, canny_results.items()):
    ax.imshow(edges, cmap='gray')
    ax.set_title(f"Canny {name}\nпікселів країв: {cv2.countNonZero(edges)}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/canny_thresholds_study.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
