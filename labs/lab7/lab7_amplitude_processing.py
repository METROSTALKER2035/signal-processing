#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №7
Амплітудні перетворення, вирівнювання гістограми та бінаризація за методом Оцу

Варіант 2: МРТ головного мозку
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, OpenCV, Pillow, Matplotlib
"""

# %% Блок 1. Підготовка середовища, імпорт та генерація низькоконтрастної сцени
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

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
K_LIN, B_LIN = 1.8, -40       # лінійне перетворення s = k*r + b
GAMMA_1, GAMMA_2 = 0.40, 2.5  # гамма-корекція: g1 < 1 та g2 > 1
T_FIXED = 95                  # фіксований поріг
S_WIN, C_OFFSET = 15, 4       # вікно та зсув адаптивного порога


def generate_brain_mri_phantom(size=512, seed=3):
    """
    Імітація низькоконтрастного знімка МРТ головного мозку: череп, сіра та
    біла речовина, шлуночки, кілька яскравих осередків, нерівномірне
    освітлення (чутливість котушки) та шум. Повертає зображення uint8 та
    маску внутрішньої області мозку.
    """
    rng = np.random.default_rng(seed)
    c = size // 2
    img = np.full((size, size), 10, dtype=np.float32)          # повітря

    cv2.ellipse(img, (c, c), (215, 240), 0, 0, 360, 60, -1)    # м'які тканини голови
    cv2.ellipse(img, (c, c), (195, 220), 0, 0, 360, 25, -1)    # череп
    brain_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(brain_mask, (c, c), (178, 202), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (c, c), (178, 202), 0, 0, 360, 85, -1)    # сіра речовина
    cv2.ellipse(img, (c, c), (140, 168), 0, 0, 360, 105, -1)   # біла речовина
    cv2.ellipse(img, (c - 28, c - 10), (22, 55), 10, 0, 360, 35, -1)  # шлуночки
    cv2.ellipse(img, (c + 28, c - 10), (22, 55), -10, 0, 360, 35, -1)
    for (x, y, r) in [(c - 85, c + 70, 16), (c + 95, c + 40, 12), (c + 60, c - 100, 10)]:
        cv2.circle(img, (x, y), r, 125, -1)                    # яскраві осередки

    img = cv2.GaussianBlur(img, (0, 0), 1.6)

    # нерівномірне підсилення: ліва частина темніша за праву
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    bias = 0.65 + 0.5 * xx / (size - 1) - 0.08 * ((yy - c) / c) ** 2
    img = img * bias + rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), brain_mask


scene, brain_mask = generate_brain_mri_phantom()
cv2.imwrite("data/low_contrast_input.png", scene)
print("Вхідне зображення збережено: data/low_contrast_input.png")


# %% Блок 2. Гістограма, функція розподілу та статистичні характеристики
img = cv2.imread("data/low_contrast_input.png", cv2.IMREAD_GRAYSCALE)

hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
hist_np, _ = np.histogram(img, bins=256, range=(0, 256))
pdf = hist_np / img.size
cdf = np.cumsum(pdf)


def image_stats(a):
    """Діапазон, середнє та СКВ яскравості зображення."""
    return int(a.min()), int(a.max()), float(a.mean()), float(a.std())


lo, hi, mean_in, std_in = image_stats(img)
print("--- ГІСТОГРАМА ВХІДНОГО ЗОБРАЖЕННЯ ---")
print(f"Розмір: {img.shape}, тип: {img.dtype}")
print(f"cv2.calcHist і np.histogram збігаються: {np.array_equal(hist_cv.astype(int), hist_np)}")
print(f"Діапазон яскравості: {lo}..{hi} (динамічний діапазон {hi - lo})")
print(f"Середнє = {mean_in:.2f}, СКВ = {std_in:.2f}")
print(f"Мінімальний рівень з ненульовою частотою: {np.nonzero(hist_np)[0][0]}, "
      f"рівень, що охоплює 99 % пікселів: {np.searchsorted(cdf, 0.99)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Вхідне зображення (МРТ, низький контраст)")
axes[0].axis('off')
axes[1].bar(np.arange(256), hist_np, width=1.0, color='#4c72b0')
axes[1].set_xlabel("Рівень яскравості r")
axes[1].set_ylabel("Кількість пікселів n_k")
axes[1].set_title("Гістограма та функція розподілу")
axes[1].set_xlim(0, 255)
axes[1].grid(True)
ax_cdf = axes[1].twinx()
ax_cdf.plot(np.arange(256), cdf, color='#c44e52', linewidth=2)
ax_cdf.set_ylabel("CDF")
ax_cdf.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig("figures/histogram_analysis.png", dpi=300)
plt.close()


# %% Блок 3. Лінійне та гамма-перетворення за допомогою таблиць LUT
r_levels = np.arange(256)
lut_lin = np.clip(K_LIN * r_levels + B_LIN, 0, 255).astype(np.uint8)
lut_g1 = np.clip(255 * (r_levels / 255.0) ** GAMMA_1, 0, 255).astype(np.uint8)
lut_g2 = np.clip(255 * (r_levels / 255.0) ** GAMMA_2, 0, 255).astype(np.uint8)

img_lin = cv2.LUT(img, lut_lin)
img_g1 = cv2.LUT(img, lut_g1)
img_g2 = cv2.LUT(img, lut_g2)

# яка частка пікселів у лінійному перетворенні втрачена через відсікання
clip_low = 100 * np.mean(K_LIN * img.astype(float) + B_LIN < 0)
clip_high = 100 * np.mean(K_LIN * img.astype(float) + B_LIN > 255)

print("\n--- ЛІНІЙНЕ ТА ГАММА-ПЕРЕТВОРЕННЯ ---")
print(f"Лінійне: k = {K_LIN}, b = {B_LIN}; r у [0, 255] -> s у [0, 255] при r від "
      f"{-B_LIN / K_LIN:.1f} до {(255 - B_LIN) / K_LIN:.1f}")
print(f"Відсічено знизу (s < 0): {clip_low:.2f} % пікселів, зверху (s > 255): {clip_high:.2f} %")
print(f"{'Зображення':<26}{'min':>5}{'max':>6}{'діапазон':>10}{'середнє':>10}{'СКВ':>9}")
for name, a in [("Вхідне", img), (f"Лінійне (k={K_LIN}, b={B_LIN})", img_lin),
                (f"Гамма {GAMMA_1}", img_g1), (f"Гамма {GAMMA_2}", img_g2)]:
    mn, mx, me, sd = image_stats(a)
    print(f"{name:<26}{mn:>5}{mx:>6}{mx - mn:>10}{me:>10.2f}{sd:>9.2f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes[0, 0].plot(r_levels, r_levels, '--', color='gray', label="s = r")
axes[0, 0].plot(r_levels, lut_lin, color='#4c72b0', linewidth=2, label=f"лінійне k={K_LIN}, b={B_LIN}")
axes[0, 0].plot(r_levels, lut_g1, color='#55a868', linewidth=2, label=f"γ = {GAMMA_1}")
axes[0, 0].plot(r_levels, lut_g2, color='#c44e52', linewidth=2, label=f"γ = {GAMMA_2}")
axes[0, 0].set_xlabel("Вхід r")
axes[0, 0].set_ylabel("Вихід s")
axes[0, 0].set_title("Передавальні характеристики")
axes[0, 0].set_xlim(0, 255)
axes[0, 0].set_ylim(0, 255)
axes[0, 0].legend(loc='upper left')
axes[0, 0].grid(True)
for ax, a, title in [(axes[0, 1], img, "Вхідне"),
                     (axes[0, 2], img_lin, f"Лінійне (k = {K_LIN}, b = {B_LIN})"),
                     (axes[1, 0], img_g1, f"Гамма γ = {GAMMA_1}"),
                     (axes[1, 1], img_g2, f"Гамма γ = {GAMMA_2}")]:
    ax.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
for a, label, col in [(img, "вхідне", 'black'), (img_lin, "лінійне", '#4c72b0'),
                      (img_g1, f"γ={GAMMA_1}", '#55a868'), (img_g2, f"γ={GAMMA_2}", '#c44e52')]:
    axes[1, 2].plot(np.bincount(a.ravel(), minlength=256), color=col, linewidth=1.2, label=label)
axes[1, 2].set_title("Гістограми після перетворень")
axes[1, 2].set_xlabel("Рівень яскравості")
axes[1, 2].set_xlim(0, 255)
axes[1, 2].set_yscale('log')
axes[1, 2].legend()
axes[1, 2].grid(True)
plt.tight_layout()
plt.savefig("figures/linear_and_gamma_transf.png", dpi=300)
plt.close()


# %% Блок 4. Глобальне вирівнювання гістограми
eq_cv = cv2.equalizeHist(img)
eq_pil = np.array(ImageOps.equalize(Image.fromarray(img)))
diff_eq = np.abs(eq_cv.astype(int) - eq_pil.astype(int)).max()

hist_eq = np.bincount(eq_cv.ravel(), minlength=256)
cdf_eq = np.cumsum(hist_eq) / eq_cv.size

print("\n--- ВИРІВНЮВАННЯ ГІСТОГРАМИ ---")
for name, a in [("До вирівнювання", img), ("Після (cv2.equalizeHist)", eq_cv)]:
    mn, mx, me, sd = image_stats(a)
    print(f"{name:<26} діапазон {mn}..{mx} ({mx - mn}), середнє {me:.2f}, СКВ {sd:.2f}, "
          f"дисперсія {sd ** 2:.1f}")
print(f"Максимальна різниця cv2.equalizeHist і PIL.ImageOps.equalize: {diff_eq} рівнів")
print(f"Кількість зайнятих рівнів яскравості: {np.count_nonzero(hist_np)} -> {np.count_nonzero(hist_eq)}")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("Оригінал")
axes[0, 1].imshow(eq_cv, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title("Після вирівнювання гістограми")
for ax in axes[0]:
    ax.axis('off')
for ax, h_arr, c_arr, title in [(axes[1, 0], hist_np, cdf, "Гістограма та CDF до вирівнювання"),
                                (axes[1, 1], hist_eq, cdf_eq, "Гістограма та CDF після вирівнювання")]:
    ax.bar(np.arange(256), h_arr, width=1.0, color='#4c72b0')
    ax.set_xlim(0, 255)
    ax.set_xlabel("Рівень яскравості")
    ax.set_ylabel("Кількість пікселів")
    ax.set_title(title)
    ax.grid(True)
    twin = ax.twinx()
    twin.plot(np.arange(256), c_arr, color='#c44e52', linewidth=2)
    twin.set_ylim(0, 1.02)
    twin.set_ylabel("CDF")
plt.tight_layout()
plt.savefig("figures/histogram_equalization.png", dpi=300)
plt.close()


# %% Блок 5. Порівняння методів бінаризації: фіксований поріг, Оцу, адаптивні
_, bin_fixed = cv2.threshold(img, T_FIXED, 255, cv2.THRESH_BINARY)
t_otsu, bin_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bin_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, S_WIN, C_OFFSET)
bin_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, S_WIN, C_OFFSET)

# перевірка критерію Оцу прямим перебором порогів (макс. міжкласова дисперсія)
levels = np.arange(256)
best_t, best_var = 0, -1.0
for t in range(1, 255):
    w0 = pdf[:t + 1].sum()
    w1 = 1.0 - w0
    if w0 == 0 or w1 == 0:
        continue
    mu0 = (levels[:t + 1] * pdf[:t + 1]).sum() / w0
    mu1 = (levels[t + 1:] * pdf[t + 1:]).sum() / w1
    var_b = w0 * w1 * (mu0 - mu1) ** 2
    if var_b > best_var:
        best_t, best_var = t, var_b


def white_share(bw, region):
    """Частка білих пікселів у заданій області."""
    return 100 * np.count_nonzero(bw[region]) / np.count_nonzero(region)


cols = np.arange(img.shape[1])
left = (brain_mask > 0) & (cols[None, :] < img.shape[1] // 2)
right = (brain_mask > 0) & (cols[None, :] >= img.shape[1] // 2)

print("\n--- БІНАРИЗАЦІЯ ---")
print(f"Фіксований поріг T = {T_FIXED}; поріг Оцу (cv2) = {t_otsu:.0f}; "
      f"поріг за власним перебором = {best_t}; max sigma_B^2 = {best_var:.1f}")
print(f"Адаптивна бінаризація: вікно S = {S_WIN}, зсув C = {C_OFFSET}")
print(f"{'Метод':<30}{'білих, %':>10}{'зліва в мозку, %':>20}{'справа в мозку, %':>20}")
for name, bw in [("Фіксований поріг", bin_fixed), ("Оцу", bin_otsu),
                 ("Адаптивний (середнє)", bin_mean), ("Адаптивний (Гаус)", bin_gauss)]:
    print(f"{name:<30}{100 * np.count_nonzero(bw) / bw.size:>10.2f}"
          f"{white_share(bw, left):>20.2f}{white_share(bw, right):>20.2f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, bw, title in [(axes[0, 0], bin_fixed, f"Фіксований поріг T = {T_FIXED}"),
                      (axes[0, 1], bin_otsu, f"Метод Оцу, T = {t_otsu:.0f}"),
                      (axes[1, 0], bin_mean, f"Адаптивний (середнє), S = {S_WIN}, C = {C_OFFSET}"),
                      (axes[1, 1], bin_gauss, f"Адаптивний (Гаус), S = {S_WIN}, C = {C_OFFSET}")]:
    ax.imshow(bw, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/binarization_methods.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
