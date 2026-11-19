#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №8
Просторова фільтрація зображень: згладжування, шумозаглушення та підкреслення меж

Варіант 2
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, OpenCV, SciPy (scipy.ndimage), Matplotlib
"""

# %% Блок 1. Підготовка середовища, імпорт та генерація еталонної сцени
import os
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
SIGMA_G = 25.0            # СКВ адитивного гаусового шуму
P_SP = 0.08               # щільність шуму "сіль та перець"
K_BOX = 5                 # ядро Box-фільтра 5x5
K_GAUSS, SIGMA = 5, 1.5   # ядро Гауса 5x5 та його sigma
K_MED = 5                 # апертура медіанного фільтра
BIL_D, BIL_SC, BIL_SS = 9, 80, 80   # білатеральний фільтр: d, sigma_color, sigma_space
K_SHARP = 0.60            # вага підвищення різкості


def generate_reference_scene(height=400, width=600):
    """
    Еталонна сцена з контрастними фігурами, тонкими лініями та текстом:
    прямокутник, концентричні кола, штрихи, промені й напис.
    """
    img = np.full((height, width), 60, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (240, 180), 200, -1)
    cv2.rectangle(img, (40, 40), (240, 180), 255, 2)
    for r, col in [(90, 230), (70, 90), (50, 210), (30, 110), (12, 250)]:
        cv2.circle(img, (430, 120), r, col, -1)
    for i in range(8):                                    # вертикальні штрихи
        x = 50 + i * 12
        cv2.line(img, (x, 220), (x, 330), 255 if i % 2 == 0 else 120, 2)
    for ang in range(0, 180, 15):                         # промені з одного центру
        a = np.deg2rad(ang)
        p1 = (int(190 + 90 * np.cos(a)), int(290 + 90 * np.sin(a)))
        p2 = (int(190 - 90 * np.cos(a)), int(290 - 90 * np.sin(a)))
        cv2.line(img, p1, p2, 190, 1)
    cv2.putText(img, "FILTER 8", (330, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3)
    cv2.putText(img, "Spatial", (340, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 170, 2)
    cv2.line(img, (330, 365), (570, 365), 255, 1)
    return img


reference = generate_reference_scene()
cv2.imwrite("data/reference_pattern.png", reference)
print("Еталонну сцену збережено: data/reference_pattern.png")


# %% Блок 2. Моделі шумів та метрики якості (MSE, PSNR)
def add_gaussian_noise(image, sigma=SIGMA_G):
    """Адитивний гаусів шум N(0, sigma^2) з відсіканням до [0, 255]."""
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image, prop=P_SP):
    """Імпульсний шум: частка prop пікселів стає 255 (сіль) або 0 (перець)."""
    noisy = image.copy()
    rnd = np.random.random(image.shape)
    noisy[rnd < prop / 2] = 0
    noisy[(rnd >= prop / 2) & (rnd < prop)] = 255
    return noisy


def calculate_metrics(ref_img, proc_img):
    """MSE та PSNR (дБ) відносно еталона."""
    mse = np.mean((ref_img.astype(np.float64) - proc_img.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    return mse, psnr


np.random.seed(42)
noisy_g = add_gaussian_noise(reference)
noisy_sp = add_salt_and_pepper_noise(reference)

mse_g, psnr_g = calculate_metrics(reference, noisy_g)
mse_sp, psnr_sp = calculate_metrics(reference, noisy_sp)
share_sp = 100 * np.mean(noisy_sp != reference)

print("--- МОДЕЛІ ШУМІВ ---")
print(f"Гаусів шум (sigma = {SIGMA_G}): MSE = {mse_g:.2f}, PSNR = {psnr_g:.2f} дБ "
      f"(теоретично MSE ~ sigma^2 = {SIGMA_G ** 2:.0f}, менше через відсікання)")
print(f"Шум 'сіль та перець' (p = {P_SP}): MSE = {mse_sp:.2f}, PSNR = {psnr_sp:.2f} дБ, "
      f"змінено {share_sp:.2f} % пікселів")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, a, title in [(axes[0], reference, "Чисте зображення"),
                     (axes[1], noisy_g, f"Гаусів шум\nMSE = {mse_g:.1f}, PSNR = {psnr_g:.2f} дБ"),
                     (axes[2], noisy_sp, f"Сіль та перець\nMSE = {mse_sp:.1f}, PSNR = {psnr_sp:.2f} дБ")]:
    ax.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/noise_models_comparison.png", dpi=300)
plt.close()


# %% Блок 3. Шумозаглушення: Box, Gauss, Median, Bilateral
def apply_filters(noisy):
    """Повертає словник відфільтрованих зображень для одного зашумленого кадру."""
    return {
        "Box": cv2.blur(noisy, (K_BOX, K_BOX)),
        "Gauss": cv2.GaussianBlur(noisy, (K_GAUSS, K_GAUSS), SIGMA),
        "Median": cv2.medianBlur(noisy, K_MED),
        "Bilateral": cv2.bilateralFilter(noisy, BIL_D, BIL_SC, BIL_SS),
    }


results = {"Гаусів шум": apply_filters(noisy_g), "Сіль та перець": apply_filters(noisy_sp)}
noisy_by_type = {"Гаусів шум": noisy_g, "Сіль та перець": noisy_sp}

# перевірка: Box-фільтр cv2 збігається зі згорткою ядром з однаковими вагами в scipy.ndimage
box_kernel = np.full((K_BOX, K_BOX), 1.0 / K_BOX ** 2)
box_nd = ndimage.convolve(noisy_g.astype(np.float64), box_kernel, mode='mirror')
diff_box = np.abs(box_nd - results["Гаусів шум"]["Box"]).max()
# для несиметричного ядра згортка та кореляція дають різні результати
asym = np.array([[0, 0, 0], [1, 2, 4], [0, 0, 0]], dtype=np.float64) / 7
conv = ndimage.convolve(reference.astype(np.float64), asym, mode='mirror')
corr = ndimage.correlate(reference.astype(np.float64), asym, mode='mirror')
print("\n--- ПЕРЕВІРКА ЗГОРТКИ ТА КОРЕЛЯЦІЇ ---")
print(f"max|cv2.blur - ndimage.convolve| для Box 5x5 = {diff_box:.2f} рівня (різниця від округлення до uint8)")
print(f"Для несиметричного ядра max|згортка - кореляція| = {np.abs(conv - corr).max():.2f}; "
      f"для симетричних ядер Box та Gauss вони збігаються")

print("\n--- ТАБЛИЦЯ MSE ТА PSNR ---")
print(f"{'Фільтр':<22}{'Гаус: MSE':>12}{'Гаус: PSNR, дБ':>16}{'S&P: MSE':>12}{'S&P: PSNR, дБ':>16}")
print(f"{'Без обробки':<22}{mse_g:>12.2f}{psnr_g:>16.2f}{mse_sp:>12.2f}{psnr_sp:>16.2f}")
metrics = {}
for name in ["Box", "Gauss", "Median", "Bilateral"]:
    mg, pg = calculate_metrics(reference, results["Гаусів шум"][name])
    ms, ps = calculate_metrics(reference, results["Сіль та перець"][name])
    metrics[name] = (mg, pg, ms, ps)
    label = {"Box": f"Box ({K_BOX}x{K_BOX})", "Gauss": f"Gauss ({K_GAUSS}x{K_GAUSS}, s={SIGMA})",
             "Median": f"Median ({K_MED})", "Bilateral": f"Bilateral (d={BIL_D})"}[name]
    print(f"{label:<22}{mg:>12.2f}{pg:>16.2f}{ms:>12.2f}{ps:>16.2f}")
best_g = min(metrics, key=lambda k: metrics[k][0])
best_sp = min(metrics, key=lambda k: metrics[k][2])
print(f"Найкращий фільтр для гаусового шуму: {best_g}; для імпульсного шуму: {best_sp}")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, (noise_name, filt) in enumerate(results.items()):
    for j, (fname, out) in enumerate(filt.items()):
        mse_v, psnr_v = calculate_metrics(reference, out)
        axes[i, j].imshow(out, cmap='gray', vmin=0, vmax=255)
        axes[i, j].set_title(f"{noise_name}: {fname}\nMSE = {mse_v:.1f}, PSNR = {psnr_v:.2f} дБ", fontsize=10)
        axes[i, j].axis('off')
plt.tight_layout()
plt.savefig("figures/smoothing_filters_results.png", dpi=300)
plt.close()


# %% Блок 4. Підвищення різкості оператором Лапласа
laplace_kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)


def sharpen_laplacian(image, k=K_SHARP):
    """I_sharp = I + k * (Лапласіан I), обчислення у float32 з відсіканням."""
    f = image.astype(np.float32)
    lap = cv2.filter2D(f, -1, laplace_kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(f + k * lap, 0, 255).astype(np.uint8), lap


# для відновлення беремо найкращий фільтр для гаусового шуму
filtered = results["Гаусів шум"][best_g]
sharp_ref, lap_ref = sharpen_laplacian(reference)
sharp_filt, _ = sharpen_laplacian(filtered)
sharp_noisy, _ = sharpen_laplacian(noisy_g)

print("\n--- ПІДВИЩЕННЯ РІЗКОСТІ (k = {:.2f}) ---".format(K_SHARP))
print(f"Сума коефіцієнтів ядра Лапласа = {laplace_kernel.sum():.0f} (однорідні ділянки не змінюються)")
for name, a in [("Еталон", reference), ("Еталон після підвищення різкості", sharp_ref),
                (f"Гаусів шум, фільтр {best_g}", filtered),
                (f"Те саме після підвищення різкості", sharp_filt),
                ("Гаусів шум без фільтрації, з підвищенням різкості", sharp_noisy)]:
    m, p = calculate_metrics(reference, a)
    print(f"{name:<52} MSE = {m:>8.2f}, PSNR = {p:>6.2f} дБ")
grad = lambda a: np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)).max()
print(f"Максимальний градієнт Собеля: еталон {grad(reference):.0f}, після підвищення різкості {grad(sharp_ref):.0f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].imshow(reference, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("Еталонне зображення")
axes[0, 1].imshow(np.clip(lap_ref + 128, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title("Відгук оператора Лапласа (зсув +128)")
axes[1, 0].imshow(sharp_ref, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title(f"Підвищена різкість еталона (k = {K_SHARP})")
axes[1, 1].imshow(sharp_filt, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title(f"Підвищена різкість відфільтрованого кадру ({best_g})")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/sharpening_laplacian.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
