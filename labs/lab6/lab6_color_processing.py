#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №6
Базові операції з цифровими зображеннями, аналіз каналів та колірні простори

Варіант 2: темно-червоний колір (Red Part 2)
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, OpenCV, Pillow, Matplotlib
"""

# %% Блок 1. Імпорт бібліотек, налаштування стилю та генерація тестової сцени
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# каталоги для графіків і даних
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


def generate_synthetic_scene(height=400, width=600):
    """
    Формує тестовий кадр у форматі BGR на нейтральному сірому тлі.
    Окрім стандартних фігур, у кадр додано темно-червону пляму (імітація
    крововиливу на медичному знімку), яка потрапляє в діапазон варіанта 2.
    Повертає зображення та еталонну маску цієї плями (для перевірки).
    """
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # перший ряд: червоний круг, зелений прямокутник, синій трикутник (BGR!)
    cv2.circle(img, (100, 110), 55, (0, 0, 255), -1)
    cv2.rectangle(img, (240, 60), (360, 160), (0, 200, 0), -1)
    triangle = np.array([[440, 160], [560, 160], [500, 55]], dtype=np.int32)
    cv2.fillPoly(img, [triangle], (255, 0, 0))

    # другий ряд: жовтий прямокутник, оранжевий еліпс, блакитний круг
    cv2.rectangle(img, (190, 250), (290, 340), (0, 255, 255), -1)
    cv2.ellipse(img, (370, 295), (55, 38), 0, 0, 360, (0, 128, 255), -1)
    cv2.circle(img, (515, 295), 45, (255, 255, 0), -1)

    # темно-червона пляма з трьох накладених еліпсів, колір BGR (50, 10, 190)
    blotch_color = (50, 10, 190)
    for centre, axes, angle in [((75, 295), (48, 34), 20),
                                ((95, 310), (30, 40), -15),
                                ((60, 320), (25, 22), 0)]:
        cv2.ellipse(img, centre, axes, angle, 0, 360, blotch_color, -1)
        cv2.ellipse(gt_mask, centre, axes, angle, 0, 360, 255, -1)

    return img, gt_mask


scene_bgr, gt_blotch = generate_synthetic_scene()
cv2.imwrite("data/test_image.png", scene_bgr)
print("Тестову сцену збережено: data/test_image.png")


# %% Блок 2. Етап 1: тензорне подання зображення та розкладання на канали
img_bgr = cv2.imread("data/test_image.png")            # OpenCV читає як BGR
img_pil = Image.open("data/test_image.png")            # Pillow читає як RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

h, w, ch = img_bgr.shape
print("--- ТЕНЗОРНЕ ПОДАННЯ ЗОБРАЖЕННЯ ---")
print(f"Форма масиву (висота, ширина, канали): {img_bgr.shape}")
print(f"Кількість пікселів: {h * w}, значень у тензорі: {img_bgr.size}")
print(f"Тип даних: {img_bgr.dtype}, обсяг пам'яті: {img_bgr.nbytes / 1024:.1f} КБ")
print(f"Pillow: режим {img_pil.mode}, розмір (ширина, висота) {img_pil.size}")

# піксель, який у файлі має колір кола (100, 110): червоний
px_bgr = img_bgr[110, 100]
px_rgb = img_rgb[110, 100]
print(f"Піксель [110, 100]: у BGR = {px_bgr.tolist()}, у RGB = {px_rgb.tolist()}")

# розкладання на канали двома способами: cv2.split і зрізи NumPy
b_cv, g_cv, r_cv = cv2.split(img_bgr)
b_np, g_np, r_np = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
same_split = (np.array_equal(b_cv, b_np) and np.array_equal(g_cv, g_np)
              and np.array_equal(r_cv, r_np))
print(f"cv2.split і зрізи NumPy дають однакові канали: {same_split}")

# окремі канали у вигляді кольорових зображень (інші канали обнулено)
zeros = np.zeros_like(r_cv)
only_red = cv2.cvtColor(cv2.merge([zeros, zeros, r_cv]), cv2.COLOR_BGR2RGB)
only_green = cv2.cvtColor(cv2.merge([zeros, g_cv, zeros]), cv2.COLOR_BGR2RGB)
only_blue = cv2.cvtColor(cv2.merge([b_cv, zeros, zeros]), cv2.COLOR_BGR2RGB)

# перехід до півтонів за ваговими коефіцієнтами ITU-R BT.601
gray_manual = (0.2989 * r_np + 0.5870 * g_np + 0.1140 * b_np).astype(np.uint8)
gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
diff_gray = np.abs(gray_manual.astype(int) - gray_cv.astype(int)).max()
print(f"Півтони: max|власна формула - cv2.cvtColor| = {diff_gray} рівень")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0, 0].imshow(img_bgr)  # масив BGR без перетворення: червоний і синій міняються місцями
axes[0, 0].set_title("BGR, відображений у Matplotlib (неправильно)")
axes[0, 1].imshow(img_rgb)
axes[0, 1].set_title("RGB (правильне відображення)")
axes[0, 2].imshow(gray_manual, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title("Півтони (BT.601)")
axes[1, 0].imshow(only_red)
axes[1, 0].set_title("Канал R")
axes[1, 1].imshow(only_green)
axes[1, 1].set_title("Канал G")
axes[1, 2].imshow(only_blue)
axes[1, 2].set_title("Канал B")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/bgr_vs_rgb_channels.png", dpi=300)
plt.close()


# %% Блок 3. Етап 2: перетворення в простори HSV та CIELab
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
H, S, V = cv2.split(img_hsv)
L_ch, a_ch, b_ch = cv2.split(img_lab)

print("\n--- ДІАПАЗОНИ КАНАЛІВ (8-бітне подання OpenCV) ---")
print(f"HSV: H {H.min()}..{H.max()} (шкала 0..180), S {S.min()}..{S.max()}, V {V.min()}..{V.max()}")
print(f"Lab: L {L_ch.min()}..{L_ch.max()} (0..255 замість 0..100), "
      f"a {a_ch.min()}..{a_ch.max()}, b {b_ch.min()}..{b_ch.max()} (нуль зсунуто на 128)")

# значення кольору плями та чистого червоного у просторі HSV
for name, (x, y) in {"темно-червона пляма": (75, 295), "чисто червоний круг": (100, 110)}.items():
    hh, ss, vv = img_hsv[y, x]
    print(f"{name}: H = {hh}, S = {ss}, V = {vv}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [
    (H, "H (відтінок)", 'hsv', 0, 180),
    (S, "S (насиченість)", 'plasma', 0, 255),
    (V, "V (яскравість)", 'gray', 0, 255),
    (L_ch, "L* (світлота)", 'gray', 0, 255),
    (a_ch, "a* (зелений - червоний)", 'coolwarm', 0, 255),
    (b_ch, "b* (синій - жовтий)", 'coolwarm', 0, 255),
]
for ax, (data, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
plt.tight_layout()
plt.savefig("figures/hsv_and_lab_spaces.png", dpi=300)
plt.close()


# %% Блок 4. Етап 3: порогова сегментація в HSV (варіант 2: Red Part 2)
# діапазон варіанта 2 у шкалі OpenCV: H 0..180, S і V 0..255
lower_v2 = np.array([170, 120, 100])
upper_v2 = np.array([180, 255, 255])
mask_v2 = cv2.inRange(img_hsv, lower_v2, upper_v2)
segmented_v2 = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_v2)

# повний червоний колір: відтінок переходить через 0, тому потрібні два діапазони
mask_red_low = cv2.inRange(img_hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
mask_red_full = cv2.bitwise_or(mask_red_low, mask_v2)
segmented_full = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_red_full)

total_px = h * w
area_v2 = cv2.countNonZero(mask_v2)
area_low = cv2.countNonZero(mask_red_low)
area_full = cv2.countNonZero(mask_red_full)

# перевірка за еталонною маскою плями
inter = np.logical_and(mask_v2 > 0, gt_blotch > 0).sum()
union = np.logical_or(mask_v2 > 0, gt_blotch > 0).sum()
gt_area = cv2.countNonZero(gt_blotch)

print("\n--- СЕГМЕНТАЦІЯ ЗА HSV ---")
print(f"Діапазон варіанта 2: нижня межа {lower_v2.tolist()}, верхня межа {upper_v2.tolist()}")
print(f"{'Маска':<36}{'Площа, пікс.':>14}{'Частка кадру, %':>18}")
print(f"{'Варіант 2 (H 170..180)':<36}{area_v2:>14}{100 * area_v2 / total_px:>18.2f}")
print(f"{'Червоний, частина 1 (H 0..10)':<36}{area_low:>14}{100 * area_low / total_px:>18.2f}")
print(f"{'Червоний повний (частини 1 + 2)':<36}{area_full:>14}{100 * area_full / total_px:>18.2f}")
print(f"Еталонна площа плями: {gt_area} пікс.; перетин з маскою: {inter} пікс.; IoU = {inter / union:.4f}")
print(f"Червоний круг у маску варіанта 2 потрапляє: "
      f"{bool(mask_v2[110, 100])} (його відтінок H = {img_hsv[110, 100, 0]})")
print(f"Зведення: зображення {h}x{w}, {total_px} пікс., {img_bgr.dtype}, "
      f"{img_bgr.nbytes / 1024:.1f} КБ")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0, 0].imshow(mask_v2, cmap='gray')
axes[0, 0].set_title("Маска варіанта 2: H 170..180")
axes[0, 1].imshow(cv2.cvtColor(segmented_v2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Виділений темно-червоний об'єкт")
axes[1, 0].imshow(mask_red_full, cmap='gray')
axes[1, 0].set_title("Повний червоний: H 0..10 та 170..180 (bitwise_or)")
axes[1, 1].imshow(cv2.cvtColor(segmented_full, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Виділений червоний колір (обидва діапазони)")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/hsv_color_segmentation.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
