#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №10
Математична морфологія бінарних та напівтонових зображень

Варіант 2: дефектоскопія доріжок друкованих плат
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, OpenCV, SciPy, Matplotlib
"""

# %% Блок 1. Підготовка середовища, імпорт та генерація тестових зображень
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
K_SIZE = 5                      # розмір структуруючого елемента (СЕ) 5x5
N_ITER = 1                      # кількість ітерацій
TW = 12                         # ширина доріжки, пікс.
TH_BLACKHAT, TH_TOPHAT = 40, 40  # пороги бінаризації відгуків Black-Hat та Top-Hat

# розриви доріжок: (x центра, y осі доріжки, ширина розриву, пікс.)
BREAKS = [(130, 70, 2), (420, 70, 7), (200, 150, 3), (480, 150, 5),
          (110, 230, 4), (400, 230, 9), (220, 310, 4), (450, 310, 3)]


def generate_morphology_test_scenes(height=400, width=600):
    """
    Бінарна та напівтонова сцени з доріжками друкованої плати.
    Повертає ідеальну маску топології, бінарний кадр із шумом
    (сіль: 120 цяток на тлі, перець: 60 пор усередині доріжок)
    та напівтоновий кадр з нерівномірним освітленням.
    """
    layout = np.zeros((height, width), dtype=np.uint8)
    for y in (70, 150, 230, 310):                               # горизонтальні доріжки
        cv2.rectangle(layout, (40, y - TW // 2), (560, y + TW // 2 - 1), 255, -1)
        cv2.circle(layout, (40, y), 18, 255, -1)                # контактні площадки
        cv2.circle(layout, (560, y), 18, 255, -1)
    cv2.rectangle(layout, (294, 70 - TW // 2), (294 + TW - 1, 310 + TW // 2 - 1), 255, -1)  # з'єднувач

    breaks_zone = np.zeros_like(layout)
    for bx, by, w in BREAKS:                                    # прорізаємо розриви
        x0 = bx - w // 2
        layout[by - TW // 2 - 1:by + TW // 2 + 1, x0:x0 + w] = 0
        breaks_zone[by - TW // 2 - 8:by + TW // 2 + 8, x0 - 8:x0 + w + 8] = 255

    rng = np.random.default_rng(42)
    raw = layout.copy()
    # шум "сіль": окремі яскраві цятки на тлі
    bg = np.argwhere((cv2.dilate(layout, np.ones((5, 5), np.uint8)) == 0))
    for y, x in bg[rng.choice(len(bg), 120, replace=False)]:
        raw[y, x] = 255
    # шум "перець": пори 2x2 усередині доріжок, подалі від розривів
    interior = cv2.erode(layout, np.ones((7, 7), np.uint8))
    interior[breaks_zone > 0] = 0
    inner = np.argwhere(interior > 0)
    for y, x in inner[rng.choice(len(inner), 60, replace=False)]:
        raw[y:y + 2, x:x + 2] = 0

    # напівтоновий кадр: мідь яскрава, підкладка темна, параболічне освітлення,
    # кілька яскравих крапинок пилу, розмиття та шум
    gray = np.where(layout > 0, 200.0, 50.0).astype(np.float32)
    far = np.argwhere(cv2.erode(255 - layout, np.ones((11, 11), np.uint8)) > 0)
    for y, x in far[rng.choice(len(far), 14, replace=False)]:
        gray[y - 1:y + 2, x - 1:x + 2] = 255
    yy, xx = np.mgrid[0:height, 0:width]
    illum = 1.0 - 0.4 * (0.5 * ((xx - width / 2) / (width / 2)) ** 2 + 0.5 * ((yy - height / 2) / (height / 2)) ** 2)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0) * illum + rng.normal(0, 4, gray.shape)
    return layout, raw, np.clip(gray, 0, 255).astype(np.uint8)


layout, img_raw, img_gray = generate_morphology_test_scenes()
cv2.imwrite("data/binary_tracks_raw.png", img_raw)
cv2.imwrite("data/grayscale_pcb_uneven.png", img_gray)
print("Тестові зображення збережено: data/binary_tracks_raw.png, data/grayscale_pcb_uneven.png")
print(f"Розмір кадру: {img_raw.shape[1]}x{img_raw.shape[0]}, розривів у топології: {len(BREAKS)}, "
      f"ширини розривів: {[b[2] for b in BREAKS]} пікс.")


# %% Блок 2. Структуруючі елементи
se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (K_SIZE, K_SIZE))
se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (K_SIZE, K_SIZE))
se_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_SIZE, K_SIZE))

print("\n--- СТРУКТУРУЮЧІ ЕЛЕМЕНТИ {}x{} ---".format(K_SIZE, K_SIZE))
for name, se in [("MORPH_RECT (елемент варіанта 2)", se_rect), ("MORPH_CROSS", se_cross),
                 ("MORPH_ELLIPSE", se_ellipse)]:
    print(f"{name}: одиниць {int(se.sum())} з {se.size}")
    print(se)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, se, title in [(axes[0], se_rect, "MORPH_RECT 5x5"), (axes[1], se_cross, "MORPH_CROSS 5x5"),
                      (axes[2], se_ellipse, "MORPH_ELLIPSE 5x5")]:
    ax.imshow(se, cmap='gray_r', vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, K_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K_SIZE, 1), minor=True)
    ax.grid(which='minor', color='#888888', linewidth=0.8, linestyle='-')
    ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("figures/structuring_elements.png", dpi=300)
plt.close()


# %% Блок 3. Ерозія та діляція
img_eroded = cv2.erode(img_raw, se_rect, iterations=N_ITER)
img_dilated = cv2.dilate(img_raw, se_rect, iterations=N_ITER)

area_raw = cv2.countNonZero(img_raw)
area_er = cv2.countNonZero(img_eroded)
area_dil = cv2.countNonZero(img_dilated)


def break_closed(mask, bx, by, w):
    """Розрив 'закритий', якщо вздовж осі доріжки на ділянці розриву всі пікселі ненульові."""
    x0 = bx - w // 2
    return bool(mask[by, x0:x0 + w].all())


print("\n--- ЕРОЗІЯ ТА ДІЛЯЦІЯ (RECT {0}x{0}, {1} ітер.) ---".format(K_SIZE, N_ITER))
print(f"Початкова площа: {area_raw} пікс.")
print(f"Після ерозії: {area_er} пікс. ({100 * (area_er - area_raw) / area_raw:+.1f} %)")
print(f"Після діляції: {area_dil} пікс. ({100 * (area_dil - area_raw) / area_raw:+.1f} %)")
print("Розриви після діляції (закритий = розрив ширше за СЕ не зникає):")
for bx, by, w in BREAKS:
    print(f"  x = {bx:3d}, y = {by:3d}, ширина {w} пікс.: "
          f"{'закритий' if break_closed(img_dilated, bx, by, w) else 'залишився відкритим'}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, a, title in [(axes[0], img_raw, f"Вихідний кадр, S = {area_raw}"),
                     (axes[1], img_eroded, f"Ерозія, S = {area_er} ({100 * (area_er - area_raw) / area_raw:+.1f} %)"),
                     (axes[2], img_dilated, f"Діляція, S = {area_dil} ({100 * (area_dil - area_raw) / area_raw:+.1f} %)")]:
    ax.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/erosion_dilation_comp.png", dpi=300)
plt.close()


# %% Блок 4. Відкриття, закриття та зв'язні компоненти
img_opened = cv2.morphologyEx(img_raw, cv2.MORPH_OPEN, se_rect, iterations=N_ITER)
img_closed = cv2.morphologyEx(img_raw, cv2.MORPH_CLOSE, se_rect, iterations=N_ITER)
img_cleaned = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, se_rect, iterations=N_ITER)


def count_components(mask):
    """Кількість зв'язних компонент (8-зв'язність) без фону, мітки та статистика."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return n - 1, labels, stats


n_raw, _, st_raw = count_components(img_raw)
n_open, _, _ = count_components(img_opened)
n_close, _, _ = count_components(img_closed)
n_clean, lab_clean, st_clean = count_components(img_cleaned)

err = lambda m: int(np.count_nonzero(m != layout))
big_raw = int(np.sum(st_raw[1:, cv2.CC_STAT_AREA] >= 100))

print("\n--- ВІДКРИТТЯ ТА ЗАКРИТТЯ ---")
print(f"Зв'язних компонент: вихідний кадр {n_raw} (з них площею >= 100 пікс.: {big_raw}), "
      f"після відкриття {n_open}, після закриття {n_close}, після закриття та відкриття {n_clean}")
print(f"Хибних пікселів відносно ідеальної топології: вихідний {err(img_raw)}, "
      f"відкриття {err(img_opened)}, закриття {err(img_closed)}, повне очищення {err(img_cleaned)}")
print(f"Ідемпотентність: open(open(A)) == open(A): "
      f"{np.array_equal(cv2.morphologyEx(img_opened, cv2.MORPH_OPEN, se_rect), img_opened)}, "
      f"close(close(A)) == close(A): "
      f"{np.array_equal(cv2.morphologyEx(img_closed, cv2.MORPH_CLOSE, se_rect), img_closed)}")
closed_after = [break_closed(img_cleaned, bx, by, w) for bx, by, w in BREAKS]
print(f"Розривів, прихованих очищенням (закриття заповнює їх): {sum(closed_after)} з {len(BREAKS)}")

cmap_lab = plt.get_cmap('nipy_spectral').copy()
cmap_lab.set_under('black')
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].imshow(img_raw, cmap='gray')
axes[0, 0].set_title(f"Вихідний кадр: {n_raw} компонент (доріжки та шум)")
axes[0, 1].imshow(img_opened, cmap='gray')
axes[0, 1].set_title(f"Відкриття: {n_open} компонент (шум-сіль прибрано)")
axes[1, 0].imshow(img_closed, cmap='gray')
axes[1, 0].set_title(f"Закриття: {n_close} компонент (пори заповнено)")
axes[1, 1].imshow(np.where(lab_clean > 0, lab_clean, -1), cmap=cmap_lab, vmin=0, vmax=max(n_clean, 1))
axes[1, 1].set_title(f"Закриття + відкриття: {n_clean} компонент (колір = мітка)")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/opening_closing_clean.png", dpi=300)
plt.close()


# %% Блок 5. Градієнт, Top-Hat та Black-Hat; конвеєр варіанта Closing -> Black-Hat
se_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph_gradient = cv2.morphologyEx(img_cleaned, cv2.MORPH_GRADIENT, se_grad)
outer = cv2.subtract(cv2.dilate(img_cleaned, se_grad), img_cleaned)
inner = cv2.subtract(img_cleaned, cv2.erode(img_cleaned, se_grad))
print("\n--- МОРФОЛОГІЧНИЙ ГРАДІЄНТ (еліпс 3x3) ---")
print(f"Пікселів контуру: градієнт {cv2.countNonZero(morph_gradient)}, зовнішній {cv2.countNonZero(outer)}, "
      f"внутрішній {cv2.countNonZero(inner)}")

# конвеєр варіанта 2: закриття, потім Black-Hat = закриття - вихідне зображення
closing_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, se_rect, iterations=N_ITER)
blackhat_gray = cv2.subtract(closing_gray, img_gray)
diff_bh = int(np.abs(blackhat_gray.astype(int)
                     - cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, se_rect, iterations=N_ITER).astype(int)).max())
tophat_gray = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, se_rect, iterations=N_ITER)
blackhat_big = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, np.ones((11, 11), np.uint8))
blackhat_bin = cv2.morphologyEx(img_raw, cv2.MORPH_BLACKHAT, se_rect, iterations=N_ITER)

mask_bh = (blackhat_gray > TH_BLACKHAT).astype(np.uint8) * 255
mask_th = (tophat_gray > TH_TOPHAT).astype(np.uint8) * 255
mask_big = (blackhat_big > TH_BLACKHAT).astype(np.uint8) * 255


def evaluate_breaks(mask, min_area=10):
    """Для кожного відомого розриву перевіряє, чи відгук займає >= 50 % його площі;
    рахує також компоненти відгуку (площею >= min_area), що не збігаються з розривами."""
    found = []
    for bx, by, w in BREAKS:
        x0 = bx - w // 2
        win = mask[by - TW // 2 - 1:by + TW // 2 + 1, x0:x0 + w]
        found.append(np.count_nonzero(win) >= 0.5 * w * TW)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    zone = np.zeros_like(mask, dtype=bool)
    for bx, by, w in BREAKS:
        zone[by - TW // 2 - 3:by + TW // 2 + 3, bx - w // 2 - 3:bx + w - w // 2 + 3] = True
    false_alarms = sum(1 for i in range(1, n)
                       if st[i, cv2.CC_STAT_AREA] >= min_area and not zone[lab == i].any())
    small = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < min_area)
    return found, false_alarms, small


found_5, fa_5, small_5 = evaluate_breaks(mask_bh)
found_11, fa_11, small_11 = evaluate_breaks(mask_big)
found_bin, fa_bin, small_bin = evaluate_breaks((blackhat_bin > 0).astype(np.uint8) * 255)
n_dust, _, st_dust = count_components(mask_th)
dust_found = int(np.sum(st_dust[1:, cv2.CC_STAT_AREA] >= 4))

print("\n--- КОНВЕЄР ВАРІАНТА 2: CLOSING -> BLACK-HAT (RECT {0}x{0}) ---".format(K_SIZE))
print(f"cv2.MORPH_BLACKHAT збігається з (закриття - вихідне): max різниця {diff_bh} рівнів")
print(f"{'Розрив (x, y)':<16}{'ширина':>8}{'бінарний 5x5':>16}{'півтон. 5x5':>14}{'півтон. 11x11':>16}")
for (bx, by, w), fb, f5, f11 in zip(BREAKS, found_bin, found_5, found_11):
    yn = lambda v: "так" if v else "ні"
    print(f"({bx:3d}, {by:3d})      {w:>8}{yn(fb):>16}{yn(f5):>14}{yn(f11):>16}")
print(f"Виявлено розривів: бінарний кадр {sum(found_bin)} з {len(BREAKS)}, півтоновий (СЕ 5x5) {sum(found_5)} з {len(BREAKS)}, "
      f"півтоновий (СЕ 11x11) {sum(found_11)} з {len(BREAKS)}")
print(f"Хибних компонент (площа >= 10 пікс., поза розривами): бінарний {fa_bin}, півтоновий 5x5 {fa_5}, 11x11 {fa_11}")
print(f"Малих компонент (пори, площа < 10 пікс.) у бінарному відгуку: {small_bin}")
print(f"Top-Hat (СЕ 5x5, поріг {TH_TOPHAT}): виявлено яскравих крапинок пилу {dust_found} (внесено 14)")
print(f"Площа виявлених дефектів Black-Hat (поріг {TH_BLACKHAT}, СЕ 5x5): {cv2.countNonZero(mask_bh)} пікс.")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(img_cleaned, cmap='gray')
axes[0].set_title("Очищений бінарний кадр")
axes[1].imshow(morph_gradient, cmap='gray')
axes[1].set_title(f"Морфологічний градієнт: {cv2.countNonZero(morph_gradient)} пікс. контуру")
axes[2].imshow(blackhat_bin, cmap='gray')
axes[2].set_title("Black-Hat бінарного кадру (розриви та пори)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/morphological_grad.png", dpi=300)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
axes[0, 0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("Півтоновий кадр (нерівномірне освітлення)")
axes[0, 1].imshow(tophat_gray, cmap='gray')
axes[0, 1].set_title("White Top-Hat (СЕ 5x5): яскраві дрібні об'єкти")
axes[0, 2].imshow(mask_th, cmap='gray')
axes[0, 2].set_title(f"Top-Hat, поріг {TH_TOPHAT}")
axes[1, 0].imshow(blackhat_gray, cmap='gray')
axes[1, 0].set_title("Black-Hat (СЕ 5x5): розриви")
axes[1, 1].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
for (bx, by, w), ok in zip(BREAKS, found_5):
    axes[1, 1].add_patch(Rectangle((bx - w // 2 - 6, by - TW // 2 - 6), w + 12, TW + 12, fill=False,
                                   edgecolor='lime' if ok else 'red', linewidth=2))
axes[1, 1].set_title(f"Виявлено {sum(found_5)} з {len(BREAKS)} (зелені), пропущено (червоні)")
axes[1, 2].imshow(mask_big, cmap='gray')
axes[1, 2].set_title(f"Black-Hat з СЕ 11x11: {sum(found_11)} з {len(BREAKS)}")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig("figures/tophat_blackhat_comp.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
