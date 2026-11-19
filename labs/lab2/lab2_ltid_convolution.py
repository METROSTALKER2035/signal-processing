#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №2
Моделювання лінійних стаціонарних дискретних систем та обчислення дискретної згортки

Варіант 2
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, SciPy, Matplotlib
"""

# %% Блок 1. Імпорт бібліотек та налаштування стилю відображення
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# каталог для графіків
os.makedirs("figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


# %% Блок 2. Параметри варіанта 2, власні функції згортки та порівняння з бібліотечними
x_bench = np.array([1, 3, -2, 2, 0, 1, -1, 4], dtype=float)    # вхідний сигнал x[n]
h_bench = np.array([2, -1, 0, 2], dtype=float)                 # імпульсна характеристика h[n]

# коефіцієнти різницевого рівняння 2-го порядку [a1, a2, b0, b1, b2]
a1, a2, b0, b1, b2 = -0.7, 0.12, 0.8, -0.4, 0.2

L_rect = 6          # тривалість прямокутного імпульсу
A_val = 1.5         # амплітуда експоненційного сигналу
alpha_val = 0.8     # основа експоненти
M_ma = 5            # довжина ковзного середнього


def manual_linear_convolution(x_seq, h_seq):
    """Лінійна згортка за означенням: y[n] = sum_k x[k] * h[n-k].
    Довжина результату N_x + N_h - 1."""
    len_x = len(x_seq)
    len_h = len(h_seq)
    y_out = np.zeros(len_x + len_h - 1)
    for n in range(len(y_out)):
        # k пробігає лише ті значення, для яких h[n-k] існує
        for k in range(max(0, n - len_h + 1), min(n, len_x - 1) + 1):
            y_out[n] += x_seq[k] * h_seq[n - k]
    return y_out


def manual_circular_convolution(x_seq, h_seq):
    """Кругова згортка довжини N = max(N_x, N_h):
    y_c[n] = sum_m x[m] * h[(n - m) mod N]. Коротку послідовність доповнюємо нулями."""
    len_n = max(len(x_seq), len(h_seq))
    x_p = np.pad(x_seq, (0, len_n - len(x_seq)))
    h_p = np.pad(h_seq, (0, len_n - len(h_seq)))
    y_out = np.zeros(len_n)
    for n in range(len_n):
        for m in range(len_n):
            y_out[n] += x_p[m] * h_p[(n - m) % len_n]
    return y_out


def fmt(v):
    """Число для запису розрахунку: від'ємні беремо в дужки"""
    return f"({v:g})" if v < 0 else f"{v:g}"


def print_conv_steps(x_seq, h_seq, n_steps):
    """Покроковий розрахунок перших n_steps відліків згортки"""
    for n in range(n_steps):
        ks = [k for k in range(len(x_seq)) if 0 <= n - k < len(h_seq)]
        symbolic = " + ".join(f"x[{k}]·h[{n - k}]" for k in ks)
        numeric = " + ".join(f"{fmt(x_seq[k])}·{fmt(h_seq[n - k])}" for k in ks)
        values = " + ".join(fmt(x_seq[k] * h_seq[n - k]) for k in ks)
        total = sum(x_seq[k] * h_seq[n - k] for k in ks)
        if len(ks) > 1:
            print(f"y[{n}] = {symbolic} = {numeric} = {values} = {total:g}")
        else:
            print(f"y[{n}] = {symbolic} = {numeric} = {total:g}")


def measure_time(func, *args, repeats=2000):
    """Середній час одного виклику функції, мкс"""
    t_start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    return (time.perf_counter() - t_start) / repeats * 1e6


y_custom = manual_linear_convolution(x_bench, h_bench)
y_numpy = np.convolve(x_bench, h_bench)
y_scipy = signal.convolve(x_bench, h_bench, mode='full')

diff_np = np.max(np.abs(y_custom - y_numpy))
diff_sc = np.max(np.abs(y_custom - y_scipy))

t_custom = measure_time(manual_linear_convolution, x_bench, h_bench)
t_numpy = measure_time(np.convolve, x_bench, h_bench)
t_scipy = measure_time(signal.convolve, x_bench, h_bench)

print("--- ЛІНІЙНА ЗГОРТКА: ВЛАСНА РЕАЛІЗАЦІЯ ТА БІБЛІОТЕЧНІ ФУНКЦІЇ ---")
print(f"x[n]      = {x_bench.astype(int).tolist()}  (N_x = {len(x_bench)})")
print(f"h[n]      = {h_bench.astype(int).tolist()}  (N_h = {len(h_bench)})")
print(f"Довжина y = N_x + N_h - 1 = {len(y_custom)}")
print(f"y_custom  = {y_custom.astype(int).tolist()}")
print(f"y_numpy   = {y_numpy.astype(int).tolist()}")
print(f"y_scipy   = {y_scipy.astype(int).tolist()}")
print(f"Час виконання: власна {t_custom:.2f} мкс, np.convolve {t_numpy:.2f} мкс, "
      f"scipy.signal.convolve {t_scipy:.2f} мкс")
print(f"Макс. відхилення від NumPy: {diff_np:.2e}, від SciPy: {diff_sc:.2e}")

print("\nПерші чотири такти згортки (аналітичний розрахунок):")
print_conv_steps(x_bench, h_bench, 4)

# складність прямого алгоритму: кількість множень N_x * N_h, час має рости як N^2
print("\nЗалежність часу виконання від довжини сигналів (N_x = N_h = N):")
print(f"{'N':>5} {'N_x*N_h':>10} {'t, мс':>9} {'t / t_prev':>11}")
np.random.seed(1)
t_prev = None
for n_len in (100, 200, 400, 800):
    x_t = np.random.rand(n_len)
    h_t = np.random.rand(n_len)
    t_best = min(measure_time(manual_linear_convolution, x_t, h_t, repeats=1) for _ in range(3))
    ratio = f"{t_best / t_prev:11.2f}" if t_prev else f"{'-':>11}"
    print(f"{n_len:5d} {n_len * n_len:10d} {t_best / 1e3:9.2f} {ratio}")
    t_prev = t_best

# графік порівняння
n_idx = np.arange(len(y_custom))

plt.figure(figsize=(10, 4))
plt.stem(n_idx, y_custom, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='Власна реалізація')
plt.plot(n_idx, y_numpy, 'rx', markersize=10, label='np.convolve')
plt.plot(n_idx, y_scipy, 'g+', markersize=12, label='scipy.signal.convolve')
plt.title("Порівняння результатів лінійної дискретної згортки")
plt.xlabel("Індекс відліку, n")
plt.ylabel("Амплітуда, y[n]")
plt.xticks(n_idx)
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("figures/linear_vs_builtin_conv.png", dpi=300)
plt.close()


# %% Блок 3. Елементарні системи: ковзне середнє та диференціатор
n_pts = 16
n_arr = np.arange(n_pts)

# вхідні сигнали: x_rect[n] = u[n] - u[n - L_rect], x_exp[n] = A * alpha^n * u[n]
x_rect = np.where(n_arr < L_rect, 1.0, 0.0)
x_exp = A_val * alpha_val ** n_arr

# імпульсні характеристики систем
h_ma = np.ones(M_ma) / M_ma         # ковзне середнє довжини M
h_diff = np.array([1.0, -1.0])      # диференціатор y[n] = x[n] - x[n-1]

# відгуки (повна згортка обрізається до n_pts відліків)
y_rect_ma = signal.convolve(x_rect, h_ma, mode='full')[:n_pts]
y_rect_diff = signal.convolve(x_rect, h_diff, mode='full')[:n_pts]
y_exp_ma = signal.convolve(x_exp, h_ma, mode='full')[:n_pts]
y_exp_diff = signal.convolve(x_exp, h_diff, mode='full')[:n_pts]

np.set_printoptions(precision=3, suppress=True, linewidth=130)
print("\n--- ЕЛЕМЕНТАРНІ СИСТЕМИ ---")
print(f"Прямокутний імпульс: L = {L_rect}; експонента: A = {A_val}, alpha = {alpha_val}; M = {M_ma}")
print("x_rect       =", x_rect)
print("MA(x_rect)   =", y_rect_ma)
print("diff(x_rect) =", y_rect_diff)
print("x_exp        =", x_exp)
print("MA(x_exp)    =", y_exp_ma)
print("diff(x_exp)  =", y_exp_diff)

# перевірка власною згорткою та прямими формулами
chk_ma = max(np.max(np.abs(manual_linear_convolution(x_rect, h_ma)[:n_pts] - y_rect_ma)),
             np.max(np.abs(manual_linear_convolution(x_exp, h_ma)[:n_pts] - y_exp_ma)))
chk_diff = max(np.max(np.abs(np.diff(x_rect, prepend=0.0) - y_rect_diff)),
               np.max(np.abs(np.diff(x_exp, prepend=0.0) - y_exp_diff)))
print(f"Відхилення від власної згортки (MA): {chk_ma:.2e}, від різниці np.diff (диференціатор): {chk_diff:.2e}")

# для експоненти при n >= 1: y_diff[n] = A * alpha^(n-1) * (alpha - 1)
diff_exp_formula = A_val * alpha_val ** (n_arr[1:] - 1) * (alpha_val - 1)
print(f"Диференціатор для експоненти: відхилення від A*alpha^(n-1)*(alpha-1) = "
      f"{np.max(np.abs(y_exp_diff[1:] - diff_exp_formula)):.2e}")

# АЧХ у крайніх точках: ω = 0 та ω = π
_, h_ma_f = signal.freqz(h_ma, 1, worN=np.array([0.0, np.pi]))
_, h_diff_f = signal.freqz(h_diff, 1, worN=np.array([0.0, np.pi]))
print(f"|H_ma(0)| = {abs(h_ma_f[0]):.3f}, |H_ma(pi)| = {abs(h_ma_f[1]):.3f}; "
      f"|H_diff(0)| = {abs(h_diff_f[0]):.3f}, |H_diff(pi)| = {abs(h_diff_f[1]):.3f}")

# графіки 2x3
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
rows = [
    ("Прямокутний імпульс", x_rect, y_rect_ma, y_rect_diff),
    ("Експоненційний сигнал", x_exp, y_exp_ma, y_exp_diff),
]
for row, (name, x_in, y_ma, y_df) in enumerate(rows):
    panels = [
        (x_in, 'k', f"Вхід: {name.lower()}"),
        (y_ma, 'b', f"Ковзне середнє (M = {M_ma})"),
        (y_df, 'r', "Диференціатор y[n] = x[n] - x[n-1]"),
    ]
    for col, (data, c, title) in enumerate(panels):
        ax = axes[row, col]
        ax.stem(n_arr, data, linefmt=c + '-', markerfmt=c + 'o', basefmt='k-')
        ax.set_title(title)
        ax.set_xlabel("Індекс відліку, n")
        ax.set_ylabel("Амплітуда")
        ax.set_xticks(n_arr)
        ax.grid(True)

plt.tight_layout()
plt.savefig("figures/elementary_systems_resp.png", dpi=300)
plt.close()


# %% Блок 4. Різницеве рівняння 2-го порядку: імпульсна та перехідна характеристики
# y[n] + a1*y[n-1] + a2*y[n-2] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
b_coeffs = [b0, b1, b2]
a_coeffs = [1.0, a1, a2]


def simulate_difference_equation(b, a, x_input):
    """Рекурентний розрахунок виходу при нульових початкових умовах:
    y[n] = (sum_m b[m]*x[n-m] - sum_{k>=1} a[k]*y[n-k]) / a[0]"""
    y_out = np.zeros(len(x_input))
    for n in range(len(x_input)):
        acc = 0.0
        for m in range(len(b)):
            if n - m >= 0:
                acc += b[m] * x_input[n - m]
        for k in range(1, len(a)):
            if n - k >= 0:
                acc -= a[k] * y_out[n - k]
        y_out[n] = acc / a[0]
    return y_out


N_sim = 25
delta_input = np.zeros(N_sim)
delta_input[0] = 1.0                # δ[n]
step_input = np.ones(N_sim)         # u[n]

h_sim = simulate_difference_equation(b_coeffs, a_coeffs, delta_input)
g_sim = simulate_difference_equation(b_coeffs, a_coeffs, step_input)

# перевірка бібліотечною функцією
h_scipy = signal.lfilter(b_coeffs, a_coeffs, delta_input)
g_scipy = signal.lfilter(b_coeffs, a_coeffs, step_input)

poles = np.roots(a_coeffs)
gain_dc = sum(b_coeffs) / sum(a_coeffs)     # H(z) при z = 1, усталене значення g[n]

np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("\n--- СИСТЕМА 2-ГО ПОРЯДКУ ---")
print(f"y[n] = {b0}*x[n] + ({b1})*x[n-1] + {b2}*x[n-2] - ({a1})*y[n-1] - {a2}*y[n-2]")
print(f"Полюси: {np.sort(poles)} (|z| < 1, система стійка)")
print("h[n] (пряма реалізація), n = 0..7:", h_sim[:8])
print("h[n] (lfilter),          n = 0..7:", h_scipy[:8])
print("g[n] (пряма реалізація), n = 0..7:", g_sim[:8])
print("g[n] (lfilter),          n = 0..7:", g_scipy[:8])
print(f"Макс. розбіжність: h - {np.max(np.abs(h_sim - h_scipy)):.2e}, "
      f"g - {np.max(np.abs(g_sim - g_scipy)):.2e}")
print(f"g[n] = cumsum(h[n]): відхилення {np.max(np.abs(g_sim - np.cumsum(h_sim))):.2e}")
print(f"Усталене значення g[n] = H(1) = {gain_dc:.6f}; g[{N_sim - 1}] = {g_sim[-1]:.6f}")

n_sim = np.arange(N_sim)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].stem(n_sim, h_sim, linefmt='b-', markerfmt='bo', basefmt='k-', label='Пряма реалізація')
axes[0].plot(n_sim, h_scipy, 'rx', markersize=9, label='scipy.signal.lfilter')
axes[0].set_title("Імпульсна характеристика h[n]")
axes[0].set_xlabel("Індекс відліку, n")
axes[0].set_ylabel("Амплітуда, h[n]")
axes[0].grid(True)
axes[0].legend(loc='upper right')

axes[1].stem(n_sim, g_sim, linefmt='g-', markerfmt='go', basefmt='k-', label='Пряма реалізація')
axes[1].plot(n_sim, g_scipy, 'mx', markersize=9, label='scipy.signal.lfilter')
axes[1].axhline(gain_dc, color='gray', ls=':', label=f"H(1) = {gain_dc:.3f}")
axes[1].set_title("Перехідна характеристика g[n]")
axes[1].set_xlabel("Індекс відліку, n")
axes[1].set_ylabel("Амплітуда, g[n]")
axes[1].set_ylim(0, 1.8)
axes[1].grid(True)
axes[1].legend(loc='upper center', ncol=3)

plt.tight_layout()
plt.savefig("figures/system_2nd_order_resp.png", dpi=300)
plt.close()


# %% Блок 5. Кругова та лінійна згортка (без доповнення та з Zero-Padding)
x_circ_test = x_bench
h_circ_test = h_bench

y_lin = manual_linear_convolution(x_circ_test, h_circ_test)
y_circ = manual_circular_convolution(x_circ_test, h_circ_test)      # N_c = max(N_x, N_h)

N_c = max(len(x_circ_test), len(h_circ_test))
L_pad = len(x_circ_test) + len(h_circ_test) - 1                     # мінімальна довжина без накладання

x_padded = np.pad(x_circ_test, (0, L_pad - len(x_circ_test)))
h_padded = np.pad(h_circ_test, (0, L_pad - len(h_circ_test)))
y_circ_padded = manual_circular_convolution(x_padded, h_padded)

# перевірка через ДПФ: циклічна згортка = ОДПФ(ДПФ(x) * ДПФ(h))
y_fft = np.real(np.fft.ifft(np.fft.fft(x_padded) * np.fft.fft(h_padded)))

print("\n--- КРУГОВА ТА ЛІНІЙНА ЗГОРТКА ---")
print(f"Лінійна згортка (довжина {len(y_lin)}):      {y_lin.astype(int).tolist()}")
print(f"Кругова, N = {N_c} без доповнення:          {y_circ.astype(int).tolist()}")
print(f"Кругова, L = {L_pad} з Zero-Padding:        {y_circ_padded.astype(int).tolist()}")
print(f"Відхилення кругової (L = {L_pad}) від лінійної: {np.max(np.abs(y_circ_padded - y_lin)):.2e}, "
      f"від ДПФ-розрахунку: {np.max(np.abs(y_circ_padded - y_fft)):.2e}")

# накладання при N = 8: y_c[n] = y[n] + y[n + N] для n = 0 .. N_h - 2
tail = len(y_lin) - N_c
print(f"Накладання хвоста лінійної згортки на перші {tail} відліки:")
for n in range(tail):
    print(f"  y_c[{n}] = y[{n}] + y[{n + N_c}] = {fmt(y_lin[n])} + {fmt(y_lin[n + N_c])} = {y_circ[n]:g}")

# як змінюється результат при різній довжині доповнення
print(f"\n{'L':>3} {'співпадає з лінійною':>22} {'спотворених відліків':>22}")
for L_try in range(N_c, L_pad + 2):
    x_try = np.pad(x_circ_test, (0, L_try - len(x_circ_test)))
    h_try = np.pad(h_circ_test, (0, L_try - len(h_circ_test)))
    y_try = manual_circular_convolution(x_try, h_try)
    y_ref = np.pad(y_lin, (0, max(0, L_try - len(y_lin))))[:L_try]
    n_bad = int(np.sum(np.abs(y_try - y_ref) > 1e-9))
    print(f"{L_try:3d} {'так' if n_bad == 0 else 'ні':>22} {n_bad:22d}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].stem(np.arange(len(y_circ)), y_circ, linefmt='r-', markerfmt='ro', basefmt='k-',
             label=f'Кругова (N = {N_c})')
axes[0].plot(np.arange(len(y_lin)), y_lin, 'k--', marker='.', label='Лінійна')
axes[0].set_title(f"Кругова згортка без доповнення (N = {N_c}) та лінійна")
axes[0].set_xlabel("Індекс відліку, n")
axes[0].set_ylabel("Амплітуда, y[n]")
axes[0].set_xticks(np.arange(len(y_lin)))
axes[0].grid(True)
axes[0].legend(loc='lower right')

axes[1].stem(np.arange(len(y_circ_padded)), y_circ_padded, linefmt='g-', markerfmt='go', basefmt='k-',
             label=f'Кругова з Zero-Padding (L = {L_pad})')
axes[1].plot(np.arange(len(y_lin)), y_lin, 'kx', markersize=10, label='Лінійна')
axes[1].set_title(f"Кругова згортка з Zero-Padding (L = {L_pad}) та лінійна")
axes[1].set_xlabel("Індекс відліку, n")
axes[1].set_ylabel("Амплітуда, y[n]")
axes[1].set_xticks(np.arange(len(y_lin)))
axes[1].grid(True)
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig("figures/circular_vs_linear_conv.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
