#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №5
Статистичний та кореляційний аналіз випадкових дискретних сигналів

Варіант 2
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, SciPy, Matplotlib
"""

# %% Блок 1. Імпорт бібліотек та налаштування стилю відображення
import os
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


# %% Блок 2. Параметри варіанта 2, генерація шумів та статистичні характеристики
a_param = 0.85        # коефіцієнт зворотного зв'язку AR(1)
b_param = 0.52        # коефіцієнт підсилення AR(1)
sigma_v = 1.2         # СКВ білого шуму, В
N_samples = 10000     # довжина реалізації

L_s = 70              # довжина зондуючого сигналу, відліків
n0 = 220              # затримка сигналу в каналі, відліків
sigma_ch = 2.0        # СКВ шуму каналу, В

np.random.seed(42)    # щоб результат відтворювався

# білий гаусів шум v[n] ~ N(0, sigma_v^2)
v_white = np.random.normal(0.0, sigma_v, N_samples)

# марківський процес AR(1): x[n] = a*x[n-1] + b*v[n]
x_markov = signal.lfilter([b_param], [1.0, -a_param], v_white)


def calculate_statistics(seq_data, name_str):
    """Вибіркове середнє, незсунена дисперсія (ddof=1) та СКВ послідовності"""
    mean_val = np.mean(seq_data)
    var_val = np.var(seq_data, ddof=1)
    std_val = np.sqrt(var_val)
    print(f"{name_str:>26}: середнє = {mean_val:8.4f} В, дисперсія = {var_val:7.4f} В^2, СКВ = {std_val:7.4f} В")
    return mean_val, var_val, std_val


# теоретична дисперсія марківського процесу: b^2 * sigma_v^2 / (1 - a^2)
var_markov_theor = b_param ** 2 * sigma_v ** 2 / (1.0 - a_param ** 2)

print("--- СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ---")
print(f"a = {a_param}, b = {b_param}, sigma_v = {sigma_v} В, N = {N_samples}; "
      f"|a| < 1: {abs(a_param) < 1}")
mean_w, var_w, std_w = calculate_statistics(v_white, "Білий шум v[n]")
mean_m, var_m, std_m = calculate_statistics(x_markov, "Марківський процес x[n]")
print(f"Теоретичні значення: білий шум - M = 0, D = {sigma_v ** 2:.4f} В^2, СКВ = {sigma_v:.4f} В; "
      f"AR(1) - M = 0, D = {var_markov_theor:.4f} В^2, СКВ = {np.sqrt(var_markov_theor):.4f} В")
print(f"Відносне відхилення дисперсій: білий шум {100 * (var_w / sigma_v ** 2 - 1):+.2f} %, "
      f"AR(1) {100 * (var_m / var_markov_theor - 1):+.2f} %")

# гістограми з теоретичними кривими Гаусса
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, sig_theor, c, title in (
        (axes[0], v_white, sigma_v, '#1f77b4', "Білий гаусів шум v[n]"),
        (axes[1], x_markov, np.sqrt(var_markov_theor), '#2ca02c', "Марківський процес x[n] (AR(1))")):
    ax.hist(data, bins=50, density=True, color=c, alpha=0.6, edgecolor='white', label='Гістограма')
    u_axis = np.linspace(data.min(), data.max(), 400)
    pdf = np.exp(-u_axis ** 2 / (2 * sig_theor ** 2)) / (sig_theor * np.sqrt(2 * np.pi))
    ax.plot(u_axis, pdf, 'r-', lw=1.8, label=f'Теоретична крива N(0; {sig_theor:.3f}²)')
    ax.set_title(title)
    ax.set_xlabel("Амплітуда, В")
    ax.set_ylabel("Густина ймовірності")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("figures/noise_histograms.png", dpi=300)
plt.close()


# %% Блок 3. Нормовані автокореляційні функції білого шуму та марківського процесу
def compute_autocorrelation(data_seq, max_lags):
    """Нормована АКФ r[m] = R[m] / R[0] для зсувів m = -max_lags ... +max_lags"""
    centered = data_seq - np.mean(data_seq)
    full_corr = signal.correlate(centered, centered, mode='full')
    mid = len(data_seq) - 1                       # положення нульового зсуву
    r_vals = full_corr[mid - max_lags: mid + max_lags + 1] / full_corr[mid]
    return np.arange(-max_lags, max_lags + 1), r_vals


max_lags = 30
lags, r_white = compute_autocorrelation(v_white, max_lags)
_, r_markov = compute_autocorrelation(x_markov, max_lags)

# теоретична АКФ марківського процесу r[m] = a^|m|
r_markov_theor = a_param ** np.abs(lags)

# інтервал кореляції: зсув, на якому АКФ спадає в e разів
level_e = np.exp(-1.0)
m_k_theor = -1.0 / np.log(abs(a_param))
pos = lags >= 0
below = np.where(r_markov[pos] < level_e)[0]
m_k_pract = int(below[0])                     # перший зсув, на якому АКФ стала меншою за 1/e
m_k_interp = (below[0] - 1) + (r_markov[pos][below[0] - 1] - level_e) / (r_markov[pos][below[0] - 1] - r_markov[pos][below[0]])

mask_nz = lags != 0
print("\n--- АВТОКОРЕЛЯЦІЙНІ ФУНКЦІЇ ---")
print(f"Білий шум: max|r[m]| при m != 0 = {np.max(np.abs(r_white[mask_nz])):.4f} "
      f"(очікується порядку 1/sqrt(N) = {1 / np.sqrt(N_samples):.4f}), інтервал кореляції = 0")
print(f"{'m':>3} {'r_xx[m] (оцінка)':>18} {'a^|m| (теорія)':>16}")
for m in (0, 1, 2, 3, 5, 6, 7, 10):
    i_m = m + max_lags
    print(f"{m:3d} {r_markov[i_m]:18.4f} {r_markov_theor[i_m]:16.4f}")
print(f"Оцінка коефіцієнта a за r[1]: {r_markov[max_lags + 1]:.4f} (задано a = {a_param})")
print(f"Інтервал кореляції марківського процесу: теоретичний m_k = -1/ln(a) = {m_k_theor:.2f} відліків, "
      f"практичний (перший зсув з r < 1/e = {level_e:.4f}) = {m_k_pract} відліків "
      f"(з лінійною інтерполяцією {m_k_interp:.2f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].stem(lags, r_white, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title("Нормована АКФ білого шуму v[n]")
axes[0].set_xlabel("Зсув, m")
axes[0].set_ylabel("r_vv[m]")
axes[0].grid(True)

axes[1].stem(lags, r_markov, linefmt='g-', markerfmt='go', basefmt='k-', label='Оцінка за реалізацією')
axes[1].plot(lags, r_markov_theor, 'r--', lw=1.5, label=f'Теорія a^|m|, a = {a_param}')
axes[1].axhline(level_e, color='gray', ls=':', label='Рівень 1/e')
axes[1].axvline(m_k_theor, color='k', ls='-.', lw=1.0, label=f'τ_k = {m_k_theor:.2f}')
axes[1].axvline(-m_k_theor, color='k', ls='-.', lw=1.0)
axes[1].set_title("Нормована АКФ марківського процесу x[n]")
axes[1].set_xlabel("Зсув, m")
axes[1].set_ylabel("r_xx[m]")
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("figures/acf_comparison.png", dpi=300)
plt.close()


# %% Блок 4. Локаційне виявлення сигналу за допомогою взаємної кореляційної функції
# зондуючий сигнал: код Баркера довжиною 7 біт, кожен біт займає L_s / 7 = 10 відліків
barker_code = np.array([1, 1, 1, -1, -1, 1, -1], dtype=float)
A0 = 1.0
chip_len = L_s // len(barker_code)
s_probe = A0 * np.repeat(barker_code, chip_len)

# модель каналу: y[n] = s[n - n0] + xi[n]
N_y = 600
y_received = np.random.normal(0.0, sigma_ch, N_y)
y_received[n0:n0 + L_s] += s_probe

# ВКФ R_ys[m] = sum_n y[n] * s[n - m]
corr_full = signal.correlate(y_received, s_probe, mode='full')
lags_full = signal.correlation_lags(len(y_received), len(s_probe), mode='full')
r_ys_pos = corr_full[lags_full >= 0]              # від'ємні зсуви відкидаємо
lags_pos = lags_full[lags_full >= 0]

n0_estimated = int(lags_pos[np.argmax(r_ys_pos)])
snr_channel_db = 10.0 * np.log10(np.mean(s_probe ** 2) / sigma_ch ** 2)
e_probe = np.sum(s_probe ** 2)                    # енергія зондуючого сигналу
gain_db = 10.0 * np.log10(L_s)                    # виграш від когерентного накопичення

# автокореляція самого коду (для бічних пелюсток)
acf_code = np.correlate(s_probe, s_probe, mode='full')
lags_code = np.arange(-(len(s_probe) - 1), len(s_probe))
sidelobe = np.max(np.abs(acf_code[np.abs(lags_code) >= chip_len]))     # поза головною пелюсткою (|m| < 10)

print("\n--- ЛОКАЦІЙНЕ ВИЯВЛЕННЯ ЗА ВКФ ---")
print(f"Зондуючий сигнал: код Баркера-7 {barker_code.astype(int).tolist()}, "
      f"L_s = {L_s} відліків ({chip_len} відліків на біт), A0 = {A0} В, енергія E_s = {e_probe:.0f}")
print(f"Канал: N_y = {N_y}, n0 = {n0}, sigma_ch = {sigma_ch} В")
print(f"SNR каналу (на інтервалі сигналу): {snr_channel_db:.2f} дБ; "
      f"виграш від накопичення 10*lg(L_s) = {gain_db:.2f} дБ")
print(f"Пік ВКФ: R_ys[{n0_estimated}] = {np.max(r_ys_pos):.2f} (очікується E_s = {e_probe:.0f}), "
      f"СКВ шуму на виході кореляції = sigma_ch*sqrt(E_s) = {sigma_ch * np.sqrt(e_probe):.2f}")
second_peak = np.max(np.delete(r_ys_pos, slice(max(n0_estimated - chip_len, 0), n0_estimated + chip_len + 1)))
print(f"Найбільший побічний максимум ВКФ: {second_peak:.2f} (відношення піка до нього {np.max(r_ys_pos) / second_peak:.2f})")
print(f"Істинна затримка: {n0}, оцінена: {n0_estimated}, абсолютна похибка: {abs(n0 - n0_estimated)} відліків")
print(f"Бічні пелюстки АКФ коду (|m| >= {chip_len}): {sidelobe:.0f} при піку {np.max(acf_code):.0f} "
      f"({20 * np.log10(sidelobe / np.max(acf_code)):.1f} дБ)")

# статистика виявлення за 1000 незалежних реалізацій шуму каналу
n_trials = 1000
errors = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    y_trial = np.random.normal(0.0, sigma_ch, N_y)
    y_trial[n0:n0 + L_s] += s_probe
    r_trial = signal.correlate(y_trial, s_probe, mode='full')[len(s_probe) - 1:]
    errors[i] = int(np.argmax(r_trial)) - n0
print(f"Статистика за {n_trials} реалізаціями шуму: точно (похибка 0) - {100 * np.mean(errors == 0):.1f} %, "
      f"з похибкою не більше 1 відліку - {100 * np.mean(np.abs(errors) <= 1):.1f} %, "
      f"не більше {chip_len} відліків (1 біт) - {100 * np.mean(np.abs(errors) <= chip_len):.1f} %, "
      f"хибне виявлення - {100 * np.mean(np.abs(errors) > chip_len):.1f} %")

fig, axes = plt.subplots(3, 1, figsize=(13, 9))

axes[0].stem(np.arange(L_s), s_probe, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title("Опорний зондуючий сигнал s[n] (код Баркера-7)")
axes[0].set_xlabel("Індекс відліку, n")
axes[0].set_ylabel("Амплітуда, В")
axes[0].grid(True)

axes[1].plot(np.arange(N_y), y_received, color='gray', lw=0.8)
axes[1].axvspan(n0, n0 + L_s, color='orange', alpha=0.3, label=f'Прихований сигнал (n0 = {n0})')
axes[1].set_title(f"Прийнята суміш y[n] (SNR = {snr_channel_db:.2f} дБ)")
axes[1].set_xlabel("Індекс відліку, n")
axes[1].set_ylabel("Амплітуда, В")
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize=9)

axes[2].plot(lags_pos, r_ys_pos, 'g-', lw=1.0)
axes[2].axvline(n0_estimated, color='r', ls='--', label=f'Пік: n0 = {n0_estimated}')
axes[2].set_title("Взаємна кореляційна функція R_ys[m]")
axes[2].set_xlabel("Зсув, m")
axes[2].set_ylabel("R_ys[m]")
axes[2].grid(True)
axes[2].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("figures/radar_detection_ccf.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
