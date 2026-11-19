#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №3
Спектральний аналіз за допомогою ДПФ/ШПФ та віконних функцій

Варіант 2
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, SciPy, Matplotlib
"""

# %% Блок 1. Імпорт бібліотек та налаштування стилю відображення
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# каталог для графіків
os.makedirs("figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


# %% Блок 2. Параметри варіанта 2 та порівняння швидкодії ДПФ і ШПФ
fs = 1200.0           # частота дискретизації, Гц
A1 = 3.0              # амплітуда сильної гармоніки, В
f1 = 200.0            # частота сильної гармоніки, Гц
phi1 = np.pi / 6      # початкова фаза першої гармоніки, рад
A2 = 0.10             # амплітуда слабкої гармоніки, В
f2 = 225.0            # частота слабкої гармоніки, Гц
phi2 = 0.0            # початкова фаза другої гармоніки, рад
sigma_noise = 0.04    # СКВ гаусового шуму, В
N = 256               # базова довжина вибірки

df = fs / N           # частотна роздільність, Гц
T_obs = N / fs        # тривалість спостереження, с

print("--- ПАРАМЕТРИ ВАРІАНТА 2 ---")
print(f"fs = {fs:.0f} Гц, N = {N}, T_obs = {T_obs * 1000:.2f} мс, df = fs/N = {df:.4f} Гц")
for name, f_i in (("f1", f1), ("f2", f2)):
    bins = f_i / df
    kind = "кратна df (витоку немає)" if abs(bins - round(bins)) < 1e-9 else "НЕ кратна df (витік)"
    print(f"{name} = {f_i:.1f} Гц -> f/df = {bins:.4f} біна, {f_i * N / fs:.4f} періодів у вікні: {kind}")
print(f"Відстань між гармоніками: {f2 - f1:.1f} Гц = {(f2 - f1) / df:.3f} біна; "
      f"A2/A1 = {20 * np.log10(A2 / A1):.2f} дБ")


def manual_dft(x_seq):
    """Пряме ДПФ за означенням: X[m] = sum_n x[n] * W_N^(mn), W_N = exp(-j*2*pi/N).
    Будуємо матрицю W розміром N x N і множимо її на вектор відліків."""
    n_len = len(x_seq)
    k_idx = np.arange(n_len)
    # добуток mn беремо за модулем N, щоб не втрачати точність для великих N
    W_matrix = np.exp(-2j * np.pi * (np.outer(k_idx, k_idx) % n_len) / n_len)
    return W_matrix @ x_seq


def measure_time(func, x_seq, repeats):
    """Мінімальний за кілька серій середній час одного виклику, мкс"""
    best = float('inf')
    for _ in range(3):
        t_start = time.perf_counter()
        for _ in range(repeats):
            func(x_seq)
        best = min(best, (time.perf_counter() - t_start) / repeats)
    return best * 1e6


N_sizes = [64, 256, 1024, 4096]
t_dft_list, t_fft_list, diff_list = [], [], []
np.random.seed(0)

print("\n--- ПОРІВНЯННЯ ШВИДКОДІЇ МАТРИЧНОГО ДПФ ТА ШПФ ---")
print(f"{'N':>6} {'t_DFT, мкс':>14} {'t_FFT, мкс':>12} {'Прискорення':>12} {'Макс. похибка':>15}")
for n_size in N_sizes:
    x_test = np.random.randn(n_size)
    t_dft = measure_time(manual_dft, x_test, 1 if n_size >= 1024 else 20)
    t_fft = measure_time(np.fft.fft, x_test, 2000)
    diff = np.max(np.abs(manual_dft(x_test) - np.fft.fft(x_test)))
    speedup = t_dft / t_fft
    t_dft_list.append(t_dft)
    t_fft_list.append(t_fft)
    diff_list.append(diff)
    print(f"{n_size:6d} {t_dft:14.1f} {t_fft:12.1f} {speedup:12.1f} {diff:15.2e}")

# графік швидкодії
plt.figure(figsize=(9, 5))
plt.plot(N_sizes, t_dft_list, 'ro-', label='Матричне ДПФ, O(N²)')
plt.plot(N_sizes, t_fft_list, 'bs-', label='np.fft.fft (ШПФ), O(N log N)')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xticks(N_sizes, [str(n_size) for n_size in N_sizes])
plt.title("Порівняння швидкодії ДПФ та ШПФ")
plt.xlabel("Розмірність вибірки N")
plt.ylabel("Час виконання, мкс")
plt.grid(True, which='both')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("figures/dft_vs_fft_benchmark.png", dpi=300)
plt.close()


# %% Блок 3. Амплітудний та фазовий спектри сигналу
np.random.seed(7)
n_arr = np.arange(N)
t_axis = n_arr / fs

s_clean = (A1 * np.sin(2 * np.pi * f1 * n_arr / fs + phi1)
           + A2 * np.cos(2 * np.pi * f2 * n_arr / fs + phi2))
s_total = s_clean + np.random.normal(0.0, sigma_noise, N)

X_spectrum = np.fft.fft(s_total)
freqs_axis = np.fft.fftfreq(N, 1.0 / fs)[:N // 2]

amp_spectrum = 2.0 * np.abs(X_spectrum[:N // 2]) / N
amp_spectrum[0] /= 2.0                       # постійна складова не має дзеркального двійника
phase_spectrum = np.arctan2(X_spectrum[:N // 2].imag, X_spectrum[:N // 2].real)
phase_spectrum[amp_spectrum < 0.03] = 0.0    # фаза шумових бінів не інформативна

# контроль: для кратної частоти ШПФ має давати рівно амплітуду гармоніки
m1 = int(round(f1 / df))
m2 = int(round(f2 / df))
print("\n--- АМПЛІТУДНИЙ ТА ФАЗОВИЙ СПЕКТРИ ---")
print(f"{'Гармоніка':>10} {'біни':>8} {'f_біна, Гц':>11} {'A, В':>8} {'A, дБВ':>8} "
      f"{'фаза, рад':>10} {'похибка f, Гц':>14}")
for m in (m1 - 1, m1, m1 + 1):
    print(f"{'сильна':>10} {m:8d} {freqs_axis[m]:11.4f} {amp_spectrum[m]:8.4f} "
          f"{20 * np.log10(amp_spectrum[m]):8.2f} {phase_spectrum[m]:10.4f} {freqs_axis[m] - f1:14.4f}")
print(f"{'слабка':>10} {m2:8d} {freqs_axis[m2]:11.4f} {amp_spectrum[m2]:8.4f} "
      f"{20 * np.log10(amp_spectrum[m2]):8.2f} {phase_spectrum[m2]:10.4f} {freqs_axis[m2] - f2:14.4f}")
m_peak = int(np.argmax(amp_spectrum))
print(f"Найбільший пік: біна {m_peak}, f = {freqs_axis[m_peak]:.4f} Гц, A = {amp_spectrum[m_peak]:.4f} В "
      f"(істинні f1 = {f1:.1f} Гц, A1 = {A1:.1f} В)")
print(f"Середній рівень шумових бінів (f > 400 Гц): {np.mean(amp_spectrum[freqs_axis > 400]):.4f} В")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].stem(freqs_axis, amp_spectrum, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].axvline(f1, color='r', ls=':', label=f"f1 = {f1:.0f} Гц")
axes[0].axvline(f2, color='g', ls=':', label=f"f2 = {f2:.0f} Гц")
axes[0].set_xlim(0, 300)
axes[0].set_title("Амплітудний спектр сигналу (прямокутне вікно)")
axes[0].set_xlabel("Частота, Гц")
axes[0].set_ylabel("Амплітуда, В")
axes[0].grid(True)
axes[0].legend(loc='upper right')

axes[1].stem(freqs_axis, phase_spectrum, linefmt='m-', markerfmt='mo', basefmt='k-')
axes[1].set_xlim(0, 300)
axes[1].set_title("Фазовий спектр (для бінів з амплітудою > 0.03 В)")
axes[1].set_xlabel("Частота, Гц")
axes[1].set_ylabel("Фаза, рад")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("figures/harmonic_spectrum.png", dpi=300)
plt.close()


# %% Блок 4. Витік спектра: кратна та некратна кількість періодів
f_mult = 48 * df          # 48 періодів у вікні, рівно в бін (збігається з f2)
f_nonmult = f1            # 42.667 періоду у вікні, між бінами 42 і 43

x_mult = A1 * np.sin(2 * np.pi * f_mult * n_arr / fs)
x_nonmult = A1 * np.sin(2 * np.pi * f_nonmult * n_arr / fs)


def amplitude_spectrum(x_seq):
    """Односторонній амплітудний спектр (В) за допомогою np.fft.fft"""
    n_len = len(x_seq)
    amp = 2.0 * np.abs(np.fft.fft(x_seq)[:n_len // 2]) / n_len
    amp[0] /= 2.0
    return amp


amp_mult = amplitude_spectrum(x_mult)
amp_nonmult = amplitude_spectrum(x_nonmult)

print("\n--- ВИТІК СПЕКТРА ---")
print(f"{'':>26} {'f, Гц':>8} {'періодів':>9} {'пік, В':>8} {'бінів > -60 дБ':>15} {'частка поза піком':>18}")
for name, f_x, amp in (("кратна кількість періодів", f_mult, amp_mult),
                       ("некратна кількість періодів", f_nonmult, amp_nonmult)):
    m_pk = int(np.argmax(amp))
    n_above = int(np.sum(20 * np.log10(np.maximum(amp, 1e-12) / A1) > -60))
    p_total = np.sum(amp ** 2)
    p_out = 1.0 - np.sum(amp[max(m_pk - 1, 0):m_pk + 2] ** 2) / p_total
    print(f"{name:>26} {f_x:8.3f} {f_x * N / fs:9.3f} {amp[m_pk]:8.4f} {n_above:15d} {p_out:18.4%}")
print(f"Амплітуда витоку на бінах поблизу f2 (біна {m2}) при некратній f1: {amp_nonmult[m2]:.4f} В "
      f"({20 * np.log10(amp_nonmult[m2]):.2f} дБВ) при A2 = {A2:.2f} В ({20 * np.log10(A2):.2f} дБВ)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].stem(freqs_axis, amp_mult, linefmt='g-', markerfmt='go', basefmt='k-',
             label=f'f = {f_mult:.2f} Гц (48 періодів)')
axes[0].stem(freqs_axis, amp_nonmult, linefmt='r-', markerfmt='r.', basefmt='k-',
             label=f'f = {f_nonmult:.2f} Гц (42.667 періоду)')
axes[0].set_xlim(100, 320)
axes[0].set_title("Амплітудний спектр, лінійний масштаб")
axes[0].set_xlabel("Частота, Гц")
axes[0].set_ylabel("Амплітуда, В")
axes[0].grid(True)
axes[0].legend(loc='upper right', fontsize=9)

axes[1].plot(freqs_axis, 20 * np.log10(np.maximum(amp_mult, 1e-12) / A1), 'g-o', ms=3,
             label='кратна кількість періодів')
axes[1].plot(freqs_axis, 20 * np.log10(np.maximum(amp_nonmult, 1e-12) / A1), 'r-o', ms=3,
             label='некратна кількість періодів')
axes[1].set_xlim(0, fs / 2)
axes[1].set_ylim(-80, 5)
axes[1].set_title("Амплітудний спектр, логарифмічний масштаб")
axes[1].set_xlabel("Частота, Гц")
axes[1].set_ylabel("Рівень відносно A1, дБ")
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("figures/spectral_leakage.png", dpi=300)
plt.close()


# %% Блок 5. Порівняння вікон: прямокутне, Ханна, Хеммінга, Блекмана
win_names = ["Прямокутне", "Ханна", "Хеммінга", "Блекмана"]
win_list = [np.ones(N), windows.hann(N), windows.hamming(N), windows.blackman(N)]

# сильна гармоніка окремо (без шуму і без слабкої): це рівень витоку на біні f2
s_strong = A1 * np.sin(2 * np.pi * f1 * n_arr / fs + phi1)

print("\n--- ПОРІВНЯННЯ ВІКОН ---")
print(f"{'Вікно':>12} {'CG':>7} {'пік f1, дБВ':>12} {'біна f2, дБВ':>13} {'витік f1 на f2, дБВ':>20} "
      f"{'f2 над витоком, дБ':>19}")
win_spectra = []
for name, w in zip(win_names, win_list):
    coherent_gain = np.sum(w) / N
    # нормування на суму вагових коефіцієнтів повертає справжню амплітуду гармоніки
    amp_w = 2.0 * np.abs(np.fft.fft(s_total * w)[:N // 2]) / np.sum(w)
    leak_w = 2.0 * np.abs(np.fft.fft(s_strong * w)[:N // 2]) / np.sum(w)
    db_w = 20 * np.log10(np.maximum(amp_w, 1e-12))
    win_spectra.append(db_w)
    peak_f1 = np.max(amp_w[m1 - 2:m1 + 3])
    print(f"{name:>12} {coherent_gain:7.3f} {20 * np.log10(peak_f1):12.2f} {db_w[m2]:13.2f} "
          f"{20 * np.log10(leak_w[m2]):20.2f} {db_w[m2] - 20 * np.log10(leak_w[m2]):19.2f}")
print(f"Істинні рівні: A1 = {A1:.2f} В ({20 * np.log10(A1):.2f} дБВ), "
      f"A2 = {A2:.2f} В ({20 * np.log10(A2):.2f} дБВ)")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
for ax, name, db_w, c in zip(axes.flat, win_names, win_spectra, colors):
    ax.plot(freqs_axis, db_w, color=c, lw=1.2)
    ax.axvline(f1, color='gray', ls='--', lw=1, label=f"f1 = {f1:.0f} Гц")
    ax.axvline(f2, color='k', ls=':', lw=1, label=f"f2 = {f2:.0f} Гц")
    ax.axhline(20 * np.log10(A2), color='b', ls='-.', lw=0.8, label="A2 = 0.10 В (-20 дБВ)")
    ax.set_xlim(100, 350)
    ax.set_ylim(-100, 15)
    ax.set_title(f"Вікно: {name}")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Рівень, дБВ")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("figures/window_functions_comp.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
