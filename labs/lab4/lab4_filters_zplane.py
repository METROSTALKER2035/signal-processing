#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №4
Синтез цифрових КІХ та НІХ фільтрів та аналіз стійкості у z-площині

Варіант 2
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, SciPy, Matplotlib
"""

# %% Блок 1. Імпорт бібліотек, налаштування стилю та функція побудови z-площини
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# каталог для графіків
os.makedirs("figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


def plot_custom_zplane(b_coeffs, a_coeffs, ax, title_str):
    """Діаграма нулів (○) та полюсів (×) на z-площині з одиничним колом"""
    zeros, poles, gain = signal.tf2zpk(b_coeffs, a_coeffs)

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.0, label='Одиничне коло')
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)

    ax.plot(zeros.real, zeros.imag, 'o', color='g', mfc='none', ms=10, mew=2, label='Нулі')
    ax.plot(poles.real, poles.imag, 'x', color='r', ms=10, mew=2, label='Полюси')

    # кратні нулі підписуємо цифрою
    for z_val in np.unique(np.round(zeros, 6)):
        mult = int(np.sum(np.abs(zeros - z_val) < 1e-5))
        if mult > 1:
            ax.annotate(f"×{mult}", (z_val.real, z_val.imag), textcoords="offset points",
                        xytext=(8, 8), fontsize=11, color='g')

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.set_title(title_str)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)
    return zeros, poles


# %% Блок 2. Аналіз розташування полюсів та стійкості систем
# базова передавальна функція варіанта 2: H_base(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
b_stable = [0.15, 0.3, 0.15]
a_stable = [1.0, -0.8, 0.64]

# для порівняння: той самий чисельник, полюси того ж кута, але з модулем 1.2 (за колом)
b_unstable = [0.15, 0.3, 0.15]
a_unstable = [1.0, -1.2, 1.44]

# імпульсні характеристики на 30 тактів
n_imp = np.arange(30)
delta_sig = np.zeros(30)
delta_sig[0] = 1.0
h_stable = signal.lfilter(b_stable, a_stable, delta_sig)
h_unstable = signal.lfilter(b_unstable, a_unstable, delta_sig)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
z_st, p_st = plot_custom_zplane(b_stable, a_stable, axes[0, 0], "z-площина: стійка система (H_base)")

axes[0, 1].stem(n_imp, h_stable, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 1].set_title("Імпульсна характеристика h[n]: стійка система")
axes[0, 1].set_xlabel("Індекс відліку, n")
axes[0, 1].set_ylabel("Амплітуда, h[n]")
axes[0, 1].grid(True)

z_un, p_unst = plot_custom_zplane(b_unstable, a_unstable, axes[1, 0], "z-площина: нестійка система")

axes[1, 1].stem(n_imp, h_unstable, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 1].set_title("Імпульсна характеристика h[n]: нестійка система")
axes[1, 1].set_xlabel("Індекс відліку, n")
axes[1, 1].set_ylabel("Амплітуда, h[n]")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("figures/zplane_stable_vs_unstable.png", dpi=300)
plt.close()

print("--- АНАЛІЗ РОЗТАШУВАННЯ ПОЛЮСІВ ТА СТІЙКОСТІ СИСТЕМ ---")
print(f"Нулі H_base:  {np.round(z_st, 4)} (кратність 2)")
print(f"Підсилення k0 = {signal.tf2zpk(b_stable, a_stable)[2]:.2f}; "
      f"H(z=1) = {np.sum(b_stable) / np.sum(a_stable):.4f}")
for name, p_vec, h_vec in (("Стійка система (H_base)", p_st, h_stable),
                           ("Нестійка система", p_unst, h_unstable)):
    print(f"{name}:")
    print(f"  полюси p = {np.round(p_vec, 4)}")
    print(f"  модулі |p| = {np.round(np.abs(p_vec), 4)}, кут = {np.degrees(np.angle(p_vec[0])):.1f}°")
    print(f"  стійка (усі |p| < 1): {np.all(np.abs(p_vec) < 1.0)}")
    print(f"  h[29] = {h_vec[29]:.3e}, max|h[n]| = {np.max(np.abs(h_vec)):.3e}")


# %% Блок 3. Синтез ФВЧ: КІХ (метод вікон) та НІХ (Баттерворт, білінійне перетворення)
fs = 1200.0                      # частота дискретизації, Гц
f_cutoff = 350.0                 # частота зрізу, Гц
nyq_rate = fs / 2.0              # частота Найквіста, Гц
norm_cutoff = f_cutoff / nyq_rate

# КІХ-фільтр верхніх частот: усічена ідеальна характеристика, вікно Хеммінга
N_fir = 51
fir_coeffs = signal.firwin(N_fir, norm_cutoff, window='hamming', pass_zero=False)

# НІХ-фільтр верхніх частот Баттерворта 4-го порядку
N_iir = 4
b_iir, a_iir = signal.butter(N_iir, norm_cutoff, btype='high')

# частотні характеристики (1024 точки на інтервалі 0 ... fs/2)
w_fir, h_fir = signal.freqz(fir_coeffs, 1.0, worN=1024)
w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=1024)
f_hz = w_fir * fs / (2 * np.pi)

# групова затримка, відліків (на ω = 0 у ФВЧ АЧХ дорівнює нулю, тому scipy попереджає про особливу точку)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, gd_fir = signal.group_delay((fir_coeffs, [1.0]), w=w_fir)
    _, gd_iir = signal.group_delay((b_iir, a_iir), w=w_iir)

poles_iir = np.roots(a_iir)
print("\n--- СИНТЕЗ ФІЛЬТРІВ ВЕРХНІХ ЧАСТОТ ---")
print(f"fs = {fs:.0f} Гц, f_c = {f_cutoff:.0f} Гц, нормована частота зрізу = {norm_cutoff:.4f}")
print(f"КІХ: {N_fir} коефіцієнтів, вікно Хеммінга, затримка (N-1)/2 = {(N_fir - 1) // 2} відліків")
print(f"НІХ: Баттерворт {N_iir}-го порядку, b = {np.round(b_iir, 5)}")
print(f"     a = {np.round(a_iir, 5)}")
print(f"     модулі полюсів НІХ: {np.round(np.sort(np.abs(poles_iir)), 4)}, "
      f"стійкий: {np.all(np.abs(poles_iir) < 1.0)}")
i_c = np.argmin(np.abs(f_hz - f_cutoff))
print(f"|H(f_c)|: КІХ {20 * np.log10(abs(h_fir[i_c])):.2f} дБ, НІХ {20 * np.log10(abs(h_iir[i_c])):.2f} дБ")
for f_probe in (100.0, 300.0, 450.0):
    i_p = np.argmin(np.abs(f_hz - f_probe))
    print(f"|H({f_probe:.0f} Гц)|: КІХ {20 * np.log10(abs(h_fir[i_p])):.2f} дБ, "
          f"НІХ {20 * np.log10(abs(h_iir[i_p])):.2f} дБ")
band = f_hz >= f_cutoff
print(f"Групова затримка: КІХ {gd_fir[np.argmin(np.abs(f_hz - 450.0))]:.2f} відліків (стала на всіх частотах); "
      f"НІХ у смузі пропускання від {np.min(gd_iir[band]):.2f} до {np.max(gd_iir[band]):.2f} відліків, "
      f"на 450 Гц {gd_iir[np.argmin(np.abs(f_hz - 450.0))]:.2f}")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(f_hz, 20 * np.log10(np.maximum(np.abs(h_fir), 1e-5)), 'b-', label=f'КІХ ({N_fir} коеф., Хеммінг)')
axes[0].plot(f_hz, 20 * np.log10(np.maximum(np.abs(h_iir), 1e-5)), 'r-', label=f'НІХ (Баттерворт, {N_iir}-й порядок)')
axes[0].axvline(f_cutoff, color='gray', ls=':', label=f'f_c = {f_cutoff:.0f} Гц')
axes[0].axhline(-3, color='k', ls='--', lw=0.8, label='-3 дБ')
axes[0].set_ylim(-80, 5)
axes[0].set_xlim(0, fs / 2)
axes[0].set_title("АЧХ фільтрів верхніх частот")
axes[0].set_xlabel("Частота, Гц")
axes[0].set_ylabel("Модуль, дБ")
axes[0].grid(True)
axes[0].legend(loc='lower right', fontsize=9)

axes[1].plot(f_hz, np.unwrap(np.angle(h_fir)), 'b-', label='КІХ')
axes[1].plot(f_hz, np.unwrap(np.angle(h_iir)), 'r-', label='НІХ')
axes[1].axvline(f_cutoff, color='gray', ls=':')
axes[1].set_xlim(0, fs / 2)
axes[1].set_title("ФЧХ фільтрів")
axes[1].set_xlabel("Частота, Гц")
axes[1].set_ylabel("Фаза, рад")
axes[1].grid(True)
axes[1].legend(loc='lower left', fontsize=9)

axes[2].plot(f_hz, gd_fir, 'b-', label='КІХ')
axes[2].plot(f_hz, gd_iir, 'r-', label='НІХ')
axes[2].axvline(f_cutoff, color='gray', ls=':')
axes[2].set_ylim(0, 50)
axes[2].set_xlim(0, fs / 2)
axes[2].set_title("Групова затримка")
axes[2].set_xlabel("Частота, Гц")
axes[2].set_ylabel("Затримка, відліків")
axes[2].grid(True)
axes[2].legend(loc='center right', fontsize=9)

plt.tight_layout()
plt.savefig("figures/filter_frequency_responses.png", dpi=300)
plt.close()


# %% Блок 4. Фільтрація зашумленого сигналу та оцінка придушення завади
# для ФВЧ корисна складова - високочастотна (f2 = 450 Гц), завада - низькочастотна (f1 = 100 Гц)
A1, f1 = 2.0, 100.0       # низькочастотна завада
A2, f2 = 1.5, 450.0       # корисний високочастотний сигнал
sigma = 0.30              # СКВ гаусового шуму, В

T_sig = 0.5
N_total = int(T_sig * fs)
t_sig = np.arange(N_total) / fs

np.random.seed(123)
s_clean = A2 * np.sin(2 * np.pi * f2 * t_sig)          # корисна складова
s_interf = A1 * np.sin(2 * np.pi * f1 * t_sig)         # завада
s_noise = np.random.normal(0.0, sigma, N_total)        # гаусів шум
s_input = s_clean + s_interf + s_noise

s_filtered_fir = signal.lfilter(fir_coeffs, 1.0, s_input)
s_filtered_iir = signal.lfilter(b_iir, a_iir, s_input)

# компенсація сталої затримки КІХ-фільтра
delay_fir = (N_fir - 1) // 2
s_filtered_fir_aligned = np.roll(s_filtered_fir, -delay_fir)


def get_magnitude_spectrum(data, rate):
    """Односторонній амплітудний спектр (В) та вісь частот (Гц)"""
    n_len = len(data)
    mag = 2.0 * np.abs(fft(data)[:n_len // 2]) / n_len
    mag[0] /= 2.0
    return fftfreq(n_len, 1.0 / rate)[:n_len // 2], mag


# спектри рахуємо на усталеній ділянці (після перехідних процесів фільтрів, поза "хвостом" np.roll)
# 432 відліки = 0.36 с: на ній вкладається рівно 36 періодів завади та 162 періоди корисного сигналу
n_start, n_seg = 120, 432
steady = slice(n_start, n_start + n_seg)

f_axis, mag_input = get_magnitude_spectrum(s_input[steady], fs)
_, mag_fir = get_magnitude_spectrum(s_filtered_fir_aligned[steady], fs)
_, mag_iir = get_magnitude_spectrum(s_filtered_iir[steady], fs)

i_f1 = int(np.argmin(np.abs(f_axis - f1)))
i_f2 = int(np.argmin(np.abs(f_axis - f2)))

print("\n--- ОЦІНКА РЕЗУЛЬТАТІВ ФІЛЬТРАЦІЇ ---")
print(f"Сигнал: {N_total} відліків, спектри на ділянці {n_seg} відліків (df = {fs / n_seg:.3f} Гц); "
      f"завада {f1:.0f} Гц ({A1} В), корисна складова {f2:.0f} Гц ({A2} В), шум sigma = {sigma} В")
print(f"{'':>10} {'A(100 Гц), В':>13} {'A(450 Гц), В':>13} {'придушення завади, дБ':>22} {'збереження корисної, %':>23}")
print(f"{'Вхід':>10} {mag_input[i_f1]:13.4f} {mag_input[i_f2]:13.4f} {'-':>22} {'-':>23}")
for name, mag in (("КІХ", mag_fir), ("НІХ", mag_iir)):
    sup_db = 20 * np.log10(mag_input[i_f1] / mag[i_f1])
    keep = 100.0 * mag[i_f2] / mag_input[i_f2]
    print(f"{name:>10} {mag[i_f1]:13.4f} {mag[i_f2]:13.4f} {sup_db:22.2f} {keep:23.2f}")

# фазовий зсув корисної складової після фільтра (КІХ - після компенсації затримки)
i_450 = int(np.argmin(np.abs(f_hz - f2)))
phase_fir = np.angle(h_fir[i_450] * np.exp(1j * w_fir[i_450] * delay_fir))
phase_iir = np.angle(h_iir[i_450])
print(f"Фазовий зсув на {f2:.0f} Гц: КІХ після компенсації затримки {phase_fir:.3f} рад, "
      f"НІХ {phase_iir:.3f} рад ({gd_iir[i_450]:.2f} відліку групової затримки)")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
t_ms = t_sig * 1000
mask = t_ms <= 150

axes[0, 0].plot(t_ms[mask], s_input[mask], color='gray', lw=1.0, label='Вхідний сигнал + завада + шум')
axes[0, 0].plot(t_ms[mask], s_clean[mask], 'b-', lw=1.5, label='Корисна складова')
axes[0, 0].set_title("Вхідний зашумлений сигнал")
axes[0, 0].set_xlabel("Час, мс")
axes[0, 0].set_ylabel("Амплітуда, В")
axes[0, 0].grid(True)
axes[0, 0].legend(loc='upper right', fontsize=8)

axes[0, 1].plot(t_ms[mask], s_clean[mask], 'k--', lw=1.2, label='Корисна складова')
axes[0, 1].plot(t_ms[mask], s_filtered_fir_aligned[mask], 'b-', lw=1.2, label='КІХ (затримку компенсовано)')
axes[0, 1].plot(t_ms[mask], s_filtered_iir[mask], 'r-', lw=1.2, label='НІХ')
axes[0, 1].set_ylim(-2.7, 2.7)
axes[0, 1].set_title("Результат фільтрації")
axes[0, 1].set_xlabel("Час, мс")
axes[0, 1].set_ylabel("Амплітуда, В")
axes[0, 1].grid(True)
axes[0, 1].legend(loc='upper center', ncol=3, fontsize=8)

axes[1, 0].stem(f_axis, mag_input, linefmt='k-', markerfmt='ko', basefmt='k-')
axes[1, 0].axvline(f_cutoff, color='gray', ls=':', label=f'f_c = {f_cutoff:.0f} Гц')
axes[1, 0].set_title("Амплітудний спектр вхідного сигналу")
axes[1, 0].set_xlabel("Частота, Гц")
axes[1, 0].set_ylabel("Амплітуда, В")
axes[1, 0].grid(True)
axes[1, 0].legend(loc='upper left', fontsize=8)

axes[1, 1].plot(f_axis, 20 * np.log10(np.maximum(mag_input, 1e-6)), color='lightgray', lw=1.0, label='Вхід')
axes[1, 1].plot(f_axis, 20 * np.log10(np.maximum(mag_fir, 1e-6)), 'b-', lw=1.2, label='Після КІХ')
axes[1, 1].plot(f_axis, 20 * np.log10(np.maximum(mag_iir, 1e-6)), 'r-', lw=1.2, label='Після НІХ')
axes[1, 1].axvline(f_cutoff, color='gray', ls=':')
axes[1, 1].set_ylim(-80, 25)
axes[1, 1].set_title("Амплітудні спектри після фільтрації")
axes[1, 1].set_xlabel("Частота, Гц")
axes[1, 1].set_ylabel("Рівень, дБВ")
axes[1, 1].grid(True)
axes[1, 1].legend(loc='upper center', ncol=3, fontsize=8)

plt.tight_layout()
plt.savefig("figures/signal_filtering_results.png", dpi=300)
plt.close()

print("\nМоделювання завершено. Графіки збережено в 'figures/'.")
