#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №1
Дослідження дискретизації, квантування сигналів та ефекту аліасингу

Варіант 2
Виконав: Афанасьєв Сергій, група КІ 24-1
Стек: Python 3.10+, NumPy, SciPy, Matplotlib
"""

# %% Блок 1. Імпорт бібліотек та налаштування стилю відображення
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

# каталоги для графіків та даних
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6


# %% Блок 2. Параметри варіанта 2 та аналітичний розрахунок
A1 = 3.0           # амплітуда першої гармоніки, В
f1 = 150.0         # частота першої гармоніки, Гц
A2 = 2.0           # амплітуда другої гармоніки, В
f2 = 800.0         # частота другої гармоніки, Гц
phi = np.pi / 4    # початкова фаза другої гармоніки, рад

f_max = max(f1, f2)     # гранична частота спектра, 800 Гц
f_s1 = 4.0 * f_max      # 3200 Гц, теорема відліків виконується
f_s2 = 2.0 * f_max      # 1600 Гц, критична частота
f_s3 = 950.0            # 950 Гц, піддискретизація (з таблиці варіантів)

T_obs = 0.02            # вікно для графіків у часі, 20 мс
T_long = 1.0            # інтервал для спектрального аналізу, 1 с (роздільність 1 Гц)
fs_cont = 100000.0      # частота для "неперервного" сигналу, Гц


def s(t):
    """Сигнал варіанта: s(t) = A1*sin(2*pi*f1*t) + A2*cos(2*pi*f2*t + phi)"""
    return A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t + phi)


def f_alias(f0, fs):
    """Частота, на якій гармоніка f0 спостерігається після дискретизації з fs"""
    k = np.floor(f0 / fs + 0.5)
    return abs(f0 - k * fs)


def sample(fs, duration):
    """Відліки сигналу s(t) з частотою fs на інтервалі duration"""
    n = np.arange(int(round(fs * duration)))
    t = n / fs
    return t, s(t)


print(f"f_max = {f_max:.0f} Гц")
for name, fs in (("fs1", f_s1), ("fs2", f_s2), ("fs3", f_s3)):
    print(f"{name} = {fs:6.0f} Гц | f_Nyq = {fs / 2:6.1f} Гц | "
          f"f1 -> {f_alias(f1, fs):6.1f} Гц | f2 -> {f_alias(f2, fs):6.1f} Гц")


# %% Блок 3. Дискретизація сигналу, спектральний аналіз та аліасинг
# короткий інтервал (для графіків у часі)
t_cont, s_cont = sample(fs_cont, T_obs)
t_s1, s_s1 = sample(f_s1, T_obs)
t_s2, s_s2 = sample(f_s2, T_obs)
t_s3, s_s3 = sample(f_s3, T_obs)

# довгий інтервал (для спектрів)
t_l1, s_l1 = sample(f_s1, T_long)
t_l2, s_l2 = sample(f_s2, T_long)
t_l3, s_l3 = sample(f_s3, T_long)


def compute_spectrum(x, fs):
    """Односторонній амплітудний спектр сигналу x (FFT)"""
    n_pts = len(x)
    X = rfft(x)
    freqs = rfftfreq(n_pts, 1.0 / fs)
    amp = 2.0 * np.abs(X) / n_pts
    amp[0] /= 2.0              # постійна складова
    if n_pts % 2 == 0:
        amp[-1] /= 2.0         # частота Найквіста не має дзеркального двійника
    return freqs, amp


spectra = {}
for name, fs, x in (("fs1", f_s1, s_l1), ("fs2", f_s2, s_l2), ("fs3", f_s3, s_l3)):
    fa, sp = compute_spectrum(x, fs)
    spectra[name] = (fa, sp)
    lines = np.where(sp > 0.05)[0]
    txt = ", ".join(f"{fa[i]:.0f} Гц ({sp[i]:.3f} В)" for i in lines)
    print(f"{name} = {fs:.0f} Гц: {txt}")

# перевірка амплітуд аналітично
# fs3: 800 Гц переходить у 150 Гц (f2 = fs3 - 150), тому cos(...+phi) -> cos(theta - phi)
# і складова на 150 Гц складається з sin- та cos-частин
a_sin = A1 + A2 * np.sin(phi)
a_cos = A2 * np.cos(phi)
print(f"Очікувана амплітуда на 150 Гц при fs3: {np.hypot(a_sin, a_cos):.3f} В")
# fs2: 800 Гц дорівнює частоті Найквіста, cos(pi*n + phi) = (-1)^n * cos(phi)
print(f"Очікувана амплітуда на 800 Гц при fs2: {A2 * abs(np.cos(phi)):.3f} В")

# графіки
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
cases = [
    ("fs1", f_s1, t_s1, s_s1, "r", f"fs1 = {f_s1:.0f} Гц = 4·fmax"),
    ("fs2", f_s2, t_s2, s_s2, "g", f"fs2 = {f_s2:.0f} Гц = 2·fmax"),
    ("fs3", f_s3, t_s3, s_s3, "m", f"fs3 = {f_s3:.0f} Гц < 2·fmax (аліасинг)"),
]
for col, (name, fs, t_d, s_d, c, title) in enumerate(cases):
    # верхній ряд - сигнал у часі
    ax = axes[0, col]
    ax.plot(t_cont * 1000, s_cont, color='#1f77b4', lw=1.2, label='Неперервний s(t)')
    ax.stem(t_d * 1000, s_d, linefmt=c + '-', markerfmt=c + 'o', basefmt='k-', label='Відліки')
    ax.set_title(title)
    ax.set_xlabel("Час, мс")
    ax.set_ylabel("Амплітуда, В")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

    # нижній ряд - амплітудний спектр
    fa, sp = spectra[name]
    ax = axes[1, col]
    idx = np.where(sp > 1e-3)[0]
    ax.stem(fa[idx], sp[idx], linefmt=c + '-', markerfmt=c + 'o', basefmt='k-')
    for i in idx:
        ax.annotate(f"{fa[i]:.0f} Гц\n{sp[i]:.2f} В", (fa[i], sp[i]),
                    textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    ax.axvline(fs / 2, color='gray', ls=':', label=f"f_Nyq = {fs / 2:.0f} Гц")
    ax.set_xlim(0, 1.1 * fs / 2)
    ax.set_ylim(0, 5.5)
    ax.set_title(f"Спектр при fs = {fs:.0f} Гц")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплітуда, В")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("figures/sampling_aliasing.png", dpi=300)
plt.close()


# %% Блок 4. Квантування за рівнем, похибки та розрахунок SNR
def uniform_quantize(x, bits, v_min, v_max):
    """Рівномірне квантування з округленням до найближчого рівня.
    Повертає квантований сигнал, похибку e = xq - x та крок Delta."""
    levels = 2 ** bits
    delta = (v_max - v_min) / (levels - 1)
    idx = np.round((x - v_min) / delta)
    idx = np.clip(idx, 0, levels - 1)       # захист від насичення
    xq = v_min + idx * delta
    return xq, xq - x, delta


def calc_snr(x, err):
    """Співвідношення сигнал/шум, дБ"""
    p_s = np.mean(x ** 2)
    p_e = np.mean(err ** 2)
    if p_e == 0:
        return float('inf')
    return 10.0 * np.log10(p_s / p_e)


bit_depths = [2, 4, 8, 16]
v_min, v_max = -5.0, 5.0          # діапазон АЦП, A1 + A2 = 5 В
p_signal = np.mean(s_l1 ** 2)     # потужність сигналу, В^2

quant = {}
print(f"Діапазон АЦП: [{v_min}; {v_max}] В, max|s| = {np.max(np.abs(s_l1)):.3f} В, "
      f"Ps = {p_signal:.3f} В^2 (теор. (A1^2 + A2^2)/2 = {(A1 ** 2 + A2 ** 2) / 2:.3f})")
print("--- РЕЗУЛЬТАТИ КВАНТУВАННЯ ЗА РІВНЕМ ---")
print(f"{'N':>3} {'L':>6} {'Delta, В':>10} {'sigma_e^2, В^2':>15} {'Delta^2/12':>12} "
      f"{'SNR, дБ':>9} {'6.02N+1.76':>11} {'Ps/(D^2/12)':>12}")
for bits in bit_depths:
    xq, err, delta = uniform_quantize(s_l1, bits, v_min, v_max)
    var_e = np.mean(err ** 2)
    snr = calc_snr(s_l1, err)
    snr_rule = 6.02 * bits + 1.76
    snr_exact = 10.0 * np.log10(p_signal / (delta ** 2 / 12.0))
    quant[bits] = {'xq': xq, 'err': err, 'delta': delta, 'var': var_e,
                   'snr': snr, 'snr_rule': snr_rule, 'snr_exact': snr_exact}
    print(f"{bits:3d} {2 ** bits:6d} {delta:10.6f} {var_e:15.3e} {delta ** 2 / 12:12.3e} "
          f"{snr:9.2f} {snr_rule:11.2f} {snr_exact:12.2f}")

# графіки: перші 64 відліки (20 мс) для кожної розрядності
n_show = len(t_s1)
t_ms = t_l1[:n_show] * 1000
colors = {2: '#d62728', 4: '#2ca02c', 8: '#ff7f0e', 16: '#9467bd'}

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
for row, bits in enumerate(bit_depths):
    q = quant[bits]
    ax = axes[row, 0]
    ax.plot(t_ms, s_l1[:n_show], '--', color='#1f77b4', label='Вихідний дискретний')
    ax.step(t_ms, q['xq'][:n_show], where='mid', color=colors[bits],
            label=f'Квантований ({bits} біт)')
    ax.set_title(f"Квантування, N = {bits} біт ({2 ** bits} рівнів)")
    ax.set_xlabel("Час, мс")
    ax.set_ylabel("Амплітуда, В")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[row, 1]
    ax.plot(t_ms, q['err'][:n_show], color=colors[bits], lw=1.3)
    ax.axhline(q['delta'] / 2, color='k', ls=':', lw=1)
    ax.axhline(-q['delta'] / 2, color='k', ls=':', lw=1, label='±Δ/2')
    ax.set_title(f"Похибка квантування e[n], N = {bits} біт (Δ = {q['delta']:.4g} В)")
    ax.set_xlabel("Час, мс")
    ax.set_ylabel("Похибка, В")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("figures/quantization_error.png", dpi=300)
plt.close()


# %% Блок 5. Дослідження дизеризації (Dithering)
np.random.seed(42)      # щоб результат відтворювався

bits_d = 3                                       # груба розрядність, щоб ефект було видно
delta_d = (v_max - v_min) / (2 ** bits_d - 1)
A_d = 0.5 * delta_d                              # амплітуда дизер-шуму за варіантом

# квантування без дизеризації
xq_nd, err_nd, _ = uniform_quantize(s_l1, bits_d, v_min, v_max)

# квантування з дизеризацією: d[n] ~ U(-A_d, +A_d)
d = np.random.uniform(-A_d, A_d, size=len(s_l1))
xq_d, _, _ = uniform_quantize(s_l1 + d, bits_d, v_min, v_max)
err_d = xq_d - s_l1                               # похибка відносно чистого сигналу

snr_nd = calc_snr(s_l1, err_nd)
snr_d = calc_snr(s_l1, err_d)

# спектри похибок (FFT)
f_e1, sp_e1 = compute_spectrum(err_nd, f_s1)
f_e2, sp_e2 = compute_spectrum(err_d, f_s1)


def peak_to_mean(amp):
    """Найбільша спектральна лінія відносно середнього рівня спектра, дБ (без постійної складової)"""
    return 20.0 * np.log10(np.max(amp[1:]) / np.mean(amp[1:]))


# оцінка спектральної щільності потужності похибки (метод Велча)
f_w, psd_nd = signal.welch(err_nd, fs=f_s1, nperseg=1024)
_, psd_d = signal.welch(err_d, fs=f_s1, nperseg=1024)

print(f"Розрядність N = {bits_d}, Delta = {delta_d:.4f} В, A_d = 0.5*Delta = {A_d:.4f} В")
print(f"Без дизеризації:  SNR = {snr_nd:6.2f} дБ, "
      f"макс. лінія / середнє = {peak_to_mean(sp_e1):5.1f} дБ")
print(f"З дизеризацією:   SNR = {snr_d:6.2f} дБ, "
      f"макс. лінія / середнє = {peak_to_mean(sp_e2):5.1f} дБ")
print(f"Дисперсія похибки: без дизеру {np.mean(err_nd ** 2):.4f} В^2, "
      f"з дизером {np.mean(err_d ** 2):.4f} В^2, Delta^2/12 = {delta_d ** 2 / 12:.4f} В^2")

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[1, :])

ax1.plot(f_e1, 20 * np.log10(np.maximum(sp_e1, 1e-6)), color='#d62728', lw=1.0)
ax1.set_title("Спектр похибки БЕЗ дизеризації")
ax2.plot(f_e2, 20 * np.log10(np.maximum(sp_e2, 1e-6)), color='#2ca02c', lw=1.0)
ax2.set_title("Спектр похибки З дизеризацією")
for ax in (ax1, ax2):
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Рівень, дБВ")
    ax.set_xlim(0, f_s1 / 2)
    ax.set_ylim(-100, 5)
    ax.grid(True)

# нижня межа -120 дБ, щоб нульові відрізки спектра не розтягували масштаб
ax3.plot(f_w, 10 * np.log10(np.maximum(psd_nd, 1e-12)), color='#d62728', label='без дизеризації')
ax3.plot(f_w, 10 * np.log10(np.maximum(psd_d, 1e-12)), color='#2ca02c', label='з дизеризацією')
ax3.set_title("Спектральна щільність потужності похибки (метод Велча)")
ax3.set_xlabel("Частота, Гц")
ax3.set_ylabel("СЩП, дБ(В²/Гц)")
ax3.set_xlim(0, f_s1 / 2)
ax3.grid(True)
ax3.legend(loc='lower right')

plt.tight_layout()
plt.savefig("figures/dither_spectrum.png", dpi=300)
plt.close()


# %% Блок 6. Експорт числових масивів у CSV
export = [
    ("s_fs1", f_s1, s_l1),
    ("s_fs2", f_s2, s_l2),
    ("s_fs3", f_s3, s_l3),
]
for bits in bit_depths:
    export.append((f"xq_N{bits}", f_s1, quant[bits]['xq']))
    export.append((f"err_N{bits}", f_s1, quant[bits]['err']))
export.append((f"xq_N{bits_d}_no_dither", f_s1, xq_nd))
export.append((f"err_N{bits_d}_no_dither", f_s1, err_nd))
export.append((f"dither_N{bits_d}", f_s1, d))
export.append((f"xq_N{bits_d}_dither", f_s1, xq_d))
export.append((f"err_N{bits_d}_dither", f_s1, err_d))

with open("data/output_signals.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["signal", "n", "t_s", "value_V"])
    for name, fs, arr in export:
        for n, v in enumerate(arr):
            writer.writerow([name, n, f"{n / fs:.8f}", f"{v:.8f}"])

print(f"\nУ CSV записано {sum(len(a) for _, _, a in export)} рядків ({len(export)} сигналів).")
print("Моделювання завершено. Графіки збережено в 'figures/', дані - в 'data/'.")
