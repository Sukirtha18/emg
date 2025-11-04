# emg
emg signal code for project
# ===========================================
# EMG Movement Analysis - Python (PC Version)
# ===========================================
# Works with simulated or real EMG data from ESP32
# Detects muscle activity using RMS envelope

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import random

# -----------------------------
# 1. Generate / load EMG signal
# -----------------------------
fs = 1000             # Sampling frequency (Hz)
duration = 5          # seconds
t = np.linspace(0, duration, fs * duration)

# Simulated EMG baseline noise
emg = np.random.normal(0, 0.05, len(t))

# Add random muscle contractions (bursts)
for start in [1, 2.3, 3.8]:
    emg[int(start*fs):int((start+0.5)*fs)] += np.random.normal(0, 0.25, int(0.5*fs))

# -----------------------------
# 2. Band-pass filter 20–450 Hz
# -----------------------------
def bandpass_filter(signal, lowcut=20, highcut=450, fs=1000, order=4):
    nyq = 0.5 * fs
    low, high = lowcut/nyq, highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

emg_filt = bandpass_filter(emg)

# -----------------------------
# 3. Rectify and smooth (RMS)
# -----------------------------
window = int(0.1 * fs)  # 100 ms window
emg_rect = np.abs(emg_filt)
emg_rms = np.sqrt(np.convolve(emg_rect**2, np.ones(window)/window, mode='same'))

# -----------------------------
# 4. Movement detection
# -----------------------------
threshold = np.mean(emg_rms) + 2*np.std(emg_rms)
active = emg_rms > threshold

# Find start & end of active regions
diff = np.diff(active.astype(int))
start_idx = np.where(diff == 1)[0]
end_idx   = np.where(diff == -1)[0]

# -----------------------------
# 5. Feature summary
# -----------------------------
rms_val = np.mean(emg_rms)
zero_cross = ((emg_filt[:-1] * emg_filt[1:]) < 0).sum()
print("=== EMG Movement Analysis ===")
print(f"RMS Value         : {rms_val:.4f}")
print(f"Zero Crossings    : {zero_cross}")
print(f"Detected Movements: {len(start_idx)}")

# -----------------------------
# 6. Plot results
# -----------------------------
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(t, emg_filt, color='gray')
plt.title("Filtered EMG Signal")
plt.ylabel("Amplitude (V)")

plt.subplot(2,1,2)
plt.plot(t, emg_rms, label="RMS Envelope")
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
for s, e in zip(start_idx, end_idx):
    plt.axvspan(t[s], t[e], color='green', alpha=0.3)
plt.title("EMG RMS Envelope & Movement Detection")
plt.xlabel("Time (s)")
plt.ylabel("RMS Value")
plt.legend()
plt.tight_layout()
plt.show()
