import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import deque

# === CONFIG ===
PORT = 'COM5'             # <-- Change this to your actual COM port
BAUD_RATE = 115200
SAMPLE_RATE = 500         # Must match your Arduino's read_freq
WINDOW_SECONDS = 2        # Duration of rolling buffer (e.g., 2s)
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SECONDS

# === INIT ===
ser = serial.Serial(PORT, BAUD_RATE)
data_buffer = deque(maxlen=BUFFER_SIZE)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, SAMPLE_RATE // 2)
ax.set_ylim(0, 1)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_title("Real-Time PSD of LDR Signal")

def update_psd(data):
    f, Pxx = welch(data, fs=SAMPLE_RATE, nperseg=min(1024, len(data)))
    line.set_data(f, Pxx)
    ax.set_xlim(1, 60)  # for SSVEP frequency range
    ax.set_ylim(0, np.max(Pxx) * 1.1 if len(Pxx) else 1)
    fig.canvas.draw()
    fig.canvas.flush_events()

# === STREAM AND PLOT ===
print("[INFO] Reading from Arduino and visualizing PSD... Press Ctrl+C to stop.")
try:
    while True:
        try:
            line_raw = ser.readline().decode('utf-8').strip()
            if ',' in line_raw:
                timestamp_str, value_str = line_raw.split(',')
                value = int(value_str)
                data_buffer.append(value)

                if len(data_buffer) >= BUFFER_SIZE:
                    update_psd(np.array(data_buffer))
        except Exception as e:
            print(f"[WARNING] Skipped malformed line: {e}")
            continue
except KeyboardInterrupt:
    print("\n[STOPPED]")
    ser.close()
