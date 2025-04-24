import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math

# === Data Preprocessing ===
def preprocess_ldr(df):
    """
    Preprocess the dataframe that are desire format.
    Ensure 'LDR Value' is numeric and fill missing 'Marker' with empty string.

    Parameters:
        df (DataFrame): The LDR data with markers.

    Returns:
        df (DataFrame): Preprocess dataframe
    """
    df['LDR Value'] = pd.to_numeric(df['LDR Value'], errors='coerce')
    df['Marker'] = df['Marker'].fillna('').astype(str)
    return df

# === Marker Check ===
def check_missing_marker(df, total_trials=64):
    """
    Ensure Dataframe has all desire 'Markers'.

    Parameters:
        df (DataFrame): The LDR data with markers.
        total_trials (int): Number of trials
    """
    missing = []
    for i in range(1, total_trials + 1):
        if f"trial_{i}" not in df['Marker'].values:
            missing.append(f"trial_{i}")
        if f"end_{i}" not in df['Marker'].values:
            missing.append(f"end_{i}")
    if missing:
        print("Missing markers:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All markers present.")
        
# === Plot Full LDR Signal with Markers ===
def plot_ldr_data_with_markers(df):
    """
    Plot every ldr_value with all markers.

    Parameters:
        (DataFrame): The LDR data with markers.
    """    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time (s from start)'], df['LDR Value'], label='LDR Value', linewidth=1.5)
    for idx, row in df.iterrows():
        if pd.notna(row['Marker']) and row['Marker'].strip() != '':
            plt.axvline(x=row['Time (s from start)'], color='red', linestyle='--', alpha=0.7)
            plt.text(row['Time (s from start)'], max(df['LDR Value']), row['Marker'], rotation=90,
                     verticalalignment='bottom', color='red', fontsize=9)
    plt.title("LDR Value Over Time with Markers")
    plt.xlabel("Time (s from start)")
    plt.ylabel("LDR Value")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# === Frequency Estimation for One Trial ===
def estimate_frequency(df, marker_start, marker_end, monitor_freq, sr=500, prominence=1, threshold=250):
    """
    Compute the estimate frequency of the signal.

    Parameters:
        df (DataFrame): The dataframe
        marker_start (string): Beginning Marker for visualize
        marker_end (string): Endding Marker for visualize
        monitor_freq (integer): Monitor's Refresh rate
        sr (integer): Arduino (Photocell) Sampling rate
        prominence (float): Required prominence of peaks. Higher = only strong peaks are detected.
        threshold (float): Threshold for find peak of signal (based on overall ldr_value)
    
    Returns:
        float: Estimated frequency, based on average time difference between peaks.
    """
    start_time = df[df['Marker'] == marker_start]['Time (s from start)'].values[0]
    end_time = df[df['Marker'] == marker_end]['Time (s from start)'].values[0]
    segment = df[(df['Time (s from start)'] >= start_time) & (df['Time (s from start)'] <= end_time + 0.2)]
    ldr_values = segment['LDR Value'].values
    time_values = segment['Time (s from start)'].values

    dist = int(sr * (2 / monitor_freq)) - 1
    peaks, _ = find_peaks(ldr_values, distance=dist, prominence=prominence, height=threshold)
    if len(peaks) < 2:
        return None
    periods = np.diff(time_values[peaks])
    return 1 / np.mean(periods)

# === Frequency Collection for All Trials ===
def collect_frequencies(df, num_trial=64, monitor_freq=60, sr=500, prominence=1, threshold=250):
    """
    Collect estimated frequencies for a range of trials using LDR signal.

    Parameters:
        df (DataFrame): The LDR data with markers.
        trial_range (range): Trial numbers to process.
        monitor_freq (float): Monitor refresh frequency (used in estimate).
        sr (int): Sampling rate of the LDR signal.
        prominence (float): Minimum prominence of peaks.
        threshold (float): Minimum peak height to detect.

    Returns:
        list: Trial numbers.
        list: Estimated frequency per trial (NaN if failed).
    """
    freqs = []
    for i in range(1, num_trial + 1):
        try:
            freq = estimate_frequency(df, f"trial_{i}", f"end_{i}", monitor_freq, sr, prominence, threshold)
            freqs.append(freq)
        except Exception as e:
            print(f"trial_{i} error: {e}")
            freqs.append(np.nan)
    return list(range(1, num_trial+1)), freqs

# === Plot Peaks in a Trial Segment ===
def plot_peaks_in_trial(df, marker_start, marker_end, distance=15, prominence=1, height=250):
    """
    Plot LDR signal between two markers and highlight detected peaks.

    Parameters:
        df (DataFrame): LDR data with time and marker columns.
        marker_start (str): Marker label for start of the segment.
        marker_end (str): Marker label for end of the segment.
        distance (int): Minimum distance between peaks (in samples).
        prominence (float): Minimum prominence of peaks.
        height (float): Minimum height of peaks.
    """
    start_time = df[df['Marker'] == marker_start]['Time (s from start)'].values[0]
    end_time = df[df['Marker'] == marker_end]['Time (s from start)'].values[0]
    segment = df[(df['Time (s from start)'] >= start_time - 0.2) & (df['Time (s from start)'] <= end_time + 0.2)]

    ldr_values = segment['LDR Value'].values
    time_values = segment['Time (s from start)'].values
    peaks, _ = find_peaks(ldr_values, distance=distance, prominence=prominence, height=height)

    if len(peaks) < 2:
        print("Not enough peaks detected.")
        return

    periods = np.diff(time_values[peaks])
    frequency = 1 / np.mean(periods)

    plt.figure(figsize=(12, 6))
    plt.plot(time_values, ldr_values, label='LDR Signal')
    plt.plot(time_values[peaks], ldr_values[peaks], 'ro', label='Detected Peaks', markersize=3, alpha=0.6)
    plt.axvline(start_time, color='green', linestyle='--', label=marker_start)
    plt.axvline(end_time, color='red', linestyle='--', label=marker_end)
    plt.text(start_time, max(ldr_values) + 5, marker_start, color='green')
    plt.text(end_time, max(ldr_values) + 5, marker_end, color='red')
    plt.title(f"LDR Signal with Peaks ({marker_start} to {marker_end})")
    plt.xlabel("Time (s)")
    plt.ylabel("LDR Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Frequency: {frequency:.4f} Hz, Peaks: {len(peaks)}, Est. Period: {np.mean(periods):.4f} sec")

# === Plot Frequency Stability ===
def plot_frequency_stability(trial_labels, frequencies):
    """
    Plot estimated frequency per trial with mean and standard deviation bands.

    Parameters:
        trial_labels (list): Trial numbers.
        frequencies (list): Frequency estimates (can include NaN).
    """
    freqs = np.array(frequencies)
    trials = np.array(trial_labels)
    mean_freq = np.nanmean(freqs)
    std_freq = np.nanstd(freqs)

    plt.figure(figsize=(12, 5))
    plt.plot(trials, freqs, marker='o', linestyle='-', color='teal', label='Trial Frequency')
    plt.axhline(mean_freq, color='red', linestyle='--', label=f'Mean: {mean_freq:.4f} Hz')
    plt.fill_between(trials, mean_freq - std_freq, mean_freq + std_freq, alpha=0.2, color='gray', label='±1 Std Dev')
    plt.title("Frequency Stability Across Trials")
    plt.xlabel("Trial Number")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Histogram Plot ===
def plot_hist_freq(freqs):
    """
    Plot histogram of estimated frequencies with mean and ±1 SD shading.

    Parameters:
        freqs (list): Frequency values (may include NaN).
    """
    freqs = np.array(freqs)
    freqs = freqs[~np.isnan(freqs)]
    counts, bins, _ = plt.hist(freqs, bins=30, color='teal', edgecolor='teal')

    mean_freq = np.mean(freqs)
    std_freq = np.std(freqs)
    min_x, max_x = bins[0], bins[-1]
    range_x = max_x - min_x

    plt.axvspan(mean_freq - std_freq, mean_freq + std_freq, color='gray', alpha=0.2, label='±1 SD')
    plt.axvline(mean_freq, color='red', linestyle='--', label=f'Mean: {mean_freq:.4f}')
    plt.axvline(mean_freq - std_freq, color='gray', linestyle=':', linewidth=1)
    plt.axvline(mean_freq + std_freq, color='gray', linestyle=':', linewidth=1)

    label_text = (f"min x: {min_x:.4f}\nmax x: {max_x:.4f}\nrange: {range_x:.4f}\n"
                  f"mean: {mean_freq:.4f}\nstd: {std_freq:.4f}")
    plt.gca().text(0.98, 0.98, label_text,
                   transform=plt.gca().transAxes, ha='right', va='top', fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.title("Histogram of Estimated Frequencies (with ±1 SD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_all_frequency_and_ldr(df, block_num, trial_labels, frequencies, n_trial, trials_per_subplot=4, ncols=4):
    """
    Combines: frequency stability, histogram, and LDR-by-blocks in a single figure.
    """
    # Clean frequency array
    freqs = np.array(frequencies)
    trials = np.array(trial_labels)
    valid_freqs = freqs[~np.isnan(freqs)]
    valid_trials = trials[~np.isnan(freqs)]

    mean_freq = np.nanmean(freqs)
    std_freq = np.nanstd(freqs)

    # LDR subplot grid size
    total_subplots = math.ceil(n_trial / trials_per_subplot)
    nrows_grid = math.ceil(total_subplots / ncols)

    # Build layout
    fig = plt.figure(figsize=(5 * ncols, 4 * (nrows_grid + 2)))  # +2 rows for top plots

    # --- Plot 1: Frequency Stability ---
    ax1 = plt.subplot2grid((nrows_grid + 2, ncols), (0, 0), colspan=ncols)
    ax1.plot(valid_trials, valid_freqs, marker='o', linestyle='-', color='teal', label='Trial Frequency')
    ax1.axhline(mean_freq, color='red', linestyle='--', label=f'Mean: {mean_freq:.4f} Hz')
    ax1.fill_between(valid_trials, mean_freq - std_freq, mean_freq + std_freq, alpha=0.2, color='gray', label='±1 SD')

    # Highlight and label trials outside ±1 SD
    for trial, freq in zip(valid_trials, valid_freqs):
        if freq < mean_freq - std_freq or freq > mean_freq + std_freq:
            ax1.text(trial, freq, f'{int(trial)}', color='crimson', fontsize=10, ha='center', va='bottom', weight='bold')

    ax1.set_title("Frequency Stability Across Trials", fontsize=16, fontname='DejaVu Sans')
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.grid(True)
    ax1.legend()

    # --- Plot 2: Histogram ---
    ax2 = plt.subplot2grid((nrows_grid + 2, ncols), (1, 0), colspan=ncols)
    counts, bins, _ = ax2.hist(valid_freqs, bins=30, color='teal', edgecolor='teal')
    ax2.axvspan(mean_freq - std_freq, mean_freq + std_freq, color='gray', alpha=0.2, label='±1 SD')
    ax2.axvline(mean_freq, color='red', linestyle='--', label=f'Mean: {mean_freq:.4f}')
    ax2.axvline(mean_freq - std_freq, color='gray', linestyle=':', linewidth=1)
    ax2.axvline(mean_freq + std_freq, color='gray', linestyle=':', linewidth=1)

    min_x, max_x = bins[0], bins[-1]
    range_x = max_x - min_x
    label_text = (f"min x: {min_x:.4f}\nmax x: {max_x:.4f}\nrange: {range_x:.4f}\n"
                  f"mean: {mean_freq:.4f}\nstd: {std_freq:.4f}")
    ax2.text(0.98, 0.98, label_text, transform=ax2.transAxes, ha='right', va='top',
             fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
    ax2.set_title("Histogram of Estimated Frequencies (with ±1 SD)", fontsize=16, fontname='DejaVu Sans')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Count")
    ax2.grid(True)
    ax2.legend()

    # --- Plot 3: LDR Grid (Start from row 2) ---
    for i in range(total_subplots):
        row = i // ncols + 2  # start from row 2
        col = i % ncols
        ax = plt.subplot2grid((nrows_grid + 2, ncols), (row, col))

        start_trial = i * trials_per_subplot + 1
        end_trial = min(start_trial + trials_per_subplot - 1, n_trial)

        try:
            t_start = df[df['Marker'] == f"trial_{start_trial}"]['Time (s from start)'].values[0]
            t_end = df[df['Marker'] == f"end_{end_trial}"]['Time (s from start)'].values[0]
        except IndexError:
            print(f"Skipping subplot {i}: missing marker for trials {start_trial}-{end_trial}")
            continue

        window_df = df[(df['Time (s from start)'] >= t_start) & (df['Time (s from start)'] <= t_end)]
        ax.plot(window_df['Time (s from start)'], window_df['LDR Value'], color='gray', linewidth=0.8)

        for trial_num in range(start_trial, end_trial + 1):
            try:
                t0 = df[df['Marker'] == f"trial_{trial_num}"]['Time (s from start)'].values[0]
                t1 = df[df['Marker'] == f"end_{trial_num}"]['Time (s from start)'].values[0]
                segment = df[(df['Time (s from start)'] >= t0) & (df['Time (s from start)'] <= t1)]
                ax.plot(segment['Time (s from start)'], segment['LDR Value'], label=f'Trial {trial_num}', color='cadetblue')
                ax.axvline(t0, color='green', linestyle='--')
                ax.axvline(t1, color='red', linestyle='--')
                ax.text(t0, 1.02, f'{trial_num}', color='darkgreen', fontsize=9, ha='center', transform=ax.get_xaxis_transform())
                ax.text(t1, 1.02, f'{trial_num}', color='maroon', fontsize=9, ha='center', transform=ax.get_xaxis_transform())
            except IndexError:
                continue

        ax.set_title(f"{start_trial}-{end_trial}", fontname='DejaVu Sans')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("LDR Value")
        ax.grid(True)

    # Add a section title for the LDR block
    # ldr_title_y = 1 - (2 / (nrows_grid + 2))  
    fig.text(0.5, 0.635, "LDR Signal with Trial Markers", ha='center', fontsize=16, fontname='DejaVu Sans')

    # Global title
    fig.suptitle(f"Frequency and LDR Signal Overview Block_{block_num}", fontsize=20, y=0.995, weight='bold', fontname='DejaVu Sans')

    # Layout fix
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    plt.show()


import numpy as np
import os
import pandas as pd  # you had `import pandas`, it should be `import pandas as pd`

def handle_overflow(timestamps):
    """
    Adjust timestamps to handle overflow from Arduino micros().
    
    Parameters:
        timestamps (list): List of timestamps in microseconds.
    
    Returns:
        list: Adjusted timestamps.
    """
    adjusted_timestamps = []
    overflow_count = 0
    max_micros = 2**32 - 1  # Maximum value before overflow
    
    for i in range(len(timestamps)):
        if i > 0 and timestamps[i] < timestamps[i - 1]:
            overflow_count += 1
        adjusted_timestamp = timestamps[i] + overflow_count * max_micros
        adjusted_timestamps.append(adjusted_timestamp)
    
    return adjusted_timestamps

def process_ldr_with_markers(subject, block_id, total_trials=64,
                              raw_data_path='raw_data', process_data_path='process_data'):
    """
    Process raw LDR and Marker data, align timestamps, handle overflow, and save cleaned output.

    Parameters:
        subject (int): Subject ID
        block_id (int or str): Block number or label
        total_trials (int): Total number of expected trials
        raw_data_path (str): Folder where raw data is stored
        process_data_path (str): Folder to save cleaned output
    """

    # === Load raw file ===
    file_name = f's{subject}_block_{block_id}_raw.csv'
    file_path = os.path.join(raw_data_path, f's{subject}', file_name)
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return
    
    print(f"[INFO] Loaded: {file_name} ({len(df)} rows)")

    # === Split marker and LDR rows ===
    marker_row = df[df['Type'] == 'Marker'].copy()
    ldr_row = df[df['Type'] == 'LDR'].copy()

    # === Check for missing trial markers ===
    missing = []
    for i in range(1, total_trials + 1):
        if f"trial_{i}" not in marker_row['Value'].values:
            missing.append(f"trial_{i}")
        if f"end_{i}" not in marker_row['Value'].values:
            missing.append(f"end_{i}")

    if missing:
        print("[WARN] Missing markers:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("[INFO] All trial markers present.")

    # === Align timestamps using Start Block as 0s ===
    start_marker = marker_row[marker_row['Value'].str.contains('Start Block', case=False)]
    end_marker = marker_row[marker_row['Value'].str.contains('End Block', case=False)]

    if start_marker.empty or end_marker.empty:
        print("[ERROR] Start or End Block marker missing.")
        return

    init_time = start_marker['Timestamp'].values[0]
    end_time = end_marker['Timestamp'].values[0]

    marker_aligned = marker_row.copy()
    marker_aligned['Timestamp'] = marker_aligned['Timestamp'] - init_time

    # === Handle overflow + align LDR ===
    # ldr_row['Timestamp'] = handle_overflow(ldr_row['Timestamp'].values)

    ldr_rows_aligned = ldr_row.copy()
    ldr_rows_aligned['Timestamp'] = (ldr_rows_aligned['Timestamp'] - ldr_rows_aligned['Timestamp'].iloc[0]) / 1e6

    # === Merge Markers into LDR stream ===
    ldr_marker = ldr_rows_aligned.copy()
    ldr_marker['Marker'] = None
    ldr_marker = ldr_marker.dropna(subset=['Timestamp', 'Value']).reset_index(drop=True)

    ldr_times = ldr_marker['Timestamp'].values
    for _, row in marker_aligned.iterrows():
        marker_time = row['Timestamp']
        label = row['Value']
        idx = np.argmin(np.abs(ldr_times - marker_time))
        ldr_marker.at[idx, 'Marker'] = label

    # === Final format ===
    ldr_marker_cleaned = ldr_marker.drop(columns=['Type']).reset_index(drop=True)
    ldr_marker_cleaned = ldr_marker_cleaned.rename(columns={
        'Timestamp': 'Time (s from start)',
        'Value': 'LDR Value'
    })

    # === Save processed file ===
    folder = os.path.join(process_data_path, f's{subject}')
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f's{subject}_block_{block_id}_process.csv')
    ldr_marker_cleaned.to_csv(save_path, index=False)

    print(f"[DONE] Processed data saved to: {save_path}")
