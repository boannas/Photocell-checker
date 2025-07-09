import csv
# import time
import serial
import threading
from pylsl import StreamInlet, resolve_byprop
import os

def run_ldr_recording_raw(subject, num_block='test101', port='COM5', baudrate=115200):
    # initial som variables
    raw_data_path = 'raw_data'
    raw_data = []
    block_number_from_marker = [str(num_block)]
    print(f'[INFO] Record Data from Subject: {subject}')

    # Try to connect to Serial (Arduino)
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[Serial] Connected to {port} at {baudrate} baud.")
    except serial.SerialException as e:
        print(f"[ERROR] Error opening serial port: {e}")
        return

    # initial 2 threading logic
    stop_event = threading.Event()
    recording = threading.Event() 
    recording.set()  # Start recording immediately

    # === LDR Reading Thread ===
    def read_ldr():
        # Always read LDR between start and stop blocks
        # Without collect LDR when the block doesn't start
        while not stop_event.is_set():
            if ser and ser.in_waiting:
                try:
                    # Extract LDR value from arduino 
                    # data contain timestamp and LDR value
                    data = ser.readline().strip()
                    parts = data.split(b',')
                    if len(parts) == 2:
                        raw_time = float(parts[0])
                        ldr_value = parts[1].decode('latin-1')
                        if recording.is_set():
                            raw_data.append((raw_time, ldr_value, 'LDR'))
                    else:
                        print(f"[WARN] Bad format: {data}")
                except Exception as e:
                    print(f"[ERROR] {e} with data: {data}")

    # # === LSL Marker Listener Thread ===  
    def marker_listener():
        print("[LSL] Looking for marker stream...")
        streams = resolve_byprop('type', 'Markers')
        if not streams:
            print("[ERROR] No LSL marker stream found.")
            stop_event.set()
            return

        inlet = StreamInlet(streams[0])
        print("[LSL] Connected to LSL. Listening for markers...")
        # Collect Marker until block end
        while not stop_event.is_set():
            sample, timestamp = inlet.pull_sample()

            if sample:
                trigger = sample[0]
                print(f"[Marker] {trigger}, timestep: {timestamp:.6f}s")

                # Update block number if Start Block marker is received
                if trigger.startswith('Start Block'):
                    parts = trigger.split()
                    if len(parts) >= 3:
                        block_number_from_marker[0] = parts[2]

                if trigger.startswith('End Block'):
                    # Stop record for every value and end the program
                    recording.clear()
                    print("[INFO] Recording ENDED. Stopping...")
                    raw_data.append((timestamp, trigger, 'Marker'))
                    stop_event.set()
                else:
                    # Record for any data between Starting and ending blocks
                    if recording.is_set():
                        raw_data.append((timestamp, trigger, 'Marker'))

    # === Start Threads ===
    # Create multiple thread computing at the same time
    threads = [
        threading.Thread(target=read_ldr),
        threading.Thread(target=marker_listener)
    ]
    # Start thread
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Construct folder and filename
    block_id = block_number_from_marker[0]
    folder = os.path.join(raw_data_path, f's{subject}')
    os.makedirs(folder, exist_ok=True)  # create folder if not exists

    raw_name = os.path.join(folder, f's{subject}_block_{block_id}_raw.csv') 
    
    # Check if file already exist
    base_name = f's{subject}_block_{block_id}_raw'
    ext = '.csv'
    counter = 1

    while os.path.exists(raw_name):
        raw_name = os.path.join(folder, f'{base_name}_dup{counter}{ext}')
        counter += 1

    # === Save Raw Data ===
    with open(raw_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Value', 'Type'])
        for row in raw_data:
            writer.writerow(row)

    print(f"\n[SUMMARY] Saved {len(raw_data)} filtered entries to {raw_name}")

    if ser:
        # Disconnect Serial port
        ser.close()
        print("[Serial] Connection closed.")

def run_multiple_blocks(subject, n_blocks=3, port='COM5', baudrate=115200):
    for i in range(1, n_blocks + 1):
        block_id = f"block_{i}"
        print(f"\n===== Starting {block_id} =====")
        run_ldr_recording_raw(subject, num_block=str(i), port=port, baudrate=baudrate)
        print(f"===== Finished {block_id} =====\n")
