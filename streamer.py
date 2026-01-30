#!/usr/bin/env python3
"""
Data Streamer - Streams EMG data over a socket for the demo GUI.
Reads from Playback/emg/user1/adc_raw_[trial]_21_[setting].npz files.
"""

import socket
import struct
import numpy as np
import time
import argparse

# Constants
HOST = 'localhost'
PORT = 5555
NUM_CHANNELS = 64
CHUNK_SIZE = 33  # Samples per chunk (~33ms at 1kHz, sent at 30Hz)
SAMPLE_RATE = 1000  # Original sample rate in Hz
STREAM_RATE = 30  # How often we send chunks (Hz)


class DataStreamer:
    def __init__(self, trial=0, data_dir='Playback/emg/user1'):
        self.trial = trial
        self.data_dir = data_dir
        self.data_cache = {}  # Cache loaded data by setting
        self.current_settings = np.ones(NUM_CHANNELS, dtype=np.int32)  # Default setting 1
        self.position = 0  # Current read position in data
        
        # Preload all settings for the trial
        self._load_all_settings()
        
    def _load_all_settings(self):
        """Load data for all 4 settings."""
        # First pass: find minimum length across all arrays
        min_length = float('inf')
        for setting in range(4):
            filename = f"{self.data_dir}/adc_raw_{self.trial}_21_{setting}.npz"
            data = np.load(filename)
            for key in ['arr_0', 'arr_1', 'arr_2', 'arr_3']:
                min_length = min(min_length, data[key].shape[1])
        
        print(f"Minimum length across all arrays: {min_length}")
        
        # Second pass: load and truncate to minimum length
        for setting in range(4):
            filename = f"{self.data_dir}/adc_raw_{self.trial}_21_{setting}.npz"
            data = np.load(filename)
            # Combine arr_0, arr_1, arr_2, arr_3 into 64 channels, truncating to min length
            combined = np.vstack([
                data['arr_0'][:, :min_length], 
                data['arr_1'][:, :min_length], 
                data['arr_2'][:, :min_length], 
                data['arr_3'][:, :min_length]
            ])
            self.data_cache[setting] = combined
            print(f"Loaded setting {setting}: shape {combined.shape}")
        
        self.data_length = min_length
        print(f"Total samples per channel: {self.data_length}")
        
    def get_chunk(self):
        """Get next chunk of data based on current per-channel settings."""
        # Build output array selecting from appropriate setting per channel
        chunk = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)
        
        # Handle wraparound
        end_pos = self.position + CHUNK_SIZE
        
        for ch in range(NUM_CHANNELS):
            # Setting is 1-4, but data files are 0-3
            setting = self.current_settings[ch] - 1
            setting = max(0, min(3, setting))  # Clamp to valid range
            
            if end_pos <= self.data_length:
                chunk[ch] = self.data_cache[setting][ch, self.position:end_pos]
            else:
                # Wrap around
                first_part = self.data_length - self.position
                chunk[ch, :first_part] = self.data_cache[setting][ch, self.position:]
                chunk[ch, first_part:] = self.data_cache[setting][ch, :CHUNK_SIZE - first_part]
        
        self.position = end_pos % self.data_length
        return chunk
    
    def update_settings(self, settings):
        """Update per-channel resolution settings."""
        self.current_settings = np.array(settings, dtype=np.int32)
        
    def run_server(self):
        """Run the socket server."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Streamer listening on {HOST}:{PORT}")
        
        while True:
            print("Waiting for connection...")
            conn, addr = server.accept()
            print(f"Connected by {addr}")
            
            try:
                self._handle_client(conn)
            except (ConnectionResetError, BrokenPipeError) as e:
                print(f"Client disconnected: {e}")
            finally:
                conn.close()
                
    def _handle_client(self, conn):
        """Handle a connected client."""
        conn.setblocking(False)
        last_send = time.time()
        
        while True:
            # Check for incoming settings update
            try:
                header = conn.recv(4, socket.MSG_PEEK)
                if header:
                    # Read message type
                    msg_type = conn.recv(4)
                    if msg_type == b'SETT':
                        # Read 64 int32 settings
                        settings_data = b''
                        while len(settings_data) < NUM_CHANNELS * 4:
                            chunk = conn.recv(NUM_CHANNELS * 4 - len(settings_data))
                            if not chunk:
                                break
                            settings_data += chunk
                        
                        if len(settings_data) == NUM_CHANNELS * 4:
                            settings = np.frombuffer(settings_data, dtype=np.int32)
                            self.update_settings(settings)
                            print(f"Updated settings: min={settings.min()}, max={settings.max()}")
            except BlockingIOError:
                pass  # No data available
            
            # Send data at target rate (30 Hz, each chunk is ~33ms of data)
            now = time.time()
            if now - last_send >= 1.0 / STREAM_RATE:
                chunk = self.get_chunk()
                
                # Send header + data
                header = b'DATA'
                data_bytes = chunk.astype(np.float32).tobytes()
                size = struct.pack('I', len(data_bytes))
                
                try:
                    conn.sendall(header + size + data_bytes)
                except BlockingIOError:
                    pass
                
                last_send = now
            
            time.sleep(0.005)  # Small sleep to prevent busy loop


def main():
    parser = argparse.ArgumentParser(description='Stream EMG data over socket')
    parser.add_argument('--trial', type=int, default=0, help='Trial number (0-4)')
    parser.add_argument('--port', type=int, default=PORT, help='Port number')
    args = parser.parse_args()
    
    streamer = DataStreamer(trial=args.trial)
    streamer.run_server()


if __name__ == '__main__':
    main()
