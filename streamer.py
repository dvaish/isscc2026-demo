#!/usr/bin/env python3
"""
Data Streamer - Streams EMG data over a socket for the demo GUI.
Reads from Playback/emg/user1/adc_raw_[trial]_21_[setting].npz files
or from a dynamically updated circular buffer file.
"""

import socket
import struct
import numpy as np
import time
import argparse
from dataclasses import dataclass

# Constants
HOST = 'localhost'
PORT = 5555
NUM_CHANNELS = 16
CHUNK_SIZE = 33  # Samples per chunk (~33ms at 1kHz, sent at 30Hz)
SAMPLE_RATE = 1000  # Original sample rate in Hz
STREAM_RATE = 30  # How often we send chunks (Hz)

BUFFER_MAGIC = b'EMGB'
BUFFER_VERSION = 1
BUFFER_HEADER_FORMAT = '<4sIIIIQ'
BUFFER_HEADER_SIZE = 32  # Bytes (padded)


@dataclass
class BufferHeader:
    magic: bytes
    version: int
    num_channels: int
    capacity: int
    write_pos: int
    total_written: int


class CircularBufferReader:
    def __init__(self, buffer_file: str, expected_channels: int = NUM_CHANNELS):
        self.buffer_file = buffer_file
        self.expected_channels = expected_channels
        self._last_total_written = 0
        self._data_memmap = None
        self._capacity = None
        self._num_channels = None

    def _read_header(self) -> BufferHeader:
        with open(self.buffer_file, 'rb') as f:
            header_bytes = f.read(BUFFER_HEADER_SIZE)
        if len(header_bytes) < struct.calcsize(BUFFER_HEADER_FORMAT):
            raise ValueError('Buffer file header is too small.')
        unpacked = struct.unpack(BUFFER_HEADER_FORMAT, header_bytes[:struct.calcsize(BUFFER_HEADER_FORMAT)])
        header = BufferHeader(*unpacked)
        if header.magic != BUFFER_MAGIC:
            raise ValueError('Invalid buffer file magic.')
        if header.version != BUFFER_VERSION:
            raise ValueError('Unsupported buffer file version.')
        if header.num_channels != self.expected_channels:
            raise ValueError(
                f'Buffer channels mismatch: expected {self.expected_channels}, got {header.num_channels}'
            )
        return header

    def _ensure_memmap(self, header: BufferHeader):
        if self._data_memmap is None or self._capacity != header.capacity or self._num_channels != header.num_channels:
            offset = BUFFER_HEADER_SIZE
            shape = (header.num_channels, header.capacity)
            self._data_memmap = np.memmap(
                self.buffer_file,
                mode='r',
                dtype=np.float32,
                offset=offset,
                shape=shape,
                order='C'
            )
            self._capacity = header.capacity
            self._num_channels = header.num_channels

    def get_chunk(self, chunk_size: int) -> tuple[np.ndarray, bool]:
        """Return (chunk, updated). updated=False if no new data since last call."""
        header = self._read_header()
        self._ensure_memmap(header)

        if header.total_written == self._last_total_written:
            return np.zeros((header.num_channels, chunk_size), dtype=np.float32), False

        if header.total_written < chunk_size:
            self._last_total_written = header.total_written
            return np.zeros((header.num_channels, chunk_size), dtype=np.float32), False

        end_pos = header.write_pos
        start_pos = (end_pos - chunk_size) % header.capacity

        if start_pos < end_pos:
            chunk = self._data_memmap[:, start_pos:end_pos].copy()
        else:
            first = self._data_memmap[:, start_pos:].copy()
            second = self._data_memmap[:, :end_pos].copy()
            chunk = np.hstack([first, second])

        self._last_total_written = header.total_written
        return chunk.astype(np.float32, copy=False), True


class DataStreamer:
    def __init__(self, trial=0, data_dir='Playback/emg/user1', source='static', buffer_file=None):
        self.trial = trial
        self.data_dir = data_dir
        self.source = source
        self.buffer_file = buffer_file
        self.data_cache = {}  # Cache loaded data by setting
        self.current_settings = np.ones(NUM_CHANNELS, dtype=np.int32)  # Default setting 1
        self.position = 0  # Current read position in data
        self.buffer_reader = None

        if self.source == 'static':
            # Preload all settings for the trial
            self._load_all_settings()
        elif self.source == 'buffer':
            if not self.buffer_file:
                raise ValueError('buffer_file must be set when source is "buffer"')
            self.buffer_reader = CircularBufferReader(self.buffer_file, expected_channels=NUM_CHANNELS)
        else:
            raise ValueError('source must be "static" or "buffer"')
        
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
        if self.source == 'buffer':
            chunk, updated = self.buffer_reader.get_chunk(CHUNK_SIZE)
            return chunk, updated

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
        return chunk, True
    
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
                chunk, updated = self.get_chunk()

                if updated:
                    # Send header + data
                    header = b'DATA'
                    data_bytes = chunk.astype(np.float32).tobytes()
                    size = struct.pack('I', len(data_bytes))

                    try:
                        conn.sendall(header + size + data_bytes)
                    except BlockingIOError:
                        pass

                    last_send = now
                else:
                    # No new data; throttle to avoid busy loop
                    last_send = now
            
            time.sleep(0.005)  # Small sleep to prevent busy loop


def main():
    parser = argparse.ArgumentParser(description='Stream EMG data over socket')
    parser.add_argument('--trial', type=int, default=0, help='Trial number (0-4)')
    parser.add_argument('--port', type=int, default=PORT, help='Port number')
    parser.add_argument('--source', type=str, default='static', choices=['static', 'buffer'],
                        help='Data source: static npz files or circular buffer file')
    parser.add_argument('--buffer-file', type=str, default=None,
                        help='Path to circular buffer file (required for source=buffer)')
    args = parser.parse_args()

    streamer = DataStreamer(trial=args.trial, source=args.source, buffer_file=args.buffer_file)
    streamer.run_server()


if __name__ == '__main__':
    main()
