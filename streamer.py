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

from testboard.interface import AURATestBoard
import argparse
import numpy as np
import time
from typing import Any, List, Tuple
from testboard.adc import *
from testboard.setup import setup_adc_test
from tqdm import tqdm
import pickle

from train_hardware import build_dataset, build_dataset, run_adaptive_model

# Constants
HOST = 'localhost'
PORT = 5555
NUM_CHANNELS = 16
NUM_CLASSES = 12
SPLIT = 0
CHUNK_SIZE = 33  # Samples per chunk (~33ms at 1kHz, sent at 30Hz)
SAMPLE_RATE = 1000  # Original sample rate in Hz
STREAM_RATE = 30  # How often we send chunks (Hz)

BUFFER_MAGIC = b'EMGB'
BUFFER_VERSION = 1
BUFFER_HEADER_FORMAT = '<4sIIIIQ'
BUFFER_HEADER_SIZE = 32  # Bytes (padded)


class DataStreamer:
    def __init__(self, trial=0, data_dir='Playback/emg/user1', source='static', buffer_file=None):
        self.trial = trial
        self.data_dir = data_dir
        self.source = source
        self.data_cache = {}  # Cache loaded data by setting
        self.current_settings = np.ones(NUM_CHANNELS, dtype=np.int32)  # Default setting 1
        self.write_position = 0  # Current write position to chip
        self.read_position = 0  # Current read position in data
        self.brd = None
        self.labels = np.array([], dtype=np.int32)
        self.coefs = np.zeros((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)
        self.pending_meta = True

        if self.source == 'static':
            # Preload all settings for the trial
            self._load_all_settings()
            self._init_default_labels()
        elif self.source == 'chip':
            self._setup_dataset()
            self._setup_chip()
        else:
            raise ValueError('source must be "static" or "chip"')
        
        
    def _setup_dataset(self, trial:int=0, split:int=0, n_samples:int=256, user:int=5):

        self.trial_lst, self.label_lst = build_dataset(
            filename=f'playback/emg/user{user}/adc_raw_{trial}_21_{3}.npz',
            splits=4, raw=False
        )
        self.trial_lst = np.array(self.trial_lst)
        self.label_lst = np.array(self.label_lst)

        acc_arr, weights_arr, settings_arr, coefs_arr = run_adaptive_model(
            trials_data=self.trial_lst, 
            trials_labels=self.label_lst, 
            selected_idcs=np.ones(64).astype(bool), 
            sim_settings=[3]*64, 
            trial=trial
        )
        print(f"Adaptive model run complete. Accuracy: {acc_arr[-1]:.4f}, Settings: {settings_arr[-1]}, Weights: {weights_arr[-1]}")
        self.selected_idcs = np.zeros(64, dtype=bool)
        self.selected_idcs[np.argsort(weights_arr[-1])[-NUM_CHANNELS:]] = True
        self.coefs_arr = coefs_arr
        self.coefs = np.array(coefs_arr, dtype=np.float32) if coefs_arr is not None else np.zeros((0, 0), dtype=np.float32)
        if len(self.label_lst) > self.trial:
            self.labels = np.array(self.label_lst[self.trial], dtype=np.int32)
        self.pending_meta = True
        
        data_file = f"datasets/emg/user{user}.npy" # TODO: Get the path to the correct data file
        dataset = np.load(data_file)
        ntrials, nsplits, nch, npts = dataset.shape

        dataset = dataset[trial]
        dataset = dataset.reshape(-1, npts)  # Shape (npts, NUM_CHANNELS)
        input_data_arr = np.zeros((npts, NUM_CHANNELS), dtype=int)
        input_data_arr = dataset[self.selected_idcs, :].T
        input_data_byte_arr = bytearray(
                input_data_arr.astype(dtype='<u4', order='C').tobytes())
        

        nstrides = len(input_data_byte_arr) // (n_samples * 4 * 16)
        curr_dataset = [input_data_byte_arr[i*n_samples*16*4:(i+1)*n_samples*16*4] 
                        for i in range(nstrides)]
        
        self.dataset = curr_dataset
        

    def _setup_chip(self, reprogram=True, src=22):
        brd = AURATestBoard(reprogram=reprogram)
        
        self.brd = brd
        self.src = src

        brd.enable_spi_stream(src=src, dac=True, readin=True,
                          dac_count=1039) # classifier_src=16, adc_raw=22
        time.sleep(0.1)
        brd.disable_spi_stream()
        for i in range(16):
            # the added 1000 corrects for some offset present in the DAC
            brd.write_dac(i, 2**15+500, edo=True) 
            brd.write_dac(i, 2**15+500, edo=False) 
        time.sleep(0.1)

        time.sleep(30)

        ptat = 8
        ncm = 11
        pcm = 5
        vcasc = 15
        vref_ldo = 10
        vref = 8
        external_vref = True
        n_curr_ctrl = 13
        p_curr_ctrl = 0
        edo_code_override = 32
        lsb_shift = 30
        npackets = 100
        channel = 0
        setting = 3
        lut_file = ''
        cal_int = False


        brd.wake_up_chip()
        setup_test(brd, ptat, ncm, pcm, vcasc, 
                n_curr_ctrl, p_curr_ctrl, 
                vref_ldo, vref, external_vref,
                edo_code_override)
        time.sleep(10)

        setup_adc_test(brd, src, [setting] * 16, lut_file, 
                lsb_shift, cal_int)
        
        brd.write_register(4, (0 << 8) | 15) 
        brd.write_register(5, (3 << 8) | 1)

        brd.enable_spi_stream(src=src, dac=True, readin=True,
                            dac_count=1039, delay=True, classifier_src=18) # classifier_src=16, adc_raw=22



      
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

    def _init_default_labels(self):
        """Initialize default labels if not provided by dataset."""
        if self.labels.size == 0:
            # 11 classes x 80 windows per class (matches build_dataset)
            labels = np.array(sum([[i] * 80 for i in range(11)], []), dtype=np.int32)
            self.labels = labels
            self.pending_meta = True

    def _normalize_coefs(self, coefs):
        """Ensure coef matrix has shape (NUM_CLASSES, NUM_CHANNELS)."""
        if coefs is None:
            return np.zeros((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)
        coefs = np.array(coefs, dtype=np.float32)
        if coefs.ndim != 2:
            return np.zeros((NUM_CLASSES, NUM_CHANNELS), dtype=np.float32)
        n_classes, n_channels = coefs.shape
        if n_channels < NUM_CHANNELS:
            pad = np.zeros((n_classes, NUM_CHANNELS - n_channels), dtype=np.float32)
            coefs = np.concatenate([coefs, pad], axis=1)
        elif n_channels > NUM_CHANNELS:
            coefs = coefs[:, :NUM_CHANNELS]
        if n_classes < NUM_CLASSES:
            pad = np.zeros((NUM_CLASSES - n_classes, NUM_CHANNELS), dtype=np.float32)
            coefs = np.concatenate([coefs, pad], axis=0)
        elif n_classes > NUM_CLASSES:
            coefs = coefs[:NUM_CLASSES, :]
        return coefs
        
    def get_chunk(self):
        """Get next chunk of data based on current per-channel settings."""
        if self.source == 'chip':
            num_write, num_read, readout = self.brd.end_to_end_stream_buffer(
                self.dataset[self.read_position % len(self.dataset)], 
                adc=True,
            )
            
            if num_write is not None:
                if num_write > 0:
                    self.write_position += 1
            if num_read is not None:
                if num_read > 0:
                    self.read_position += 1
                    data = process_bytearrs([readout])
                    data_proc = post_process_data(data, self.src)
                    return data_proc, True
                else:
                    return None, False
            else:
                return None, False

        else:
            # Build output array selecting from appropriate setting per channel
            chunk = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)
            
            # Handle wraparound
            end_pos = self.read_position + CHUNK_SIZE
            
            for ch in range(NUM_CHANNELS):
                # Setting is 1-4, but data files are 0-3
                setting = self.current_settings[ch] - 1
                setting = max(0, min(3, setting))  # Clamp to valid range
                
                if end_pos <= self.data_length:
                    chunk[ch] = self.data_cache[setting][ch, self.read_position:end_pos]
                else:
                    # Wrap around
                    first_part = self.data_length - self.read_position
                    chunk[ch, :first_part] = self.data_cache[setting][ch, self.read_position:]
                    chunk[ch, first_part:] = self.data_cache[setting][ch, :CHUNK_SIZE - first_part]
            
            self.read_position = end_pos % self.data_length
            return chunk, True
    
    def update_settings(self, settings):
        """Update per-channel resolution settings."""
        self.current_settings = np.array(settings, dtype=np.int32)
        print(self.current_settings)
        setup_adc_settings(self.brd, settings[SPLIT*NUM_CHANNELS:(SPLIT+1)*NUM_CHANNELS])  # Assuming all channels use the same setting for simplicity
        acc_arr, weights_arr, settings_arr, coefs_arr = run_adaptive_model(
            trials_data=self.trial_lst,
            trials_labels=self.label_lst,
            selected_idcs=self.selected_idcs,
            sim_settings=4-settings,
            trial=self.trial
        )
        self.coefs = np.array(coefs_arr, dtype=np.float32) if coefs_arr is not None else np.zeros((0, 0), dtype=np.float32)
        self.pending_meta = True

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
                    if self.pending_meta:
                        self._send_labels(conn)
                        self._send_coefs(conn)
                        self.pending_meta = False

                    # Send header + data
                    header = b'DATA'
                    data_bytes = chunk.astype(np.float32).tobytes()
                    dims = struct.pack('<II', chunk.shape[0], chunk.shape[1])

                    try:
                        conn.sendall(header + dims + data_bytes)
                    except BlockingIOError:
                        pass

                    last_send = now
                else:
                    # No new data; throttle to avoid busy loop
                    last_send = now
            
            time.sleep(0.005)  # Small sleep to prevent busy loop

    def _send_labels(self, conn):
        """Send labels packet to the client."""
        labels = np.array(self.labels, dtype=np.int32).reshape(-1)
        header = b'LABL'
        meta = struct.pack('<I', labels.size)
        payload = labels.tobytes()
        try:
            conn.sendall(header + meta + payload)
        except BlockingIOError:
            pass

    def _send_coefs(self, conn):
        """Send coef matrix packet to the client."""
        coefs = np.array(self.coefs, dtype=np.float32)
        if coefs.ndim != 2:
            coefs = np.zeros((0, 0), dtype=np.float32)
        header = b'COEF'
        meta = struct.pack('<II', coefs.shape[0], coefs.shape[1])
        payload = coefs.tobytes()
        try:
            conn.sendall(header + meta + payload)
        except BlockingIOError:
            pass


def main():
    parser = argparse.ArgumentParser(description='Stream EMG data over socket')
    parser.add_argument('--trial', type=int, default=0, help='Trial number (0-4)')
    parser.add_argument('--port', type=int, default=PORT, help='Port number')
    parser.add_argument('--source', type=str, default='static', choices=['static', 'chip'],
                        help='Data source: static npz files or circular buffer file')
    parser.add_argument('--buffer-file', type=str, default=None,
                        help='Path to circular buffer file (required for source=buffer)')
    args = parser.parse_args()

    streamer = DataStreamer(trial=args.trial, source=args.source, buffer_file=args.buffer_file)
    streamer.run_server()


if __name__ == '__main__':
    main()
