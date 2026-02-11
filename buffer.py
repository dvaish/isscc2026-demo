#!/usr/bin/env python3
"""
Example circular buffer writer for the streamer.
Writes synthetic EMG-like data to a fixed-size buffer file.
"""

import argparse
import struct
import time
import numpy as np

BUFFER_MAGIC = b'EMGB'
BUFFER_VERSION = 1
BUFFER_HEADER_FORMAT = '<4sIIIIQ'
BUFFER_HEADER_SIZE = 32  # Bytes (padded)


def _write_header(f, num_channels, capacity, write_pos, total_written):
    header = struct.pack(
        BUFFER_HEADER_FORMAT,
        BUFFER_MAGIC,
        BUFFER_VERSION,
        num_channels,
        capacity,
        write_pos,
        total_written
    )
    header += b'\x00' * (BUFFER_HEADER_SIZE - len(header))
    f.seek(0)
    f.write(header)
    f.flush()


def init_buffer_file(path, num_channels, capacity):
    data_bytes = num_channels * capacity * 4
    total_size = BUFFER_HEADER_SIZE + data_bytes
    with open(path, 'wb') as f:
        _write_header(f, num_channels, capacity, 0, 0)
        f.seek(total_size - 1)
        f.write(b'\x00')


class CircularBufferWriter:
    def __init__(self, path, num_channels, capacity):
        self.path = path
        self.num_channels = num_channels
        self.capacity = capacity
        self.write_pos = 0
        self.total_written = 0
        self._data = np.memmap(
            self.path,
            mode='r+',
            dtype=np.float32,
            offset=BUFFER_HEADER_SIZE,
            shape=(num_channels, capacity),
            order='C'
        )

    def write_block(self, block: np.ndarray):
        if block.shape[0] != self.num_channels:
            raise ValueError('Block channels mismatch.')
        block_len = block.shape[1]
        end_pos = (self.write_pos + block_len) % self.capacity

        if self.write_pos < end_pos:
            self._data[:, self.write_pos:end_pos] = block
        else:
            first = self.capacity - self.write_pos
            self._data[:, self.write_pos:] = block[:, :first]
            self._data[:, :end_pos] = block[:, first:]

        self.write_pos = end_pos
        self.total_written += block_len

        with open(self.path, 'r+b') as f:
            _write_header(f, self.num_channels, self.capacity, self.write_pos, self.total_written)


def main():
    parser = argparse.ArgumentParser(description='Write synthetic EMG data to circular buffer')
    parser.add_argument('--buffer-file', type=str, required=True, help='Path to buffer file')
    parser.add_argument('--channels', type=int, default=64, help='Number of channels')
    parser.add_argument('--capacity', type=int, default=30000, help='Samples per channel in buffer')
    parser.add_argument('--chunk-size', type=int, default=33, help='Samples per write')
    parser.add_argument('--rate', type=float, default=30.0, help='Writes per second')
    args = parser.parse_args()

    init_buffer_file(args.buffer_file, args.channels, args.capacity)
    writer = CircularBufferWriter(args.buffer_file, args.channels, args.capacity)

    t = 0.0
    dt = 1.0 / args.rate
    freqs = np.linspace(10.0, 100.0, args.channels)

    print(f"Writing to {args.buffer_file} at {args.rate} Hz")
    while True:
        # Generate synthetic EMG-like data
        time_axis = t + np.arange(args.chunk_size) / (args.rate * args.chunk_size)
        signal = 10 * np.sin(2 * np.pi * freqs[:, None] * time_axis[None, :])
        noise = 0.05 * np.random.randn(args.channels, args.chunk_size)
        block = (signal + noise).astype(np.float32)

        print(block.shape)

        writer.write_block(block)
        t += dt
        time.sleep(dt)


if __name__ == '__main__':
    main()
