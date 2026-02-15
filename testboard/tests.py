"""
Tests, this file contains a bunch of sanity check tests that can be run on the 
EarEEG chip/Firmware
"""


from .interface import EarEEGTestBoard
from .plot import get_spectrum, get_noise_spec, get_noise_spec_no_harm

import matplotlib.pyplot as plt
import numpy as np


def test_firmware_memory(bit_file):
    """ Tests that the firmware memory (on the FPGA) is working """
    brd = EarEEGTestBoard(bit_file)
    for i in range(15):
        brd.write_firmware_register(i, i)
        val = brd.read_firmware_register(i)
        print(i, val)


def test_spi(bit_file):
    """ Tests the first few SPI registers in the memory """
    brd = EarEEGTestBoard(bit_file)
    brd.wake_up_chip()
    for i in range(10):
        value = brd.read_register(i)
        value = brd.read_register(i)
        print(value)
    brd.tuck_in_chip()


def test_stim_vref_dacs(bit_file):
    brd = EarEEGTestBoard(bit_file)
    brd.set_reset()
    time.sleep(1)
    brd.enable_clk()
    time.sleep(1)
    brd.clear_reset()
    time.sleep(1)
    brd.write_register(50, 0x4441)
    time.sleep(0.1)
    brd.write_register(50, 0x4441)
    time.sleep(0.1)
    brd.write_register(0, 0x0120)
    time.sleep(0.1)
    brd.write_register(0, 0x0120)
    time.sleep(0.1)
    brd.read_register(0)
    time.sleep(0.1)
    brd.read_register(0)
    time.sleep(0.1)
    for i in range(1):
        print("Enabling stim DAC")
        brd.write_register(43, 0x00FF)
        time.sleep(0.1)
        brd.write_register(43, 0x00FF)
        time.sleep(0.1)
        brd.write_register(42, 1 << i)
        time.sleep(0.1)
        brd.write_register(42, 1 << i)
        time.sleep(0.1)
        brd.write_register(41, 1 << i)
        time.sleep(0.1)
        brd.write_register(41, 1 << i)
        time.sleep(0.1)
        brd.read_register(43)
        brd.read_register(43)
        time.sleep(0.1)
        brd.read_register(42)
        brd.read_register(42)
        time.sleep(0.1)
        brd.read_register(41)
        brd.read_register(41)
        time.sleep(0.1)
        for j in range(15):
            print(f"Setting DAC to {j}")
            brd.write_register(51, j | (j << 4) | (j << 8) | (j << 12))
            time.sleep(0.1)
            brd.write_register(52, j | (j << 4) | (j << 8) | (j << 12))
            time.sleep(0.1)
            brd.write_register(53, j | (j << 4) | (j << 8) | (j << 12))
            time.sleep(0.1)
            brd.write_register(54, j | (j << 4) | (j << 8) | (j << 12))
            time.sleep(2)
    brd.set_reset()
    time.sleep(1)
    brd.disable_clk()
    brd.print_debug_info()


def test_fft():
    # Test the FFT based SNDR and SNR functions to make sure they are consistent
    t = np.arange(1, 10000) * 1/1e3
    fs = 1e3
    fsig = 13
    test_sig = np.cos(2 * np.pi * fsig * t)
    noise_amp = 5e-6
    test_sig += np.random.randn(len(t)) * noise_amp
    freq, spec = get_spectrum(test_sig, fs)
    freq2, spec2 = get_spectrum(test_sig, fs)
    noise_spec = get_noise_spec(freq, spec, fsig, margin=14)
    sig_idx = np.argmin(np.abs(freq-fsig))
    sig_pwr = sum(spec[sig_idx-14:sig_idx+15])
    noise_pwr = sum(noise_spec)
    avg_noise = np.average(noise_spec)
    snr = sig_pwr / (noise_pwr)
    snr2 = sig_pwr / (avg_noise * len(test_sig))
    print(10*np.log10(snr))
    print(10*np.log10(snr2))
    plt.figure()
    plt.plot(test_sig)
    plt.figure()
    plt.semilogx(freq, 10*np.log10(spec))
    plt.semilogx(freq, 10*np.log10(noise_spec))
    plt.semilogx(freq2, 10*np.log10(spec2))
    plt.show()
