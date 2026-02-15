from testboard.interface import AURATestBoard
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Any, List, Tuple


def program_lut(brd: Any, adc_channel: int, lut_filename: str):
    lut = np.loadtxt(lut_filename).astype(int)
    for i, val in enumerate(lut):
        if val < 0:
            val = 2**16 + val
        brd.write_cal_lut(int(adc_channel), int(i), int(val))


def program_lut_all(brd: Any, lut_filename: str):
    lut = np.loadtxt(lut_filename).astype(int)
    for ch in range(18):
        for i, val in enumerate(lut[ch]):
            sval = val
            if val < 0:
                val = 2**16 + val
            print(f"Programming LUT for channel {ch}, address {i}, signed {sval}, unsigned {val}")
            brd.write_cal_lut(int(ch), int(i), int(val))


def read_lut(brd: Any, adc_channel: int):
    return [brd.read_cal_lut(int(adc_channel), int(i)) for i in range(33)]


def setup_curr_ctrl(brd: Any, n_curr_ctrl: int, p_curr_ctrl: int):
    code = (p_curr_ctrl << 4) | n_curr_ctrl
    code = (code << 8) | code
    for i in range(9):
        brd.write_register(16+i, code)


def setup_edo_override(brd: Any, code: int):
    print("Setting EDO Mode to override")
    for i in range(8):
        brd.write_register(8 + i, code | code << 6)
    # Set the EDO Mode to EDO Mode override
    brd.write_register(25, 0x4000)


def setup_edo_gain(brd: Any, code: int):
    print("Setting EDO Gain Calibration Codes")
    for i in range(16):
        brd.write_register(35+i, code)


def setup_adc_settings(brd: Any, setting: List[int]):
    for i in range(2):
        val = 0
        for j in range(8):
            val = (val | (setting[i*8+j] << (j*2)))
        brd.write_register(29+i, val)

def write_model_params(brd):
    # SOUL model parameters
    num_features = 8
    num_channels = 16
    spi_addr = 96
    path = "/Users/dhruvvaish/Documents/Berkeley/Muller/AURA/aura_firmware/scripts/datasets/seizure/chb16_seizure17_test_{}.txt"
    test_weights_path = path.format("weights")
    test_mean_path = path.format("mean")
    test_stdev_path = path.format("std")
    test_prenorm_path = path.format("prenorm")
    with open(test_weights_path, 'r') as f:
        test_weights = list(f.readlines())[0].strip('\n').split(',')
        test_weights = [int(i, 16) for i in test_weights]
        # write intercept (or bias)
        brd.write_register(spi_addr, test_weights[0])
        # write weights
        spi_addr += 1
        print("=== Write weights ===", spi_addr)
        for i in range(num_features):
            for j in range(num_channels):
                brd.write_register(spi_addr, test_weights[i*num_channels+j+1])
                spi_addr += 1
    with open(test_mean_path, 'r') as f:
        test_mean = list(f.readlines())[0].strip('\n').split(',')
        test_mean = [int(i, 16) for i in test_mean]
        # write mean
        print("=== Write mean ===", spi_addr)
        for i in range(num_features):
            for j in range(num_channels):
                brd.write_register(spi_addr, test_mean[i*num_channels+j])
                spi_addr += 1
    with open(test_stdev_path, 'r') as f:
        test_stdev = list(f.readlines())[0].strip('\n').split(',')
        test_stdev = [int(i, 16) for i in test_stdev]
        # write stdev
        print("=== Write stdev ===", spi_addr)
        for i in range(num_features):
            for j in range(num_channels):
                brd.write_register(spi_addr, test_stdev[i*num_channels+j])
                spi_addr += 1
    with open(test_prenorm_path, 'r') as f:
        test_prenorm = list(f.readlines())[0].strip('\n').split(',')
        test_prenorm = [int(i, 16) for i in test_prenorm]
        # write prenorm
        spi_addr = 481
        brd.write_register(spi_addr, (test_prenorm[7] << 12) + (test_prenorm[6] << 8) + (test_prenorm[5] << 4) + test_prenorm[4])
        spi_addr += 1
        brd.write_register(spi_addr, (test_prenorm[3] << 12) + (test_prenorm[2] << 8) + (test_prenorm[1] << 4) + test_prenorm[0])
    # other configuration parameters
    ictal_window = 10
    interictal_window = 100
    spi_addr += 1
    brd.write_register(spi_addr, (ictal_window << 8) + interictal_window)
    ictal_confidence = 0x02CD
    interictal_confidence = 0x0133
    spi_addr += 1
    brd.write_register(spi_addr, ictal_confidence)
    spi_addr += 1
    brd.write_register(spi_addr, interictal_confidence)
    spi_addr = 1
    brd.write_register(spi_addr, 4 + (4 << 8)) # timing_gen_en + digtest_out_sel = 4
    spi_addr = 3
    brd.write_register(spi_addr, 255) # soul_clk_half_period


def setup_test(brd: Any, ptat: int, ncm: int, pcm: int, vcasc: int, 
               n_curr_ctrl: int, p_curr_ctrl: int,
               vref_ldo: int, vref: int, external_vref: int, 
               edo_code_override: int):
    """ Setup the chip """
    # Enable ptat, write the ptat code, pcm, and vcasc
    brd.write_register(32, 2 << 12)
    brd.write_register(34, pcm << 12 | vcasc << 8 | 4 << 4 | ptat)
    # Enable the VREF LDO and enable common mode block and set ncm 
    if external_vref:
        print("Using external vref")
        brd.write_register(33, 16 | ncm)
    else:
        brd.write_register(33, (vref_ldo << 5) | (vref << 10) | 16 | ncm)
        brd.write_register(32, 14 << 12 | 1)
    # Setup the current controls
    setup_curr_ctrl(brd, n_curr_ctrl, p_curr_ctrl)
    # Setup timing for each of the settings
    rst_analog_thres = 31 # Sets when the rst_analog is released
    num_rst_clk_periods = 63 # Sets the number of rst clock periods
    soul_clk_half_period = 255
    num_soul_half_period_per_adc_out_half_period = 8
    chop_clk_num_cycles = 1 # 3
    num_cycles_setting0 = 0
    num_cycles_setting1 = 3
    num_cycles_setting2 = 15
    num_cycles_setting3 = 63
    num_cycles_overall = 63
    brd.write_register(2, (rst_analog_thres << 8) | num_rst_clk_periods)
    brd.write_register(3, soul_clk_half_period)
    brd.write_register(4, (chop_clk_num_cycles << 8) | 
        num_soul_half_period_per_adc_out_half_period)
    brd.write_register(5, (num_cycles_setting1 << 8) | num_cycles_setting0)
    brd.write_register(6, (num_cycles_setting3 << 8) | num_cycles_setting2)
    brd.write_register(7, num_cycles_overall)
    # Setup the settings for each adc
    # Set the EDO to override
    if edo_code_override >= 0:
        setup_edo_override(brd, edo_code_override)
    else:
        setup_edo_gain(brd, 8192)
        brd.write_register(25, 0x0000)

def process_bytearrs(bytearr_list):
    nbytes = len(bytearr_list[0])
    npts = nbytes // 4
    word_data = []
    for bytearr in bytearr_list:
        for i in range(npts):
            word_data.append(int.from_bytes(
                bytearr[i*4:(i+1)*4], byteorder="little", signed=False))
    indices = [x>>28 for x in word_data[:32]]
    start_index = np.where(np.array(indices)==0)[0][0]
    num_pts = (len(word_data) - start_index) // 16
    return [word_data[start_index+i::16][:num_pts] for i in range(16)]

def process_bytearrs_classifier(bytearr_list):
    nbytes = len(bytearr_list[0])
    npts = nbytes // 4
    word_data = []
    for bytearr in bytearr_list:
        for i in range(npts):
            word_data.append(int.from_bytes(
                bytearr[i*4:(i+1)*4], byteorder="little", signed=False))
    word_data = np.array(word_data)
    signed_data = convert_unsigned_to_signed(word_data, 16)
    return signed_data


def convert_unsigned_to_signed(d_raw: np.ndarray, nbits: int):
    msb = d_raw > (2 ** (nbits-1))
    return d_raw + msb * -1 * (2 ** (nbits))


def post_process_data(data, data_src):
    d_raw = np.array(data) & 0x0FFFFFFF
    if data_src in (22, 26, 27):
        return convert_unsigned_to_signed(d_raw, 26)
    elif data_src in (24, 25):
        return convert_unsigned_to_signed(d_raw, 19)
    else:
        return convert_unsigned_to_signed(d_raw, 16)

