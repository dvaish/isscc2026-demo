from buffer import CircularBufferWriter, init_buffer_file
from testboard.interface import AURATestBoard
import argparse
import numpy as np
import time
from typing import Any, List, Tuple
from testboard.adc import *
from test_adc import setup_adc_test
from tqdm import tqdm
import pickle

import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='Test DAC Stream',
            description='Run data through the ADC')
    parser.add_argument('--data_file', help='Dataset file',
                        type=str)
    parser.add_argument('--reprogram', action='store_true', 
                        help='Reprogram the FPGA')
    parser.add_argument('--adc', help="If passed, read from ADC (readout_fifo). Else, read from classifier (classifier_fifo)",
                        action='store_true', default=False)
    parser.add_argument('--ptat', help='PTAT Code to set',
                        type=int, default=8)
    parser.add_argument('--ncm', help='NMOS common mode code to set',
                        type=int, default=11)
    parser.add_argument('--pcm', help='PMOS common mode code to set',
                        type=int, default=5)
    parser.add_argument('--vcasc', help='Cascode votlage code to set',
                        type=int, default=15)
    parser.add_argument('--vref_ldo', help='VREF LDO Code',
                        type=int, default=10)
    parser.add_argument('--vref', help='VREF Code',
                        type=int, default=8)
    parser.add_argument('--external_vref', help='Use External VREF', 
                        action='store_true', default=True)
    parser.add_argument('--n_curr_ctrl', help='NMOS Current DAC Code',
                        type=int, default=13)
    parser.add_argument('--p_curr_ctrl', help='PMOS Current DAC Code',
                        type=int, default=0)
    parser.add_argument('--edo_code_override', help='EDO Code to Override',
                        type=int, default=32)
    parser.add_argument('--lsb_shift', help='LSB Shift to Apply',
                        type=int, default=30)
    parser.add_argument('--npackets', help="Number of samples to record",
                        type=int, default=100)
    parser.add_argument('--src', help="Data source to read out",
                        type=int, default=22)
    parser.add_argument('--channel', help='Channel to read out',
                        type=int, default=0)
    parser.add_argument('--setting', help='ADC Setting (0, 1, 2, or 3)',
                        type=int, default=3)
    parser.add_argument('--lut_file', help='File Containing LUT Data',
                        type=str, default='')
    parser.add_argument('--cal_int', help='Calibrate using the internal system',
                        action='store_true', default=False)
    
    args = parser.parse_args()

    brd = AURATestBoard(reprogram=args.reprogram)
    
    logging.info("BEGIN SETUP")

    brd.enable_spi_stream(src=args.src, dac=True, readin=True,
                          dac_count=1039) # classifier_src=16, adc_raw=22
    time.sleep(0.1)
    brd.disable_spi_stream()

    logging.info("SET DAC")
    for i in range(16):
        # the added 1000 corrects for some offset present in the DAC
        brd.write_dac(i, 2**15+500, edo=True) 
        brd.write_dac(i, 2**15+500, edo=False) 
        time.sleep(0.1)
    time.sleep(30)

    logging.info("WAKE UP CHIP")

    brd.wake_up_chip()
    setup_test(brd, args.ptat, args.ncm, args.pcm, args.vcasc, 
               args.n_curr_ctrl, args.p_curr_ctrl, 
               args.vref_ldo, args.vref, args.external_vref,
               args.edo_code_override)
    time.sleep(10)

    logging.info("END SETUP")

    dataset = np.load(args.data_file)
    ntrials, nsplits, nch, npts = dataset.shape
    trial_lst = []

    setup_adc_test(brd, args.src, [args.setting] * 16, args.lut_file, 
                   args.lsb_shift, args.cal_int)
    
    # Some manual setup
    brd.write_register(4, (0 << 8) | 15) 
    brd.write_register(5, (3 << 8) | 1)


    logging.info(f"On Trial {trial+1} of {ntrials}")
    logging.info(f"On Split {split+1} of {nsplits}")

    input_data_arr = np.zeros((npts, 16), dtype=int)
    input_data_arr[:, :nch] = dataset[trial, split, :, :].T
    input_data_byte_arr = bytearray(
            input_data_arr.astype(dtype='<u4', order='C').tobytes())
    
    
    # Stride and rearange the data
    num_samples = 256
    nstrides = len(input_data_byte_arr) // (num_samples * 4 * 16)
    curr_dataset = [input_data_byte_arr[i*num_samples*16*4:(i+1)*num_samples*16*4] 
                    for i in range(nstrides)]
    
    pbar = tqdm(total=ntrials * nsplits * nstrides, desc="Processing Trials")

    logging.info("Enable SPI Stream")
    # brd.write_register(486, (1 << 15))
    brd.enable_spi_stream(src=args.src, dac=True, readin=True,
                            dac_count=1039, delay=True, classifier_src=18) # classifier_src=16, adc_raw=22
    
    readout_arrs = []

    i, j = 0, 0
    while True:
        num_write, num_read, readout = brd.end_to_end_stream_buffer(
            curr_dataset[i % len(curr_dataset)], 
            adc=args.adc,
        )
        
        if num_write is not None:
            if num_write > 0:
                i += 1
            else:
                logging.info(f"num_write={num_write}, i={i}")
        if num_read is not None:
            if num_read > 0:
                j += 1
                readout_arrs.append(readout)
                pbar.update(1)
                if len(readout_arrs) == 1:
                    data = process_bytearrs(readout_arrs)
                    data_proc = post_process_data(data, args.src)
                    writer.write_block(data_proc)
                    readout_arrs = []
            else:
                logging.info(f"num_read={num_read}, j={j}")


    ### BEGIN CLEANUP ###

    brd._reset_fifo()
    logging.info(f"Data shape: {data_proc.shape}")
    brd.tuck_in_chip()

    ### END CLEANUP ###