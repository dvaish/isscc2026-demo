from .interface import AURATestBoard
from .adc import *
import time

def setup_adc_test(brd, data_src: int, setting: List[int],
                   lut_filename: str, lsb_shift: int, cal_int: bool): 
    setup_adc_settings(brd, setting)
    brd.write_register(31, 0x5000) # Fixes sign error in ADC
    brd.write_register(77, lsb_shift | 0x40)
    if lut_filename != "":
        brd.write_register(0, 0xFFFF)
        brd.write_register(1, 0xE104)
        print("Programming LUT")
        # for i in range(18):
        #     program_lut(brd, lut_filename)
        program_lut_all(brd, lut_filename)
        brd.write_register(1, 0xE105)
        # brd.write_register(28, (3 << 12) | (5 << 8) | (7 << 4) | 9)
    else:
        brd.write_register(1, 0xE104)
    shift = 2
    brd.write_register(28, ((shift) << 12) | ((shift + 2) << 8) | ((shift + 4) << 4) | (shift + 5))
    brd.write_register(0, 0xFFFF)
    print("Channels Enabled")
    time.sleep(10)
    if cal_int:
        print("Running Calibration")

        brd.write_register(25, 0x4000 | 15)
        brd.write_register(26, 0x00FF)
        brd.write_register(27, 0x00FF)
        brd.write_register(1, 0xE107)
        time.sleep(60)
        brd.write_register(1, 0xE104)
        time.sleep(10)
        lut = np.zeros((18, 33))
        for i in range(18):
            rv = read_lut(brd, i)
            lut[i] = np.array(rv)
            for j, v in enumerate(rv):
                print(f"CH={i}; ADDR={j}; VALUE={v}")
        np.save(f"luts/lut_before_{time.time()}", lut)
        brd.write_register(1, 0xE107)
        time.sleep(10)
        print("Capturing Data now")