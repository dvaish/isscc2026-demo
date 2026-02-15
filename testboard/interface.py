from collections import namedtuple
from functools import reduce
import sys
sys.path.append("./API_python3")
import ok
import numpy as np
import time
import threading
import tqdm


class AURATestBoard:
    # Configuration parameters from the Firmware
    rd_wire_addr_error = 0x22
    rd_wire_addr_read = 0x20
    wr_wire_addr_debug = 0x00
    wr_wire_addr_command = 0x02
    wr_wire_addr_addr = 0x04
    wr_wire_addr_data = 0x06
    trigger_addr_command_update = 0x42
    
    pipeout_addr_spi = 0xA0
    pipeout_addr_spi_soul = 0xA8
    pipein_addr = 0x80

    # Clock frequency from the firmware
    clock_frequency = 8e6
    
    # 1bit read specifications
    num_blocks_1bit = 2
    block_size_1bit = 16
    
    # SPI read specifications
    num_blocks_spi = 1
    wr_block_size_spi = 4096 * 4 # bytes
    rd_block_size_spi = 64 * 4 # 64 * 4 # bytes
    
    block_size_spi = 4096 * 4 # bytes

    fpga_freq = 16e6

    opcodes = dict(
        GETID=0xC0,
        CLRRS=0xC2,
        SETRS=0xC1,
        ENCLK=0xC3,
        DISCLK=0xC4,
        RDMEM=0xC5,
        WRMEM=0xC6,
        ENSTR=0xC7,
        DISTR=0xC8,
        ENFMEAS=0xC9,
        RDFMEASP=0xCA,
        RDFMEASN=0xCB,
        WRDTI=0xCC,
        DISFMEAS=0xCD,
        RDADCF=0xCE,
        WRIBF=0xCF,
        RDIBF=0xD0,
        RDDBG=0xD1,
        RDNWD=0xD2,
        RDDTC=0xD3,
        RDSCC=0xD4,
        ENCALDAC=0xD5,
        WRCALDAC=0xD6,
        DISCALDAC=0xD7,
        WRLUT=0xD8,
        RDLUT=0xD9,
        WRDAC=0xDA,
    )

    def __init__(self, bitfile="/Users/dhruvvaish/Documents/Berkeley/Muller/AURA/aura_firmware/Bitstreams/AuraFirmware.bit", reprogram=True):
        xem = ok.okCFrontPanel()
        if xem.NoError != xem.OpenBySerial(""):
            raise ValueError("Unable to open a device!")
        dev_info = ok.okTDeviceInfo()
        if xem.NoError != xem.GetDeviceInfo(dev_info):
            raise ValueError("Error reading Device information")

        print("Product         : {}".format(dev_info.productName))
        print("Firmware Version: {}".format(dev_info.deviceMajorVersion))
        print("Serial Number   : {}".format(dev_info.serialNumber))
        print("Device ID       : {}".format(dev_info.deviceID))
        print("USB Speed       : {}".format(dev_info.usbSpeed))

        if reprogram:
            print(bitfile)
            if xem.NoError != xem.ConfigureFPGA(bitfile):
                raise ValueError("FPGA configuration failed")
        self.xem = xem
        self.input_data_byte_arr = None
        self._reset_fifo()
        self._set_version(3)

    def turn_on_debug_led(self, idx):
        """ Turn on a debug LED """
        self.xem.SetWireInValue(
            self.wr_wire_addr_debug, (1 << idx), 0xFFFFFFFF)
        self.xem.UpdateWireIns()

    def turn_off_debug_led(self):
        """ Turn off all the debug LEDs """
        self.xem.SetWireInValue(
            self.wr_wire_addr_debug, 0, 0xFFFFFFFF)
        self.xem.UpdateWireIns()

    def _parse_error_wire(self, x):
        command_number = (x & 0xFF000000) >> 24
        opcode = (x & 0x00FF0000) >> 16
        error_code = (x & 0x00FFFF)
        return command_number, opcode, error_code

    def _reset_fifo(self):
        self.xem.SetWireInValue(0x08, 0b111, 0xFFFFFFFF)
        self.xem.UpdateWireIns()
        self.xem.SetWireInValue(0x08, 0b000, 0xFFFFFFFF)
        self.xem.UpdateWireIns()

    def _set_version(self, version):
        """ Set the version of the FPGA """
        self.xem.SetWireInValue(0x0A, version, 0xFFFFFFFF)
        self.xem.UpdateWireIns()

    def _update_readback(self):
        """ Read back the error and read wire values """
        self.xem.UpdateWireOuts()
        error_wire = self.xem.GetWireOutValue(self.rd_wire_addr_error)
        read_wire = self.xem.GetWireOutValue(self.rd_wire_addr_read)
        return error_wire, read_wire

    def _send_command(self, opcode, addr, data):
        """ Send a command to the FPGA"""
        self.xem.SetWireInValue(
            self.wr_wire_addr_command, self.opcodes[opcode], 0xFFFFFFFF)
        self.xem.SetWireInValue(
            self.wr_wire_addr_addr, addr, 0xFFFFFFFF)
        self.xem.SetWireInValue(
            self.wr_wire_addr_data, data, 0xFFFFFFFF)
        self.xem.UpdateWireIns()
        self.xem.ActivateTriggerIn(self.trigger_addr_command_update, 0)
        time.sleep(0.1)
        error_wire, read_wire = self._update_readback()
        command_num, opcode_out, error_code = self._parse_error_wire(error_wire)
        if opcode_out != self.opcodes[opcode] or error_code != 0:
            print(f"[{command_num}], Error Code: {0}".format(error_code))
            raise ValueError(f"Failed to send command OPCODE={opcode}, DATA={data}, ADDR={addr}, OPCODE_OUT={opcode_out}, EXPECTED={self.opcodes[opcode]}")
        return read_wire

    def set_reset(self):
        """ Set reset on the Ear EEG Chip """
        self._send_command("SETRS", 0, 0)

    def clear_reset(self):
        """ Clear the reset on the EarEEG chip """
        self._send_command("CLRRS", 0, 0)

    def enable_clk(self, clk_div_code=0, spi_div_code=0):
        """ Enable the EarEEG clock to come out of the FPGA """
        self._send_command("ENCLK", 0, clk_div_code | (spi_div_code << 16))
    
    def disable_clk(self):
        """ Disable the EarEEG clock to come out of the FPGA """
        self._send_command("DISCLK", 0, 0)

    def write_digtest_in(self, value):
        """ Write the chip digtest in """
        self._send_command("WRDTI", 0, value)

    def write_register(self, addr, value):
        """ Write to a register address on the chip """
        self._send_command("WRMEM", addr, value)
    
    def read_register(self, addr):
        """ Read from a register address on the chip """
        self._send_command("RDMEM", addr, 0)
        time.sleep(0.01)
        return self._send_command("RDMEM", addr, 0)

    def write_cal_lut(self, channel, addr, value):
        """ Write to a register address on the chip """
        self._send_command("WRLUT", (channel << 6) | addr, value)

    def read_cal_lut(self, channel, addr):
        """ Read from calibration LUT on chip chip """
        self._send_command("RDLUT", (channel << 6) | addr, 0)
        time.sleep(0.01)
        return self._send_command("RDLUT", (channel << 6) | addr, 0)

    def enable_freq_meas(self, count_max):
        """ Enable Frequency Measurement up to count count_max """
        self._send_command("ENFMEAS", 0, count_max)
    
    def disable_freq_meas(self):
        """ Disable Frequency Measurement """
        self._send_command("DISFMEAS", 0, 0)
    
    def read_p_freq(self):
        """ Read the result of a Frequency measurment (P side)"""
        return self._send_command("RDFMEASP", 0, 0)
        
    def read_n_freq(self):
        """ Read the result of a Frequency measurment (N side)"""
        return self._send_command("RDFMEASN", 0, 0)
    
    def enable_cal_dac(self):
        """ Enable the calibration DAC and the DAC code ramp generator"""
        return self._send_command("ENCALDAC", 0, 0)

    def write_cal_dac(self, cal_dac_code):
        """" Write a 15-bit DAC code to the calibration DAC"""
        return self._send_command("WRCALDAC", 0, cal_dac_code)
    
    def write_dac(self, addr, data, edo=False):
        assert addr < 16
        addr = (int(edo) << 4) + addr
        return self._send_command("WRDAC", addr, data)
    
    def disable_cal_dac(self):
        """ Disable the calibration DAC and the DAC code ramp generator"""
        return self._send_command("DISCALDAC", 0, 0)

    def enable_spi_stream(self, src, dac=False, delay=False, readin=False, classifier_src=0,
                          dac_count=0, dac_sclk_count=15):
        """ Enables the SPI Streaming mode, used for the on chip decimators """
        code = 0x3 if readin else 0x1
        src_code = src | classifier_src << 6
        self.xem.SetWireInValue(
            self.wr_wire_addr_command, self.opcodes["ENSTR"], 0xFFFFFFFF)
        self.xem.SetWireInValue(
            self.wr_wire_addr_addr, (dac_sclk_count << 20) | dac_count, 0xFFFFFFFF)
        self.xem.SetWireInValue(
            self.wr_wire_addr_data, (src_code << 2) | (code) | (int(dac) << 14) | (int(delay) << 15), 0xFFFFFFFF)
        self.xem.UpdateWireIns()
        self.xem.ActivateTriggerIn(self.trigger_addr_command_update, 0)

    def disable_spi_stream(self, fifo_status=0):
        """ Disables the SPI Streaming mode, used for the on chip decimators """
        out_val = self._send_command("DISTR", 0, fifo_status)
        return int(out_val & 0xFF), int(out_val >> 16)

    def read_spi_fifo_soul(self):
        """ Read SOUL output data from spi_fifo_soul """
        pipeByteOut = bytearray(self.num_blocks_spi * self.rd_block_size_spi)
        tic = time.time()
        num_read = self.xem.ReadFromBlockPipeOut(
            self.pipeout_addr_spi_soul, self.rd_block_size_spi, pipeByteOut)
        toc = time.time()
        numberOfWords = int(self.num_blocks_spi * self.rd_block_size_spi / 4) # USB 3.0 transfers 8-bit words
        wordsBinaryValues = [('{0:08b}'.format(pipeByteOut[4*wordIndex+3]) + 
                        '{0:08b}'.format(pipeByteOut[4*wordIndex+2]) +
                        '{0:08b}'.format(pipeByteOut[4*wordIndex+1]) +
                        '{0:08b}'.format(pipeByteOut[4*wordIndex+0])) 
                        for wordIndex in range(numberOfWords)]
        wordsHexValues = [hex(int(words, 2)) for words in wordsBinaryValues]        

        if num_read < 0:
            print("Failed to read from device | Error code: {0} | Elapsed time: {1}".format(num_read, toc-tic))
            return False
            #raise ValueError("Failed to read from device")
        else:
            print("Successfully read from device | Read num: {0} | Elapsed time: {1}".format(num_read, toc-tic))
            print("pipeByteOut: ", pipeByteOut)
            print("wordsHexValues: ", wordsHexValues)
            return True

    def write_input_buffer(self, addr, value):
        """ Write SOUL test data to FPGA input buffer """
        self._send_command("WRIBF", addr, value)

    def streamin_input_buffer(self, num_window):
        """ Write test data to FPGA input buffer via BTPipein interface """
        start_idx = num_window * 16 * 4 # 16 channels per window, 4 bytes per channel
        for i in range(16):
            tic = time.time()
            #print(start_idx + i * self.wr_block_size_spi, start_idx + (i + 1) * self.wr_block_size_spi)
            num_write = self.xem.WriteToBlockPipeIn(
                self.pipein_addr, self.wr_block_size_spi, self.input_data_byte_arr[start_idx + i * self.wr_block_size_spi : start_idx + (i + 1) * self.wr_block_size_spi])
            toc = time.time()
            if num_write < 0:
                print("Failed to write to device | Error code: {0} | Elapsed time: {1}".format(num_write, toc-tic))
                break
            else:
                print("Successfully write to device | Write num: {0} | Elapsed time: {1}".format(num_write, toc-tic))

    def end_to_end_stream_buffer(self, readin: bytearray, adc=False, rd_en=True, wr_en=True):
        """ Write test data to FPGA input buffer via BTPipein interface """
        block_size = self.rd_block_size_spi if not adc else self.block_size_spi
        pipeout_addr = self.pipeout_addr_spi_soul if not adc else self.pipeout_addr_spi

        self.xem.UpdateWireOuts()
        read_addr = 0x24 if adc else 0x26
        read = self.xem.GetWireOutValue(read_addr) & 0x01
        write = self.xem.GetWireOutValue(0x28) & 0x01

        if read and rd_en:
            if adc:
                buffer = bytearray(len(readin))
            else:
                buffer = bytearray(len(readin) // 16)
            num_read = self.xem.ReadFromBlockPipeOut(
                    pipeout_addr, block_size, buffer)
        else:
            buffer = None
            num_read = None

        if write and wr_en:
            num_write = self.xem.WriteToBlockPipeIn(
                self.pipein_addr, 
                self.wr_block_size_spi, 
                readin
            )
        else:
            num_write = None
            
        return num_write, num_read, buffer
        
    def read_adcf_output(self, addr, channel):
        """ Read SOUL adc feature selection data """
        self._send_command("RDADCF", addr >> 4 + channel, 0)
        time.sleep(0.01)
        return self._send_command("RDADCF", addr >> 4 + channel, 0)
        
    def read_input_buffer(self, addr, sel=0):
        """ Read SOUL test data from FPGA input buffer """
        sel_addr = (sel >> 16) + addr
        self._send_command("RDIBF", sel_addr, 0)
        time.sleep(0.01)
        return self._send_command("RDIBF", sel_addr, 0)

    def read_fifo_debug(self, addr, value):
        """ Read firmware SOUL output FIFO debug data """
        debug_info = self._send_command("RDDBG", addr, value)
        time.sleep(0.01)
        debug_info = self._send_command("RDDBG", addr, value)
        fifo_data = debug_info & 0xFFFF
        fifo_count = debug_info >> 16
        return fifo_count, fifo_data, debug_info

    def read_debug_counter(self):
        num_windows_counter = self._send_command("RDNWD", 0, 0)
        time.sleep(0.01)
        num_windows_counter = self._send_command("RDNWD", 0, 0)
        time.sleep(0.01)
        detected_counter = self._send_command("RDDTC", 0, 0)
        time.sleep(0.01)
        detected_counter = self._send_command("RDDTC", 0, 0)
        return hex(num_windows_counter), hex(detected_counter)
    
    def read_spi_fifo(self, bytearrs):
        for arr in tqdm.tqdm(bytearrs):
            num_read = self.xem.ReadFromBlockPipeOut(
                self.pipeout_addr_spi, self.block_size_spi, arr)
            if num_read < 0:
                error_info = self.xem.GetLastErrorMessage()
                print(f"Error Info: {error_info}")
                raise ValueError(f"Failed to Read; Code: {num_read}")

    def wake_up_chip(self, spi_clk_counter_max=0, clk_counter_max=0):
        self.set_reset()
        time.sleep(1)
        self.enable_clk(clk_div_code=clk_counter_max, spi_div_code=spi_clk_counter_max)
        time.sleep(1)
        self.clear_reset()
        time.sleep(1)

    def tuck_in_chip(self):
        self.set_reset()
        time.sleep(1)
        self.disable_clk()

    def print_debug_info(self):
        print(self.xem.NoError, "No Error")
        print(self.xem.Failed, "Failed")
        print(self.xem.Timeout, "Timeout")
        print(self.xem.DoneNotHigh, "DoneNotHigh")
        print(self.xem.TransferError, "TransferError")
        print(self.xem.CommunicationError, "Communication Error")
        print(self.xem.InvalidBitstream, "Invalid bitstream")
        print(self.xem.FileError, "File Error")
        print(self.xem.DeviceNotOpen)
        print(self.xem.InvalidEndpoint)
        print(self.xem.InvalidBlockSize)
        print(self.xem.I2CRestrictedAddress)
        print(self.xem.I2CBitError)
        print(self.xem.I2CNack)
        print(self.xem.I2CUnknownStatus)
        print(self.xem.UnsupportedFeature, "Unsupported Feature!")
        print(self.xem.FIFOUnderflow)
        print(self.xem.FIFOOverflow)
        print(self.xem.DataAlignmentError)
        print(self.xem.InvalidResetProfile)
        print(self.xem.InvalidParameter)
    
    def read_memory_and_dump(self, filename="register_file.txt"):
        registers = []
        for memory_address in range(64):
            self.read_register(memory_address)
            registers.append(self.read_register(memory_address))
        with open(filename, "w") as f:
            for idx, line in enumerate(registers):
                f.write(f'{idx}: ')
                f.write(f'{hex(line)}')
                f.write('\n')

