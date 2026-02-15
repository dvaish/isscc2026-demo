"""
Interface Class to Abstract Connections to Keysight Trueform 33500B Waveform Generator
"""
import pyvisa
import time

_instr_name = "USB0::2391::9735::MY57200377::INSTR" # for the 33500B
_instr_name_2 = "USB0::0x0957::0x5707::MY53802931::0::INSTR" # for the 33600A

def create_resource_manager():
    return pyvisa.ResourceManager()

def setup_voltage(rm, _instr_name, vmin: float = 0, vmax: float = 0):
    """
    Setup the Device to Generate Voltage
    """
    instr = rm.open_resource(_instr_name)
    
    instr.read_termination = '\n'
    instr.write_termination = '\n'
    print(f"Connected to: {instr.query('*IDN?')}")

    # Set voltage low and high limits
    instr.write(f"SOUR:VOLT:LIM:LOW {vmin}")
    instr.write(f"SOUR:VOLT:LIM:HIGH {vmax}")
    return instr


def drive_DC_voltage(instr, vin_dc, channel: int = 1):
    """
    Generate a DC voltage. "vin_dc" in units of mV.
    """
    # Set function type to be DC
    instr.write(f"SOUR{channel}:FUNC DC")

    # Set output DC voltage to "vin_dc"
    instr.write(f"SOUR{channel}:VOLT:OFFS +{vin_dc} mV")


def drive_sine_wave(instr, half_amp_low: str, half_amp_high: str, freq: str):
    """
    Generate a Sine Wave, differential by default
    """
    instr.write(f"SOUR2:TRACK INV")
    instr.write(f"SOUR1:FUNC SIN")
    instr.write(f"FREQ {freq}")
    instr.write(f"VOLT:HIGH {half_amp_high}")
    instr.write(f"VOLT:LOW {half_amp_low}")


def enable_voltage(instr, channel: int = 1):
    """
    Enable Voltage Generation
    """
    instr.write(f"OUTP{channel} ON")


def disable_voltage(instr, channel: int = 1):
    """
    Disable Voltage Generation
    """
    instr.write(f"OUTP{channel} OFF")

def set_hiZ(instr, channel: int = 1):
    """
    Set the expected output termination to high impedance.
    """
    instr.write(f"OUTP{channel}:LOAD INF")


def test_both_sources():
    rm = create_resource_manager()

    instr = setup_voltage(rm, _instr_name, vmax = 1)
    instr2 = setup_voltage(rm, _instr_name_2, vmax = 1)

    set_hiZ(instr, 1)
    drive_DC_voltage(instr, 146, 1)
    enable_voltage(instr, 1)
    disable_voltage(instr, 1)

    set_hiZ(instr, 2)
    drive_DC_voltage(instr, 146, 2)
    enable_voltage(instr, 2)
    disable_voltage(instr, 2)

    set_hiZ(instr2, 1)
    drive_DC_voltage(instr2, 156, 1)
    enable_voltage(instr2, 1)
    disable_voltage(instr2, 1)

    set_hiZ(instr2, 2)
    drive_DC_voltage(instr2, 156, 2)
    enable_voltage(instr2, 2)
    disable_voltage(instr2, 2)


def test_sine_wave():
    rm = create_resource_manager()
    breakpoint()
    instr = setup_voltage(rm, _instr_name, vmax = 1)
    drive_sine_wave(instr, "-10.0E-3", "+10.0E-3", 6)
    enable_voltage(instr, 1)
    enable_voltage(instr, 2)
    time.sleep(20)
    disable_voltage(instr, 1)
    disable_voltage(instr, 2)



if __name__ == '__main__':
    # test_both_sources()
    test_sine_wave()

