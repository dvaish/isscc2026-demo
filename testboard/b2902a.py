"""
Interface Class to Abstract Connections to B2962A Precision Supply Unit
"""
import pyvisa

_instr_name = "USB0::2391::35864::MY51143152::0::INSTR"


def create_resource_manager():
    return pyvisa.ResourceManager('@py')


def setup_voltage(rm, compliance_voltage="3.3"):
    """
    Setup the Device to Measure Voltage
    """
    instr = rm.open_resource(_instr_name)
    print(f"Connected to: {instr.query('*IDN?')}")
    instr.write(":SOUR:FUNC:MODE CURR") # Source Current for V meas
    instr.write(":SOUR:CURR 0") # Source 0A of current
    instr.write(f":SOUR:VOLT:PROT {compliance_voltage}")
    instr.write(f":SENS:VOLT:RANG:AUTO ON") # AUTO Ranging
    return instr


def enable_voltage(instr):
    """
    Enable Voltage Measurement
    """
    instr.write(":OUTP1 ON")


def disable_voltage(instr):
    """
    Enable Voltage Measurement
    """
    instr.write(":OUTP1 OFF")


def read_voltage(instr) -> float:
    """
    Read Values from the B2925A
    """
    return float(instr.query(":MEAS:VOLT?").strip())

