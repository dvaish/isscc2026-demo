"""
Interface Class to Abstract Connections to Keysight 34461A 6 1/2 Digit Multimeter
"""
import pyvisa

_instr_name = "USB0::10893::5121::MY53223885::0::INSTR"


def create_resource_manager():
    return pyvisa.ResourceManager('@py')


def setup_voltage(rm, compliance_voltage="3.3", ):
    """
    Setup the Device to Measure Voltage
    """
    instr = rm.open_resource(_instr_name)
    print(f"Connected to: {instr.query('*IDN?')}")
    instr.write(f"FUNC \"VOLT:DC\"")
    instr.write("VOLT:DC:RANG:AUTO ONCE") #Autorange thresholds: Down range at <10% of range; Up range at>120%ofrange.
    instr.write("VOLT:DC:NPLC 1") # Integrate one PLC cycle
    return instr


def enable_voltage(instr):
    """
    Enable Voltage Measurement
    """
    print("Enabled voltage measurement") #Don't think enabling is required, MEAS? does it for us



def disable_voltage(instr):
    """
    Enable Voltage Measurement
    """
    print("Disabled voltage measurement") #Don't think enabling is required, MEAS? does it for us

def read_voltage(instr) -> float:
    """
    Read Values from the 34461A
    """
    return float(instr.query(":MEAS:VOLT:DC? MIN").strip())

