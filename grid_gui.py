#!/usr/bin/env python3
"""
Grid GUI with 4 buttons and 32 plots in 4x8 grid
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph as pg
import scipy.signal as signal

pg.setConfigOptions(antialias=True, background='w', foreground='k')

# Configuration
FFT_POINTS = 8192  # N-point FFT (user configurable)
DATA_FILE = 'adc_raw_4_0.npy'
NUM_CHANNELS = 16
SAMPLING_RATE = 1000  # Hz (adjust based on your system)


def load_and_process_data(filename, button_id, fft_points=FFT_POINTS):
    """
    Load 16-channel raw data from .npy file and compute FFT for each channel.
    
    Args:
        filename: Path to the .npy file
        button_id: ID of the button pressed (0-3) corresponding to bit mode
        fft_points: Number of FFT points (N)
        
    Returns:
        List of 32 (x, y) tuples: first 16 are time-domain, next 16 are FFT
    """
    # Load raw data
    raw_data = np.load(filename)
    
    # Data shape is (num_sweep_params, 16, samples)
    # Use the first sweep parameter (index 0)
    if len(raw_data.shape) == 3:
        # Select the first sweep parameter
        channel_data = raw_data[0]  # Shape: (16, samples)
    else:
        # If data is already 2D (16, samples), use it directly
        channel_data = raw_data
    
    plot_data = []
    
    # First 16 plots: Time-domain data for each channel
    for ch in range(NUM_CHANNELS):
        ch_signal = channel_data[ch]
        
        # Display a portion of the time-domain signal
        display_samples = min(2000, len(ch_signal))
        time_x = np.arange(display_samples) / SAMPLING_RATE  # Convert to seconds
        time_y = ch_signal[:display_samples].astype(float)
        plot_data.append((time_x, time_y))
    
    # Next 16 plots: FFT for each channel
    for ch in range(NUM_CHANNELS):
        ch_signal = channel_data[ch]
        freq, spec = signal.welch(ch_signal[-fft_points:], fs=SAMPLING_RATE,
                                  nperseg=fft_points,
                                  window='blackmanharris', scaling='density')
        plot_data.append((freq, 10*np.log10(spec)))
    
    return plot_data


class CommandThread(QThread):
    """Background thread for running command line operations."""
    finished = pyqtSignal(object)  # Emits result data when complete
    
    def __init__(self, button_id):
        super().__init__()
        self.button_id = button_id
        
    def run(self):
        """Execute the command - placeholder for now."""
        result = self._execute_command(self.button_id)
        self.finished.emit(result)
    
    def _execute_command(self, button_id):
        """
        Load data and process for the selected bit mode.
        
        Args:
            button_id: ID of the button pressed (0-3)
            
        Returns:
            Data to update plots with
        """
        # TODO: Implement actual command line execution here if needed
        import time
        time.sleep(0.5)  # Simulate processing time
        
        try:
            # Load and process the actual data
            plot_data = load_and_process_data(DATA_FILE, button_id, FFT_POINTS)
            return plot_data
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            
            # Return dummy data on error
            dummy_data = []
            for i in range(32):
                x = np.linspace(0, 10, 100)
                y = np.sin(x + button_id + i * 0.1) + np.random.normal(0, 0.1, 100)
                dummy_data.append((x, y))
            return dummy_data


class GridGUI(QMainWindow):
    """Main GUI window with buttons and plot grid."""
    
    def __init__(self):
        super().__init__()
        self.command_thread = None
        self.plot_widgets = []
        self.buttons = []
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('Grid GUI - 4 Buttons × 32 Plots')
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        button_labels = ['10 Bit Mode', '11 Bit Mode', '12 Bit Mode', '13 Bit Mode']
        for i, label in enumerate(button_labels):
            btn = QPushButton(label)
            btn.setMinimumHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    font-weight: bold;
                    background-color: #2196F3;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                    color: #757575;
                }
            """)
            btn.clicked.connect(lambda checked, btn_id=i: self.on_button_clicked(btn_id))
            self.buttons.append(btn)
            button_layout.addWidget(btn)
        
        main_layout.addLayout(button_layout)
        
        # Create plot grid (4 rows × 8 columns = 32 plots)
        plot_grid = QGridLayout()
        plot_grid.setSpacing(5)
        
        for row in range(4):
            for col in range(8):
                plot_idx = row * 8 + col
                
                # Label left 16 plots as CH0-CH15, right 16 as FFT0-FFT15
                if row < 2:
                    # Left half: CH plots
                    ch_idx = row * 8 + col
                    plot_title = f'CH{ch_idx}'
                else:
                    # Right half: FFT plots
                    fft_idx = (row-2) * 8 + col 
                    plot_title = f'FFT{fft_idx}'
                
                # Create plot widget
                plot_widget = pg.PlotWidget()
                plot_widget.setMinimumSize(150, 150)
                plot_widget.setTitle(plot_title, size='10pt')
                plot_widget.setLabel('bottom', 'X')
                plot_widget.setLabel('left', 'Y')
                plot_widget.showGrid(x=True, y=True, alpha=0.3)
                
                # Initialize with empty data
                self._initialize_plot(plot_widget)
                
                self.plot_widgets.append(plot_widget)
                plot_grid.addWidget(plot_widget, row, col)
        
        main_layout.addLayout(plot_grid)
        
    def _initialize_plot(self, plot_widget):
        """Initialize a plot with default data."""
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2))
        
    def on_button_clicked(self, button_id):
        """
        Handle button click events.
        
        Args:
            button_id: ID of the button pressed (0-3)
        """
        print(f"Button {button_id + 1} clicked")
        
        # Disable all buttons
        self._set_buttons_enabled(False)
        
        # Start command execution in background thread
        self.command_thread = CommandThread(button_id)
        self.command_thread.finished.connect(self.on_command_finished)
        self.command_thread.start()
    
    def on_command_finished(self, result_data):
        """
        Handle command completion and update plots.
        
        Args:
            result_data: Data returned from command execution
        """
        print("Command finished, updating plots...")
        
        # Update all plots with new data
        self._update_plots(result_data)
        
        # Re-enable all buttons
        self._set_buttons_enabled(True)
        
    def _set_buttons_enabled(self, enabled):
        """
        Enable or disable all buttons.
        
        Args:
            enabled: True to enable, False to disable
        """
        for btn in self.buttons:
            btn.setEnabled(enabled)
    
    def _update_plots(self, plot_data):
        """
        Update all plots with new data.
        
        Args:
            plot_data: List of (x, y) tuples for each plot
        """
        # TODO: Implement actual plot update logic here
        if plot_data is None or len(plot_data) != 32:
            print("Invalid plot data")
            return
        
        for i, (x, y) in enumerate(plot_data):
            if i < len(self.plot_widgets):
                plot_widget = self.plot_widgets[i]
                plot_widget.clear()
                plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2))


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    gui = GridGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
