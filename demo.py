#!/usr/bin/env python3
"""
Neural Recording Chip - Resolution Control Demo
Clean, responsive PyQt5 + pyqtgraph implementation.
Receives streaming data from streamer.py over socket.
"""

import sys
import socket
import struct
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QScrollArea, QPushButton, QFrame,
    QSplitter, QGroupBox, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg

from train import settings_to_pow, build_dataset, run_adaptive_model

# Configure pyqtgraph for clean look
pg.setConfigOptions(antialias=True, background='w', foreground='k')

# Socket settings
STREAMER_HOST = 'localhost'
STREAMER_PORT = 5555
NUM_CHANNELS = 64
CHUNK_SIZE = 33  # Samples per chunk from streamer
DISPLAY_SAMPLES = 1000  # 1 second of data to display (1kHz)


class SocketReceiver(QThread):
    """Background thread for receiving data from streamer."""
    data_received = pyqtSignal(np.ndarray)
    connected = pyqtSignal(bool)
    disconnected = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.socket = None
        self.settings_to_send = None
        self._lock = False
        
    def run(self):
        while self.running:
            try:
                self._connect_and_receive()
            except Exception as e:
                print(f"Connection error: {e}")
                self.disconnected.emit()
                if self.running:
                    QThread.msleep(1000)  # Wait before reconnecting
                    
    def _connect_and_receive(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((STREAMER_HOST, STREAMER_PORT))
        self.socket.setblocking(True)
        self.connected.emit(True)
        print("Connected to streamer")
        
        while self.running:
            # Send settings if queued
            if self.settings_to_send is not None and not self._lock:
                self._lock = True
                settings = self.settings_to_send
                self.settings_to_send = None
                try:
                    msg = b'SETT' + settings.astype(np.int32).tobytes()
                    self.socket.sendall(msg)
                except Exception as e:
                    print(f"Error sending settings: {e}")
                self._lock = False
            
            # Receive data
            try:
                header = self._recv_exact(4)
                if header == b'DATA':
                    size_data = self._recv_exact(4)
                    size = struct.unpack('I', size_data)[0]
                    data_bytes = self._recv_exact(size)
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    data = data.reshape(NUM_CHANNELS, -1)
                    self.data_received.emit(data)
            except Exception as e:
                raise e
                
    def _recv_exact(self, n):
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data
    
    def send_settings(self, settings):
        """Queue settings to send to streamer."""
        self.settings_to_send = np.array(settings, dtype=np.int32)
        
    def stop(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


class DataGenerator:
    """Provides channel importance and accuracy data."""
    
    def __init__(self, trial_lst, label_lst, num_channels=64):
        self.trial_lst = trial_lst
        self.label_lst = label_lst
        self.num_channels = num_channels
        
        self.channel_importances = np.load("sparse_weights.npy")
        self.channel_importances = np.mean(self.channel_importances, axis=(0, 1))
        
        self.power_levels = settings_to_pow[0] * (1 + np.arange(self.num_channels))[::-1]
        self.accuracy = np.load("sparse_accuracies.npy")
        self.accuracy = np.mean(self.accuracy, axis=(0, 1))

        self.reconfig_accs = None
        self.reconfig_weights = None
        self.reconfig_settings = None

    
    def get_channel_importances(self):
        return self.channel_importances
    
    def get_power_accuracy_curve(self):
        return self.power_levels, self.accuracy
    
    def get_reconfiguration_point(self, settings=None, selection=0, mode='linear'):
        acc_arr, weights_arr, settings_arr = run_adaptive_model(
            self.trial_lst, self.label_lst, 
            enabled_settings=[0, 1, 2, 3], sim_settings=settings, selection=selection, mode=mode
        )
        self.reconfig_accs = acc_arr
        self.reconfig_weights = weights_arr
        self.reconfig_settings = settings_arr
        powers = settings_to_pow[settings_arr]
        return np.array(acc_arr), np.sum(np.array(powers), axis=-1)


class ChannelSlider(QWidget):
    """Compact channel resolution slider."""
    
    def __init__(self, channel_id, init, callback):
        super().__init__()
        self.channel_id = channel_id
        self.callback = callback
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)
        
        # Channel label
        self.label = QLabel(f"{channel_id:02d}")
        self.label.setFixedWidth(22)
        self.label.setFont(QFont("Menlo", 9))
        self.label.setStyleSheet("color: #555;")
        layout.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 4)
        self.slider.setValue(init)
        self.slider.setFixedWidth(60)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #ddd;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                height: 12px;
                margin: -4px 0;
                background: #2196F3;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #1976D2;
            }
        """)
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider)
        
        # Value display
        self.value_label = QLabel(f"{init}")
        self.value_label.setFixedWidth(12)
        self.value_label.setFont(QFont("Menlo", 9, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
    def _on_change(self, value):
        self.value_label.setText(str(value))
        colors = {1: '#E53935', 2: '#FB8C00', 3: '#C0CA33', 4: '#43A047'}
        self.value_label.setStyleSheet(f"color: {colors[value]}; font-weight: bold;")
        self.callback(self.channel_id, value)
    
    def value(self):
        return self.slider.value()
    
    def setValue(self, v):
        self.slider.setValue(v)


class Demo(QMainWindow):
    def __init__(self, trial_lst, label_lst):
        super().__init__()
        self.setWindowTitle("Resolution Reconfiguration")
        self.setGeometry(100, 100, 1500, 900)
        self.setStyleSheet("""
            QMainWindow { background: #fafafa; }
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #e0e0e0; 
                border-radius: 6px; 
                margin-top: 8px; 
                padding-top: 8px;
                background: white;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 12px; 
                padding: 0 6px;
                color: #333;
            }
            QPushButton {
                background: #2196F3;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #1976D2; }
            QPushButton:pressed { background: #1565C0; }
            QLabel { color: #333; }
        """)
        
        # Data
        self.data_gen = DataGenerator(trial_lst, label_lst, num_channels=64)
        self.num_channels = 64
        self.initial_setting = np.ones(self.num_channels, dtype=int) * 1
        self.resolution_settings = self.initial_setting.copy()
        self.committed_settings = self.initial_setting.copy()
        self.voltage_buffer = np.zeros((self.num_channels, DISPLAY_SAMPLES))  # Rolling 1s buffer
        self.previous = []
        self.sliders = []
        
        self._setup_ui()
        self._init_plots()
        
        # Socket receiver for streaming data
        self.socket_receiver = SocketReceiver()
        self.socket_receiver.data_received.connect(self._on_data_received)
        self.socket_receiver.connected.connect(self._on_socket_connected)
        self.socket_receiver.disconnected.connect(self._on_socket_disconnected)
        self.socket_receiver.start()
        
        # Timer for updating traces at 30Hz (matches streamer rate)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_traces)
        self.timer.start(33)  # ~30 Hz
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        
        # Left panel
        left = QWidget()
        left.setFixedWidth(520)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Accuracy plot
        acc_group = QGroupBox("Accuracy vs Power")
        acc_layout = QVBoxLayout(acc_group)
        acc_layout.setContentsMargins(8, 12, 8, 8)
        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setLabel('left', 'Accuracy')
        self.acc_plot.setLabel('bottom', 'Power (μW)')
        self.acc_plot.showGrid(x=True, y=True, alpha=0.15)
        self.acc_plot.setLogMode(x=True, y=False)
        self.acc_plot.getAxis('left').setStyle(tickFont=QFont("Helvetica", 9))
        self.acc_plot.getAxis('bottom').setStyle(tickFont=QFont("Helvetica", 9))
        acc_layout.addWidget(self.acc_plot)
        left_layout.addWidget(acc_group, stretch=2)
        
        # Channel importance plot
        imp_group = QGroupBox("Channel Importance")
        imp_layout = QVBoxLayout(imp_group)
        imp_layout.setContentsMargins(8, 12, 8, 8)
        self.imp_plot = pg.PlotWidget()
        self.imp_plot.setLabel('left', 'Importance')
        self.imp_plot.setLabel('bottom', 'Channel')
        self.imp_plot.showGrid(x=False, y=True, alpha=0.15)
        self.imp_plot.getAxis('left').setStyle(tickFont=QFont("Helvetica", 9))
        self.imp_plot.getAxis('bottom').setStyle(tickFont=QFont("Helvetica", 9))
        imp_layout.addWidget(self.imp_plot)
        left_layout.addWidget(imp_group, stretch=2)
        
        # Resolution controls
        ctrl_group = QGroupBox("Resolution Settings")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setContentsMargins(8, 12, 8, 8)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        
        slider_container = QWidget()
        slider_container.setStyleSheet("background: transparent;")
        grid = QGridLayout(slider_container)
        grid.setSpacing(2)
        grid.setContentsMargins(0, 0, 0, 0)
        
        for i in range(self.num_channels):
            slider = ChannelSlider(i, self.resolution_settings[i], self._on_resolution_changed)
            self.sliders.append(slider)
            grid.addWidget(slider, i // 4, i % 4)
        
        scroll.setWidget(slider_container)
        ctrl_layout.addWidget(scroll)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Submit")
        self.submit_btn.clicked.connect(self._on_submit)
        btn_layout.addWidget(self.submit_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: #757575;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #616161; }
            QPushButton:pressed { background: #424242; }
        """)
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)
        
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setStyleSheet("""
            QPushButton {
                background: #43A047;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #388E3C; }
            QPushButton:pressed { background: #2E7D32; }
        """)
        self.auto_btn.clicked.connect(self._on_auto)
        btn_layout.addWidget(self.auto_btn)
        
        self.auto_mode = QComboBox()
        self.auto_mode.addItem("Distribution")  # Placeholder
        self.auto_mode.addItems(["linear", "quadratic", "exponential"])
        self.auto_mode.setCurrentIndex(0)
        self.auto_mode.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
                color: #333;
                font-size: 12px;
                min-width: 90px;
            }
            QComboBox:hover { border-color: #43A047; }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #ccc;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background: white;
                color: #333;
                selection-background-color: #43A047;
                selection-color: white;
                border: 1px solid #ccc;
            }
        """)
        btn_layout.addWidget(self.auto_mode)
        
        self.channel_drop = QSpinBox()
        self.channel_drop.setRange(0, 64)
        self.channel_drop.setValue(0)
        self.channel_drop.setPrefix("Selection: ")
        self.channel_drop.setStyleSheet("""
            QSpinBox {
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
                color: #333;
                font-size: 12px;
                min-width: 90px;
            }
            QSpinBox:hover { border-color: #43A047; }
        """)
        btn_layout.addWidget(self.channel_drop)
        
        self.status = QLabel("")
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        btn_layout.addWidget(self.status)
        btn_layout.addStretch()
        ctrl_layout.addLayout(btn_layout)
        
        left_layout.addWidget(ctrl_group, stretch=3)
        main_layout.addWidget(left)
        
        # Right panel - Voltage traces
        trace_group = QGroupBox("Voltage Traces (64 Ch)")
        trace_layout = QVBoxLayout(trace_group)
        trace_layout.setContentsMargins(8, 12, 8, 8)
        
        # Create 8x8 grid of plots
        self.trace_widget = pg.GraphicsLayoutWidget()
        self.trace_widget.setBackground('w')
        self.trace_plots = []
        self.trace_curves = []
        
        for row in range(8):
            for col in range(8):
                ch = row * 8 + col
                p = self.trace_widget.addPlot(row=row, col=col)
                p.hideAxis('left')
                p.hideAxis('bottom')
                p.setYRange(-100, 100)
                p.setXRange(0, 1)
                p.setMouseEnabled(x=False, y=False)
                
                # Channel label
                label = pg.TextItem(f"{ch}", color='#888', anchor=(0, 0))
                label.setFont(QFont("Helvetica", 7))
                label.setPos(0.02, 90)
                p.addItem(label)
                
                curve = p.plot(pen=pg.mkPen('#333', width=1))
                self.trace_plots.append(p)
                self.trace_curves.append(curve)
        
        trace_layout.addWidget(self.trace_widget)
        main_layout.addWidget(trace_group, stretch=1)
        
    def _init_plots(self):
        self._plot_accuracy()
        self._plot_importance()
        self._update_traces()
        
    def _on_resolution_changed(self, channel, value):
        self.resolution_settings[channel] = value
        self._plot_importance()
        
    def _on_submit(self):
        self.committed_settings = self.resolution_settings.copy()
        self._plot_accuracy()
        # Send settings to streamer
        self.socket_receiver.send_settings(self.committed_settings)
        self.status.setText("✓ Submitted")
        QTimer.singleShot(2000, lambda: self.status.setText(""))

    def _on_reset(self):
        self.resolution_settings = self.initial_setting.copy()
        self.committed_settings = self.initial_setting.copy()
        self.previous = []
        for slider in self.sliders:
            slider.setValue(self.initial_setting[slider.channel_id])
        self._plot_accuracy()
        # Send settings to streamer
        self.socket_receiver.send_settings(self.committed_settings)
        self.status.setText("✓ Reset")
        QTimer.singleShot(2000, lambda: self.status.setText(""))

    def _on_auto(self):
        """Auto-configure resolution settings based on channel importance."""
        mode = self.auto_mode.currentText()
        if mode == "Distribution":
            mode = "linear"  # Default if placeholder selected
        num_drop = self.channel_drop.value()
        self._plot_accuracy(auto=True, selection=num_drop, mode=mode)
        self.resolution_settings = 4-self.data_gen.reconfig_settings[0].copy()
        self.committed_settings = self.resolution_settings.copy()
        for slider in self.sliders:
            slider.setValue(self.resolution_settings[slider.channel_id])
        # Send settings to streamer
        self.socket_receiver.send_settings(self.committed_settings)
        self.status.setText("✓ Submitted")
        QTimer.singleShot(2000, lambda: self.status.setText(""))
        
    def _plot_accuracy(self, auto=False, selection=0, mode='linear'):
        self.acc_plot.clear()
        
        power, accuracy = self.data_gen.get_power_accuracy_curve()
        rr_acc, rr_power = self.data_gen.get_reconfiguration_point(
            settings=[4-self.committed_settings for _ in range(5)] if not auto else None, selection=selection, mode=mode
        )
        
        # Tradeoff curve
        self.acc_plot.plot(power, accuracy, pen=pg.mkPen('#2196F3', width=2))
        
        # Fill under curve
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(power, accuracy),
            pg.PlotDataItem(power, np.full_like(accuracy, 0.5)),
            brush=pg.mkBrush('#2196F320')
        )
        self.acc_plot.addItem(fill)
        
        # Current point scatter
        self.acc_plot.plot(
            rr_power, rr_acc, 
            pen=None, symbol='o', symbolSize=6,
            symbolBrush='#E5393580', symbolPen=None
        )
        
        # Mean current point
        mean_pow, mean_acc = np.mean(rr_power), np.mean(rr_acc)
        self.acc_plot.plot(
            [mean_pow], [mean_acc],
            pen=None, symbol='star', symbolSize=14,
            symbolBrush='#E53935', symbolPen=pg.mkPen('#B71C1C', width=1)
        )
        
        # Previous point connection
        buffer = (mean_pow, mean_acc)
        if self.previous:
            for prev_pow, prev_acc in self.previous[::-1]:
                self.acc_plot.plot(
                    [prev_pow, buffer[0]], [prev_acc, buffer[1]],
                    pen=pg.mkPen('#43A047', width=2, style=Qt.DashLine)
                )
                self.acc_plot.plot(
                    [prev_pow], [prev_acc],
                    pen=None, symbol='star', symbolSize=10,
                    symbolBrush='#43A047', symbolPen=None
                )
                buffer = (prev_pow, prev_acc)
            
        
        self.previous.append((mean_pow, mean_acc))
        
    def _plot_importance(self):
        self.imp_plot.clear()
        
        importances = self.data_gen.get_channel_importances()
        channels = np.arange(self.num_channels)
        
        colors = {1: '#E53935', 2: '#FB8C00', 3: '#C0CA33', 4: '#43A047', 5:'#A9A9A9'}
        brushes = [pg.mkBrush(colors[r]) for r in self.resolution_settings]
        
        # Black stems
        for i, imp in enumerate(importances):
            self.imp_plot.plot(
                [i, i], [0, imp], 
                pen=pg.mkPen('#333', width=1.5)
            )
        
        # Colored dots
        scatter = pg.ScatterPlotItem(
            x=channels, y=importances,
            size=10, brush=brushes,
            pen=pg.mkPen('#333', width=0.5)
        )
        self.imp_plot.addItem(scatter)
        self.imp_plot.setXRange(-1, self.num_channels)
        self.imp_plot.setYRange(0, max(importances) * 1.15)
        
    def _update_traces(self):
        """Update voltage trace plots from buffered socket data."""
        t = np.linspace(0, 1, 100)  # 100 display points over 1 second
        
        for ch in range(self.num_channels):
            # Downsample 1000 samples to 100 points for display
            data = self.voltage_buffer[ch, ::10]
            self.trace_curves[ch].setData(t, data)
    
    def _on_data_received(self, data):
        """Handle incoming voltage data from socket - append to rolling buffer."""
        num_new = data.shape[1]
        # Shift buffer left and append new data
        self.voltage_buffer = np.roll(self.voltage_buffer, -num_new, axis=1)
        self.voltage_buffer[:, -num_new:] = data
    
    def _on_socket_connected(self):
        """Handle socket connection."""
        self.status.setText("● Connected")
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        QTimer.singleShot(2000, lambda: self.status.setText(""))
    
    def _on_socket_disconnected(self):
        """Handle socket disconnection."""
        self.status.setText("● Disconnected")
        self.status.setStyleSheet("color: #E53935; font-weight: bold;")
            
    def closeEvent(self, event):
        self.timer.stop()
        self.socket_receiver.stop()
        self.socket_receiver.wait()
        event.accept()


def main():
    trial_lst, label_lst = build_dataset(
        filename='Playback/emg/user1/adc_raw_{trial}_21_{setting}.npz', 
        splits=4, raw=False
    )
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Light palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    app.setPalette(palette)
    
    demo = Demo(trial_lst, label_lst)
    demo.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
