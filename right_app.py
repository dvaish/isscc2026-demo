#!/usr/bin/env python3
"""
Neural Recording Chip - Live Signal Viewer (Right Panel)
Displays streaming raw voltage, MAV, and logit traces from streamer.py.
Also includes resolution controls to send settings to the streamer.
"""

import sys
import socket
import struct
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QScrollArea, QPushButton, QFrame,
    QGroupBox, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg

pg.setConfigOptions(antialias=True, background='w', foreground='k')

# Socket settings
STREAMER_HOST = 'localhost'
STREAMER_PORT = 5555
NUM_CHANNELS = 64
DISPLAY_CHANNELS = 16
NUM_CLASSES = 12
CHUNK_SIZE = 33
DISPLAY_SAMPLES = 1000   # 1 second at 1kHz
MAV_WINDOW = 50          # 50ms MAV window
MAV_SAMPLES = DISPLAY_SAMPLES // MAV_WINDOW


class SocketReceiver(QThread):
    """Background thread for receiving data from streamer and sending settings."""
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
                    QThread.msleep(1000)

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
                    data = data.reshape(DISPLAY_CHANNELS, -1)
                    self.data_received.emit(data)
            except Exception as e:
                raise e

    def _recv_exact(self, n):
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data

    def send_settings(self, settings):
        self.settings_to_send = np.array(settings, dtype=np.int32)

    def stop(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


class ChannelSlider(QWidget):
    """Compact channel resolution slider."""

    def __init__(self, channel_id, init, callback):
        super().__init__()
        self.channel_id = channel_id
        self.callback = callback

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        self.label = QLabel(f"{channel_id:02d}")
        self.label.setFixedWidth(22)
        self.label.setFont(QFont("Menlo", 9))
        self.label.setStyleSheet("color: #555;")
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 4)
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

        self.value_label = QLabel(f"{init}")
        self.value_label.setFixedWidth(12)
        self.value_label.setFont(QFont("Menlo", 9, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

    def _on_change(self, value):
        self.value_label.setText(str(value))
        colors = {0: '#9E9E9E', 1: '#E53935', 2: '#FB8C00', 3: '#C0CA33', 4: '#43A047'}
        self.value_label.setStyleSheet(f"color: {colors[value]}; font-weight: bold;")
        self.callback(self.channel_id, value)

    def value(self):
        return self.slider.value()

    def setValue(self, v):
        self.slider.setValue(v)


class RightApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resolution Reconfiguration — Live Viewer")
        self.setGeometry(660, 100, 1000, 900)
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

        self.num_channels = NUM_CHANNELS
        self.initial_setting = np.ones(self.num_channels, dtype=int)
        self.resolution_settings = self.initial_setting.copy()
        self.committed_settings = self.initial_setting.copy()

        # Signal buffers
        self.voltage_buffer = np.zeros((NUM_CHANNELS, DISPLAY_SAMPLES))
        self.mav_buffer = np.zeros((DISPLAY_CHANNELS, MAV_SAMPLES))
        self.logits_buffer = np.zeros((NUM_CLASSES, MAV_SAMPLES))
        self.mav_accumulator = np.zeros((DISPLAY_CHANNELS,))
        self.mav_count = 0

        self.sliders = []

        self._setup_ui()

        # Socket receiver
        self.socket_receiver = SocketReceiver()
        self.socket_receiver.data_received.connect(self._on_data_received)
        self.socket_receiver.connected.connect(self._on_socket_connected)
        self.socket_receiver.disconnected.connect(self._on_socket_disconnected)
        self.socket_receiver.start()

        # Refresh timer at ~30Hz
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_traces)
        self.timer.start(33)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # ── Resolution controls (mirrored from left app) ──────────────────────
        ctrl_group = QGroupBox("Resolution Settings")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setContentsMargins(8, 12, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll.setFixedHeight(180)

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
            QPushButton { background: #757575; color: white; border: none;
                padding: 10px 24px; border-radius: 4px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background: #616161; }
            QPushButton:pressed { background: #424242; }
        """)
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        btn_layout.addWidget(self.status)
        btn_layout.addStretch()
        ctrl_layout.addLayout(btn_layout)

        main_layout.addWidget(ctrl_group, stretch=0)

        # ── Raw Voltage Traces ────────────────────────────────────────────────
        trace_group = QGroupBox("Raw Voltage Traces (16 Ch)")
        trace_layout = QVBoxLayout(trace_group)
        trace_layout.setContentsMargins(8, 12, 8, 8)

        self.trace_widget = pg.GraphicsLayoutWidget()
        self.trace_widget.setBackground('w')
        self.trace_plots = []
        self.trace_curves = []

        for row in range(4):
            for col in range(4):
                ch = row * 4 + col
                p = self.trace_widget.addPlot(row=row, col=col)
                p.hideAxis('left')
                p.hideAxis('bottom')
                p.enableAutoRange(axis='y', enable=True)
                p.setXRange(0, 1)
                p.setMouseEnabled(x=False, y=False)
                label = pg.TextItem(f"{ch}", color='#888', anchor=(0, 0))
                label.setFont(QFont("Helvetica", 7))
                label.setPos(0.02, 90)
                p.addItem(label)
                curve = p.plot(pen=pg.mkPen('#333', width=1))
                self.trace_plots.append(p)
                self.trace_curves.append(curve)

        trace_layout.addWidget(self.trace_widget)
        main_layout.addWidget(trace_group, stretch=2)

        # ── MAV Traces ────────────────────────────────────────────────────────
        mav_group = QGroupBox("Mean Absolute Value (50ms windows)")
        mav_layout = QVBoxLayout(mav_group)
        mav_layout.setContentsMargins(8, 12, 8, 8)

        self.mav_widget = pg.GraphicsLayoutWidget()
        self.mav_widget.setBackground('w')
        self.mav_plots = []
        self.mav_curves = []

        for row in range(4):
            for col in range(4):
                ch = row * 4 + col
                p = self.mav_widget.addPlot(row=row, col=col)
                p.hideAxis('left')
                p.hideAxis('bottom')
                p.enableAutoRange(axis='y', enable=True)
                p.setXRange(0, 1)
                p.setMouseEnabled(x=False, y=False)
                label = pg.TextItem(f"{ch}", color='#888', anchor=(0, 0))
                label.setFont(QFont("Helvetica", 7))
                label.setPos(0.02, 45)
                p.addItem(label)
                curve = p.plot(pen=pg.mkPen('#2196F3', width=1))
                self.mav_plots.append(p)
                self.mav_curves.append(curve)

        mav_layout.addWidget(self.mav_widget)
        main_layout.addWidget(mav_group, stretch=2)

        # ── Logits Traces ─────────────────────────────────────────────────────
        logits_group = QGroupBox("Class Logits (12 Classes)")
        logits_layout = QVBoxLayout(logits_group)
        logits_layout.setContentsMargins(8, 12, 8, 8)

        self.logits_widget = pg.GraphicsLayoutWidget()
        self.logits_widget.setBackground('w')
        self.logits_plots = []
        self.logits_curves = []

        for i in range(NUM_CLASSES):
            p = self.logits_widget.addPlot(row=0, col=i)
            p.hideAxis('left')
            p.hideAxis('bottom')
            p.setYRange(-1, 1)
            p.setXRange(0, 1)
            p.setMouseEnabled(x=False, y=False)
            label = pg.TextItem(f"C{i}", color='#888', anchor=(0, 0))
            label.setFont(QFont("Helvetica", 7))
            label.setPos(0.02, 0.9)
            p.addItem(label)
            curve = p.plot(pen=pg.mkPen('#E53935', width=1))
            self.logits_plots.append(p)
            self.logits_curves.append(curve)

        logits_layout.addWidget(self.logits_widget)
        main_layout.addWidget(logits_group, stretch=1)

    # ── Resolution control handlers ───────────────────────────────────────────

    def _on_resolution_changed(self, channel, value):
        self.resolution_settings[channel] = value

    def _on_submit(self):
        self.committed_settings = self.resolution_settings.copy()
        self.socket_receiver.send_settings(self.committed_settings)
        self._flash_status("✓ Submitted")

    def _on_reset(self):
        self.resolution_settings = self.initial_setting.copy()
        self.committed_settings = self.initial_setting.copy()
        for slider in self.sliders:
            slider.setValue(self.initial_setting[slider.channel_id])
        self.socket_receiver.send_settings(self.committed_settings)
        self._flash_status("✓ Reset")

    def _flash_status(self, msg):
        self.status.setText(msg)
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        QTimer.singleShot(2000, lambda: self.status.setText(""))

    # ── Live data handlers ────────────────────────────────────────────────────

    def _on_data_received(self, data):
        num_new = data.shape[1]

        # Roll voltage buffer and insert new samples
        self.voltage_buffer = np.roll(self.voltage_buffer, -num_new, axis=1)
        self.voltage_buffer[:DISPLAY_CHANNELS, -num_new:] = data

        # Compute MAV in 50-sample windows
        for i in range(num_new):
            self.mav_accumulator += np.abs(data[:DISPLAY_CHANNELS, i])
            self.mav_count += 1

            if self.mav_count >= MAV_WINDOW:
                mav_value = self.mav_accumulator / MAV_WINDOW

                self.mav_buffer = np.roll(self.mav_buffer, -1, axis=1)
                self.mav_buffer[:, -1] = mav_value

                self.logits_buffer = np.roll(self.logits_buffer, -1, axis=1)
                self.logits_buffer[:, -1] = np.zeros(NUM_CLASSES)  # Replace with model output

                self.mav_accumulator = np.zeros((DISPLAY_CHANNELS,))
                self.mav_count = 0

    def _update_traces(self):
        t_raw = np.linspace(0, 1, DISPLAY_SAMPLES)
        t_mav = np.linspace(0, 1, MAV_SAMPLES)

        for ch in range(DISPLAY_CHANNELS):
            self.trace_curves[ch].setData(t_raw, self.voltage_buffer[ch, :])

        for ch in range(DISPLAY_CHANNELS):
            self.mav_curves[ch].setData(t_mav, self.mav_buffer[ch])

        for i in range(NUM_CLASSES):
            self.logits_curves[i].setData(t_mav, self.logits_buffer[i])

    # ── Socket status ─────────────────────────────────────────────────────────

    def _on_socket_connected(self):
        self.status.setText("● Connected")
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        QTimer.singleShot(2000, lambda: self.status.setText(""))

    def _on_socket_disconnected(self):
        self.status.setText("● Disconnected")
        self.status.setStyleSheet("color: #E53935; font-weight: bold;")

    def closeEvent(self, event):
        self.timer.stop()
        self.socket_receiver.stop()
        self.socket_receiver.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    app.setPalette(palette)

    window = RightApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
