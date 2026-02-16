#!/usr/bin/env python3
"""
Neural Recording Chip - Resolution Control App (Left Panel)
Accuracy/power tradeoff visualization and channel resolution configuration.
Standalone — no streamer connection required.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QScrollArea, QPushButton, QFrame,
    QGroupBox, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg

from train import settings_to_pow, build_dataset, run_adaptive_model

pg.setConfigOptions(antialias=True, background='w', foreground='k')

NUM_CHANNELS = 64


class DataGenerator:
    """Provides channel importance and accuracy data."""

    def __init__(self, trial_lst, label_lst, num_channels=64):
        self.trial_lst = trial_lst
        self.label_lst = label_lst
        self.num_channels = num_channels

        initial_accs, _, initial_settings, initial_weights = run_adaptive_model(
            self.trial_lst, self.label_lst,
            enabled_settings=[0, 1, 2, 3]
        )
        self.channel_importances = initial_weights[0]
        self.initial_settings = initial_settings[0]

        self.power_levels = settings_to_pow[0] * (1 + np.arange(self.num_channels))[::-1]
        self.accuracy = np.load("sparse_accuracies.npy")
        self.accuracy = np.mean(self.accuracy, axis=(0, 1))

    def get_channel_importances(self):
        return self.channel_importances

    def get_power_accuracy_curve(self):
        return self.power_levels, self.accuracy

    def get_reconfiguration_point(self, settings=None, selection=0, mode='linear'):
        acc_arr, _, settings_arr, weights_arr = run_adaptive_model(
            self.trial_lst, self.label_lst,
            enabled_settings=[0, 1, 2, 3], sim_settings=settings,
            mode=mode
        )
        settings_arr = np.array(settings_arr)
        powers = settings_to_pow[settings_arr]
        return np.array(acc_arr), np.sum(powers, axis=-1)


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
        self.label.setStyleSheet("color: #FFFFFF;")
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
        self.value_label.setStyleSheet(f"color: #E53935; font-weight: bold;")
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


class LeftApp(QMainWindow):
    def __init__(self, trial_lst, label_lst):
        super().__init__()
        self.setWindowTitle("Resolution Control — Configuration")
        self.setGeometry(100, 100, 540, 900)
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

        self.data_gen = DataGenerator(trial_lst, label_lst, num_channels=NUM_CHANNELS)
        self.num_channels = NUM_CHANNELS
        self.initial_setting = np.ones(self.num_channels, dtype=int)
        self.resolution_settings = self.initial_setting.copy()
        self.committed_settings = self.initial_setting.copy()
        self.previous = []
        self.sliders = []

        self._setup_ui()
        self._init_plots()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

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
        layout.addWidget(acc_group, stretch=2)

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
        layout.addWidget(imp_group, stretch=2)

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
            QPushButton { background: #757575; color: white; border: none;
                padding: 10px 24px; border-radius: 4px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background: #616161; }
            QPushButton:pressed { background: #424242; }
        """)
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)

        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setStyleSheet("""
            QPushButton { background: #43A047; color: white; border: none;
                padding: 10px 24px; border-radius: 4px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background: #388E3C; }
            QPushButton:pressed { background: #2E7D32; }
        """)
        self.auto_btn.clicked.connect(self._on_auto)
        btn_layout.addWidget(self.auto_btn)

        self.auto_mode = QComboBox()
        self.auto_mode.addItem("Distribution")
        self.auto_mode.addItems(["linear", "quadratic", "exponential"])
        self.auto_mode.setCurrentIndex(0)
        self.auto_mode.setStyleSheet("""
            QComboBox { padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px;
                background: white; color: #333; font-size: 12px; min-width: 90px; }
            QComboBox:hover { border-color: #43A047; }
            QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right;
                width: 20px; border-left: 1px solid #ccc; }
            QComboBox QAbstractItemView { background: white; color: #333;
                selection-background-color: #43A047; selection-color: white; border: 1px solid #ccc; }
        """)
        btn_layout.addWidget(self.auto_mode)

        self.channel_drop = QSpinBox()
        self.channel_drop.setRange(0, 64)
        self.channel_drop.setValue(0)
        self.channel_drop.setPrefix("Selection: ")
        self.channel_drop.setStyleSheet("""
            QSpinBox { padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px;
                background: white; color: #333; font-size: 12px; min-width: 90px; }
            QSpinBox:hover { border-color: #43A047; }
        """)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        btn_layout.addWidget(self.status)
        btn_layout.addStretch()
        ctrl_layout.addLayout(btn_layout)

        layout.addWidget(ctrl_group, stretch=3)

    def _init_plots(self):
        self._plot_accuracy()
        self._plot_importance()

    def _on_resolution_changed(self, channel, value):
        self.resolution_settings[channel] = value
        self._plot_importance()

    def _on_submit(self):
        self.committed_settings = self.resolution_settings.copy()
        self._plot_accuracy()
        self._flash_status("✓ Submitted")

    def _on_reset(self):
        self.resolution_settings = self.initial_setting.copy()
        self.committed_settings = self.initial_setting.copy()
        self.previous = []
        for slider in self.sliders:
            slider.setValue(self.initial_setting[slider.channel_id])
        self._plot_accuracy()
        self._flash_status("✓ Reset")

    def _on_auto(self):
        mode = self.auto_mode.currentText()
        if mode == "Distribution":
            mode = "linear"
        num_drop = self.channel_drop.value()
        # self._plot_accuracy(auto=True, selection=num_drop, mode=mode)
        self.resolution_settings = 4 - self.data_gen.initial_settings.copy()
        self.committed_settings = self.resolution_settings.copy()
        importances = self.data_gen.get_channel_importances()
        for slider in self.sliders:
            slider.setValue(self.resolution_settings[slider.channel_id])
        self._plot_importance()
        self._flash_status("✓ Auto Applied")

    def _flash_status(self, msg):
        self.status.setText(msg)
        self.status.setStyleSheet("color: #43A047; font-weight: bold;")
        QTimer.singleShot(2000, lambda: self.status.setText(""))

    def _plot_accuracy(self, auto=False, selection=0, mode='linear'):
        self.acc_plot.clear()
        power, accuracy = self.data_gen.get_power_accuracy_curve()
        rr_acc, rr_power = self.data_gen.get_reconfiguration_point(
            settings=[4 - self.committed_settings for _ in range(5)] if not auto else None,
            selection=selection, mode=mode
        )

        self.acc_plot.plot(power, accuracy, pen=pg.mkPen('#2196F3', width=2))
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(power, accuracy),
            pg.PlotDataItem(power, np.full_like(accuracy, 0.5)),
            brush=pg.mkBrush('#2196F320')
        )
        self.acc_plot.addItem(fill)
        self.acc_plot.plot(
            rr_power, rr_acc,
            pen=None, symbol='o', symbolSize=6,
            symbolBrush='#E5393580', symbolPen=None
        )
        mean_pow, mean_acc = np.mean(rr_power), np.mean(rr_acc)
        self.acc_plot.plot(
            [mean_pow], [mean_acc],
            pen=None, symbol='star', symbolSize=14,
            symbolBrush='#E53935', symbolPen=pg.mkPen('#B71C1C', width=1)
        )
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
        colors = {0: '#9E9E9E', 1: '#E53935', 2: '#FB8C00', 3: '#C0CA33', 4: '#43A047'}
        brushes = [pg.mkBrush(colors[r]) for r in self.resolution_settings]
        for i, imp in enumerate(importances):
            self.imp_plot.plot([i, i], [0, imp], pen=pg.mkPen('#333', width=1.5))
        scatter = pg.ScatterPlotItem(
            x=channels, y=importances, size=10, brush=brushes,
            pen=pg.mkPen('#333', width=0.5)
        )
        self.imp_plot.addItem(scatter)
        self.imp_plot.setXRange(-1, self.num_channels)
        self.imp_plot.setYRange(0, max(importances) * 1.15)

    def closeEvent(self, event):
        event.accept()


def main():
    import traceback
    try:
        trial_lst, label_lst = build_dataset(
            filename='Playback/emg/user1/adc_raw_{trial}_21_{setting}.npz',
            splits=4, raw=False
        )

        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(250, 250, 250))
        palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        app.setPalette(palette)
        
        breakpoint()
        window = LeftApp(trial_lst, label_lst)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
