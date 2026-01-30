#!/usr/bin/env python3
"""
Neural Recording Chip - Resolution Control Demo
Demonstrates how channel resolution settings affect classifier performance.
Optimized for performance with efficient matplotlib updates.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from train import settings_to_pow, build_dataset, run_adaptive_model


class DummyDataGenerator:
    """Generate synthetic neural recording data for demonstration."""
    
    def __init__(self, trial_lst, label_lst, num_channels=64, sampling_rate=30000):
        self.trial_lst = trial_lst
        self.label_lst = label_lst
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.time_step = 0
        
        # Generate channel importances (between 0 and 1)
        np.random.seed(42)  # For reproducibility
        self.channel_importances = np.random.beta(2, 5, num_channels)
        
        # Power-accuracy tradeoff curve (simulated)
        self.power_levels = settings_to_pow[0] * (1+np.arange(self.num_channels))[::-1]  # uW
        self.accuracy = np.load("sparse.npy") # 4 x 5 x 64
        self.accuracy = np.mean(self.accuracy, axis=(0, 1)) # 64
        
    def get_voltage_data(self, duration_sec=1):
        """Generate 1 second of voltage data for all 64 channels at 1kHz (downsampled)."""
        # Generate at 1kHz for display (not full 30kHz)
        num_samples = int(duration_sec * 1000)  # 1kHz rate
        
        # Generate realistic neural-like data
        data = np.zeros((self.num_channels, num_samples))
        
        t = np.arange(num_samples) / 1000.0  # Time in seconds
        
        for ch in range(self.num_channels):
            # Add some frequency content (like neural oscillations)
            freq1 = 10 + ch * 2  # Different frequency per channel
            freq2 = 50 + ch * 3
            
            signal = (self.channel_importances[ch] * 
                     (50 * np.sin(2 * np.pi * freq1 * t) + 
                      30 * np.sin(2 * np.pi * freq2 * t)))
            
            noise = np.random.normal(0, 10, num_samples)
            data[ch] = signal + noise
        
        self.time_step += 1
        return data
    
    def get_channel_importances(self):
        """Get the importance score for each channel."""
        return self.channel_importances
    
    def get_power_accuracy_curve(self):
        """Get power consumption vs classifier accuracy."""
        return self.power_levels, self.accuracy
    
    def get_reconfiguration_point(self, settings=None):
        """Get a simulated reconfiguration point (not used in demo)."""
        acc_arr, weights_arr, settings_arr = run_adaptive_model(self.trial_lst, self.label_lst, enabled_settings=[0, 1, 2, 3], sim_weights=settings)
        powers = settings_to_pow[settings_arr]
        acc_arr = np.array(acc_arr)
        powers = np.array(powers)
        return acc_arr, np.sum(powers, axis=-1)


class Demo:
    def __init__(self, root, trial_lst, label_lst):
        self.root = root
        self.root.title('Neural Recording Chip - Resolution Control Demo')
        self.root.geometry('1600x1000')
        
        # Data generator
        self.data_gen = DummyDataGenerator(trial_lst, label_lst, num_channels=64)
        self.num_channels = 64
        
        # Resolution settings: current (being edited) and committed
        self.resolution_settings = np.ones(self.num_channels, dtype=int) * 2  # Current slider values
        self.committed_settings = np.ones(self.num_channels, dtype=int) * 2   # Submitted values
        
        # Data buffer for streaming (1 second at 1kHz)
        self.voltage_buffer = self.data_gen.get_voltage_data(1)
        
        # Flag for running
        self.is_running = True
        
        # Slider references
        self.resolution_sliders = []
        
        # Plot line references for efficient updates
        self.trace_lines = []
        self.trace_axes = None

        # Store previous accuracy
        self.previous = (None, None)
        
        # Create UI
        self.create_ui()
        
        # Initialize plots
        self.init_voltage_traces()
        self.plot_accuracy_vs_power()
        self.plot_channel_importances()
        
        # Start streaming updates at ~60 Hz (every 16ms)
        self.schedule_update()
        
    def create_ui(self):
        """Create the main UI with 4 quadrants."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel
        left_frame = ttk.Frame(main_frame, width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_frame.pack_propagate(False)
        
        # Upper left: Accuracy vs Power
        upper_left = ttk.LabelFrame(left_frame, text="Accuracy vs Power (Updates on Submit)")
        upper_left.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_accuracy = Figure(figsize=(5, 3), dpi=100)
        self.canvas_accuracy = FigureCanvasTkAgg(self.fig_accuracy, upper_left)
        self.canvas_accuracy.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Lower left: Channel importances
        lower_left = ttk.LabelFrame(left_frame, text="Channel Importances & Resolution")
        lower_left.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_channels = Figure(figsize=(5, 2.5), dpi=100)
        self.canvas_channels = FigureCanvasTkAgg(self.fig_channels, lower_left)
        self.canvas_channels.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Resolution control area with scrollbar
        control_frame = ttk.LabelFrame(left_frame, text="Channel Resolution Settings (1=Low, 4=High)")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbar for sliders
        canvas_container = ttk.Frame(control_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.slider_canvas = tk.Canvas(canvas_container, yscrollcommand=scrollbar.set, height=150)
        self.slider_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.slider_canvas.yview)
        
        slider_frame = ttk.Frame(self.slider_canvas)
        self.slider_canvas.create_window((0, 0), window=slider_frame, anchor='nw')
        
        # Create sliders in a grid (4 per row)
        for i in range(self.num_channels):
            row = i // 4
            col = i % 4
            
            cell_frame = ttk.Frame(slider_frame)
            cell_frame.grid(row=row, column=col, padx=2, pady=1, sticky='ew')
            
            label = ttk.Label(cell_frame, text=f"Ch{i:02d}", width=5)
            label.pack(side=tk.LEFT)
            
            value_label = ttk.Label(cell_frame, text="2", width=2)
            
            slider = ttk.Scale(cell_frame, from_=1, to=4, orient=tk.HORIZONTAL, length=80)
            
            # Store reference BEFORE setting value
            self.resolution_sliders.append((slider, value_label))
            
            # Configure command and set value
            slider.configure(command=lambda val, ch=i: self.on_resolution_changed(ch, val))
            slider.set(2)
            slider.pack(side=tk.LEFT, padx=2)
            
            value_label.pack(side=tk.LEFT)
        
        slider_frame.update_idletasks()
        self.slider_canvas.config(scrollregion=self.slider_canvas.bbox("all"))
        
        # Submit button
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.submit_button = ttk.Button(
            button_frame, 
            text="Submit Channel Assignments", 
            command=self.on_submit_settings
        )
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(button_frame, text="Settings not yet submitted", foreground='gray')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Right panel: Voltage traces
        right_frame = ttk.LabelFrame(main_frame, text="Raw Voltage Traces (64 Channels) - 60Hz Update")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_traces = Figure(figsize=(8, 8), dpi=100)
        self.canvas_traces = FigureCanvasTkAgg(self.fig_traces, right_frame)
        self.canvas_traces.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def on_resolution_changed(self, channel, value):
        """Handle resolution slider changes - only updates stem plot."""
        int_value = int(round(float(value)))
        self.resolution_settings[channel] = int_value
        
        # Update the value label
        if channel < len(self.resolution_sliders):
            slider, label = self.resolution_sliders[channel]
            label.config(text=str(int_value))
        
        # Only redraw the stem plot (not accuracy plot)
        self.plot_channel_importances()
        
    def on_submit_settings(self):
        """Handle submit button - commits settings and updates accuracy plot."""
        # Copy current settings to committed settings
        self.committed_settings = self.resolution_settings.copy()
        
        # Update the accuracy plot with new settings
        self.plot_accuracy_vs_power()
        
        # Update status
        self.status_label.config(text="Settings submitted!", foreground='green')
        self.root.after(2000, lambda: self.status_label.config(text="", foreground='gray'))
        
    def plot_accuracy_vs_power(self):
        """Plot accuracy vs power tradeoff - only called on submit."""
        self.fig_accuracy.clear()
        ax = self.fig_accuracy.add_subplot(111)
        
        power, accuracy = self.data_gen.get_power_accuracy_curve()
        rr_acc, rr_power = self.data_gen.get_reconfiguration_point(settings=[self.committed_settings for i in range(5)])

        # Calculate current operating point based on COMMITTED settings
        avg_resolution = np.mean(self.committed_settings)
        current_power = 50 + avg_resolution * 200  # Simplified power model
        current_accuracy = 0.5 + 0.45 * (1 - np.exp(-current_power / 200))
        
        ax.plot(power, accuracy, 'b-', linewidth=2, label='Tradeoff Curve')
        ax.fill_between(power, 0.5, accuracy, alpha=0.2)
        ax.scatter(rr_power, rr_acc, c='red', s=10, zorder=5, alpha=0.5)
        ax.scatter(np.mean(rr_power), np.mean(rr_acc), c='red', s=10, zorder=5, 
                   label=f'Current: {current_accuracy:.2%}', marker='*')
        if (self.previous != (None, None)):
            prev_power, prev_acc = self.previous
            ax.plot([prev_power, np.mean(rr_power)], [prev_acc, np.mean(rr_acc)], 'k--', linewidth=1, marker='*', color='green',
                    label='Change')
        ax.set_xlabel('Power (uW)')
        ax.set_ylabel('Classifier Accuracy')
        ax.set_title('Accuracy vs Power Tradeoff')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.semilogx()
        # ax.set_ylim([0.5, 1.0])
        # ax.set_xlim([0, 1100])
        self.previous = (np.mean(rr_power), np.mean(rr_acc))
        
        self.fig_accuracy.tight_layout()
        self.canvas_accuracy.draw()
        
    def plot_channel_importances(self):
        """Plot stem plot of channel importances with resolution overlay."""
        self.fig_channels.clear()
        ax = self.fig_channels.add_subplot(111)
        
        importances = self.data_gen.get_channel_importances()
        channels = np.arange(self.num_channels)
        
        # Color map for resolution settings (1=red, 2=orange, 3=yellow-green, 4=green)
        color_map = {1: '#d62728', 2: '#ff7f0e', 3: '#bcbd22', 4: '#2ca02c'}
        colors = [color_map[r] for r in self.resolution_settings]
        
        # Draw BLACK stem lines
        ax.vlines(channels, 0, importances, colors='black', linewidth=1.5)
        
        # Add colored markers (bulbs) on top based on resolution setting
        ax.scatter(channels, importances, c=colors, s=50, zorder=5, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Channel #')
        ax.set_ylabel('Importance')
        ax.set_title('Channel Importances (Bulb Color = Resolution: Red=1, Orange=2, Yellow-Green=3, Green=4)')
        ax.set_xlim([-1, self.num_channels])
        ax.set_ylim([0, max(importances) * 1.15])
        ax.grid(True, alpha=0.3, axis='y')
        
        self.fig_channels.tight_layout()
        self.canvas_channels.draw()
        
    def init_voltage_traces(self):
        """Initialize voltage trace subplots with line objects for efficient updates."""
        self.fig_traces.clear()
        
        # Create 64 subplots in an 8x8 grid
        self.trace_axes = self.fig_traces.subplots(8, 8)
        
        # Time vector (1 second at 1kHz, downsampled to ~60 points for display)
        self.display_samples = 60  # ~60 points for smooth display
        t = np.linspace(0, 1, self.display_samples)
        
        self.trace_lines = []
        
        for ch in range(self.num_channels):
            row = ch // 8
            col = ch % 8
            ax = self.trace_axes[row, col]
            
            # Create line object (will update data later)
            line, = ax.plot(t, np.zeros(self.display_samples), 'k-', linewidth=0.5)
            self.trace_lines.append(line)
            
            ax.set_ylim([-100, 100])
            ax.set_xlim([0, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add channel label
            ax.text(0.02, 0.95, f'{ch}', transform=ax.transAxes, 
                   fontsize=6, verticalalignment='top', fontweight='bold')
        
        self.fig_traces.suptitle('Streaming Neural Data (1kHz sampled, 60Hz display)', fontsize=10)
        self.fig_traces.tight_layout(rect=[0, 0, 1, 0.97])
        self.canvas_traces.draw()
        
    def update_voltage_traces(self):
        """Efficiently update voltage traces using blitting-like approach."""
        # Downsample 1000 samples to display_samples for plotting
        downsample_factor = self.voltage_buffer.shape[1] // self.display_samples
        
        for ch in range(self.num_channels):
            # Downsample data
            data = self.voltage_buffer[ch, ::downsample_factor][:self.display_samples]
            self.trace_lines[ch].set_ydata(data)
        
        # Redraw only the canvas (more efficient than full redraw)
        self.canvas_traces.draw_idle()
        
    def update_data(self):
        """Update streaming data at ~60Hz."""
        if not self.is_running:
            return
            
        # Generate new voltage data (1 second buffer at 1kHz)
        self.voltage_buffer = self.data_gen.get_voltage_data(1)
        
        # Update voltage traces efficiently
        self.update_voltage_traces()
        
        # Schedule next update
        self.schedule_update()
    
    def schedule_update(self):
        """Schedule the next data update at ~60Hz."""
        if self.is_running:
            self.root.after(16, self.update_data)  # ~60 Hz (1000ms / 60 ≈ 16ms)


def main():
    splits = 4
    trial_lst, label_lst = build_dataset(filename='Playback/emg/user1/adc_raw_{trial}_21_{setting}.npz', splits=4, raw=False)

    root = tk.Tk()
    
    # Set a nice style
    style = ttk.Style()
    style.theme_use('clam')
    
    demo = Demo(root, trial_lst, label_lst)
    
    def on_closing():
        demo.is_running = False
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
