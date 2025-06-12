import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QSlider, QLabel, QSpinBox, QHBoxLayout, QVBoxLayout, QGridLayout, QCheckBox, QComboBox, QPushButton

# Constants
FS = 250  # sample rate (Hz)

class FFTBandpowerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG FFT & Bandpower Explorer")

        # Main layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Load CSV button
        load_btn = QPushButton("Load CSV File")
        load_btn.clicked.connect(self.load_csv)
        layout.addWidget(load_btn)

        # Controls
        ctrl = QGridLayout()
        layout.addLayout(ctrl)

        # Channel selector
        ctrl.addWidget(QLabel("Channel:"), 0, 0)
        self.channelBox = QComboBox()
        self.channelBox.currentIndexChanged.connect(self.update_plots)
        ctrl.addWidget(self.channelBox, 0, 1)

        # All channels
        self.allChk = QCheckBox("All Channels")
        self.allChk.stateChanged.connect(self.update_plots)
        ctrl.addWidget(self.allChk, 0, 2)

        # Start time slider (samples)
        ctrl.addWidget(QLabel("Start Time (s):"), 1, 0)
        self.sl_start = QSlider(QtCore.Qt.Horizontal)
        self.sl_start.setMinimum(0)
        self.sl_start.setSingleStep(1)            # 1 sample (~4 ms)
        self.sl_start.setPageStep(FS)              # 1 second
        self.sl_start.valueChanged.connect(self.on_slider_move)
        ctrl.addWidget(self.sl_start, 1, 1)
        self.lbl_time = QLabel("00:00:00.000")
        ctrl.addWidget(self.lbl_time, 1, 2)

        # Window length (seconds)
        ctrl.addWidget(QLabel("Window (s):"), 2, 0)
        self.sb_window = QSpinBox()
        self.sb_window.setRange(1, 30)
        self.sb_window.setValue(5)
        self.sb_window.valueChanged.connect(self.on_window_change)
        ctrl.addWidget(self.sb_window, 2, 1)

        # Plot area
        plots = QHBoxLayout()
        layout.addLayout(plots)

        self.fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_plot.setLabel('left', 'Amplitude')
        self.fft_plot.showGrid(x=True, y=True)
        plots.addWidget(self.fft_plot)

        self.bp_plot = pg.PlotWidget(title="Band Powers")
        self.bp_plot.setLabel('bottom', 'Band')
        self.bp_plot.setLabel('left', 'Power (uV^2/Hz)')
        self.bp_plot.showGrid(x=True, y=True)
        plots.addWidget(self.bp_plot)

        # Data placeholders
        self.timestamps = None
        self.raw_df = None
        self.filt_df = None
        self.sample_count = 0

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open CSV File', '', 'CSV Files (*.csv)')
        if not path:
            return
        df = pd.read_csv(path)
        # Parse timestamps and build numeric index
        self.timestamps = pd.to_datetime(df['Timestamp (Formatted)'])
        self.sample_count = len(self.timestamps)

        # Identify columns
        raw_cols = [c for c in df.columns if not c.endswith('_filt') and c != 'Timestamp (Formatted)']
        filt_cols = [c for c in df.columns if c.endswith('_filt')]
        self.raw_df = df[raw_cols]
        self.filt_df = df[filt_cols]

        # Populate controls
        self.channelBox.clear()
        self.channelBox.addItems(raw_cols)
        self.allChk.setChecked(False)

        # Configure slider range in samples
        window_samps = self.sb_window.value() * FS
        max_start = max(0, self.sample_count - window_samps)
        self.sl_start.setMaximum(max_start)
        self.sl_start.setValue(0)
        self.on_slider_move(0)

    def on_slider_move(self, sample_idx):
        # Update timestamp label and plots
        idx = min(sample_idx, self.sample_count - 1)
        if self.timestamps is not None:
            ts = self.timestamps.iloc[idx]
            self.lbl_time.setText(ts.strftime('%H:%M:%S.%f')[:-3])
        self.update_plots()

    def on_window_change(self, window_s):
        # Adjust slider max when window changes
        if self.sample_count:
            window_samps = window_s * FS
            self.sl_start.setMaximum(max(0, self.sample_count - window_samps))
        self.update_plots()

    def update_plots(self):
        if self.raw_df is None:
            return
        # Clear previous plots
        self.fft_plot.clear()
        self.bp_plot.clear()

        # Window indices
        start_idx = self.sl_start.value()
        win_samps = self.sb_window.value() * FS
        end_idx = min(start_idx + win_samps, self.sample_count)

        # Channel selection
        chans = list(self.raw_df.columns) if self.allChk.isChecked() else [self.channelBox.currentText()]

        # Band definitions
        bands = [(1,4),(4,8),(8,13),(13,30),(30,55)]
        names = ['Delta','Theta','Alpha','Beta','Gamma']
        bp_vals = np.zeros(len(bands))

        # Compute FFT and bandpower
        for i, ch in enumerate(chans):
            data = self.filt_df[f"{ch}_filt"].values[start_idx:end_idx]
            if len(data) < 2:
                continue
            freqs = np.fft.rfftfreq(len(data), 1/FS)
            spec = np.abs(np.fft.rfft(data)) / len(data)
            pen = pg.mkPen(color=pg.intColor(i, hues=len(chans)), width=2)
            self.fft_plot.plot(freqs, spec, pen=pen, name=ch)
            for j, (lo, hi) in enumerate(bands):
                mask = (freqs >= lo) & (freqs < hi)
                if np.any(mask):
                    bp_vals[j] += np.trapz(spec[mask]**2, freqs[mask])

        # Average across channels
        if self.allChk.isChecked() and chans:
            bp_vals /= len(chans)

                # Plot band-power bars individually
        x = np.arange(len(bands))
        for j, height in enumerate(bp_vals):
            brush = pg.mkBrush(pg.intColor(j, hues=len(bands)))
            pen = pg.mkPen(color=brush.color(), width=1)
            bar = pg.BarGraphItem(x=[j], height=[height], width=0.6, brush=brush, pen=pen)
            self.bp_plot.addItem(bar)
        self.bp_plot.getAxis('bottom').setTicks([list(zip(x, names))])

        # Auto-range bar plot so values are visible
        self.bp_plot.enableAutoRange('y')

        # Auto-range FFT plot
        self.fft_plot.enableAutoRange()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = FFTBandpowerGUI()
    gui.show()
    sys.exit(app.exec_())
