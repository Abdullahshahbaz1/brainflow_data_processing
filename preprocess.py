import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog

from brainflow.data_filter import (
    DataFilter,
    FilterTypes,
    DetrendOperations,
    NoiseTypes
)

# ─── 1. LOAD RAW DATA ─────────────────────────────────────────────────────

FILENAME = r"C:\Users\DELL\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2025-05-30_15-23-14\concentration1sub2.txt"
FS = 250

df = pd.read_csv(FILENAME, comment='%', skipinitialspace=True, low_memory=False)
df.columns = df.columns.str.strip()
exg_cols = [c for c in df.columns if c.startswith("EXG Channel")]
timestamp_col = 'Timestamp (Formatted)'

data_raw = df[exg_cols].values
timestamps = df[timestamp_col].astype(str).values
n_samples, n_channels = data_raw.shape

# ─── 2. THE MAIN WINDOW ───────────────────────────────────────────────────

class FilterControlGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Filter Control")

        # Central widget + layout
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        vbox = QtWidgets.QVBoxLayout(cw)

        # ─ Controls layout ────────────────────────────────────────────────────
        ctrl = QtWidgets.QGridLayout()
        vbox.addLayout(ctrl)

        # Channel selector for visualization
        ctrl.addWidget(QtWidgets.QLabel("Channel:"), 0, 0)
        self.channelBox = QtWidgets.QComboBox()
        self.channelBox.addItems(exg_cols)
        self.channelBox.currentIndexChanged.connect(self.update_plot)
        ctrl.addWidget(self.channelBox, 0, 1)

        # Detrend checkbox
        self.cb_detrend = QtWidgets.QCheckBox("Detrend")
        self.cb_detrend.setChecked(True)
        self.cb_detrend.stateChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_detrend, 1, 0)

        # Band-pass toggle + cutoffs
        self.cb_bp = QtWidgets.QCheckBox("Band-pass")
        self.cb_bp.setChecked(True)
        self.cb_bp.stateChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_bp, 2, 0)

        ctrl.addWidget(QtWidgets.QLabel("Low (Hz):"), 2, 1)
        self.sb_bp_low = QtWidgets.QDoubleSpinBox()
        self.sb_bp_low.setRange(0.1, 120.0)
        self.sb_bp_low.setSingleStep(0.5)
        self.sb_bp_low.setValue(5.0)
        self.sb_bp_low.valueChanged.connect(self.update_plot)
        ctrl.addWidget(self.sb_bp_low, 2, 2)

        ctrl.addWidget(QtWidgets.QLabel("High (Hz):"), 2, 3)
        self.sb_bp_high = QtWidgets.QDoubleSpinBox()
        self.sb_bp_high.setRange(1.0, 250.0)
        self.sb_bp_high.setSingleStep(1.0)
        self.sb_bp_high.setValue(50.0)
        self.sb_bp_high.valueChanged.connect(self.update_plot)
        ctrl.addWidget(self.sb_bp_high, 2, 4)

        # Notch toggles
        self.cb_notch50 = QtWidgets.QCheckBox("Notch 50 Hz")
        self.cb_notch50.setChecked(True)
        self.cb_notch50.stateChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_notch50, 3, 0)

        self.cb_notch60 = QtWidgets.QCheckBox("Notch 60 Hz")
        self.cb_notch60.setChecked(False)
        self.cb_notch60.stateChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_notch60, 3, 1)

        # Filter order
        ctrl.addWidget(QtWidgets.QLabel("Filter order:"), 4, 0)
        self.sb_order = QtWidgets.QSpinBox()
        self.sb_order.setRange(1, 10)
        self.sb_order.setValue(4)
        self.sb_order.valueChanged.connect(self.update_plot)
        ctrl.addWidget(self.sb_order, 4, 1)

        # Filter type dropdown
        ctrl.addWidget(QtWidgets.QLabel("Filter type:"), 5, 0)
        self.cb_type = QtWidgets.QComboBox()
        for f in FilterTypes:
            self.cb_type.addItem(f.name, f.value)
        self.cb_type.setCurrentIndex(self.cb_type.findData(FilterTypes.BUTTERWORTH_ZERO_PHASE.value))
        self.cb_type.currentIndexChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_type, 5, 1, 1, 2)

        # Mains-noise removal
        self.cb_rm50 = QtWidgets.QCheckBox("Remove 50 Hz")
        self.cb_rm50.setChecked(True)
        self.cb_rm50.stateChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_rm50, 6, 0)

        self.cb_rm60 = QtWidgets.QCheckBox("Remove 60 Hz")
        self.cb_rm60.setChecked(True)
        self.cb_rm60.stateChanged.connect(self.update_plot)
        ctrl.addWidget(self.cb_rm60, 6, 1)

        # ─ SAVE BUTTON ─────────────────────────────────────────────────────────
        self.save_button = QtWidgets.QPushButton("Save All Channels")
        self.save_button.clicked.connect(self.save_all_channels)
        ctrl.addWidget(self.save_button, 7, 0, 1, 2)

        # ─ Plot ────────────────────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        vbox.addWidget(self.plot)
        self.raw_curve = self.plot.plot(pen=pg.mkPen('gray'), name='Raw')
        self.filt_curve = self.plot.plot(pen=pg.mkPen('teal'), name='Filtered')
        self.plot.getAxis('bottom').setLabel('Time')
        self.plot.getAxis('left').setLabel('µV')
        self.plot.getViewBox().enableAutoRange('xy')

        # initial plotting
        self.update_plot()

    def apply_filter(self, arr):
        data = arr.copy()
        if self.cb_detrend.isChecked():
            DataFilter.detrend(data, DetrendOperations.LINEAR.value)
        if self.cb_bp.isChecked():
            lo, hi = self.sb_bp_low.value(), self.sb_bp_high.value()
            DataFilter.perform_bandpass(data, FS, lo, hi,
                                        self.sb_order.value(),
                                        self.cb_type.currentData(),
                                        0)
        if self.cb_notch50.isChecked():
            DataFilter.perform_bandstop(data, FS, 49.0, 51.0,
                                        self.sb_order.value(),
                                        self.cb_type.currentData(),
                                        0)
        if self.cb_notch60.isChecked():
            DataFilter.perform_bandstop(data, FS, 59.0, 61.0,
                                        self.sb_order.value(),
                                        self.cb_type.currentData(),
                                        0)
        if self.cb_rm50.isChecked():
            DataFilter.remove_environmental_noise(data, FS, NoiseTypes.FIFTY.value)
        if self.cb_rm60.isChecked():
            DataFilter.remove_environmental_noise(data, FS, NoiseTypes.SIXTY.value)
        return data - np.mean(data)

    def update_plot(self):
        ch = self.channelBox.currentIndex()
        raw = data_raw[:, ch].astype(float)
        raw_z = raw - np.mean(raw)
        filt_z = self.apply_filter(raw)
        # use timestamp strings on x-axis
        x = list(timestamps)
        self.raw_curve.setData(x=range(len(x)), y=raw_z)
        self.filt_curve.setData(x=range(len(x)), y=filt_z)

    def save_all_channels(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save All Filtered Data', '', 'CSV Files (*.csv)')
        if not path:
            return
        raw_mat = data_raw.astype(float)
        filt_mat = np.zeros_like(raw_mat)
        for idx in range(n_channels):
            filt_mat[:, idx] = self.apply_filter(raw_mat[:, idx])
        # build dataframe including original timestamp
        df_save = pd.DataFrame(
            data=np.hstack([timestamps.reshape(-1,1), raw_mat, filt_mat]),
            columns=[timestamp_col] + exg_cols + [f + '_filt' for f in exg_cols]
        )
        df_save.to_csv(path, index=False)
        QtWidgets.QMessageBox.information(self, 'Saved', f'Data with timestamps saved to:\n{path}')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = FilterControlGUI()
    w.show()
    sys.exit(app.exec_())
