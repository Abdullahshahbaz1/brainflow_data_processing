import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont
from collections import deque

# Try to import BrainFlow; if unavailable, disable those options
try:
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
    from brainflow.ml_model import MLModel, BrainFlowModelParams, BrainFlowClassifiers, BrainFlowMetrics
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False

class EEGViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Concentration & Relaxation Viewer")
        self.geometry("1200x900")

        # --- Load CSV ---
        csv_file = filedialog.askopenfilename(
            title="Select EEG Filtered Data CSV",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not csv_file:
            messagebox.showerror("Error", "No file selected.")
            self.destroy()
            return

        self.data = pd.read_csv(csv_file, parse_dates=["Timestamp (Formatted)"])
        self.filtered_cols = [c for c in self.data.columns if c.endswith("_filt")]
        self.sr = 250

        # Frequency bands
        self.bands = {
            "Delta": (1.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta":  (12, 30),
            "Gamma": (30, 45)
        }
        self.metric_history = deque(maxlen=5)

        # --- Control Panel ---
        ctrl = tk.Frame(self); ctrl.pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="Window (s):").pack(side=tk.LEFT, padx=(10,2))
        self.window_var = tk.StringVar(value="4")
        self.window_combo = ttk.Combobox(ctrl, textvariable=self.window_var,
                                         values=["1","2","3","4","5","10"],
                                         width=5, state="readonly")
        self.window_combo.pack(side=tk.LEFT, padx=(0,10))

        # Detrend / Bandpass checkboxes (always available)
        self.detrend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Detrend", variable=self.detrend_var).pack(side=tk.LEFT, padx=(10,2))
        self.bp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Bandpass 1.5-45Hz", variable=self.bp_var).pack(side=tk.LEFT, padx=(10,2))

        # Pipeline selection
        pipelines = ["Welch PSD", "Raw FFT"]
        if HAS_BRAINFLOW:
            pipelines.insert(0, "BrainFlow")
        tk.Label(ctrl, text="Pipeline:").pack(side=tk.LEFT, padx=(10,2))
        self.pipeline_var = tk.StringVar(value=pipelines[0])
        self.pipeline_combo = ttk.Combobox(ctrl, textvariable=self.pipeline_var,
                                           values=pipelines, width=12, state="readonly")
        self.pipeline_combo.pack(side=tk.LEFT, padx=(0,10))

        # Metric source
        metricsrc = ["Raw α/β Ratio"]
        if HAS_BRAINFLOW:
            metricsrc.insert(0, "MLModel")
        tk.Label(ctrl, text="Metric:").pack(side=tk.LEFT, padx=(10,2))
        self.metric_src = tk.StringVar(value=metricsrc[0])
        self.metric_src_combo = ttk.Combobox(ctrl, textvariable=self.metric_src,
                                             values=metricsrc, width=12, state="readonly")
        self.metric_src_combo.pack(side=tk.LEFT, padx=(0,10))

        # BrainFlow model params
        if HAS_BRAINFLOW:
            tk.Label(ctrl, text="Classifier:").pack(side=tk.LEFT, padx=(10,2))
            self.classifier_var = tk.StringVar(value=BrainFlowClassifiers.DEFAULT_CLASSIFIER.name)
            ttk.Combobox(ctrl, textvariable=self.classifier_var,
                         values=[c.name for c in BrainFlowClassifiers],
                         width=12, state="readonly").pack(side=tk.LEFT, padx=(0,10))
            tk.Label(ctrl, text="Metric Type:").pack(side=tk.LEFT, padx=(10,2))
            self.metric_type_var = tk.StringVar(value=BrainFlowMetrics.MINDFULNESS.name)
            ttk.Combobox(ctrl, textvariable=self.metric_type_var,
                         values=[m.name for m in BrainFlowMetrics],
                         width=12, state="readonly").pack(side=tk.LEFT, padx=(0,10))

        ttk.Button(ctrl, text="Apply", command=self._on_apply).pack(side=tk.LEFT, padx=10)

        # --- Slider ---
        self.scale = tk.Scale(self, from_=0, to=len(self.data)-1,
                              orient=tk.HORIZONTAL, length=1100,
                              command=self.update_view)
        self.scale.pack(padx=10, pady=5)

        # --- Timestamp ---
        self.time_label = tk.Label(self, text="", font=("Arial", 12))
        self.time_label.pack(pady=(0,10))

        # --- Smoothed Concentration & Relaxation ---
        box = tk.Frame(self); box.pack(fill=tk.X, pady=5)
        bold = tkfont.Font(size=16, weight="bold")
        self.conc_lbl  = tk.Label(box, text="Concentration: --", font=bold,
                                  relief="groove", padx=10, pady=10)
        self.relax_lbl = tk.Label(box, text="Relaxation: --",    font=bold,
                                  relief="groove", padx=10, pady=10)
        self.conc_lbl.pack(side=tk.LEFT, padx=20)
        self.relax_lbl.pack(side=tk.LEFT, padx=20)

        # --- Global Relative Band Powers ---
        band_frame = tk.Frame(self); band_frame.pack(fill=tk.X, pady=5)
        self.band_lbls = {}
        for b in self.bands:
            lbl = tk.Label(band_frame, text=f"{b}: --", font=bold,
                           relief="ridge", padx=10, pady=5)
            lbl.pack(side=tk.LEFT, padx=5)
            self.band_lbls[b] = lbl

        # --- Filtered Channel Values (single row) ---
        tree_f = tk.Frame(self); tree_f.pack(fill=tk.X, padx=10, pady=5)
        self.tree = ttk.Treeview(tree_f, columns=self.filtered_cols, show='headings', height=1)
        for c in self.filtered_cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X)

        # --- Relative Band-Power Matrix per Channel ---
        matrix_f = tk.Frame(self); matrix_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        mat_cols = ["Channel"] + list(self.bands.keys())
        self.matrix = ttk.Treeview(matrix_f, columns=mat_cols, show='headings', height=8)
        for c in mat_cols:
            self.matrix.heading(c, text=c)
            self.matrix.column(c, width=100, anchor=tk.CENTER)
        self.matrix.pack(fill=tk.BOTH, expand=True)

        # --- Metric History Table ---
        hist_f = tk.Frame(self); hist_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        hist_cols = ["Idx","RawConc","RawRelax","ModelConc","ModelRelax"]
        self.hist = ttk.Treeview(hist_f, columns=hist_cols, show='headings', height=5)
        for c in hist_cols:
            self.hist.heading(c, text=c)
            self.hist.column(c, width=100, anchor=tk.CENTER)
        self.hist.pack(fill=tk.BOTH, expand=True)

        # Build ML model (if available)
        self._build_model()
        # Initial view
        self.update_view()

    def _on_apply(self):
        if self.pipeline_var.get()=="BrainFlow" and not HAS_BRAINFLOW:
            messagebox.showerror("Error", "BrainFlow library isn’t installed.")
            return
        if self.metric_src.get()=="MLModel" and not HAS_BRAINFLOW:
            messagebox.showerror("Error", "BrainFlow library isn’t installed.")
            return
        self._build_model()
        self.update_view()

    def _build_model(self):
        """Prepare MLModel if requested."""
        if HAS_BRAINFLOW and self.metric_src.get()=="MLModel":
            mt = BrainFlowMetrics[self.metric_type_var.get()].value
            cf = BrainFlowClassifiers[self.classifier_var.get()].value
            params = BrainFlowModelParams(mt, cf)
            self.model = MLModel(params)
            self.model.prepare()
        else:
            self.model = None

    def update_view(self, *args):
        idx = self.scale.get()
        w_sec = int(self.window_var.get())
        half  = (w_sec * self.sr)//2
        start, end = max(0,idx-half), min(len(self.data), idx+half)
        ts = self.data["Timestamp (Formatted)"].iloc[idx]
        self.time_label.config(text=f"Timestamp: {ts:%Y-%m-%d %H:%M:%S.%f}"[:-3])

        # extract window
        wf = self.data[self.filtered_cols].iloc[start:end].values
        n  = wf.shape[0]
        if n < 2: return

        # preprocess each channel
        proc = wf.copy()
        for ch in range(proc.shape[1]):
            if self.detrend_var.get() and HAS_BRAINFLOW:
                DataFilter.detrend(proc[:,ch], DetrendOperations.LINEAR.value)
            if self.bp_var.get() and HAS_BRAINFLOW:
                DataFilter.perform_bandpass(proc[:,ch], self.sr,
                                           1.5, 45, 4,
                                           FilterTypes.BUTTERWORTH.value, 0)

        # compute band powers
        abs_p, rel_p = {}, {}
        freqs = np.fft.rfftfreq(n, d=1/self.sr)

        if self.pipeline_var.get()=="BrainFlow" and HAS_BRAINFLOW:
            arr = np.ascontiguousarray(proc.T)
            avg, _ = DataFilter.get_avg_band_powers(arr,
                        list(range(arr.shape[0])), self.sr, True)
            abs_p = {b: avg[i] for i,b in enumerate(self.bands)}
            total = sum(abs_p.values()) or 1e-6
            rel_p = {b: abs_p[b]/total for b in abs_p}

        else:
            # Raw FFT or Welch PSD (identical here in code; to add Welch use DataFilter.get_psd_welch)
            abs_ch = {b:[] for b in self.bands}
            for ch in range(proc.shape[1]):
                psd = np.abs(np.fft.rfft(proc[:,ch]))**2
                for b,(lo,hi) in self.bands.items():
                    mask = (freqs>=lo)&(freqs<hi)
                    abs_ch[b].append(np.trapz(psd[mask], freqs[mask]))
            abs_p = {b: np.mean(abs_ch[b]) for b in abs_ch}
            total = sum(abs_p.values()) or 1e-6
            rel_p = {b: abs_p[b]/total for b in abs_p}

        # update global band boxes
        for b,lbl in self.band_lbls.items():
            lbl.config(text=f"{b}: {rel_p[b]:.3f}")

        # filtered-chan last value
        last_vals = [f"{v:.4f}" for v in proc[-1]]
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", "end", values=last_vals)

        # per-channel rel band matrix
        self.matrix.delete(*self.matrix.get_children())
        for i,ch in enumerate(self.filtered_cols):
            psd = np.abs(np.fft.rfft(proc[:,i]))**2
            tot_ch = sum(np.trapz(psd[(freqs>=lo)&(freqs<hi)],
                                  freqs[(freqs>=lo)&(freqs<hi)])
                         for lo,hi in self.bands.values()) or 1e-6
            row = [ch] + [ f"{np.trapz(psd[(freqs>=lo)&(freqs<hi)],
                                        freqs[(freqs>=lo)&(freqs<hi)])/tot_ch:.3f}"
                           for lo,hi in self.bands.values() ]
            self.matrix.insert("", "end", values=row)

        # compute metrics
        alpha = rel_p["Alpha"]
        beta  = rel_p["Beta"]
        raw_c = beta/(alpha+beta or 1e-6)
        raw_r = alpha/(alpha+beta or 1e-6)
        mdl_c, mdl_r = (None,None)
        if self.model:
            feat = np.array([abs_p[b] for b in self.bands]).reshape(1,-1)
            mdl_c = float(self.model.predict(feat)[0])
            mdl_r = 1 - mdl_c

        # update history & smooth
        self.metric_history.append((idx, raw_c, raw_r, mdl_c, mdl_r))
        cs = [m[1] for m in self.metric_history]
        rs = [m[2] for m in self.metric_history]
        ms = [m[3] for m in self.metric_history if m[3] is not None]
        ds = [m[4] for m in self.metric_history if m[4] is not None]
        self.conc_lbl.config(text=f"Concentration: {np.mean(ms or cs):.3f}")
        self.relax_lbl.config(text=f"Relaxation: {np.mean(ds or rs):.3f}")

        # history table
        self.hist.delete(*self.hist.get_children())
        for rec in self.metric_history:
            self.hist.insert("", "end", values=[
                rec[0], f"{rec[1]:.3f}", f"{rec[2]:.3f}",
                "" if rec[3] is None else f"{rec[3]:.3f}",
                "" if rec[4] is None else f"{rec[4]:.3f}"
            ])


if __name__=="__main__":
    EEGViewer().mainloop()
