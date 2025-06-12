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
        ttk.Combobox(ctrl, textvariable=self.window_var,
                     values=["1","2","3","4","5","10"],
                     width=5, state="readonly").pack(side=tk.LEFT)

        self.detrend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Detrend", variable=self.detrend_var).pack(side=tk.LEFT, padx=10)
        self.bp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Bandpass 1.5-45Hz", variable=self.bp_var).pack(side=tk.LEFT)

        pipelines = []
        if HAS_BRAINFLOW: pipelines.append("BrainFlow")
        pipelines += ["Welch PSD", "Raw FFT"]
        tk.Label(ctrl, text="Pipeline:").pack(side=tk.LEFT, padx=10)
        self.pipeline_var = tk.StringVar(value=pipelines[0])
        ttk.Combobox(ctrl, textvariable=self.pipeline_var,
                     values=pipelines, width=12, state="readonly").pack(side=tk.LEFT)

        metricsrc = []
        if HAS_BRAINFLOW: metricsrc.append("MLModel")
        metricsrc += ["Raw α/β Ratio"]
        tk.Label(ctrl, text="Metric:").pack(side=tk.LEFT, padx=10)
        self.metric_src = tk.StringVar(value=metricsrc[-1])  # Default to Raw α/β Ratio to avoid model issues
        ttk.Combobox(ctrl, textvariable=self.metric_src,
                     values=metricsrc, width=12, state="readonly").pack(side=tk.LEFT)

        if HAS_BRAINFLOW:
            tk.Label(ctrl, text="Classifier:").pack(side=tk.LEFT, padx=10)
            self.classifier_var = tk.StringVar(value=BrainFlowClassifiers.DEFAULT_CLASSIFIER.name)
            ttk.Combobox(ctrl, textvariable=self.classifier_var,
                         values=[c.name for c in BrainFlowClassifiers],
                         width=12, state="readonly").pack(side=tk.LEFT)
            tk.Label(ctrl, text="Metric Type:").pack(side=tk.LEFT, padx=10)
            self.metric_type_var = tk.StringVar(value=BrainFlowMetrics.MINDFULNESS.name)
            ttk.Combobox(ctrl, textvariable=self.metric_type_var,
                         values=[m.name for m in BrainFlowMetrics],
                         width=12, state="readonly").pack(side=tk.LEFT)

        ttk.Button(ctrl, text="Apply", command=self._on_apply).pack(side=tk.LEFT, padx=10)

        # --- Slider ---
        self.scale = tk.Scale(self, from_=0, to=len(self.data)-1,
                              orient=tk.HORIZONTAL, length=1100, command=self.update_view)
        self.scale.pack(padx=10, pady=5)

        # --- Timestamp ---
        self.time_label = tk.Label(self, text="", font=("Arial", 12))
        self.time_label.pack(pady=(0,10))

        # --- Smoothed Concentration & Relaxation ---
        box = tk.Frame(self); box.pack(fill=tk.X, pady=5)
        bold = tkfont.Font(size=16, weight="bold")
        self.conc_lbl = tk.Label(box, text="Concentration: --", font=bold,
                                  relief="groove", padx=10, pady=10)
        self.relax_lbl = tk.Label(box, text="Relaxation: --", font=bold,
                                   relief="groove", padx=10, pady=10)
        self.conc_lbl.pack(side=tk.LEFT, padx=20)
        self.relax_lbl.pack(side=tk.LEFT, padx=20)

        # Global relative band powers
        band_frame = tk.Frame(self); band_frame.pack(fill=tk.X, pady=5)
        self.band_lbls = {}
        for b in self.bands:
            lbl = tk.Label(band_frame, text=f"{b}: --", font=bold,
                           relief="ridge", padx=10, pady=5)
            lbl.pack(side=tk.LEFT, padx=5)
            self.band_lbls[b] = lbl

        # Filtered channel values
        tree_f = tk.Frame(self); tree_f.pack(fill=tk.X, padx=10, pady=5)
        self.tree = ttk.Treeview(tree_f, columns=self.filtered_cols, show='headings', height=1)
        for c in self.filtered_cols:
            self.tree.heading(c, text=c); self.tree.column(c, width=100)
        self.tree.pack(fill=tk.X)

        # Per-channel band-power matrix
        matrix_f = tk.Frame(self); matrix_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        mat_cols = ["Channel"] + list(self.bands.keys())
        self.matrix = ttk.Treeview(matrix_f, columns=mat_cols, show='headings', height=8)
        for c in mat_cols:
            self.matrix.heading(c, text=c); self.matrix.column(c, width=100)
        self.matrix.pack(fill=tk.BOTH, expand=True)

        # Metric history table
        hist_f = tk.Frame(self); hist_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        hist_cols = ["Idx","RawConc","RawRelax","ModelConc","ModelRelax"]
        self.hist = ttk.Treeview(hist_f, columns=hist_cols, show='headings', height=5)
        for c in hist_cols:
            self.hist.heading(c, text=c); self.hist.column(c, width=100)
        self.hist.pack(fill=tk.BOTH, expand=True)

        # Initialize model as None
        self.model = None
        self._build_model()
        self.update_view()
        
        # Bind cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _on_closing(self):
        """Clean up resources when closing"""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.release()
            except:
                pass
        self.destroy()

    def _on_apply(self):
        if (self.pipeline_var.get()=="BrainFlow" or self.metric_src.get()=="MLModel") and not HAS_BRAINFLOW:
            messagebox.showerror("Error", "Install BrainFlow to use this option.")
            return
        self._build_model()
        self.update_view()

    def _build_model(self):
        """Build ML model with proper error handling"""
        # Always release existing model first
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.release()
            except:
                pass
        self.model = None
        
        if not HAS_BRAINFLOW or self.metric_src.get() != "MLModel":
            return
            
        try:
            mt = BrainFlowMetrics[self.metric_type_var.get()].value
            cf = BrainFlowClassifiers[self.classifier_var.get()].value
            params = BrainFlowModelParams(mt, cf)
            self.model = MLModel(params)
            self.model.prepare()
            print(f"Successfully prepared ML model: {self.metric_type_var.get()} with {self.classifier_var.get()}")
            
        except Exception as e:
            print(f"Failed to build model: {e}")
            self.model = None

    def _get_brainflow_feature_vector(self, data):
        """
        Create feature vector using BrainFlow's expected format.
        Based on BrainFlow documentation: feature_vector = np.concatenate((bands[0], bands[1]))
        """
        try:
            # Ensure data is contiguous and properly shaped
            if len(data.shape) != 2:
                raise ValueError(f"Data must be 2D, got shape {data.shape}")
            
            num_channels = data.shape[1]
            print(f"Data shape: {data.shape}, Channels: {num_channels}")
            
            # Transpose to channels x samples format for BrainFlow
            data_transposed = np.ascontiguousarray(data.T, dtype=np.float64)
            
            # Get channel indices
            channels = list(range(data_transposed.shape[0]))
            
            # Get band powers - this returns TWO arrays that need to be concatenated
            avg_band_powers, std_band_powers = DataFilter.get_avg_band_powers(
                data_transposed, channels, self.sr, True
            )
            
            print(f"Avg band powers shape: {avg_band_powers.shape}, Values: {avg_band_powers}")
            print(f"Std band powers shape: {std_band_powers.shape}, Values: {std_band_powers}")
            
            # THIS IS THE KEY FIX: Concatenate both arrays as shown in BrainFlow docs
            feature_vector = np.concatenate((avg_band_powers, std_band_powers))
            
            print(f"Feature vector shape: {feature_vector.shape}, Values: {feature_vector}")
            
            return feature_vector
            
        except Exception as e:
            print(f"Error creating BrainFlow feature vector: {e}")
            return None

    def update_view(self, *args):
        idx = self.scale.get()
        w_sec = int(self.window_var.get()); half = w_sec*self.sr//2
        start,end = max(0,idx-half), min(len(self.data), idx+half)
        ts = self.data["Timestamp (Formatted)"].iloc[idx]
        self.time_label.config(text=f"Timestamp: {ts:%Y-%m-%d %H:%M:%S.%f}"[:-3])

        # Window and preprocess
        wf = self.data[self.filtered_cols].iloc[start:end].values
        n = wf.shape[0]
        if n < 2: 
            return
            
        # Ensure minimum window size for reliable analysis
        if n < self.sr:  # Less than 1 second of data
            print(f"Warning: Window too small ({n} samples), results may be unreliable")
        
        proc = wf.copy().astype(np.float64)
        
        # Apply preprocessing with BrainFlow if available
        if HAS_BRAINFLOW:
            for ch in range(proc.shape[1]):
                arr = np.ascontiguousarray(proc[:,ch], dtype=np.float64)
                try:
                    if self.detrend_var.get():
                        DataFilter.detrend(arr, DetrendOperations.LINEAR.value)
                    if self.bp_var.get():
                        DataFilter.perform_bandpass(
                            arr, self.sr, 1.5, 45, 4,
                            FilterTypes.BUTTERWORTH.value, 0
                        )
                    proc[:,ch] = arr
                except Exception as e:
                    print(f"Error preprocessing channel {ch}: {e}")

        # Manual absolute & relative band powers
        freqs = np.fft.rfftfreq(n, d=1/self.sr)
        abs_ch = {b:[] for b in self.bands}
        for ch in range(proc.shape[1]):
            psd = np.abs(np.fft.rfft(proc[:,ch]))**2
            for b,(lo,hi) in self.bands.items():
                mask = (freqs>=lo)&(freqs<hi)
                if np.any(mask):
                    abs_ch[b].append(np.trapz(psd[mask], freqs[mask]))
                else:
                    abs_ch[b].append(0.0)
        
        abs_p = {b: np.mean(abs_ch[b]) for b in abs_ch}
        total = sum(abs_p.values()) or 1e-6
        rel_p = {b: abs_p[b]/total for b in abs_p}

        # Update global band labels
        for b,lbl in self.band_lbls.items():
            lbl.config(text=f"{b}: {rel_p[b]:.3f}")

        # Filtered channel single-row
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", "end", values=[f"{v:.4f}" for v in proc[-1]])

        # Per-channel matrix
        self.matrix.delete(*self.matrix.get_children())
        for i,ch in enumerate(self.filtered_cols):
            psd = np.abs(np.fft.rfft(proc[:,i]))**2
            vals = []
            for lo,hi in self.bands.values():
                m = (freqs>=lo)&(freqs<hi)
                if np.any(m):
                    vals.append(np.trapz(psd[m], freqs[m]))
                else:
                    vals.append(0.0)
            ch_tot = sum(vals) or 1e-6
            rels = [v/ch_tot for v in vals]
            row = [ch] + [f"{r:.3f}" for r in rels]
            self.matrix.insert("", "end", values=row)

        # Compute metrics
        alpha = rel_p["Alpha"]; beta = rel_p["Beta"]
        raw_c = beta/(alpha+beta+1e-6)
        raw_r = alpha/(alpha+beta+1e-6)
        mdl_c = mdl_r = None
        
        # Try ML model prediction with corrected feature vector
        if (self.model and 
            self.pipeline_var.get() == "BrainFlow" and 
            self.metric_src.get() == "MLModel"):
            
            try:
                # Get properly formatted feature vector
                feature_vector = self._get_brainflow_feature_vector(proc)
                
                if feature_vector is not None:
                    print(f"Attempting prediction with feature vector shape: {feature_vector.shape}")
                    prediction = self.model.predict(feature_vector)
                    
                    if len(prediction) > 0:
                        # BrainFlow only has MINDFULNESS and RESTFULNESS metrics
                        if self.metric_type_var.get() == "MINDFULNESS":
                            mdl_c = float(prediction[0])  # Use mindfulness as concentration
                            mdl_r = 1.0 - mdl_c  # Approximate relaxation as inverse
                        elif self.metric_type_var.get() == "RESTFULNESS":
                            mdl_r = float(prediction[0])  # Use restfulness as relaxation
                            mdl_c = 1.0 - mdl_r  # Approximate concentration as inverse
                        else:
                            # For USER_DEFINED or other metrics
                            mdl_c = float(prediction[0])
                            mdl_r = 1.0 - mdl_c
                        
                        # Clamp values to reasonable range
                        if mdl_c is not None:
                            mdl_c = max(0.0, min(1.0, mdl_c))
                        if mdl_r is not None:
                            mdl_r = max(0.0, min(1.0, mdl_r))
                        
                        print(f"Successful prediction - Concentration: {mdl_c:.3f}, Relaxation: {mdl_r:.3f}")
                    
            except Exception as e:
                print(f"ML model prediction failed: {e}")
                # Fall back to raw calculation
                pass

        # Update history & smoothing
        self.metric_history.append((idx, raw_c, raw_r, mdl_c, mdl_r))
        cs = [m[1] for m in self.metric_history]
        rs = [m[2] for m in self.metric_history]
        ms = [m[3] for m in self.metric_history if m[3] is not None]
        ds = [m[4] for m in self.metric_history if m[4] is not None]
        
        # Use model values if available, otherwise fall back to raw
        final_conc = np.mean(ms) if ms else np.mean(cs)
        final_relax = np.mean(ds) if ds else np.mean(rs)
        
        self.conc_lbl.config(text=f"Concentration: {final_conc:.3f}")
        self.relax_lbl.config(text=f"Relaxation: {final_relax:.3f}")

        # History table
        self.hist.delete(*self.hist.get_children())
        for rec in self.metric_history:
            self.hist.insert("", "end", values=[
                rec[0], f"{rec[1]:.3f}", f"{rec[2]:.3f}",
                "" if rec[3] is None else f"{rec[3]:.3f}",
                "" if rec[4] is None else f"{rec[4]:.3f}"])
        
if __name__=="__main__":
    try:
        app = EEGViewer()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()