import sys
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

class EEGViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Concentration & Relaxation Viewer")
        self.geometry("1100x550")

        # --- 1) Load CSV ---
        csv_file = filedialog.askopenfilename(
            title="Select EEG Filtered Data CSV",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not csv_file:
            messagebox.showerror("Error", "No file selected. Exiting.")
            self.destroy()
            sys.exit(1)

        self.data = pd.read_csv(csv_file, parse_dates=["Timestamp (Formatted)"])
        self.filtered_cols = [c for c in self.data.columns if c.endswith("_filt")]
        self.sr = 250  # sampling rate in Hz

        # --- 2) Control panel ---
        ctrl = tk.Frame(self)
        ctrl.pack(fill=tk.X, pady=5)

        # Window size dropdown (seconds)
        tk.Label(ctrl, text="Window (s):").pack(side=tk.LEFT, padx=(10,2))
        self.window_var = tk.StringVar(value="1")
        window_sizes = ["1","2","3","5","10"]
        self.window_combo = ttk.Combobox(
            ctrl, textvariable=self.window_var,
            values=window_sizes, state="readonly", width=5
        )
        self.window_combo.pack(side=tk.LEFT, padx=(0,10))

        # Apply button
        ttk.Button(ctrl, text="Apply", command=self.update_view).pack(side=tk.LEFT, padx=10)

        # --- 3) Slider for time index ---
        self.scale = tk.Scale(
            self, from_=0, to=len(self.data)-1,
            orient=tk.HORIZONTAL, length=1000,
            command=self.update_view
        )
        self.scale.pack(padx=10, pady=5)

        # Timestamp display
        self.time_label = tk.Label(self, text="", font=("Arial", 12))
        self.time_label.pack(pady=(0,10))

        # --- 4) Concentration & Relaxation big boxes ---
        box_frame = tk.Frame(self)
        box_frame.pack(fill=tk.X, pady=5)
        bold_font = tkfont.Font(family="Arial", size=16, weight="bold")

        self.conc_label = tk.Label(
            box_frame,
            text="Concentration: --",
            font=bold_font,
            relief="groove", padx=10, pady=10
        )
        self.conc_label.pack(side=tk.LEFT, padx=20)

        self.relax_label = tk.Label(
            box_frame,
            text="Relaxation: --",
            font=bold_font,
            relief="groove", padx=10, pady=10
        )
        self.relax_label.pack(side=tk.LEFT, padx=20)

        # --- 5) Table to display filtered channels ---
        cols = self.filtered_cols
        self.tree = ttk.Treeview(self, columns=cols, show='headings', height=1)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # initial update
        self.update_view()

    def update_view(self, *args):
        # get current index and window
        idx = self.scale.get()
        w_sec = int(self.window_var.get())
        half = (w_sec * self.sr) // 2
        start = max(0, idx-half)
        end = min(len(self.data), idx+half)

        # update timestamp label
        ts = self.data["Timestamp (Formatted)"].iloc[idx]
        self.time_label.config(text=f"Timestamp: {ts:%Y-%m-%d %H:%M:%S.%f}"[:-3])

        # windowed data [samples, channels]
        window = self.data[self.filtered_cols].iloc[start:end].values
        n = window.shape[0]
        if n < 2:
            messagebox.showwarning("Window too small", "Increase window size.")
            return

        # FFT per channel
        freqs = np.fft.rfftfreq(n, d=1/self.sr)
        bands = {"alpha":(8,12), "beta":(12,30)}
        alpha_powers = []
        beta_powers = []
        for ch in range(window.shape[1]):
            sig = window[:,ch]
            fft_vals = np.abs(np.fft.rfft(sig))**2
            a_idx = (freqs >= bands["alpha"][0]) & (freqs <= bands["alpha"][1])
            b_idx = (freqs >= bands["beta"][0]) & (freqs <= bands["beta"][1])
            alpha_powers.append(np.trapz(fft_vals[a_idx], freqs[a_idx]))
            beta_powers.append(np.trapz(fft_vals[b_idx], freqs[b_idx]))

        # average across channels
        mean_alpha = np.mean(alpha_powers)
        mean_beta = np.mean(beta_powers)
        total = mean_alpha + mean_beta if (mean_alpha+mean_beta)>0 else 1e-6
        relax = mean_alpha/total
        conc = mean_beta/total

        # update big boxes
        self.conc_label.config(text=f"Concentration: {conc:.4f}")
        self.relax_label.config(text=f"Relaxation: {relax:.4f}")

        # update table (show channel values at idx)
        row = [f"{self.data[col].iloc[idx]:.4f}" for col in self.filtered_cols]
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree.insert("", "end", values=row)


if __name__ == "__main__":
    # ensure dependencies
    try:
        import pandas, numpy, tkinter
    except ImportError:
        print("Install dependencies: pip install pandas numpy")
        sys.exit(1)
    EEGViewer().mainloop()
