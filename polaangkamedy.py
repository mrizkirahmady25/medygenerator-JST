# ==============================================================================
# Nama  : Muhammad Rizki Rahmady
# NIM   : 03041182429009
# Tugas : Tugas 13 - Pengenalan Pola Angka 0 - 9 (BPNN + Nguyen-Widrow + Momentum)
# ==============================================================================

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as md
import matplotlib.pyplot as medy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==============================================================================
# 1. PREPROCESSING OPENCV & GENERATOR DATASET (28x28)
# ==============================================================================
def pusatkan_citra_cv2(grid_biner):
    img_u8 = (grid_biner * 255).astype(md.uint8)
    coords = cv2.findNonZero(img_u8)
    if coords is None:
        return grid_biner
    x, y, w, h = cv2.boundingRect(coords)
    crop = img_u8[y : y + h, x : x + w]

    skala = 20.0 / max(w, h)
    nw, nh = max(1, int(w * skala)), max(1, int(h * skala))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    kanvas = md.zeros((28, 28), dtype=md.uint8)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    kanvas[oy : oy + nh, ox : ox + nw] = resized
    _, biner = cv2.threshold(kanvas, 80, 1, cv2.THRESH_BINARY)
    return biner.astype(md.float64)


def buat_dataset_cv2(jumlah_per_angka=60, seed=42):
    rng = md.random.default_rng(seed)
    X_list, y_list = [], []
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    for digit in range(10):
        for _ in range(jumlah_per_angka):
            kanvas = md.zeros((28, 28), dtype=md.uint8)
            font = fonts[rng.integers(0, len(fonts))]
            skala = rng.uniform(0.70, 0.92)
            tebal = int(rng.integers(1, 3))

            teks = str(digit)
            (w, h), _ = cv2.getTextSize(teks, font, skala, tebal)
            ox = int((28 - w) // 2 + rng.integers(-2, 3))
            oy = int((28 + h) // 2 + rng.integers(-2, 3))
            cv2.putText(kanvas, teks, (ox, oy), font, skala, 255, tebal, cv2.LINE_AA)

            sudut = rng.uniform(-12, 12)
            M_rot = cv2.getRotationMatrix2D((14, 14), sudut, 1.0)
            kanvas = cv2.warpAffine(kanvas, M_rot, (28, 28), flags=cv2.INTER_LINEAR)

            _, biner = cv2.threshold(kanvas, 90, 1, cv2.THRESH_BINARY)
            biner_pusat = pusatkan_citra_cv2(biner.astype(md.float64))
            X_list.append(biner_pusat.flatten())

            target = md.zeros(10, dtype=md.float64)
            target[digit] = 1.0
            y_list.append(target)

    X, Y = md.array(X_list), md.array(y_list)

    # Pembagian Dataset 80% Latih, 10% Validasi, 10% Uji
    idx_tr, idx_va, idx_te = [], [], []
    for digit in range(10):
        idx_k = md.where(md.argmax(Y, axis=1) == digit)[0]
        rng.shuffle(idx_k)
        n_tr = int(0.8 * len(idx_k))
        n_va = int(0.1 * len(idx_k))
        idx_tr.extend(idx_k[:n_tr])
        idx_va.extend(idx_k[n_tr : n_tr + n_va])
        idx_te.extend(idx_k[n_tr + n_va :])

    rng.shuffle(idx_tr)
    rng.shuffle(idx_va)
    rng.shuffle(idx_te)
    return X[idx_tr], Y[idx_tr], X[idx_va], Y[idx_va], X[idx_te], Y[idx_te]


# ==============================================================================
# 2. ARSITEKTUR BPNN (784 - 16 - 16 - 10) + NGUYEN-WIDROW + MOMENTUM + L2
# ==============================================================================
class BPNNDigitRecognizer:
    def __init__(self, layers=(784, 16, 16, 10), seed=42):
        self.layers = layers
        self.rng = md.random.default_rng(seed)
        self.weights, self.biases = [], []
        self.vel_w, self.vel_b = [], []
        self.inisialisasi_nguyen_widrow()

    def inisialisasi_nguyen_widrow(self):
        self.weights.clear()
        self.biases.clear()
        self.vel_w.clear()
        self.vel_b.clear()

        for i in range(len(self.layers) - 1):
            n_in, n_out = self.layers[i], self.layers[i + 1]
            if i < len(self.layers) - 2:
                beta = 0.7 * (n_out ** (1.0 / n_in))
                w_rand = self.rng.uniform(-0.5, 0.5, size=(n_in, n_out))
                norm = md.linalg.norm(w_rand, axis=0, keepdims=True) + 1e-12
                W = beta * (w_rand / norm)
                b = self.rng.uniform(-beta, beta, size=(1, n_out))
            else:
                W = self.rng.uniform(-0.5, 0.5, size=(n_in, n_out))
                b = self.rng.uniform(-0.5, 0.5, size=(1, n_out))

            self.weights.append(W)
            self.biases.append(b)
            self.vel_w.append(md.zeros_like(W))
            self.vel_b.append(md.zeros_like(b))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + md.exp(-md.clip(x, -30, 30)))

    @staticmethod
    def sigmoid_turunan(y):
        return y * (1.0 - y)

    def forward(self, X):
        aktivasi = [X]
        a = X
        for W, b in zip(self.weights, self.biases):
            a = self.sigmoid(a @ W + b)
            aktivasi.append(a)
        return aktivasi

    def train_step(self, X, Y, alpha=0.03, momentum=0.7, weight_decay=1e-4):
        indeks = self.rng.permutation(len(X))
        for idx in indeks:
            x_i = X[idx : idx + 1]
            y_i = Y[idx : idx + 1]

            aktivasi = self.forward(x_i)
            error = y_i - aktivasi[-1]

            deltas = [error * self.sigmoid_turunan(aktivasi[-1])]
            for l in range(len(self.weights) - 1, 0, -1):
                delta_h = (deltas[0] @ self.weights[l].T) * self.sigmoid_turunan(aktivasi[l])
                deltas.insert(0, delta_h)

            for l in range(len(self.weights)):
                grad_w = aktivasi[l].T @ deltas[l] - weight_decay * self.weights[l]
                self.vel_w[l] = alpha * grad_w + momentum * self.vel_w[l]
                self.vel_b[l] = alpha * deltas[l] + momentum * self.vel_b[l]
                self.weights[l] += self.vel_w[l]
                self.biases[l] += self.vel_b[l]

    def evaluasi(self, X, Y):
        out = self.forward(X)[-1]
        mse = float(md.mean((Y - out) ** 2))
        pred = md.argmax(out, axis=1)
        aktual = md.argmax(Y, axis=1)
        akurasi = float(md.mean(pred == aktual))
        return mse, akurasi, pred, aktual

    def hitung_confusion_matrix(self, X, Y):
        _, _, pred, aktual = self.evaluasi(X, Y)
        cm = md.zeros((10, 10), dtype=int)
        for a, p in zip(aktual, pred):
            cm[a, p] += 1
        return cm

    def simpan_bobot(self, filepath):
        data = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            data[f"W{i}"] = W
            data[f"b{i}"] = b
        md.savez(filepath, **data)

    def muat_bobot(self, filepath):
        data = md.load(filepath)
        for i in range(len(self.weights)):
            self.weights[i] = data[f"W{i}"]
            self.biases[i] = data[f"b{i}"]


# ==============================================================================
# 3. DASHBOARD GUI MULTI-TAB & KANVAS GAMBAR HALUS
# ==============================================================================
class BPNNProDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("BPNN Studio Pro (784-16-16-10) - Muhammad Rizki Rahmady (03041182429009)")
        self.root.geometry("1440x860")
        self.root.configure(bg="#121212")

        self.model = BPNNDigitRecognizer(layers=(784, 16, 16, 10))
        self.grid_data = md.zeros((28, 28), dtype=md.float64)
        self.grid_prev = md.full((28, 28), -1.0, dtype=md.float64)
        self.sel_ids = [[None] * 28 for _ in range(28)]
        self.last_pos = None

        self.sedang_training = False
        self.stop_flag = False
        self.mode_gambar = True

        self.hist_epoch, self.hist_mse_tr, self.hist_mse_va = [], [], []
        self.hist_acc_tr, self.hist_acc_va = [], []

        self._atur_style_ttk()
        self._bangun_header_kontrol()
        self._bangun_notebook_tabs()
        self._bangun_footer_log()

        self.regenerasi_dataset()
        self._inisialisasi_grid_kanvas()
        self._update_tampilan_kanvas()
        self._gambar_arsitektur_jaringan()

    # --- TEMA & HEADER KONTROL ---
    def _atur_style_ttk(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#121212", borderwidth=0)
        style.configure("TNotebook.Tab", background="#242424", foreground="#dddddd", padding=[14, 6], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#8b1e24")], foreground=[("selected", "#ffffff")])
        style.configure("TProgressbar", troughcolor="#2a2a2a", background="#c92a2a")

    def _bangun_header_kontrol(self):
        bar = tk.Frame(self.root, bg="#1d1d1d", padx=10, pady=8, highlightbackground="#333333", highlightthickness=1)
        bar.pack(side=tk.TOP, fill=tk.X)

        def tambah_input(label_txt, default_val, width=6):
            tk.Label(bar, text=label_txt, bg="#1d1d1d", fg="#cccccc", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 2))
            var = tk.StringVar(value=str(default_val))
            ent = tk.Entry(bar, textvariable=var, width=width, bg="#2c2c2c", fg="#ffffff", insertbackground="#ffffff", relief=tk.FLAT)
            ent.pack(side=tk.LEFT, padx=(0, 4))
            return var

        self.var_alpha = tambah_input("Alpha (α):", 0.03, 5)
        self.var_momentum = tambah_input("Momentum (μ):", 0.7, 5)
        self.var_target_mse = tambah_input("Target MSE:", 0.005, 6)
        self.var_max_epoch = tambah_input("Max Epoch:", 150, 5)
        self.var_n_pola = tambah_input("Pola/Angka:", 60, 4)

        btn_cfg = {"font": ("Segoe UI", 9, "bold"), "relief": tk.FLAT, "padx": 10, "pady": 3}
        self.btn_latih = tk.Button(bar, text="▶ Latih", bg="#9b1c1c", fg="#ffffff", command=self.mulai_training, **btn_cfg)
        self.btn_latih.pack(side=tk.LEFT, padx=(12, 4))

        self.btn_stop = tk.Button(bar, text="■ Stop", bg="#444444", fg="#ffffff", state=tk.DISABLED, command=self.hentikan_training, **btn_cfg)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        tk.Button(bar, text="↻ Regenerasi Data", bg="#2c2c2c", fg="#eeeeee", command=self.regenerasi_dataset, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="💾 Simpan Bobot", bg="#2c2c2c", fg="#eeeeee", command=self.simpan_bobot_dialog, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="📂 Muat Bobot", bg="#2c2c2c", fg="#eeeeee", command=self.muat_bobot_dialog, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="📊 Ekspor Grafik (PNG)", bg="#1e4d2b", fg="#ffffff", command=self.ekspor_grafik_paper, **btn_cfg).pack(side=tk.RIGHT, padx=4)

    # --- TATA LETAK 3 TAB ---
    def _bangun_notebook_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.tab_studio = tk.Frame(self.notebook, bg="#161616")
        self.tab_eval = tk.Frame(self.notebook, bg="#161616")
        self.tab_data = tk.Frame(self.notebook, bg="#161616")

        self.notebook.add(self.tab_studio, text="  1. Studio Training & Prediksi  ")
        self.notebook.add(self.tab_eval, text="  2. Evaluasi & Analisis Overfitting  ")
        self.notebook.add(self.tab_data, text="  3. Galeri Dataset (28x28)  ")

        self._isi_tab_studio()
        self._isi_tab_evaluasi()
        self._isi_tab_dataset()

    def _isi_tab_studio(self):
        kiri = tk.Frame(self.tab_studio, bg="#1e1e1e", width=330, padx=14, pady=12)
        kiri.pack(side=tk.LEFT, fill=tk.Y)
        kiri.pack_propagate(False)

        tk.Label(kiri, text="KANVAS INPUT BINER (28x28)", bg="#1e1e1e", fg="#ff6b6b", font=("Segoe UI", 10, "bold")).pack(pady=(0, 6))

        self.ukuran_sel = 10
        self.canvas_digit = tk.Canvas(
            kiri, width=280, height=280, bg="#222222", highlightthickness=1, highlightbackground="#555555", cursor="crosshair"
        )
        self.canvas_digit.pack()
        self.canvas_digit.bind("<Button-1>", self._mulai_gambar)
        self.canvas_digit.bind("<B1-Motion>", self._gerak_gambar)
        self.canvas_digit.bind("<ButtonRelease-1>", self._lepas_gambar)

        ctrl = tk.Frame(kiri, bg="#1e1e1e", pady=8)
        ctrl.pack(fill=tk.X)
        tk.Button(ctrl, text="Prediksi", bg="#9b1c1c", fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.prediksi_kanvas).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_mode = tk.Button(ctrl, text="Mode: Gambar", bg="#333333", fg="#ffffff", relief=tk.FLAT, command=self.ganti_mode)
        self.btn_mode.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(ctrl, text="Bersihkan", bg="#333333", fg="#ffffff", relief=tk.FLAT, command=self.bersihkan_kanvas).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ctrl2 = tk.Frame(kiri, bg="#1e1e1e", pady=2)
        ctrl2.pack(fill=tk.X)
        tk.Button(ctrl2, text="Muat Citra CV2", bg="#2c3e50", fg="#ffffff", relief=tk.FLAT, command=self.muat_citra_eksternal).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(ctrl2, text="🎲 Uji Sampel Acak", bg="#3b4d61", fg="#ffffff", relief=tk.FLAT, command=self.uji_sampel_acak).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        kuas_fr = tk.Frame(kiri, bg="#1e1e1e", pady=4)
        kuas_fr.pack(fill=tk.X)
        tk.Label(kuas_fr, text="Kuas:", bg="#1e1e1e", fg="#cccccc", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.var_kuas = tk.IntVar(value=1)
        tk.Spinbox(kuas_fr, from_=1, to=3, width=3, textvariable=self.var_kuas, bg="#2c2c2c", fg="#ffffff").pack(side=tk.LEFT, padx=4)

        self.var_autocenter = tk.BooleanVar(value=True)
        tk.Checkbutton(
            kuas_fr, text="Auto-Center CV2", variable=self.var_autocenter,
            bg="#1e1e1e", fg="#cccccc", selectcolor="#2c2c2c", activebackground="#1e1e1e", font=("Segoe UI", 8)
        ).pack(side=tk.RIGHT)

        self.lbl_hasil_utama = tk.Label(kiri, text="Hasil: -", font=("Segoe UI", 15, "bold"), bg="#1e1e1e", fg="#ffffff")
        self.lbl_hasil_utama.pack(pady=(6, 2))

        medy.style.use("dark_background")
        self.fig_prob, self.ax_prob = medy.subplots(figsize=(3.0, 1.8), dpi=90)
        self.fig_prob.patch.set_facecolor("#1e1e1e")
        self.ax_prob.set_facecolor("#1e1e1e")
        self.canvas_prob = FigureCanvasTkAgg(self.fig_prob, master=kiri)
        self.canvas_prob.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._update_grafik_prob(md.zeros(10))

        kanan = tk.Frame(self.tab_studio, bg="#161616")
        kanan.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_net = tk.Canvas(kanan, bg="#161616", height=320, highlightthickness=0)
        self.canvas_net.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))
        self.canvas_net.bind("<Configure>", lambda e: self._gambar_arsitektur_jaringan())

        self.lbl_status_net = tk.Label(
            kanan, text="Arsitektur: 784 - 16 - 16 - 10 | Aktivasi: Sigmoid Biner", bg="#161616", fg="#cccccc", font=("Segoe UI", 9, "bold")
        )
        self.lbl_status_net.pack(side=tk.TOP, pady=(0, 4))

        self.fig_train, (self.ax_mse, self.ax_acc) = medy.subplots(1, 2, figsize=(9.5, 2.9), dpi=95)
        self.fig_train.patch.set_facecolor("#161616")
        for ax in (self.ax_mse, self.ax_acc):
            ax.set_facecolor("#161616")
            ax.tick_params(labelsize=8)
        self.ax_mse.set_title("MSE vs Epoch", fontsize=9)
        self.ax_acc.set_title("Akurasi vs Epoch (%)", fontsize=9)
        self.fig_train.tight_layout(pad=1.6)

        self.canvas_plot = FigureCanvasTkAgg(self.fig_train, master=kanan)
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

    def _isi_tab_evaluasi(self):
        kiri_ev = tk.Frame(self.tab_eval, bg="#161616")
        kiri_ev.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_cm, self.ax_cm = medy.subplots(figsize=(6.0, 5.2), dpi=95)
        self.fig_cm.patch.set_facecolor("#161616")
        self.ax_cm.set_facecolor("#161616")
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=kiri_ev)
        self.canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        kanan_ev = tk.Frame(self.tab_eval, bg="#1e1e1e", width=420, padx=16, pady=16)
        kanan_ev.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        kanan_ev.pack_propagate(False)

        tk.Label(kanan_ev, text="DIAGNOSIS & ANALISIS OVERFITTING", bg="#1e1e1e", fg="#ff6b6b", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))
        self.txt_analisis = tk.Text(kanan_ev, bg="#141414", fg="#e0e0e0", font=("Consolas", 9), relief=tk.FLAT, wrap=tk.WORD)
        self.txt_analisis.pack(fill=tk.BOTH, expand=True)
        self.txt_analisis.insert(tk.END, "Klik '▶ Latih' terlebih dahulu untuk melihat Confusion Matrix, akurasi per digit (0-9), dan analisis Overfitting otomatis.")
        self.txt_analisis.config(state=tk.DISABLED)

    def _isi_tab_dataset(self):
        self.lbl_info_data = tk.Label(self.tab_data, text="", bg="#161616", fg="#ffffff", font=("Segoe UI", 10, "bold"), pady=8)
        self.lbl_info_data.pack(side=tk.TOP)

        self.fig_gal, self.axes_gal = medy.subplots(2, 10, figsize=(12, 3.6), dpi=95)
        self.fig_gal.patch.set_facecolor("#161616")
        self.canvas_gal = FigureCanvasTkAgg(self.fig_gal, master=self.tab_data)
        self.canvas_gal.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    def _bangun_footer_log(self):
        prog_fr = tk.Frame(self.root, bg="#1d1d1d", padx=10, pady=3)
        prog_fr.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status_bar = tk.Label(prog_fr, text="Siap. Klik '▶ Latih' sebelum melakukan prediksi.", bg="#1d1d1d", fg="#dddddd", font=("Segoe UI", 9))
        self.lbl_status_bar.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(prog_fr, mode="determinate", length=260)
        self.progress.pack(side=tk.RIGHT)

        self.txt_log = tk.Text(self.root, height=5, bg="#0e0e0e", fg="#a3e635", font=("Consolas", 9), state=tk.DISABLED, relief=tk.FLAT)
        self.txt_log.pack(side=tk.BOTTOM, fill=tk.X)

    # --- REGENERASI DATASET & GALERI ---
    def regenerasi_dataset(self):
        try:
            n_pola = max(50, int(self.var_n_pola.get()))
        except ValueError:
            n_pola = 60
            self.var_n_pola.set("60")

        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = buat_dataset_cv2(n_pola)
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        info = f"Total Dataset: {total} citra biner (28x28) | Training 80%: {len(self.X_train)} | Validasi 10%: {len(self.X_val)} | Testing 10%: {len(self.X_test)}"
        self.lbl_info_data.config(text=info)
        self._tulis_log(info)

        labels = md.argmax(self.Y_train, axis=1)
        for d in range(10):
            idx_d = md.where(labels == d)[0]
            for row in range(2):
                ax = self.axes_gal[row, d]
                ax.clear()
                ax.imshow(self.X_train[idx_d[row]].reshape(28, 28), cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"Angka {d}", fontsize=9, color="#ffffff")
        self.fig_gal.tight_layout(pad=0.6)
        self.canvas_gal.draw_idle()

    # --- KANVAS GAMBAR HALUS (CV2.LINE + ITEMCONFIG) ---
    def _inisialisasi_grid_kanvas(self):
        s = self.ukuran_sel
        for r in range(28):
            for c in range(28):
                self.sel_ids[r][c] = self.canvas_digit.create_rectangle(
                    c * s, r * s, (c + 1) * s, (r + 1) * s, fill="#1f1f1f", outline="#2a2a2a"
                )

    def _update_tampilan_kanvas(self):
        for r in range(28):
            for c in range(28):
                val = self.grid_data[r, c]
                if val != self.grid_prev[r, c]:
                    warna = "#f8f9fa" if val > 0.5 else "#1f1f1f"
                    self.canvas_digit.itemconfig(self.sel_ids[r][c], fill=warna)
                    self.grid_prev[r, c] = val

    def _mulai_gambar(self, event):
        c = int(md.clip(event.x // self.ukuran_sel, 0, 27))
        r = int(md.clip(event.y // self.ukuran_sel, 0, 27))
        self.last_pos = (c, r)
        self._gores_garis_halus((c, r), (c, r))

    def _gerak_gambar(self, event):
        c = int(md.clip(event.x // self.ukuran_sel, 0, 27))
        r = int(md.clip(event.y // self.ukuran_sel, 0, 27))
        if self.last_pos is None:
            self.last_pos = (c, r)
        self._gores_garis_halus(self.last_pos, (c, r))
        self.last_pos = (c, r)

    def _lepas_gambar(self, event):
        self.last_pos = None

    def _gores_garis_halus(self, pt1, pt2):
        img_u8 = (self.grid_data * 255).astype(md.uint8)
        warna_garis = 255 if self.mode_gambar else 0
        tebal = max(1, int(self.var_kuas.get()))
        cv2.line(img_u8, pt1, pt2, warna_garis, thickness=tebal, lineType=cv2.LINE_8)
        self.grid_data = (img_u8 > 127).astype(md.float64)
        self._update_tampilan_kanvas()

    def ganti_mode(self):
        self.mode_gambar = not self.mode_gambar
        self.btn_mode.config(text="Mode: Gambar" if self.mode_gambar else "Mode: Hapus")

    def bersihkan_kanvas(self):
        self.grid_data.fill(0.0)
        self._update_tampilan_kanvas()
        self.lbl_hasil_utama.config(text="Hasil: -")
        self._update_grafik_prob(md.zeros(10))
        self._gambar_arsitektur_jaringan()

    def uji_sampel_acak(self):
        idx = int(md.random.randint(0, len(self.X_test)))
        label_asli = int(md.argmax(self.Y_test[idx]))
        self.grid_data = self.X_test[idx].reshape(28, 28).copy()
        self._update_tampilan_kanvas()
        self.prediksi_kanvas(label_asli=label_asli, pakai_autocenter=False)

    def muat_citra_eksternal(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Gagal membuka file citra dengan OpenCV.")
            return
        if md.mean(img) > 127:
            img = 255 - img
        _, biner = cv2.threshold(img, 80, 1, cv2.THRESH_BINARY)
        self.grid_data = pusatkan_citra_cv2(biner.astype(md.float64))
        self._update_tampilan_kanvas()
        self.prediksi_kanvas(pakai_autocenter=False)

    # --- VISUALISASI JARINGAN & PROBABILITAS ---
    def _update_grafik_prob(self, prob):
        self.ax_prob.clear()
        warna = ["#ff4d4f" if i == md.argmax(prob) and md.max(prob) > 0 else "#4a5568" for i in range(10)]
        self.ax_prob.bar(range(10), prob, color=warna)
        self.ax_prob.set_xticks(range(10))
        self.ax_prob.set_ylim(0, 1.05)
        self.ax_prob.tick_params(labelsize=7)
        self.ax_prob.set_title("Probabilitas Output (0-9)", fontsize=8)
        self.fig_prob.tight_layout(pad=0.6)
        self.canvas_prob.draw_idle()

    def _gambar_arsitektur_jaringan(self, aktivasi=None):
        self.canvas_net.delete("all")
        W_can = max(self.canvas_net.winfo_width(), 820)
        H_can = max(self.canvas_net.winfo_height(), 300)

        x_pos = [W_can * 0.12, W_can * 0.40, W_can * 0.68, W_can * 0.92]
        judul = ["Input (784)", "Hidden 1 (16)", "Hidden 2 (16)", "Output (10)"]
        n_tampil = [17, 16, 16, 10]

        for x, jdl in zip(x_pos, judul):
            self.canvas_net.create_text(x, 16, text=jdl, fill="#dddddd", font=("Segoe UI", 8, "bold"))

        koord = []
        for idx, n in enumerate(n_tampil):
            ys = md.linspace(38, H_can - 18, n)
            koord.append([(x_pos[idx], float(y)) for y in ys])

        for l_idx, (src, dst) in enumerate([(1, 2), (2, 3)]):
            W_mat = self.model.weights[l_idx + 1]
            w_max = md.max(md.abs(W_mat)) + 1e-6
            for i, (x1, y1) in enumerate(koord[src]):
                for j, (x2, y2) in enumerate(koord[dst]):
                    wv = W_mat[i, j]
                    if abs(wv) < 0.18 * w_max:
                        continue
                    warna = "#ff6b3d" if wv > 0 else "#4ea8de"
                    self.canvas_net.create_line(x1, y1, x2, y2, fill=warna, width=1 if abs(wv) < 0.55 * w_max else 2)

        r = 6
        for l_idx, pts in enumerate(koord):
            for n_idx, (x, y) in enumerate(pts):
                if l_idx == 0 and n_idx == 8:
                    self.canvas_net.create_text(x, y, text="⋮", fill="#ffffff", font=("Segoe UI", 12, "bold"))
                    continue
                isi = "#161616"
                if aktivasi is not None:
                    val = aktivasi[0][0, int((n_idx / 17.0) * 783)] if l_idx == 0 else aktivasi[l_idx][0, n_idx]
                    g = int(md.clip(val * 255, 22, 255))
                    isi = f"#{g:02x}{g:02x}{g:02x}"
                self.canvas_net.create_oval(x - r, y - r, x + r, y + r, fill=isi, outline="#dddddd", width=1.2)
                if l_idx == 3:
                    self.canvas_net.create_text(x + 16, y, text=str(n_idx), fill="#dddddd", font=("Segoe UI", 8, "bold"))

    # --- PROSES TRAINING & EVALUASI ---
    def mulai_training(self):
        if self.sedang_training:
            return
        try:
            alpha = float(self.var_alpha.get())
            momentum = float(self.var_momentum.get())
            target_mse = float(self.var_target_mse.get())
            max_epoch = int(self.var_max_epoch.get())
        except ValueError:
            messagebox.showerror("Input Tidak Valid", "Pastikan Alpha, Momentum, Target MSE, dan Max Epoch berupa angka.")
            return

        self.sedang_training = True
        self.stop_flag = False
        self.btn_latih.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress["maximum"] = max_epoch
        self.progress["value"] = 0

        self.hist_epoch.clear()
        self.hist_mse_tr.clear()
        self.hist_mse_va.clear()
        self.hist_acc_tr.clear()
        self.hist_acc_va.clear()

        self.model.inisialisasi_nguyen_widrow()
        threading.Thread(target=self._loop_training, args=(alpha, momentum, target_mse, max_epoch), daemon=True).start()

    def hentikan_training(self):
        self.stop_flag = True

    def _loop_training(self, alpha, momentum, target_mse, max_epoch):
        alasan = "Max Epoch tercapai"
        for ep in range(1, max_epoch + 1):
            if self.stop_flag:
                alasan = "Dihentikan manual"
                break

            self.model.train_step(self.X_train, self.Y_train, alpha=alpha, momentum=momentum, weight_decay=1e-4)
            mse_tr, acc_tr, _, _ = self.model.evaluasi(self.X_train, self.Y_train)
            mse_va, acc_va, _, _ = self.model.evaluasi(self.X_val, self.Y_val)

            self.hist_epoch.append(ep)
            self.hist_mse_tr.append(mse_tr)
            self.hist_mse_va.append(mse_va)
            self.hist_acc_tr.append(acc_tr)
            self.hist_acc_va.append(acc_va)

            self.root.after(0, self._update_epoch_ui, ep, mse_tr, mse_va, acc_tr, acc_va, target_mse)

            if mse_tr <= target_mse:
                alasan = f"Target MSE <= {target_mse} tercapai"
                break

        self.root.after(0, self._selesai_training, alasan, alpha, momentum)

    def _update_epoch_ui(self, ep, mse_tr, mse_va, acc_tr, acc_va, target_mse):
        self.progress["value"] = ep
        self.lbl_status_bar.config(text=f"Epoch {ep} | MSE Latih: {mse_tr:.5f} | Val: {mse_va:.5f} | Acc Latih: {acc_tr*100:.1f}% | Val: {acc_va*100:.1f}%")
        if ep == 1 or ep % 5 == 0:
            self._tulis_log(f"Epoch {ep:>3} | MSE_tr={mse_tr:.5f} | MSE_va={mse_va:.5f} | Acc_tr={acc_tr:.3f} | Acc_va={acc_va:.3f}")

        self.ax_mse.clear()
        self.ax_mse.plot(self.hist_epoch, self.hist_mse_tr, label="MSE Latih", color="#ff6b6b", linewidth=1.8)
        self.ax_mse.plot(self.hist_epoch, self.hist_mse_va, label="MSE Validasi", color="#ffd166", linestyle="--", linewidth=1.5)
        self.ax_mse.axhline(target_mse, color="#94a3b8", linestyle=":", linewidth=1.0, label=f"Target ({target_mse})")
        self.ax_mse.set_title("MSE vs Epoch", fontsize=9)
        self.ax_mse.set_xlabel("Epoch", fontsize=8)
        self.ax_mse.set_ylabel("MSE", fontsize=8)
        self.ax_mse.grid(True, alpha=0.25)
        self.ax_mse.legend(fontsize=7)

        self.ax_acc.clear()
        self.ax_acc.plot(self.hist_epoch, [a * 100 for a in self.hist_acc_tr], label="Akurasi Latih (%)", color="#4ade80", linewidth=1.8)
        self.ax_acc.plot(self.hist_epoch, [a * 100 for a in self.hist_acc_va], label="Akurasi Validasi (%)", color="#60a5fa", linestyle="--", linewidth=1.5)
        self.ax_acc.set_title("Akurasi vs Epoch (%)", fontsize=9)
        self.ax_acc.set_xlabel("Epoch", fontsize=8)
        self.ax_acc.set_ylabel("Akurasi (%)", fontsize=8)
        self.ax_acc.set_ylim(0, 105)
        self.ax_acc.grid(True, alpha=0.25)
        self.ax_acc.legend(fontsize=7)

        self.fig_train.tight_layout(pad=1.5)
        self.canvas_plot.draw_idle()

    def _selesai_training(self, alasan, alpha, momentum):
        self.sedang_training = False
        self.btn_latih.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        mse_tr, acc_tr, _, _ = self.model.evaluasi(self.X_train, self.Y_train)
        mse_va, acc_va, _, _ = self.model.evaluasi(self.X_val, self.Y_val)
        mse_te, acc_te, _, _ = self.model.evaluasi(self.X_test, self.Y_test)
        cm = self.model.hitung_confusion_matrix(self.X_test, self.Y_test)

        self._gambar_arsitektur_jaringan()
        self._update_tab_evaluasi(cm, mse_tr, mse_va, mse_te, acc_tr, acc_va, acc_te, alpha, momentum, alasan)
        self._tulis_log(f"SELESAI ({alasan}) -> Akurasi Latih: {acc_tr*100:.2f}% | Val: {acc_va*100:.2f}% | Uji: {acc_te*100:.2f}%")
        self.lbl_status_bar.config(text=f"Selesai ({alasan}) | Akurasi Uji: {acc_te*100:.2f}%")

    def _update_tab_evaluasi(self, cm, mse_tr, mse_va, mse_te, acc_tr, acc_va, acc_te, alpha, momentum, alasan):
        self.ax_cm.clear()
        self.ax_cm.imshow(cm, cmap="Reds")
        self.ax_cm.set_title("Confusion Matrix (Data Uji 10%)", fontsize=10)
        self.ax_cm.set_xlabel("Prediksi Model", fontsize=9)
        self.ax_cm.set_ylabel("Label Sebenarnya", fontsize=9)
        self.ax_cm.set_xticks(range(10))
        self.ax_cm.set_yticks(range(10))

        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                warna_teks = "#000000" if val > cm.max() * 0.6 else "#ffffff"
                self.ax_cm.text(j, i, str(val), ha="center", va="center", color=warna_teks, fontsize=8, fontweight="bold")
        self.fig_cm.tight_layout()
        self.canvas_cm.draw_idle()

        gap_acc = (acc_tr - acc_va) * 100
        if acc_tr < 0.70:
            status = "UNDERFITTING (Model belum cukup belajar; naikkan Max Epoch atau Alpha)."
        elif gap_acc > 12.0 or (mse_va > mse_tr * 2.5 and mse_va > 0.02):
            status = f"TERINDIKASI OVERFITTING (Selisih akurasi Latih-Val = {gap_acc:.2f}%)."
        else:
            status = f"GENERALISASI BAIK / OPTIMAL (Selisih akurasi Latih-Val hanya {gap_acc:.2f}%)."

        laporan = (
            f"RINGKASAN EKSPERIMEN\n"
            f"{'='*38}\n"
            f"Status Berhenti : {alasan}\n"
            f"Total Epoch     : {len(self.hist_epoch)}\n"
            f"Parameter       : α={alpha}, μ={momentum}\n\n"
            f"PERFORMA DATASET (80/10/10)\n"
            f"  - Training   : MSE={mse_tr:.5f} | Acc={acc_tr*100:.2f}%\n"
            f"  - Validasi   : MSE={mse_va:.5f} | Acc={acc_va*100:.2f}%\n"
            f"  - Testing    : MSE={mse_te:.5f} | Acc={acc_te*100:.2f}%\n\n"
            f"DIAGNOSIS OVERFITTING\n"
            f"  {status}\n\n"
            f"AKURASI PER DIGIT (DATA UJI)\n"
        )
        for d in range(10):
            total_d = cm[d].sum()
            benar_d = cm[d, d]
            persen = (benar_d / total_d * 100) if total_d > 0 else 0.0
            laporan += f"  Angka {d} : {benar_d}/{total_d} ({persen:5.1f}%)\n"

        self.txt_analisis.config(state=tk.NORMAL)
        self.txt_analisis.delete("1.0", tk.END)
        self.txt_analisis.insert(tk.END, laporan)
        self.txt_analisis.config(state=tk.DISABLED)

    def prediksi_kanvas(self, label_asli=None, pakai_autocenter=True):
        if pakai_autocenter and self.var_autocenter.get() and md.any(self.grid_data > 0):
            self.grid_data = pusatkan_citra_cv2(self.grid_data)
            self._update_tampilan_kanvas()

        x_in = self.grid_data.flatten().reshape(1, 784)
        aktivasi = self.model.forward(x_in)
        prob = aktivasi[-1][0]
        kelas = int(md.argmax(prob))
        skor = float(prob[kelas])

        if label_asli is not None:
            self.lbl_hasil_utama.config(text=f"Prediksi: {kelas} (Asli: {label_asli})")
            self._tulis_log(f"Uji Sampel Acak -> Asli: {label_asli} | Prediksi: {kelas} (aktivasi {skor:.4f})")
        else:
            self.lbl_hasil_utama.config(text=f"Hasil: {kelas} ({skor*100:.1f}%)")
            self._tulis_log(f"Prediksi Kanvas -> Digit {kelas} (aktivasi {skor:.4f})")

        self.lbl_status_net.config(text=f"Prediksi: {kelas} | Aktivasi Output Tertinggi: {skor:.4f}")
        self._update_grafik_prob(prob)
        self._gambar_arsitektur_jaringan(aktivasi=aktivasi)

    # --- SIMPAN/MUAT BOBOT & EKSPOR GRAFIK ---
    def simpan_bobot_dialog(self):
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy Weights", "*.npz")])
        if path:
            self.model.simpan_bobot(path)
            self._tulis_log(f"Bobot berhasil disimpan ke: {os.path.basename(path)}")

    def muat_bobot_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy Weights", "*.npz")])
        if path:
            self.model.muat_bobot(path)
            self._gambar_arsitektur_jaringan()
            self._tulis_log(f"Bobot berhasil dimuat dari: {os.path.basename(path)}")

    def ekspor_grafik_paper(self):
        folder = filedialog.askdirectory(title="Pilih Folder Penyimpanan Grafik Laporan Paper")
        if not folder:
            return
        self.fig_train.savefig(os.path.join(folder, "Grafik_MSE_dan_Akurasi_BPNN.png"), dpi=300, bbox_inches="tight")
        self.fig_cm.savefig(os.path.join(folder, "Confusion_Matrix_BPNN.png"), dpi=300, bbox_inches="tight")
        self.fig_gal.savefig(os.path.join(folder, "Sampel_Dataset_28x28.png"), dpi=300, bbox_inches="tight")
        messagebox.showinfo("Berhasil", f"3 grafik resolusi tinggi (300 DPI) telah disimpan di:\n{folder}")

    def _tulis_log(self, pesan):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, pesan + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)


# ==============================================================================
# 4. EKSEKUSI PROGRAM UTAMA
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BPNNProDashboard(root)
    root.mainloop()
    