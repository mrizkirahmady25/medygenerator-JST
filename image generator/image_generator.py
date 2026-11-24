# ==============================================================================
# Image Generator - Dataset Angka 0-9 (28x28)
# Untuk training BPNN Pengenalan Pola Angka
# Nama  : Muhammad Rizki Rahmady
# NIM   : 03041182429009
# ==============================================================================

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==============================================================================
# 1. FUNGSI GENERATOR CITRA ANGKA
# ==============================================================================
def pusatkan_citra(grid_biner):
    """Pusatkan objek di kanvas 28x28 menggunakan bounding box."""
    img_u8 = (grid_biner * 255).astype(np.uint8)
    coords = cv2.findNonZero(img_u8)
    if coords is None:
        return grid_biner
    x, y, w, h = cv2.boundingRect(coords)
    crop = img_u8[y : y + h, x : x + w]

    skala = 20.0 / max(w, h)
    nw, nh = max(1, int(w * skala)), max(1, int(h * skala))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    kanvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    kanvas[oy : oy + nh, ox : ox + nw] = resized
    _, biner = cv2.threshold(kanvas, 80, 1, cv2.THRESH_BINARY)
    return biner.astype(np.float64)


def generate_digit_image(digit, rng, font_idx=None, skala=None, tebal=None,
                          sudut=None, offset_x=None, offset_y=None,
                          noise_level=0.0, blur=False):
    """
    Generate satu citra angka 28x28 dengan parameter variasi.
    
    Parameters:
    -----------
    digit       : int (0-9) - Angka yang akan digenerate
    rng         : numpy random generator
    font_idx    : int (0-6) - Index font OpenCV, None = random
    skala       : float - Skala font, None = random (0.6-1.0)
    tebal       : int - Ketebalan garis, None = random (1-3)
    sudut       : float - Sudut rotasi (derajat), None = random (-15,15)
    offset_x    : int - Offset horizontal, None = random (-3,3)
    offset_y    : int - Offset vertikal, None = random (-3,3)
    noise_level : float (0.0-1.0) - Level noise salt & pepper
    blur        : bool - Tambahkan gaussian blur
    """
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    ]

    # Parameter default: random jika None
    if font_idx is None:
        font_idx = int(rng.integers(0, len(fonts)))
    font = fonts[min(font_idx, len(fonts) - 1)]

    if skala is None:
        skala = float(rng.uniform(0.60, 1.00))
    if tebal is None:
        tebal = int(rng.integers(1, 4))
    if sudut is None:
        sudut = float(rng.uniform(-15, 15))
    if offset_x is None:
        offset_x = int(rng.integers(-3, 4))
    if offset_y is None:
        offset_y = int(rng.integers(-3, 4))

    # Render teks di kanvas 28x28
    kanvas = np.zeros((28, 28), dtype=np.uint8)
    teks = str(digit)
    (w, h), _ = cv2.getTextSize(teks, font, skala, tebal)
    ox = int((28 - w) // 2 + offset_x)
    oy = int((28 + h) // 2 + offset_y)
    cv2.putText(kanvas, teks, (ox, oy), font, skala, 255, tebal, cv2.LINE_AA)

    # Rotasi
    M_rot = cv2.getRotationMatrix2D((14, 14), sudut, 1.0)
    kanvas = cv2.warpAffine(kanvas, M_rot, (28, 28), flags=cv2.INTER_LINEAR)

    # Noise salt & pepper
    if noise_level > 0:
        mask = rng.random((28, 28))
        kanvas[mask < noise_level / 2] = 0
        kanvas[mask > 1 - noise_level / 2] = 255

    # Gaussian blur
    if blur:
        kanvas = cv2.GaussianBlur(kanvas, (3, 3), 0.5)

    # Binarisasi dan centering
    _, biner = cv2.threshold(kanvas, 90, 1, cv2.THRESH_BINARY)
    biner_pusat = pusatkan_citra(biner.astype(np.float64))

    return biner_pusat


def batch_generate(digits, jumlah_per_angka, rng, noise_level=0.0, blur=False,
                   progress_callback=None):
    """
    Generate batch citra untuk beberapa angka.
    
    Returns: list of (digit, image_array_28x28)
    """
    results = []
    total = len(digits) * jumlah_per_angka
    count = 0

    for digit in digits:
        for i in range(jumlah_per_angka):
            img = generate_digit_image(digit, rng, noise_level=noise_level, blur=blur)
            results.append((digit, img))
            count += 1
            if progress_callback:
                progress_callback(count, total)

    return results


# ==============================================================================
# 2. GUI IMAGE GENERATOR
# ==============================================================================
class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Generator - Dataset Angka 0-9 (28×28) | Muhammad Rizki Rahmady")
        self.root.geometry("1280x780")
        self.root.configure(bg="#0f0f0f")
        self.root.minsize(1000, 600)

        self.rng = np.random.default_rng(42)
        self.generated_images = []
        self.preview_page = 0
        self.images_per_page = 100  # 10x10 grid

        self._setup_style()
        self._build_header()
        self._build_main_area()
        self._build_footer()

        # Generate sampel awal
        self._quick_generate_preview()

    # --- STYLE ---
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#0f0f0f", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1e1e1e", foreground="#cccccc",
                         padding=[14, 6], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab",
                   background=[("selected", "#1a5276")],
                   foreground=[("selected", "#ffffff")])
        style.configure("TProgressbar", troughcolor="#1e1e1e", background="#2ecc71")
        style.configure("Green.TButton", background="#27ae60", foreground="#ffffff")

    # --- HEADER KONTROL ---
    def _build_header(self):
        header = tk.Frame(self.root, bg="#1a1a1a", padx=12, pady=10,
                          highlightbackground="#333", highlightthickness=1)
        header.pack(side=tk.TOP, fill=tk.X)

        # Judul
        tk.Label(header, text="🎨 IMAGE GENERATOR", bg="#1a1a1a", fg="#3498db",
                 font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=(0, 20))

        # --- Kontrol input ---
        ctrl = tk.Frame(header, bg="#1a1a1a")
        ctrl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def add_input(parent, label, default, width=5):
            tk.Label(parent, text=label, bg="#1a1a1a", fg="#bbb",
                     font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 2))
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(parent, textvariable=var, width=width, bg="#2a2a2a",
                           fg="#fff", insertbackground="#fff", relief=tk.FLAT,
                           font=("Segoe UI", 9))
            ent.pack(side=tk.LEFT, padx=(0, 4))
            return var

        self.var_jumlah = add_input(ctrl, "Jumlah/Angka:", 50, 5)
        self.var_noise = add_input(ctrl, "Noise (0-1):", 0.0, 5)
        self.var_seed = add_input(ctrl, "Seed:", 42, 5)

        # Checkbox blur
        self.var_blur = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Blur", variable=self.var_blur,
                       bg="#1a1a1a", fg="#bbb", selectcolor="#2a2a2a",
                       activebackground="#1a1a1a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)

        # Angka yang dipilih
        tk.Label(ctrl, text="Angka:", bg="#1a1a1a", fg="#bbb",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 4))
        self.digit_vars = []
        for d in range(10):
            var = tk.BooleanVar(value=True)
            self.digit_vars.append(var)
            tk.Checkbutton(ctrl, text=str(d), variable=var,
                           bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a",
                           activebackground="#1a1a1a", font=("Segoe UI", 8, "bold"),
                           width=2).pack(side=tk.LEFT)

        # Tombol
        btn_frame = tk.Frame(header, bg="#1a1a1a")
        btn_frame.pack(side=tk.RIGHT)

        btn_cfg = {"font": ("Segoe UI", 9, "bold"), "relief": tk.FLAT, "padx": 12, "pady": 4}

        tk.Button(btn_frame, text="🔄 Generate", bg="#2980b9", fg="#fff",
                  command=self.generate_all, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="👁 Preview 1 Angka", bg="#8e44ad", fg="#fff",
                  command=self.preview_single, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="💾 Simpan Semua (PNG)", bg="#27ae60", fg="#fff",
                  command=self.save_all_png, **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="💾 Simpan (NPZ)", bg="#16a085", fg="#fff",
                  command=self.save_npz, **btn_cfg).pack(side=tk.LEFT, padx=3)

    # --- AREA UTAMA ---
    def _build_main_area(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Tab 1: Galeri Preview
        self.tab_gallery = tk.Frame(self.notebook, bg="#111111")
        self.notebook.add(self.tab_gallery, text="  📷 Galeri Preview  ")

        # Navigasi halaman
        nav = tk.Frame(self.tab_gallery, bg="#111111", pady=4)
        nav.pack(side=tk.TOP, fill=tk.X, padx=10)

        self.lbl_page_info = tk.Label(nav, text="Halaman 1", bg="#111111", fg="#fff",
                                       font=("Segoe UI", 10, "bold"))
        self.lbl_page_info.pack(side=tk.LEFT, padx=10)

        tk.Button(nav, text="◀ Prev", bg="#2a2a2a", fg="#fff", relief=tk.FLAT,
                  command=self.prev_page, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text="Next ▶", bg="#2a2a2a", fg="#fff", relief=tk.FLAT,
                  command=self.next_page, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)

        self.lbl_total_info = tk.Label(nav, text="Total: 0 gambar", bg="#111111", fg="#aaa",
                                        font=("Segoe UI", 9))
        self.lbl_total_info.pack(side=tk.RIGHT, padx=10)

        # Matplotlib gallery
        self.fig_gal, self.axes_gal = plt.subplots(10, 10, figsize=(11, 11), dpi=72)
        self.fig_gal.patch.set_facecolor("#111111")
        for row in self.axes_gal:
            for ax in row:
                ax.set_facecolor("#111111")
                ax.set_xticks([])
                ax.set_yticks([])
        self.fig_gal.subplots_adjust(hspace=0.4, wspace=0.15)

        self.canvas_gal = FigureCanvasTkAgg(self.fig_gal, master=self.tab_gallery)
        self.canvas_gal.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Tab 2: Single Preview (interaktif)
        self.tab_single = tk.Frame(self.notebook, bg="#111111")
        self.notebook.add(self.tab_single, text="  🔍 Preview Detail  ")
        self._build_single_preview_tab()

        # Tab 3: Statistik
        self.tab_stats = tk.Frame(self.notebook, bg="#111111")
        self.notebook.add(self.tab_stats, text="  📊 Statistik Dataset  ")
        self._build_stats_tab()

    def _build_single_preview_tab(self):
        top = tk.Frame(self.tab_single, bg="#1a1a1a", padx=10, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text="Angka:", bg="#1a1a1a", fg="#bbb",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=4)
        self.var_single_digit = tk.StringVar(value="0")
        tk.Spinbox(top, from_=0, to=9, width=3, textvariable=self.var_single_digit,
                   bg="#2a2a2a", fg="#fff", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Font:", bg="#1a1a1a", fg="#bbb",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(16, 4))
        self.var_font = tk.StringVar(value="Random")
        font_names = ["Random", "Simplex", "Duplex", "Complex", "Triplex",
                       "Complex Small", "Script Simplex", "Script Complex"]
        tk.OptionMenu(top, self.var_font, *font_names).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Skala:", bg="#1a1a1a", fg="#bbb",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(16, 4))
        self.var_skala = tk.StringVar(value="auto")
        tk.Entry(top, textvariable=self.var_skala, width=5, bg="#2a2a2a", fg="#fff",
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Rotasi (°):", bg="#1a1a1a", fg="#bbb",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(16, 4))
        self.var_rotasi = tk.StringVar(value="auto")
        tk.Entry(top, textvariable=self.var_rotasi, width=5, bg="#2a2a2a", fg="#fff",
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="🔄 Generate 1", bg="#2980b9", fg="#fff",
                  font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                  command=self._generate_single).pack(side=tk.LEFT, padx=16)
        tk.Button(top, text="🎲 Random Batch (25)", bg="#8e44ad", fg="#fff",
                  font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                  command=self._generate_single_batch).pack(side=tk.LEFT, padx=4)

        # Area preview besar
        preview_frame = tk.Frame(self.tab_single, bg="#111111")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Preview besar (kiri)
        left = tk.Frame(preview_frame, bg="#111111")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.fig_big, self.ax_big = plt.subplots(figsize=(3.5, 3.5), dpi=100)
        self.fig_big.patch.set_facecolor("#111111")
        self.ax_big.set_facecolor("#111111")
        self.canvas_big = FigureCanvasTkAgg(self.fig_big, master=left)
        self.canvas_big.get_tk_widget().pack()

        self.lbl_detail = tk.Label(left, text="Detail: -", bg="#111111", fg="#fff",
                                    font=("Consolas", 9), justify=tk.LEFT)
        self.lbl_detail.pack(pady=8)

        # Grid 5x5 variasi (kanan)
        right = tk.Frame(preview_frame, bg="#111111")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_grid, self.axes_grid = plt.subplots(5, 5, figsize=(7, 7), dpi=80)
        self.fig_grid.patch.set_facecolor("#111111")
        for row in self.axes_grid:
            for ax in row:
                ax.set_facecolor("#111111")
                ax.set_xticks([])
                ax.set_yticks([])
        self.fig_grid.suptitle("Variasi Random", color="#fff", fontsize=10)
        self.fig_grid.subplots_adjust(hspace=0.3, wspace=0.1)
        self.canvas_grid = FigureCanvasTkAgg(self.fig_grid, master=right)
        self.canvas_grid.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_stats_tab(self):
        self.fig_stats, self.axes_stats = plt.subplots(1, 2, figsize=(12, 4.5), dpi=90)
        self.fig_stats.patch.set_facecolor("#111111")
        for ax in self.axes_stats:
            ax.set_facecolor("#111111")
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=self.tab_stats)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_stats_info = tk.Label(self.tab_stats, text="Klik 'Generate' untuk melihat statistik dataset.",
                                        bg="#111111", fg="#aaa", font=("Segoe UI", 10))
        self.lbl_stats_info.pack(pady=4)

    # --- FOOTER ---
    def _build_footer(self):
        footer = tk.Frame(self.root, bg="#1a1a1a", padx=10, pady=4,
                          highlightbackground="#333", highlightthickness=1)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = tk.Label(footer, text="Siap. Klik 'Generate' untuk membuat dataset.",
                                    bg="#1a1a1a", fg="#ddd", font=("Segoe UI", 9))
        self.lbl_status.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(footer, mode="determinate", length=300)
        self.progress.pack(side=tk.RIGHT)

    # --- GENERATE ALL ---
    def generate_all(self):
        try:
            jumlah = max(1, int(self.var_jumlah.get()))
            noise = float(self.var_noise.get())
            seed = int(self.var_seed.get())
        except ValueError:
            messagebox.showerror("Error", "Pastikan Jumlah, Noise, dan Seed berupa angka valid.")
            return

        digits = [d for d in range(10) if self.digit_vars[d].get()]
        if not digits:
            messagebox.showwarning("Peringatan", "Pilih minimal satu angka!")
            return

        blur = self.var_blur.get()
        self.rng = np.random.default_rng(seed)

        total = len(digits) * jumlah
        self.progress["maximum"] = total
        self.progress["value"] = 0
        self.lbl_status.config(text=f"Generating {total} gambar...")
        self.root.update_idletasks()

        def on_progress(count, total):
            self.root.after(0, lambda: self._update_progress(count, total))

        def run():
            results = batch_generate(digits, jumlah, self.rng, noise, blur, on_progress)
            self.root.after(0, lambda: self._on_generate_done(results))

        threading.Thread(target=run, daemon=True).start()

    def _update_progress(self, count, total):
        self.progress["value"] = count
        self.lbl_status.config(text=f"Generating... {count}/{total} ({count*100//total}%)")

    def _on_generate_done(self, results):
        self.generated_images = results
        self.preview_page = 0
        self.progress["value"] = self.progress["maximum"]
        self.lbl_status.config(text=f"Selesai! {len(results)} gambar berhasil di-generate.")
        self.lbl_total_info.config(text=f"Total: {len(results)} gambar")
        self._update_gallery()
        self._update_stats()

    # --- GALLERY ---
    def _update_gallery(self):
        start = self.preview_page * self.images_per_page
        end = min(start + self.images_per_page, len(self.generated_images))
        total_pages = max(1, (len(self.generated_images) + self.images_per_page - 1) // self.images_per_page)

        self.lbl_page_info.config(text=f"Halaman {self.preview_page + 1} / {total_pages}")

        for row in self.axes_gal:
            for ax in row:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#111111")

        for idx, i in enumerate(range(start, end)):
            r, c = idx // 10, idx % 10
            digit, img = self.generated_images[i]
            self.axes_gal[r][c].imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            self.axes_gal[r][c].set_title(str(digit), fontsize=7, color="#3498db", pad=1)

        self.fig_gal.subplots_adjust(hspace=0.5, wspace=0.15)
        self.canvas_gal.draw_idle()

    def prev_page(self):
        if self.preview_page > 0:
            self.preview_page -= 1
            self._update_gallery()

    def next_page(self):
        total_pages = max(1, (len(self.generated_images) + self.images_per_page - 1) // self.images_per_page)
        if self.preview_page < total_pages - 1:
            self.preview_page += 1
            self._update_gallery()

    # --- SINGLE PREVIEW ---
    def preview_single(self):
        self.notebook.select(self.tab_single)
        self._generate_single_batch()

    def _generate_single(self):
        digit = int(self.var_single_digit.get())

        font_map = {"Random": None, "Simplex": 0, "Duplex": 1, "Complex": 2,
                     "Triplex": 3, "Complex Small": 4, "Script Simplex": 5, "Script Complex": 6}
        font_idx = font_map.get(self.var_font.get(), None)

        skala_str = self.var_skala.get().strip()
        skala = None if skala_str == "auto" else float(skala_str)

        rotasi_str = self.var_rotasi.get().strip()
        sudut = None if rotasi_str == "auto" else float(rotasi_str)

        noise = float(self.var_noise.get())
        blur = self.var_blur.get()

        img = generate_digit_image(digit, self.rng, font_idx=font_idx, skala=skala,
                                    sudut=sudut, noise_level=noise, blur=blur)

        self.ax_big.clear()
        self.ax_big.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        self.ax_big.set_title(f"Angka: {digit}", color="#3498db", fontsize=12, fontweight="bold")
        self.ax_big.set_xticks([])
        self.ax_big.set_yticks([])
        self.fig_big.tight_layout()
        self.canvas_big.draw_idle()

        piksel_aktif = int(np.sum(img > 0.5))
        self.lbl_detail.config(text=f"Digit    : {digit}\n"
                                     f"Ukuran   : 28 × 28\n"
                                     f"Piksel 1 : {piksel_aktif}\n"
                                     f"Piksel 0 : {784 - piksel_aktif}\n"
                                     f"Densitas : {piksel_aktif/784*100:.1f}%")

    def _generate_single_batch(self):
        digit = int(self.var_single_digit.get())
        noise = float(self.var_noise.get())
        blur = self.var_blur.get()

        # Generate 1 besar
        self._generate_single()

        # Generate 25 variasi
        for r in range(5):
            for c in range(5):
                ax = self.axes_grid[r][c]
                ax.clear()
                img = generate_digit_image(digit, self.rng, noise_level=noise, blur=blur)
                ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

        self.fig_grid.suptitle(f"25 Variasi Angka {digit}", color="#fff", fontsize=10)
        self.canvas_grid.draw_idle()

    # --- STATISTIK ---
    def _update_stats(self):
        if not self.generated_images:
            return

        digits = [d for d, _ in self.generated_images]
        densities = [float(np.sum(img > 0.5)) / 784 * 100 for _, img in self.generated_images]

        # Distribusi per angka
        ax1 = self.axes_stats[0]
        ax1.clear()
        counts = [digits.count(d) for d in range(10)]
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c",
                   "#3498db", "#9b59b6", "#e91e63", "#607d8b", "#795548"]
        bars = ax1.bar(range(10), counts, color=colors, edgecolor="#333", linewidth=0.5)
        ax1.set_xlabel("Angka", fontsize=9, color="#ccc")
        ax1.set_ylabel("Jumlah Citra", fontsize=9, color="#ccc")
        ax1.set_title("Distribusi Jumlah Citra per Angka", fontsize=10, color="#fff")
        ax1.set_xticks(range(10))
        ax1.tick_params(colors="#aaa", labelsize=8)
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha="center", va="bottom", color="#fff", fontsize=8)

        # Distribusi densitas piksel
        ax2 = self.axes_stats[1]
        ax2.clear()
        ax2.hist(densities, bins=20, color="#3498db", edgecolor="#222", alpha=0.85)
        ax2.set_xlabel("Densitas Piksel Aktif (%)", fontsize=9, color="#ccc")
        ax2.set_ylabel("Frekuensi", fontsize=9, color="#ccc")
        ax2.set_title("Distribusi Densitas Piksel", fontsize=10, color="#fff")
        ax2.tick_params(colors="#aaa", labelsize=8)
        ax2.axvline(np.mean(densities), color="#e74c3c", linestyle="--", linewidth=1.5,
                     label=f"Rata-rata: {np.mean(densities):.1f}%")
        ax2.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor="#fff")

        self.fig_stats.tight_layout(pad=2.0)
        self.canvas_stats.draw_idle()

        self.lbl_stats_info.config(
            text=f"Total: {len(self.generated_images)} citra | "
                 f"Densitas rata-rata: {np.mean(densities):.1f}% | "
                 f"Min: {np.min(densities):.1f}% | Max: {np.max(densities):.1f}%"
        )

    # --- SAVE PNG ---
    def save_all_png(self):
        if not self.generated_images:
            messagebox.showwarning("Peringatan", "Belum ada gambar! Klik 'Generate' dulu.")
            return

        folder = filedialog.askdirectory(title="Pilih Folder untuk Menyimpan Gambar PNG")
        if not folder:
            return

        self.lbl_status.config(text="Menyimpan file PNG...")
        self.root.update_idletasks()

        def run():
            for i, (digit, img) in enumerate(self.generated_images):
                digit_folder = os.path.join(folder, str(digit))
                os.makedirs(digit_folder, exist_ok=True)

                # Cari nomor file berikutnya
                existing = [f for f in os.listdir(digit_folder) if f.endswith(".png")]
                next_num = len(existing)

                filename = f"{digit}_{next_num:04d}.png"
                filepath = os.path.join(digit_folder, filename)

                img_u8 = (img * 255).astype(np.uint8)
                cv2.imwrite(filepath, img_u8)

            self.root.after(0, lambda: self._on_save_done(folder, "PNG"))

        threading.Thread(target=run, daemon=True).start()

    def save_npz(self):
        if not self.generated_images:
            messagebox.showwarning("Peringatan", "Belum ada gambar! Klik 'Generate' dulu.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz")],
            title="Simpan Dataset sebagai NPZ"
        )
        if not filepath:
            return

        X = np.array([img.flatten() for _, img in self.generated_images])
        labels = np.array([d for d, _ in self.generated_images])

        # One-hot encoding
        Y = np.zeros((len(labels), 10), dtype=np.float64)
        for i, label in enumerate(labels):
            Y[i, label] = 1.0

        np.savez(filepath, X=X, Y=Y, labels=labels)
        self._on_save_done(os.path.dirname(filepath), "NPZ")

    def _on_save_done(self, folder, fmt):
        self.lbl_status.config(text=f"Berhasil menyimpan {len(self.generated_images)} gambar ({fmt})!")
        messagebox.showinfo("Berhasil",
                            f"{len(self.generated_images)} gambar berhasil disimpan ({fmt})!\n\n"
                            f"Lokasi: {folder}")

    # --- QUICK PREVIEW ---
    def _quick_generate_preview(self):
        """Generate preview awal saat aplikasi dibuka."""
        results = []
        for digit in range(10):
            for _ in range(10):
                img = generate_digit_image(digit, self.rng)
                results.append((digit, img))
        self.generated_images = results
        self.lbl_total_info.config(text=f"Total: {len(results)} gambar (preview)")
        self._update_gallery()
        self._update_stats()


# ==============================================================================
# 3. MAIN
# ==============================================================================
if __name__ == "__main__":
    plt.style.use("dark_background")
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()
