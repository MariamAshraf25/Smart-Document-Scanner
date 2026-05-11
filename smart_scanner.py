# --------------------------------------------------------------------------
# Project:     Smart Document Scanner
# Author:      Mariam Ashraf
# Faculty:     Engineering - Capital University (Formerly Helwan)
# Date:        May 2026
# --------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

# ─────────────────────────────────────────────
#  SOBEL FROM SCRATCH 
# ─────────────────────────────────────────────
def sobel_from_scratch(gray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    h, w = gray.shape
    img  = gray.astype(np.float64)
    Gx   = np.zeros_like(img)
    Gy   = np.zeros_like(img)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch    = img[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(patch * Kx)
            Gy[i, j] = np.sum(patch * Ky)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255
    return magnitude.astype(np.uint8)


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────
def order_points(pts):
    pts  = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 10)
    maxH = max(int(np.linalg.norm(br - tr)), int(np.linalg.norm(bl - tl)), 10)
    dst  = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]],
                    dtype=np.float32)
    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


# ─────────────────────────────────────────────
#  PIPELINE  (original hybrid — untouched)
# ─────────────────────────────────────────────
def process_image_hybrid(image_bgr):
    steps    = {}
    original = image_bgr.copy()
    steps['00. Original'] = original.copy()

    # Resize
    h, w  = original.shape[:2]
    ratio = 800 / max(h, w) if max(h, w) > 1000 else 1.0
    proc  = cv2.resize(original, (int(w * ratio), int(h * ratio)))

    # STEP 1: Sobel after Gaussian Blur
    gray        = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    steps['01. Grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blur        = cv2.GaussianBlur(gray, (9, 9), 0)
    steps['02. Gaussian Blur'] = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    sobel_edges = sobel_from_scratch(blur)
    steps['03. Sobel Edges'] = cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2BGR)

    # STEP 2: HSV Paper Mask
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, val_mask = cv2.threshold(val, 60, 255, cv2.THRESH_BINARY)
    paper_mask  = cv2.bitwise_and(sat_mask, val_mask)

    k = np.ones((7, 7), np.uint8)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, k, iterations=4)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN,  k, iterations=3)
    steps['04. HSV Paper Mask'] = cv2.cvtColor(paper_mask, cv2.COLOR_GRAY2BGR)

    # STEP 5: HYBRID — Combine Sobel and Mask
    hybrid_edges = cv2.bitwise_and(sobel_edges, paper_mask)
    steps['05. Hybrid Edges'] = cv2.cvtColor(hybrid_edges, cv2.COLOR_GRAY2BGR)

    # STEP 6: Contour Detection on Hybrid Edges
    hybrid_dilated = cv2.dilate(hybrid_edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(hybrid_dilated.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return proc, original, False, steps

    contours   = sorted(contours, key=cv2.contourArea, reverse=True)
    screen_cnt = None
    for c in contours[:3]:
        peri = cv2.arcLength(c, True)
        for eps in [0.02, 0.05, 0.1]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is not None:
            break

    if screen_cnt is None:
        r          = cv2.minAreaRect(contours[0])
        screen_cnt = cv2.boxPoints(r).reshape(4, 1, 2).astype(np.int32)

    edge_vis = proc.copy()
    cv2.drawContours(edge_vis, [screen_cnt], -1, (0, 255, 0), 3)
    for pt in screen_cnt.reshape(4, 2):
        cv2.circle(edge_vis, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
    steps['06. Detection Visual'] = edge_vis.copy()

    # STEP 7: Perspective & Final Scan
    pts    = (screen_cnt.reshape(4, 2) / ratio).astype(np.float32)
    warped = four_point_transform(original, pts)
    steps['07. Perspective Warp'] = warped.copy()

    final_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # تحسين الجودة قبل الـ threshold عشان نشيل النقط السودا
    final_gray = cv2.fastNlMeansDenoising(final_gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_gray = clahe.apply(final_gray)
    final      = cv2.adaptiveThreshold(final_gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
    final_bgr  = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    steps['08. Final Scan'] = final_bgr.copy()

    return edge_vis, final_bgr, True, steps


# ─────────────────────────────────────────────
#  STEPS VIEWER WINDOW
# ─────────────────────────────────────────────
class StepsWindow(tk.Toplevel):
    DARK_BG  = "#0d0f14"
    PANEL_BG = "#13161e"
    CARD_BG  = "#1a1e2a"
    ACCENT   = "#00e5b0"
    ACCENT2  = "#0099ff"
    TEXT_DIM = "#5a6070"
    TEXT_LBL = "#8892a4"
    BORDER   = "#252a38"

    def __init__(self, master, steps: dict):
        super().__init__(master)
        self.title("Pipeline Steps")
        self.configure(bg=self.DARK_BG)
        self.minsize(1000, 650)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        W, H   = 1200, 760
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

        self._steps = steps
        self._names = list(steps.keys())
        self._cur   = 0
        self._build()
        self.after(100, lambda: self._show_step(0))

    def _build(self):
        # Top bar
        bar = tk.Frame(self, bg=self.PANEL_BG, height=50)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="⬛  PIPELINE STEPS VIEWER",
                 font=("Consolas", 12, "bold"),
                 bg=self.PANEL_BG, fg=self.ACCENT).pack(side="left", padx=20, pady=12)
        self._nav_lbl = tk.Label(bar, font=("Consolas", 10),
                                 bg=self.PANEL_BG, fg=self.TEXT_LBL)
        self._nav_lbl.pack(side="right", padx=20)

        # Sidebar
        side = tk.Frame(self, bg=self.PANEL_BG, width=210)
        side.pack(fill="y", side="left")
        side.pack_propagate(False)
        tk.Label(side, text="STEPS", font=("Consolas", 9, "bold"),
                 bg=self.PANEL_BG, fg=self.TEXT_DIM).pack(anchor="w", padx=16, pady=(18, 8))

        self._step_btns = []
        for i, name in enumerate(self._names):
            btn = tk.Button(
                side, text=name,
                font=("Consolas", 8),
                bg=self.PANEL_BG, fg=self.TEXT_DIM,
                activebackground=self.CARD_BG,
                activeforeground=self.ACCENT,
                relief="flat", bd=0,
                anchor="w", padx=14, pady=7,
                cursor="hand2",
                command=lambda idx=i: self._show_step(idx)
            )
            btn.pack(fill="x", padx=6, pady=1)
            self._step_btns.append(btn)

        nav = tk.Frame(side, bg=self.PANEL_BG)
        nav.pack(side="bottom", fill="x", padx=10, pady=16)
        tk.Button(nav, text="◀  PREV", font=("Consolas", 9, "bold"),
                  bg="#0d2a22", fg=self.ACCENT,
                  activebackground="#0d2a22", activeforeground=self.ACCENT,
                  relief="flat", bd=0, padx=8, pady=8,
                  cursor="hand2", command=self._prev).pack(fill="x", pady=(0, 6))
        tk.Button(nav, text="NEXT  ▶", font=("Consolas", 9, "bold"),
                  bg="#0a1e35", fg=self.ACCENT2,
                  activebackground="#0a1e35", activeforeground=self.ACCENT2,
                  relief="flat", bd=0, padx=8, pady=8,
                  cursor="hand2", command=self._next).pack(fill="x")

        # Main image area
        main = tk.Frame(self, bg=self.DARK_BG)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        self._step_title = tk.Label(main, text="",
                                    font=("Consolas", 11, "bold"),
                                    bg=self.DARK_BG, fg=self.ACCENT)
        self._step_title.pack(anchor="w", pady=(0, 8))

        card = tk.Frame(main, bg=self.CARD_BG,
                        highlightbackground=self.BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(card, bg=self.CARD_BG, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas.bind("<Configure>", lambda e: self._redraw())

        self._info_lbl = tk.Label(main, text="", font=("Consolas", 8),
                                  bg=self.DARK_BG, fg=self.TEXT_DIM)
        self._info_lbl.pack(anchor="w", pady=(6, 0))

        self.bind("<Left>",  lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())

    def _show_step(self, idx):
        self._cur         = idx
        name              = self._names[idx]
        img               = self._steps[name]
        self._current_img = img
        h, w              = img.shape[:2]

        self._step_title.config(text=f"  {name}")
        self._nav_lbl.config(text=f"Step {idx+1} / {len(self._names)}")
        self._info_lbl.config(text=f"{w} × {h} px")

        for i, btn in enumerate(self._step_btns):
            if i == idx:
                btn.config(fg=self.ACCENT,   bg=self.CARD_BG)
            elif i < idx:
                btn.config(fg=self.TEXT_LBL, bg=self.PANEL_BG)
            else:
                btn.config(fg=self.TEXT_DIM, bg=self.PANEL_BG)
        self._redraw()

    def _redraw(self):
        if not hasattr(self, '_current_img') or self._current_img is None:
            return
        img  = self._current_img
        cw   = self._canvas.winfo_width()  or 800
        ch   = self._canvas.winfo_height() or 550
        h, w = img.shape[:2]
        scale   = min((cw - 8) / w, (ch - 8) / h, 1.0)
        nw, nh  = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        self._canvas._photo = photo

    def _prev(self):
        if self._cur > 0:
            self._show_step(self._cur - 1)

    def _next(self):
        if self._cur < len(self._names) - 1:
            self._show_step(self._cur + 1)


# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
DARK_BG    = "#0d0f14"
PANEL_BG   = "#13161e"
CARD_BG    = "#1a1e2a"
ACCENT     = "#00e5b0"
ACCENT2    = "#0099ff"
TEXT_MAIN  = "#e8eaf0"
TEXT_DIM   = "#5a6070"
TEXT_LABEL = "#8892a4"
BORDER     = "#252a38"
BTN_UP_BG  = "#0d2a22"
BTN_UP_FG  = ACCENT
BTN_PR_BG  = "#0a1e35"
BTN_PR_FG  = ACCENT2
BTN_SV_BG  = "#1a1e2a"
BTN_SV_FG  = TEXT_DIM
BTN_ST_BG  = "#1a120d"
BTN_ST_FG  = "#ffb347"


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class SmartScanner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Document Scanner")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)
        self.minsize(960, 600)

        self._orig_cv  = None
        self._edges_cv = None
        self._scan_cv  = None
        self._steps    = {}
        self._success  = False

        self._build_ui()
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h   = 1180, 720
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build_ui(self):
        topbar = tk.Frame(self, bg=PANEL_BG, height=56)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="⬛  SMART DOCUMENT SCANNER",
                 font=("Consolas", 13, "bold"),
                 bg=PANEL_BG, fg=ACCENT).pack(side="left", padx=20, pady=14)
        self._status_lbl = tk.Label(topbar, text="● READY",
                                    font=("Consolas", 10),
                                    bg=PANEL_BG, fg=TEXT_DIM)
        self._status_lbl.pack(side="right", padx=20)

        sidebar = tk.Frame(self, bg=PANEL_BG, width=190)
        sidebar.pack(fill="y", side="left")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="PIPELINE",
                 font=("Consolas", 9, "bold"),
                 bg=PANEL_BG, fg=TEXT_DIM).pack(anchor="w", padx=18, pady=(22, 6))

        pipeline_steps = [
            ("00", "Original"),
            ("01", "Grayscale"),
            ("02", "Gaussian Blur"),
            ("03", "Sobel Edges ★"),
            ("04", "HSV Paper Mask"),
            ("05", "Hybrid Edges ★"),
            ("06", "Contours"),
            ("07", "Perspective Warp"),
            ("08", "Adaptive Thresh"),
        ]
        self._step_labels = []
        for num, name in pipeline_steps:
            f = tk.Frame(sidebar, bg=PANEL_BG)
            f.pack(fill="x", padx=12, pady=1)
            n = tk.Label(f, text=num, font=("Consolas", 8, "bold"),
                         bg=PANEL_BG, fg=ACCENT, width=3, anchor="w")
            n.pack(side="left")
            l = tk.Label(f, text=name, font=("Consolas", 9),
                         bg=PANEL_BG, fg=TEXT_DIM, anchor="w")
            l.pack(side="left")
            self._step_labels.append((n, l))

        tk.Frame(sidebar, bg=PANEL_BG).pack(expand=True, fill="both")

        self._btn_upload = self._make_button(
            sidebar, "⬆  UPLOAD IMAGE", self._upload, BTN_UP_BG, BTN_UP_FG)
        self._btn_upload.pack(fill="x", padx=14, pady=(0, 6))

        self._btn_process = self._make_button(
            sidebar, "▶  PROCESS", self._process, BTN_PR_BG, BTN_PR_FG)
        self._btn_process.pack(fill="x", padx=14, pady=(0, 6))
        self._btn_process.config(state="disabled")

        self._btn_steps = self._make_button(
            sidebar, "🔍  SHOW STEPS", self._show_steps, BTN_ST_BG, BTN_ST_FG)
        self._btn_steps.pack(fill="x", padx=14, pady=(0, 6))
        self._btn_steps.config(state="disabled")

        self._btn_save = self._make_button(
            sidebar, "💾  SAVE RESULT", self._save, BTN_SV_BG, BTN_SV_FG)
        self._btn_save.pack(fill="x", padx=14, pady=(0, 18))
        self._btn_save.config(state="disabled")

        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True)

        self._panels = []
        headers = ["ORIGINAL", "DETECTION  (Hybrid ★)", "FINAL SCAN"]
        colors  = [TEXT_LABEL, ACCENT, ACCENT2]
        for i, (hdr, clr) in enumerate(zip(headers, colors)):
            col = tk.Frame(main, bg=DARK_BG)
            col.pack(side="left", fill="both", expand=True,
                     padx=(18 if i == 0 else 8, 8 if i < 2 else 18), pady=18)
            tk.Label(col, text=hdr, font=("Consolas", 9, "bold"),
                     bg=DARK_BG, fg=clr).pack(anchor="w", pady=(0, 6))
            card = tk.Frame(col, bg=CARD_BG,
                            highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="both", expand=True)
            canvas = tk.Canvas(card, bg=CARD_BG, highlightthickness=0, cursor="crosshair")
            canvas.pack(fill="both", expand=True, padx=2, pady=2)
            info = tk.Label(col, text="—", font=("Consolas", 8),
                            bg=DARK_BG, fg=TEXT_DIM)
            info.pack(anchor="w", pady=(4, 0))
            self._panels.append((canvas, info))

        self.bind("<Configure>", lambda e: self._refresh_canvases())

    def _make_button(self, parent, text, cmd, bg, fg):
        return tk.Button(
            parent, text=text, command=cmd,
            font=("Consolas", 9, "bold"),
            bg=bg, fg=fg,
            activebackground=bg, activeforeground=fg,
            relief="flat", bd=0, padx=10, pady=10, cursor="hand2"
        )

    def _set_status(self, msg, color=TEXT_DIM):
        self._status_lbl.config(text=msg, fg=color)
        self.update_idletasks()

    def _highlight_step(self, idx):
        for i, (n, l) in enumerate(self._step_labels):
            if i < idx:
                n.config(fg=ACCENT);    l.config(fg=TEXT_LABEL)
            elif i == idx:
                n.config(fg="#ffdd00"); l.config(fg=TEXT_MAIN)
            else:
                n.config(fg=TEXT_DIM);  l.config(fg=TEXT_DIM)
        self.update_idletasks()

    def _show_on_canvas(self, idx, img_bgr, info_text=""):
        canvas, info = self._panels[idx]
        cw = canvas.winfo_width()  or 300
        ch = canvas.winfo_height() or 400
        h, w  = img_bgr.shape[:2]
        scale = min((cw - 8) / w, (ch - 8) / h, 1.0)
        nw, nh  = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        canvas._photo = photo
        info.config(text=info_text)

    def _refresh_canvases(self):
        for i, img in enumerate([self._orig_cv, self._edges_cv, self._scan_cv]):
            if img is not None:
                h, w = img.shape[:2]
                self._show_on_canvas(i, img, f"{w} × {h} px")

    def _upload(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        self._orig_cv  = img
        self._edges_cv = None
        self._scan_cv  = None
        self._steps    = {}
        self.after(100, lambda: self._show_on_canvas(
            0, img, f"{img.shape[1]} × {img.shape[0]} px"))
        for idx in (1, 2):
            self._panels[idx][0].delete("all")
            self._panels[idx][1].config(text="—")
        self._btn_process.config(state="normal")
        self._btn_save.config(state="disabled")
        self._btn_steps.config(state="disabled")
        self._set_status("● IMAGE LOADED", ACCENT)
        for n, l in self._step_labels:
            n.config(fg=TEXT_DIM); l.config(fg=TEXT_DIM)

    def _process(self):
        if self._orig_cv is None:
            return
        self._btn_process.config(state="disabled")
        self._btn_upload.config(state="disabled")
        self._btn_steps.config(state="disabled")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        import time
        for i in range(8):
            self.after(0, self._highlight_step, i)
            self._set_status(f"● STEP {i+1}/8 RUNNING…", "#ffdd00")
            time.sleep(0.05)
        self._set_status("● COMPUTING SOBEL (from scratch)…", "#ffdd00")
        edges, scan, ok, steps = process_image_hybrid(self._orig_cv)
        self._edges_cv = edges
        self._scan_cv  = scan
        self._success  = ok
        self._steps    = steps
        self.after(0, self._on_pipeline_done)

    def _on_pipeline_done(self):
        if self._edges_cv is not None:
            self._show_on_canvas(1, self._edges_cv,
                                 f"{self._edges_cv.shape[1]} × {self._edges_cv.shape[0]} px")
        if self._scan_cv is not None:
            self._show_on_canvas(2, self._scan_cv,
                                 f"{self._scan_cv.shape[1]} × {self._scan_cv.shape[0]} px")
        if self._success:
            self._set_status("● DONE — DOCUMENT DETECTED ✓", ACCENT)
        else:
            self._set_status("⚠  DONE — NO 4-CORNER DOCUMENT FOUND", "#ff6060")
        for n, l in self._step_labels:
            n.config(fg=ACCENT); l.config(fg=TEXT_LABEL)
        self._btn_process.config(state="normal")
        self._btn_upload.config(state="normal")
        if self._steps:
            self._btn_steps.config(state="normal", fg=BTN_ST_FG)
        if self._scan_cv is not None:
            self._btn_save.config(state="normal", fg=ACCENT)

    def _show_steps(self):
        if not self._steps:
            messagebox.showinfo("No Steps", "Process an image first.")
            return
        StepsWindow(self, self._steps)

    def _save(self):
        if self._scan_cv is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            title="Save Scanned Document"
        )
        if not path:
            return
        cv2.imwrite(path, self._scan_cv)
        self._set_status(f"● SAVED → {path.split('/')[-1]}", ACCENT2)
        messagebox.showinfo("Saved", f"Scan saved to:\n{path}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartScanner()
    app.mainloop()
