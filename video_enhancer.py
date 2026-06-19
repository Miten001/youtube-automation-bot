#!/usr/bin/env python3
"""
Video Enhancer - HDR/Super HDR + Color Enhancement
Full GUI Dashboard with progress, time estimation, and file size display.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import random
import cv2
import numpy as np


class VideoEnhancer:
    """Handles video enhancement processing with HDR and color enhancement."""

    def __init__(self):
        self.cancel_flag = False

    def enhance_frame_hdr(self, frame):
        """Apply Super HDR enhancement to a single frame."""
        # Convert to float for processing
        img = frame.astype(np.float32) / 255.0

        # Tone mapping - expand dynamic range
        # Apply gamma correction for highlights and shadows separately
        shadows = np.power(img, 0.6)  # Brighten shadows
        highlights = np.power(img, 1.4)  # Control highlights

        # Blend based on luminance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])

        # HDR blend: shadows in dark areas, highlights in bright areas
        hdr_frame = shadows * (1 - gray_3ch) + highlights * gray_3ch

        # Local contrast enhancement using CLAHE
        hdr_uint8 = np.clip(hdr_frame * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(hdr_uint8, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel for local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        hdr_result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return hdr_result

    def enhance_frame_color(self, frame):
        """Apply color enhancement to a single frame."""
        # Convert to HSV for saturation boost
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Boost saturation by 30%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)

        # Slight vibrance increase (boost less saturated colors more)
        saturation = hsv[:, :, 1] / 255.0
        boost_factor = 1.0 + 0.3 * (1.0 - saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost_factor, 0, 255)

        hsv = hsv.astype(np.uint8)
        color_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply subtle sharpening
        kernel = np.array([[-0.5, -0.5, -0.5],
                           [-0.5,  5.0, -0.5],
                           [-0.5, -0.5, -0.5]]) / 1.0
        sharpened = cv2.filter2D(color_enhanced, -1, kernel)

        # Blend original with sharpened (subtle effect)
        result = cv2.addWeighted(color_enhanced, 0.7, sharpened, 0.3, 0)

        return result

    def apply_minor_zoom(self, frame, zoom_factor=1.07):
        """Apply a slight zoom-in effect by cropping from center and resizing back."""
        h, w = frame.shape[:2]
        # Calculate crop dimensions
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        x_start = (w - new_w) // 2
        y_start = (h - new_h) // 2

        cropped = frame[y_start:y_start + new_h, x_start:x_start + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def apply_random_zoom(self, frame, frame_index, fps):
        """Apply random zoom in/out effect that changes every few seconds."""
        if not hasattr(self, '_random_zoom_state'):
            self._random_zoom_state = {
                'current_factor': 1.0,
                'next_change_frame': 0
            }

        state = self._random_zoom_state

        # Check if it's time to pick a new random zoom level
        if frame_index >= state['next_change_frame']:
            # Random zoom factor between 1.0 and 1.08 (subtle effect)
            state['current_factor'] = random.uniform(1.0, 1.08)
            # Change again after a random interval (2-5 seconds)
            interval_seconds = random.uniform(2.0, 5.0)
            state['next_change_frame'] = frame_index + int(interval_seconds * fps)

        zoom_factor = state['current_factor']

        if zoom_factor <= 1.001:
            return frame

        h, w = frame.shape[:2]
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        x_start = (w - new_w) // 2
        y_start = (h - new_h) // 2

        cropped = frame[y_start:y_start + new_h, x_start:x_start + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def enhance_frame(self, frame, hdr_enabled=True, color_enabled=True,
                      minor_zoom_enabled=False, random_zoom_enabled=False,
                      frame_index=0, fps=30.0):
        """Apply all enabled enhancements to a frame."""
        result = frame.copy()

        if hdr_enabled:
            result = self.enhance_frame_hdr(result)

        if color_enabled:
            result = self.enhance_frame_color(result)

        if minor_zoom_enabled:
            result = self.apply_minor_zoom(result)

        if random_zoom_enabled:
            result = self.apply_random_zoom(result, frame_index, fps)

        return result

    def process_video(self, input_path, output_path, hdr_enabled=True,
                      color_enabled=True, minor_zoom_enabled=False,
                      random_zoom_enabled=False, progress_callback=None):
        """Process entire video file with enhancements."""
        self.cancel_flag = False
        # Reset random zoom state for each new video
        if hasattr(self, '_random_zoom_state'):
            del self._random_zoom_state

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            raise ValueError("Cannot determine video frame count")

        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise ValueError("Cannot create output video file")

        start_time = time.time()
        processed = 0

        try:
            while True:
                if self.cancel_flag:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Enhance frame
                enhanced = self.enhance_frame(frame, hdr_enabled, color_enabled,
                                              minor_zoom_enabled, random_zoom_enabled,
                                              frame_index=processed, fps=fps)
                out.write(enhanced)

                processed += 1

                # Report progress
                if progress_callback and processed % 5 == 0:
                    elapsed = time.time() - start_time
                    progress_pct = processed / total_frames
                    if progress_pct > 0:
                        estimated_total = elapsed / progress_pct
                        remaining = estimated_total - elapsed
                    else:
                        remaining = 0

                    progress_callback(processed, total_frames, remaining)

        finally:
            cap.release()
            out.release()

        if self.cancel_flag:
            # Clean up partial output
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        # Final progress update
        if progress_callback:
            progress_callback(total_frames, total_frames, 0)

        return True

    def cancel(self):
        """Cancel the current processing."""
        self.cancel_flag = True


class VideoEnhancerGUI:
    """GUI Dashboard for the Video Enhancer."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Enhancer - Super HDR & Color Enhancement | made by @codex_here")
        self.root.geometry("700x750")
        self.root.resizable(True, True)
        self.root.minsize(700, 750)
        self.root.configure(bg="#1a1a2e")

        self.enhancer = VideoEnhancer()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.processing = False

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Title.TLabel',
                        background='#1a1a2e',
                        foreground='#00d4ff',
                        font=('Helvetica', 18, 'bold'))

        style.configure('Header.TLabel',
                        background='#1a1a2e',
                        foreground='#ffffff',
                        font=('Helvetica', 11, 'bold'))

        style.configure('Info.TLabel',
                        background='#16213e',
                        foreground='#a0a0a0',
                        font=('Helvetica', 10))

        style.configure('Status.TLabel',
                        background='#1a1a2e',
                        foreground='#00ff88',
                        font=('Helvetica', 10))

        style.configure('Dark.TFrame',
                        background='#1a1a2e')

        style.configure('Card.TFrame',
                        background='#16213e',
                        relief='flat')

        style.configure('Custom.TButton',
                        background='#0f3460',
                        foreground='#ffffff',
                        font=('Helvetica', 10, 'bold'),
                        padding=(15, 8))

        style.map('Custom.TButton',
                  background=[('active', '#1a5276')])

        style.configure('Cancel.TButton',
                        background='#e74c3c',
                        foreground='#ffffff',
                        font=('Helvetica', 10, 'bold'),
                        padding=(15, 8))

        style.map('Cancel.TButton',
                  background=[('active', '#c0392b')])

        style.configure('Green.Horizontal.TProgressbar',
                        troughcolor='#16213e',
                        background='#00ff88',
                        thickness=25)

    def _create_widgets(self):
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Title
        title_label = ttk.Label(main_frame,
                                text="Video Enhancer",
                                style='Title.TLabel')
        title_label.pack(pady=(0, 5))

        subtitle = ttk.Label(main_frame,
                             text="Super HDR + Color Enhancement",
                             style='Status.TLabel')
        subtitle.pack(pady=(0, 15))

        # Input File Section
        input_frame = ttk.Frame(main_frame, style='Card.TFrame')
        input_frame.pack(fill=tk.X, pady=5, ipady=8, ipadx=10)

        ttk.Label(input_frame, text="Input Video:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(8, 2))

        input_row = ttk.Frame(input_frame, style='Card.TFrame')
        input_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.input_entry = tk.Entry(input_row, textvariable=self.input_path,
                                    font=('Helvetica', 9),
                                    bg='#0f3460', fg='#ffffff',
                                    insertbackground='#ffffff',
                                    relief='flat')
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        browse_btn = ttk.Button(input_row, text="Browse",
                                style='Custom.TButton',
                                command=self._browse_input)
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Output File Section
        output_frame = ttk.Frame(main_frame, style='Card.TFrame')
        output_frame.pack(fill=tk.X, pady=5, ipady=8, ipadx=10)

        ttk.Label(output_frame, text="Output Video:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(8, 2))

        output_row = ttk.Frame(output_frame, style='Card.TFrame')
        output_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.output_entry = tk.Entry(output_row, textvariable=self.output_path,
                                     font=('Helvetica', 9),
                                     bg='#0f3460', fg='#ffffff',
                                     insertbackground='#ffffff',
                                     relief='flat')
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        browse_out_btn = ttk.Button(output_row, text="Browse",
                                    style='Custom.TButton',
                                    command=self._browse_output)
        browse_out_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Enhancement Options
        options_frame = ttk.Frame(main_frame, style='Card.TFrame')
        options_frame.pack(fill=tk.X, pady=5, ipady=8, ipadx=10)

        ttk.Label(options_frame, text="Enhancement Options:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(8, 5))

        self.hdr_var = tk.BooleanVar(value=True)
        self.color_var = tk.BooleanVar(value=True)
        self.minor_zoom_var = tk.BooleanVar(value=False)
        self.random_zoom_var = tk.BooleanVar(value=False)

        checks_row = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        hdr_check = tk.Checkbutton(checks_row, text="Super HDR Enhancement",
                                   variable=self.hdr_var,
                                   bg='#16213e', fg='#ffffff',
                                   selectcolor='#0f3460',
                                   activebackground='#16213e',
                                   activeforeground='#ffffff',
                                   font=('Helvetica', 10))
        hdr_check.pack(side=tk.LEFT, padx=(0, 30))

        color_check = tk.Checkbutton(checks_row, text="Color Enhancement",
                                     variable=self.color_var,
                                     bg='#16213e', fg='#ffffff',
                                     selectcolor='#0f3460',
                                     activebackground='#16213e',
                                     activeforeground='#ffffff',
                                     font=('Helvetica', 10))
        color_check.pack(side=tk.LEFT)

        checks_row2 = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row2.pack(fill=tk.X, padx=10, pady=(0, 8))

        minor_zoom_check = tk.Checkbutton(checks_row2, text="Minor Zoom",
                                          variable=self.minor_zoom_var,
                                          bg='#16213e', fg='#ffffff',
                                          selectcolor='#0f3460',
                                          activebackground='#16213e',
                                          activeforeground='#ffffff',
                                          font=('Helvetica', 10))
        minor_zoom_check.pack(side=tk.LEFT, padx=(0, 30))

        random_zoom_check = tk.Checkbutton(checks_row2, text="Random Zoom In/Out",
                                           variable=self.random_zoom_var,
                                           bg='#16213e', fg='#ffffff',
                                           selectcolor='#0f3460',
                                           activebackground='#16213e',
                                           activeforeground='#ffffff',
                                           font=('Helvetica', 10))
        random_zoom_check.pack(side=tk.LEFT)

        # Info Dashboard
        dashboard_frame = ttk.Frame(main_frame, style='Card.TFrame')
        dashboard_frame.pack(fill=tk.X, pady=5, ipady=8, ipadx=10)

        ttk.Label(dashboard_frame, text="Dashboard:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(8, 5))

        info_grid = ttk.Frame(dashboard_frame, style='Card.TFrame')
        info_grid.pack(fill=tk.X, padx=10, pady=(0, 8))

        # File size
        ttk.Label(info_grid, text="File Size:",
                  style='Info.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.size_label = ttk.Label(info_grid, text="-- MB",
                                    style='Info.TLabel')
        self.size_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 30), pady=2)

        # Duration
        ttk.Label(info_grid, text="Duration:",
                  style='Info.TLabel').grid(row=0, column=2, sticky=tk.W, pady=2)
        self.duration_label = ttk.Label(info_grid, text="--:--",
                                        style='Info.TLabel')
        self.duration_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=2)

        # Resolution
        ttk.Label(info_grid, text="Resolution:",
                  style='Info.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.resolution_label = ttk.Label(info_grid, text="-- x --",
                                          style='Info.TLabel')
        self.resolution_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 30), pady=2)

        # FPS
        ttk.Label(info_grid, text="FPS:",
                  style='Info.TLabel').grid(row=1, column=2, sticky=tk.W, pady=2)
        self.fps_label = ttk.Label(info_grid, text="--",
                                   style='Info.TLabel')
        self.fps_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0), pady=2)

        # Progress Section
        progress_frame = ttk.Frame(main_frame, style='Card.TFrame')
        progress_frame.pack(fill=tk.X, pady=5, ipady=8, ipadx=10)

        ttk.Label(progress_frame, text="Progress:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(8, 5))

        # Progress bar row with Cancel and Enter buttons beside it
        progress_bar_row = ttk.Frame(progress_frame, style='Card.TFrame')
        progress_bar_row.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.progress_bar = ttk.Progressbar(progress_bar_row,
                                            style='Green.Horizontal.TProgressbar',
                                            mode='determinate',
                                            maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.cancel_btn = tk.Button(progress_bar_row, text="Cancel",
                                    command=self._cancel_processing,
                                    bg='#e74c3c', fg='#ffffff',
                                    activebackground='#c0392b',
                                    activeforeground='#ffffff',
                                    font=('Helvetica', 10, 'bold'),
                                    relief='raised', bd=2,
                                    state=tk.DISABLED,
                                    padx=10, pady=4)
        self.cancel_btn.pack(side=tk.LEFT, padx=(10, 5))

        enter_label = tk.Label(progress_bar_row, text="Enter",
                               bg='#0f3460', fg='#00ff88',
                               font=('Helvetica', 10, 'bold'),
                               padx=10, pady=4,
                               relief='raised', bd=2)
        enter_label.pack(side=tk.LEFT, padx=(5, 0))

        progress_info = ttk.Frame(progress_frame, style='Card.TFrame')
        progress_info.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.progress_label = ttk.Label(progress_info,
                                        text="Ready - Press Enter to Start",
                                        style='Info.TLabel')
        self.progress_label.pack(side=tk.LEFT)

        self.time_label = ttk.Label(progress_info,
                                    text="Time Remaining: --:--",
                                    style='Info.TLabel')
        self.time_label.pack(side=tk.RIGHT)

        # Credit Label - at the very bottom, small font
        credit_label = tk.Label(main_frame, text="made by @codex_here",
                                bg='#1a1a2e', fg='#00d4ff',
                                font=('Helvetica', 8, 'italic'))
        credit_label.pack(side=tk.BOTTOM, pady=(20, 0))

        # Bind Enter key to start processing
        self.root.bind('<Return>', lambda event: self._start_processing())

    def _browse_input(self):
        """Open file dialog for input video."""
        filetypes = [
            ("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("All Files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Input Video",
                                          filetypes=filetypes)
        if path:
            self.input_path.set(path)
            self._update_file_info(path)

            # Auto-set output path
            base, ext = os.path.splitext(path)
            self.output_path.set(f"{base}_enhanced.mp4")

    def _browse_output(self):
        """Open file dialog for output video."""
        filetypes = [
            ("MP4 Video", "*.mp4"),
            ("AVI Video", "*.avi"),
            ("All Files", "*.*")
        ]
        path = filedialog.asksaveasfilename(title="Save Enhanced Video",
                                            filetypes=filetypes,
                                            defaultextension=".mp4")
        if path:
            self.output_path.set(path)

    def _update_file_info(self, path):
        """Update dashboard with video file information."""
        try:
            # File size
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            self.size_label.config(text=f"{size_mb:.2f} MB")

            # Video info
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                duration_sec = frame_count / fps if fps > 0 else 0
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)

                self.resolution_label.config(text=f"{width} x {height}")
                self.fps_label.config(text=f"{fps:.1f}")
                self.duration_label.config(text=f"{minutes:02d}:{seconds:02d}")

                cap.release()
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not read video info: {e}")

    def _start_processing(self):
        """Start the enhancement process in a background thread."""
        if self.processing:
            return

        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()

        if not input_path:
            messagebox.showerror("Error", "Please select an input video file.")
            return

        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input file does not exist.")
            return

        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        if not self.hdr_var.get() and not self.color_var.get() \
                and not self.minor_zoom_var.get() and not self.random_zoom_var.get():
            messagebox.showwarning("Warning",
                                   "Please enable at least one enhancement option.")
            return

        self.processing = True
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Processing...")

        # Run processing in background thread
        thread = threading.Thread(target=self._process_thread,
                                  args=(input_path, output_path),
                                  daemon=True)
        thread.start()

    def _process_thread(self, input_path, output_path):
        """Background thread for video processing."""
        try:
            success = self.enhancer.process_video(
                input_path, output_path,
                hdr_enabled=self.hdr_var.get(),
                color_enabled=self.color_var.get(),
                minor_zoom_enabled=self.minor_zoom_var.get(),
                random_zoom_enabled=self.random_zoom_var.get(),
                progress_callback=self._on_progress
            )

            self.root.after(0, self._on_complete, success)

        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _on_progress(self, current, total, remaining_sec):
        """Callback for progress updates (called from worker thread)."""
        self.root.after(0, self._update_progress, current, total, remaining_sec)

    def _update_progress(self, current, total, remaining_sec):
        """Update progress UI elements (called on main thread)."""
        if total > 0:
            pct = (current / total) * 100
            self.progress_bar['value'] = pct
            self.progress_label.config(text=f"Frame {current}/{total} ({pct:.1f}%)")

            minutes = int(remaining_sec // 60)
            seconds = int(remaining_sec % 60)
            self.time_label.config(text=f"Time Remaining: {minutes:02d}:{seconds:02d}")

    def _on_complete(self, success):
        """Called when processing completes."""
        self.processing = False
        self.cancel_btn.config(state=tk.DISABLED)

        if success:
            self.progress_bar['value'] = 100
            self.progress_label.config(text="Enhancement Complete!")
            self.time_label.config(text="Time Remaining: 00:00")

            # Show output file size
            output_path = self.output_path.get().strip()
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                messagebox.showinfo(
                    "Success",
                    f"Video enhanced successfully!\n"
                    f"Output: {output_path}\n"
                    f"Size: {size_mb:.2f} MB"
                )
        else:
            self.progress_label.config(text="Cancelled")
            self.progress_bar['value'] = 0
            self.time_label.config(text="Time Remaining: --:--")

    def _on_error(self, error_msg):
        """Called when processing encounters an error."""
        self.processing = False
        self.cancel_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Error occurred")
        self.progress_bar['value'] = 0
        messagebox.showerror("Error", f"Processing failed:\n{error_msg}")

    def _cancel_processing(self):
        """Cancel the current processing."""
        if self.processing:
            self.enhancer.cancel()
            self.progress_label.config(text="Cancelling...")

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Entry point for the Video Enhancer application."""
    app = VideoEnhancerGUI()
    app.run()


if __name__ == "__main__":
    main()
