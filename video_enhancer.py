#!/usr/bin/env python3
"""
Video Enhancer - HDR/Super HDR + Color Enhancement
Full GUI Dashboard with progress, time estimation, and file size display.
Includes YouTube video download and copyright-free edit features.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import random
import subprocess
import tempfile
import cv2
import numpy as np

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


class VideoEnhancer:
    """Handles video enhancement processing with HDR and color enhancement."""

    def __init__(self):
        self.cancel_flag = False

    def _get_skin_mask(self, frame):
        """Detect skin-tone regions in the frame using HSV color range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Skin tone HSV range: H=0-25, S=40-170, V=80-255
        lower_skin = np.array([0, 40, 80], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # Smooth the mask to avoid hard edges
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return mask.astype(np.float32) / 255.0

    def enhance_frame_hdr(self, frame):
        """Apply Super HDR enhancement to a single frame with skin tone protection."""
        # Convert to float for processing
        img = frame.astype(np.float32) / 255.0

        # Tone mapping - expand dynamic range
        # Apply gamma correction for highlights and shadows separately
        shadows = np.power(img, 0.55)  # Slightly less aggressive shadow brightening
        highlights = np.power(img, 1.3)  # Slightly less compression on highlights

        # Blend based on luminance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])

        # HDR blend: shadows in dark areas, highlights in bright areas
        hdr_frame = shadows * (1 - gray_3ch) + highlights * gray_3ch

        # Apply overall brightness boost (gamma correction < 1 = brighter)
        hdr_frame = np.power(hdr_frame, 0.85)

        # Local contrast enhancement using CLAHE with reduced clip limit
        hdr_uint8 = np.clip(hdr_frame * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(hdr_uint8, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel (reduced clipLimit to protect skin tones)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Additional brightness boost on L channel
        l_enhanced = np.clip(l_enhanced.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        hdr_result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Protect skin tones: blend original frame back in skin regions
        skin_mask = self._get_skin_mask(frame)
        skin_mask_3ch = cv2.merge([skin_mask, skin_mask, skin_mask])
        # In skin regions, blend 60% original + 40% enhanced to preserve skin tones
        hdr_result = (skin_mask_3ch * (0.6 * frame + 0.4 * hdr_result) +
                      (1 - skin_mask_3ch) * hdr_result).astype(np.uint8)

        return hdr_result

    def enhance_frame_color(self, frame):
        """Apply color enhancement to a single frame with skin tone protection."""
        # Convert to HSV for saturation and brightness boost
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Create a mask for skin-tone hues (orange/red range H=0-25)
        # Apply less saturation boost in skin-tone hue range
        hue = hsv[:, :, 0]
        skin_hue_mask = (hue <= 25).astype(np.float32)

        # Boost saturation: 45% for non-skin, 20% for skin hues
        sat_boost_full = 1.45
        sat_boost_skin = 1.20
        sat_boost = sat_boost_full * (1 - skin_hue_mask) + sat_boost_skin * skin_hue_mask
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)

        # Vibrance increase (boost less saturated colors more)
        # Reduced for skin-tone hues
        saturation = hsv[:, :, 1] / 255.0
        vibrance_full = 1.0 + 0.4 * (1.0 - saturation)
        vibrance_skin = 1.0 + 0.15 * (1.0 - saturation)
        boost_factor = vibrance_full * (1 - skin_hue_mask) + vibrance_skin * skin_hue_mask
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost_factor, 0, 255)

        # Brightness boost via Value channel
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.12, 0, 255)

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

    def apply_strong_zoom(self, frame, frame_index, fps):
        """Apply strong zoom in/out effect - more aggressive than random zoom."""
        if not hasattr(self, '_strong_zoom_state'):
            self._strong_zoom_state = {
                'current_factor': 1.0,
                'next_change_frame': 0
            }

        state = self._strong_zoom_state

        # Check if it's time to pick a new strong zoom level
        if frame_index >= state['next_change_frame']:
            # Strong zoom factor between 1.0 and 1.2 (very noticeable)
            state['current_factor'] = random.uniform(1.0, 1.2)
            # Change more frequently: every 1-3 seconds
            interval_seconds = random.uniform(1.0, 3.0)
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

    def apply_copyright_free_edit(self, frame):
        """Apply subtle transformations to make video look different from original.

        Applies: horizontal flip, slight crop, minor hue shift,
        subtle brightness/contrast variation, and grain overlay.
        """
        h, w = frame.shape[:2]

        # 1. Horizontal flip (mirror)
        result = cv2.flip(frame, 1)

        # 2. Slight crop from edges (2-3% crop)
        crop_pct = 0.025
        x_crop = int(w * crop_pct)
        y_crop = int(h * crop_pct)
        result = result[y_crop:h - y_crop, x_crop:w - x_crop]
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. Minor color hue shift (5-10 degrees)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 7) % 180  # Shift hue by ~7 degrees
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 4. Slight brightness/contrast variation
        alpha = 1.03  # Slight contrast increase
        beta = 3      # Slight brightness increase
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        # 5. Add very subtle grain/noise overlay
        noise = np.random.normal(0, 3, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return result

    def enhance_frame(self, frame, hdr_enabled=True, color_enabled=True,
                      minor_zoom_enabled=False, random_zoom_enabled=False,
                      strong_zoom_enabled=False, copyright_free_enabled=False,
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

        if strong_zoom_enabled:
            result = self.apply_strong_zoom(result, frame_index, fps)

        if copyright_free_enabled:
            result = self.apply_copyright_free_edit(result)

        return result

    def _get_ffmpeg_path(self):
        """Get the path to the ffmpeg binary from imageio-ffmpeg."""
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            return get_ffmpeg_exe()
        except ImportError:
            # Fallback: try system ffmpeg
            return "ffmpeg"

    def _has_audio_stream(self, input_path):
        """Check if the input video has an audio stream."""
        ffmpeg_path = self._get_ffmpeg_path()
        try:
            result = subprocess.run(
                [ffmpeg_path, "-i", input_path],
                capture_output=True, text=True, timeout=10
            )
            # ffmpeg prints info to stderr
            return "Audio:" in result.stderr
        except Exception:
            return False

    def _merge_audio(self, video_path, audio_source_path, output_path):
        """Merge audio from audio_source into video_path, saving to output_path."""
        ffmpeg_path = self._get_ffmpeg_path()
        cmd = [
            ffmpeg_path,
            "-y",                      # Overwrite output
            "-i", video_path,          # Enhanced video (no audio)
            "-i", audio_source_path,   # Original video (audio source)
            "-c:v", "copy",            # Copy video stream as-is
            "-c:a", "aac",             # Encode audio as AAC
            "-map", "0:v:0",           # Use video from first input
            "-map", "1:a:0",           # Use audio from second input
            "-shortest",               # End when shortest stream ends
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Failed to merge audio: {result.stderr}")

    def process_video(self, input_path, output_path, hdr_enabled=True,
                      color_enabled=True, minor_zoom_enabled=False,
                      random_zoom_enabled=False, strong_zoom_enabled=False,
                      copyright_free_enabled=False,
                      progress_callback=None):
        """Process entire video file with enhancements, preserving original audio.

        When copyright_free_enabled is True, speed is changed to 1.05x via ffmpeg
        after frame processing.
        """
        self.cancel_flag = False
        # Reset zoom states for each new video
        if hasattr(self, '_random_zoom_state'):
            del self._random_zoom_state
        if hasattr(self, '_strong_zoom_state'):
            del self._strong_zoom_state

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

        # Check if original video has audio
        has_audio = self._has_audio_stream(input_path)

        # Write enhanced frames to a temp file first (OpenCV cannot copy audio)
        temp_dir = os.path.dirname(output_path) or "."
        temp_fd, temp_video_path = tempfile.mkstemp(suffix=".avi", dir=temp_dir)
        os.close(temp_fd)

        # Setup writer - Use XVID codec with AVI container for frame writing
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            os.remove(temp_video_path)
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
                                              strong_zoom_enabled, copyright_free_enabled,
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
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        # Merge audio from original video into the enhanced video
        if has_audio:
            try:
                self._merge_audio(temp_video_path, input_path, output_path)
            except Exception as e:
                # If audio merge fails, fall back to video-only output
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_video_path, output_path)
                temp_video_path = None
            finally:
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        else:
            # No audio in original, just rename temp to output
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_video_path, output_path)

        # Apply 1.05x speed change for copyright-free edit
        if copyright_free_enabled and os.path.exists(output_path):
            try:
                ffmpeg_path = self._get_ffmpeg_path()
                temp_speed_fd, temp_speed_path = tempfile.mkstemp(suffix=".mp4",
                                                                   dir=os.path.dirname(output_path) or ".")
                os.close(temp_speed_fd)
                cmd = [
                    ffmpeg_path, "-y",
                    "-i", output_path,
                    "-filter_complex",
                    "[0:v]setpts=0.9524*PTS[v];[0:a]atempo=1.05[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac",
                    temp_speed_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    os.replace(temp_speed_path, output_path)
                else:
                    # If speed change fails (e.g. no audio), try video only
                    cmd_no_audio = [
                        ffmpeg_path, "-y",
                        "-i", output_path,
                        "-filter:v", "setpts=0.9524*PTS",
                        "-an",
                        "-c:v", "libx264", "-preset", "fast",
                        temp_speed_path
                    ]
                    result2 = subprocess.run(cmd_no_audio, capture_output=True, text=True, timeout=600)
                    if result2.returncode == 0:
                        os.replace(temp_speed_path, output_path)
                    else:
                        if os.path.exists(temp_speed_path):
                            os.remove(temp_speed_path)
            except Exception:
                if 'temp_speed_path' in locals() and os.path.exists(temp_speed_path):
                    os.remove(temp_speed_path)

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
        self.root.geometry("700x900")
        self.root.resizable(True, True)
        self.root.minsize(700, 900)
        self.root.configure(bg="#1a1a2e")

        self.enhancer = VideoEnhancer()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.yt_url = tk.StringVar()
        self.yt_format = tk.StringVar(value="16:9")
        self.processing = False
        self.downloading = False

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
        """Create all GUI widgets with scrollable canvas."""
        # Create a canvas with scrollbar for the entire content
        container = ttk.Frame(self.root, style='Dark.TFrame')
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg='#1a1a2e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL,
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set,
                              yscrollincrement=20)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollable inner frame
        main_frame = ttk.Frame(self.canvas, style='Dark.TFrame')
        self.canvas_window = self.canvas.create_window((0, 0), window=main_frame,
                                                       anchor='nw')

        # Update scroll region when frame size changes
        def _on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        main_frame.bind('<Configure>', _on_frame_configure)
        self.canvas.bind('<Configure>', _on_canvas_configure)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            # On Windows, event.delta is typically 120 or -120
            if event.delta > 0:
                self.canvas.yview_scroll(-3, "units")
            else:
                self.canvas.yview_scroll(3, "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                self.canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(3, "units")

        # Bind mousewheel only when mouse is over the canvas area
        def _bind_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        self.canvas.bind("<Enter>", _bind_mousewheel)
        self.canvas.bind("<Leave>", _unbind_mousewheel)

        # Title
        title_label = ttk.Label(main_frame,
                                text="Video Enhancer",
                                style='Title.TLabel')
        title_label.pack(pady=(10, 3), padx=20)

        subtitle = ttk.Label(main_frame,
                             text="Super HDR + Color Enhancement",
                             style='Status.TLabel')
        subtitle.pack(pady=(0, 10), padx=20)

        # YouTube / Instagram Download Section
        yt_frame = ttk.Frame(main_frame, style='Card.TFrame')
        yt_frame.pack(fill=tk.X, pady=3, ipady=5, ipadx=10, padx=20)

        ttk.Label(yt_frame, text="YouTube / Instagram Download:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(5, 2))

        yt_url_row = ttk.Frame(yt_frame, style='Card.TFrame')
        yt_url_row.pack(fill=tk.X, padx=10, pady=(0, 3))

        self.yt_url_entry = tk.Entry(yt_url_row, textvariable=self.yt_url,
                                     font=('Helvetica', 9),
                                     bg='#0f3460', fg='#a0a0a0',
                                     insertbackground='#ffffff',
                                     relief='flat')
        self.yt_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self.yt_url_entry.insert(0, "Paste YouTube or Instagram URL here")

        yt_options_row = ttk.Frame(yt_frame, style='Card.TFrame')
        yt_options_row.pack(fill=tk.X, padx=10, pady=(0, 3))

        yt_format_16_9 = tk.Radiobutton(yt_options_row, text="16:9 (Landscape)",
                                        variable=self.yt_format, value="16:9",
                                        bg='#16213e', fg='#ffffff',
                                        selectcolor='#0f3460',
                                        activebackground='#16213e',
                                        activeforeground='#ffffff',
                                        font=('Helvetica', 10))
        yt_format_16_9.pack(side=tk.LEFT, padx=(0, 20))

        yt_format_9_16 = tk.Radiobutton(yt_options_row, text="Short (9:16)",
                                        variable=self.yt_format, value="9:16",
                                        bg='#16213e', fg='#ffffff',
                                        selectcolor='#0f3460',
                                        activebackground='#16213e',
                                        activeforeground='#ffffff',
                                        font=('Helvetica', 10))
        yt_format_9_16.pack(side=tk.LEFT)

        yt_btn_row = ttk.Frame(yt_frame, style='Card.TFrame')
        yt_btn_row.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.yt_download_btn = ttk.Button(yt_btn_row, text="Download",
                                          style='Custom.TButton',
                                          command=self._download_youtube)
        self.yt_download_btn.pack(side=tk.LEFT)

        self.yt_status_label = ttk.Label(yt_btn_row, text="",
                                         style='Info.TLabel')
        self.yt_status_label.pack(side=tk.LEFT, padx=(15, 0))

        # Input File Section (compact)
        input_frame = ttk.Frame(main_frame, style='Card.TFrame')
        input_frame.pack(fill=tk.X, pady=3, ipady=4, ipadx=10, padx=20)

        ttk.Label(input_frame, text="Input Video:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(4, 2))

        input_row = ttk.Frame(input_frame, style='Card.TFrame')
        input_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.input_entry = tk.Entry(input_row, textvariable=self.input_path,
                                    font=('Helvetica', 9),
                                    bg='#0f3460', fg='#ffffff',
                                    insertbackground='#ffffff',
                                    relief='flat')
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        browse_btn = ttk.Button(input_row, text="Browse",
                                style='Custom.TButton',
                                command=self._browse_input)
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Output File Section (compact)
        output_frame = ttk.Frame(main_frame, style='Card.TFrame')
        output_frame.pack(fill=tk.X, pady=3, ipady=4, ipadx=10, padx=20)

        ttk.Label(output_frame, text="Output Video:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(4, 2))

        output_row = ttk.Frame(output_frame, style='Card.TFrame')
        output_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.output_entry = tk.Entry(output_row, textvariable=self.output_path,
                                     font=('Helvetica', 9),
                                     bg='#0f3460', fg='#ffffff',
                                     insertbackground='#ffffff',
                                     relief='flat')
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        browse_out_btn = ttk.Button(output_row, text="Browse",
                                    style='Custom.TButton',
                                    command=self._browse_output)
        browse_out_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Enhancement Options
        options_frame = ttk.Frame(main_frame, style='Card.TFrame')
        options_frame.pack(fill=tk.X, pady=3, ipady=5, ipadx=10, padx=20)

        ttk.Label(options_frame, text="Enhancement Options:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(5, 3))

        self.hdr_var = tk.BooleanVar(value=True)
        self.color_var = tk.BooleanVar(value=True)
        self.minor_zoom_var = tk.BooleanVar(value=False)
        self.random_zoom_var = tk.BooleanVar(value=False)
        self.strong_zoom_var = tk.BooleanVar(value=False)
        self.copyright_free_var = tk.BooleanVar(value=False)

        checks_row = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row.pack(fill=tk.X, padx=10, pady=(0, 2))

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
        checks_row2.pack(fill=tk.X, padx=10, pady=(0, 2))

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

        checks_row3 = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row3.pack(fill=tk.X, padx=10, pady=(0, 2))

        strong_zoom_check = tk.Checkbutton(checks_row3, text="Strong Zoom In/Out",
                                           variable=self.strong_zoom_var,
                                           bg='#16213e', fg='#ffffff',
                                           selectcolor='#0f3460',
                                           activebackground='#16213e',
                                           activeforeground='#ffffff',
                                           font=('Helvetica', 10))
        strong_zoom_check.pack(side=tk.LEFT)

        checks_row4 = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row4.pack(fill=tk.X, padx=10, pady=(0, 5))

        copyright_free_check = tk.Checkbutton(checks_row4, text="Copyright Free Edit",
                                              variable=self.copyright_free_var,
                                              bg='#16213e', fg='#ffffff',
                                              selectcolor='#0f3460',
                                              activebackground='#16213e',
                                              activeforeground='#ffffff',
                                              font=('Helvetica', 10))
        copyright_free_check.pack(side=tk.LEFT)

        # Info Dashboard
        dashboard_frame = ttk.Frame(main_frame, style='Card.TFrame')
        dashboard_frame.pack(fill=tk.X, pady=3, ipady=5, ipadx=10, padx=20)

        ttk.Label(dashboard_frame, text="Dashboard:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(5, 3))

        info_grid = ttk.Frame(dashboard_frame, style='Card.TFrame')
        info_grid.pack(fill=tk.X, padx=10, pady=(0, 5))

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
        progress_frame.pack(fill=tk.X, pady=3, ipady=5, ipadx=10, padx=20)

        ttk.Label(progress_frame, text="Progress:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=(5, 3))

        # Progress bar row with Cancel and Enter buttons beside it
        progress_bar_row = ttk.Frame(progress_frame, style='Card.TFrame')
        progress_bar_row.pack(fill=tk.X, padx=10, pady=(0, 3))

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
        progress_info.pack(fill=tk.X, padx=10, pady=(0, 5))

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
        credit_label.pack(side=tk.BOTTOM, pady=(10, 10))

        # Ensure scroll region is set after all widgets are created
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

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

            # Auto-set output path (use .mp4 for best audio/video compatibility)
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

    def _download_youtube(self):
        """Start YouTube video download in a background thread."""
        if self.downloading:
            return

        url = self.yt_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Please paste a YouTube URL.")
            return

        if yt_dlp is None:
            messagebox.showerror("Error",
                                 "yt-dlp is not installed. Run: pip install yt-dlp")
            return

        # Ask user where to save the downloaded video
        save_dir = filedialog.askdirectory(title="Select Download Folder")
        if not save_dir:
            return

        self.downloading = True
        self.yt_download_btn.config(state=tk.DISABLED)
        self.yt_status_label.config(text="Downloading...")

        thread = threading.Thread(target=self._download_youtube_thread,
                                  args=(url, save_dir),
                                  daemon=True)
        thread.start()

    def _download_youtube_thread(self, url, save_dir):
        """Background thread for downloading YouTube video."""
        try:
            video_format = self.yt_format.get()
            output_template = os.path.join(save_dir, '%(title)s.%(ext)s')

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_template,
                'merge_output_format': 'mp4',
            }

            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                # Ensure .mp4 extension
                if not filename.endswith('.mp4'):
                    base = os.path.splitext(filename)[0]
                    filename = base + '.mp4'

            # If Short (9:16) format selected, crop to 9:16 aspect ratio
            if video_format == "9:16" and os.path.exists(filename):
                self.root.after(0, lambda: self.yt_status_label.config(
                    text="Cropping to 9:16..."))
                cropped_path = self._crop_to_9_16(filename)
                if cropped_path:
                    filename = cropped_path

            self.root.after(0, self._on_download_complete, filename)

        except Exception as e:
            self.root.after(0, self._on_download_error, str(e))

    def _crop_to_9_16(self, video_path):
        """Crop a video to 9:16 aspect ratio (center crop) using ffmpeg."""
        try:
            ffmpeg_path = self.enhancer._get_ffmpeg_path()
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_short{ext}"

            # Use ffmpeg crop filter: crop to 9:16 from center
            # crop=ih*9/16:ih (width = height * 9/16, height stays same)
            cmd = [
                ffmpeg_path, "-y",
                "-i", video_path,
                "-vf", "crop=ih*9/16:ih",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                # Remove original and rename
                os.remove(video_path)
                os.rename(output_path, video_path)
                return video_path
            else:
                # If crop fails, keep original
                if os.path.exists(output_path):
                    os.remove(output_path)
                return video_path
        except Exception:
            return video_path

    def _on_download_complete(self, filename):
        """Called when YouTube download completes."""
        self.downloading = False
        self.yt_download_btn.config(state=tk.NORMAL)
        self.yt_status_label.config(text="Download complete!")

        # Auto-set the downloaded file as input
        if filename and os.path.exists(filename):
            self.input_path.set(filename)
            self._update_file_info(filename)
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_enhanced.mp4")

        messagebox.showinfo("Success", f"Video downloaded:\n{filename}")

    def _on_download_error(self, error_msg):
        """Called when YouTube download fails."""
        self.downloading = False
        self.yt_download_btn.config(state=tk.NORMAL)
        self.yt_status_label.config(text="Download failed")
        messagebox.showerror("Download Error", f"Failed to download:\n{error_msg}")

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
                and not self.minor_zoom_var.get() and not self.random_zoom_var.get() \
                and not self.strong_zoom_var.get() and not self.copyright_free_var.get():
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
                strong_zoom_enabled=self.strong_zoom_var.get(),
                copyright_free_enabled=self.copyright_free_var.get(),
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
