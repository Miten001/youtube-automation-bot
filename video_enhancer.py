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

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class VideoEnhancer:
    """Handles video enhancement processing with HDR and color enhancement."""

    def __init__(self):
        self.cancel_flag = False
        # Cache the face cascade classifier for reuse across frames
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Face detection frame-skip state
        self._face_detect_frame_count = 0
        self._face_detect_last_region = None

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
                      cinematic_bars_enabled=False, face_zoom_enabled=False,
                      color_preset="None",
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

        if face_zoom_enabled:
            result = self.apply_face_detect_zoom(result)

        if cinematic_bars_enabled:
            result = self.apply_cinematic_bars(result)

        if color_preset and color_preset != "None":
            result = self.apply_color_preset(result, color_preset)

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
                      cinematic_bars_enabled=False, face_zoom_enabled=False,
                      color_preset="None",
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
        # Reset face detection frame-skip state
        self._face_detect_frame_count = 0
        self._face_detect_last_region = None

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
                                              cinematic_bars_enabled, face_zoom_enabled,
                                              color_preset,
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

    # ========== NEW FEATURE METHODS ==========

    def apply_slow_motion(self, input_path, output_path):
        """Apply 0.5x slow motion using ffmpeg setpts filter."""
        ffmpeg_path = self._get_ffmpeg_path()
        has_audio = self._has_audio_stream(input_path)
        if has_audio:
            cmd = [
                ffmpeg_path, "-y",
                "-i", input_path,
                "-filter_complex",
                "[0:v]setpts=2.0*PTS[v];[0:a]atempo=0.5[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                output_path
            ]
        else:
            cmd = [
                ffmpeg_path, "-y",
                "-i", input_path,
                "-filter:v", "setpts=2.0*PTS",
                "-an",
                "-c:v", "libx264", "-preset", "fast",
                output_path
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Slow motion failed: {result.stderr}")

    def apply_voice_enhancement(self, input_path, output_path):
        """Enhance voice clarity using ffmpeg audio filters."""
        ffmpeg_path = self._get_ffmpeg_path()
        audio_filter = "highpass=f=200,lowpass=f=3000,compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-45|-27/-25|0/-10|20/-7,volume=1.5"
        cmd = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Voice enhancement failed: {result.stderr}")

    def apply_remove_background_noise(self, input_path, output_path):
        """Remove background noise using ffmpeg afftdn filter."""
        ffmpeg_path = self._get_ffmpeg_path()
        cmd = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-af", "afftdn=nf=-25",
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Remove background noise failed: {result.stderr}")

    def apply_voice_enhance_and_denoise(self, input_path, output_path):
        """Apply voice enhancement and noise removal in a single ffmpeg pass."""
        ffmpeg_path = self._get_ffmpeg_path()
        # Combine noise removal (afftdn) with voice enhancement filters in one chain
        audio_filter = (
            "afftdn=nf=-25,"
            "highpass=f=200,lowpass=f=3000,"
            "compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-45|-27/-25|0/-10|20/-7,"
            "volume=1.5"
        )
        cmd = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Voice enhancement + noise removal failed: {result.stderr}")

    def apply_auto_subtitles(self, input_path, output_path):
        """Extract audio, perform speech-to-text, generate SRT, burn subtitles."""
        if sr is None:
            raise ValueError("speech_recognition is not installed. Run: pip install SpeechRecognition")

        ffmpeg_path = self._get_ffmpeg_path()
        temp_dir = os.path.dirname(output_path) or "."

        # Extract audio to WAV
        temp_audio_fd, temp_audio = tempfile.mkstemp(suffix=".wav", dir=temp_dir)
        os.close(temp_audio_fd)
        try:
            cmd = [
                ffmpeg_path, "-y",
                "-i", input_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                temp_audio
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise ValueError(f"Audio extraction failed: {result.stderr}")

            # Speech-to-text using speech_recognition
            recognizer = sr.Recognizer()
            srt_entries = []
            chunk_duration = 5  # seconds per chunk

            with sr.AudioFile(temp_audio) as source:
                audio_duration = source.DURATION
                idx = 1
                offset = 0.0
                while offset < audio_duration:
                    duration = min(chunk_duration, audio_duration - offset)
                    audio_data = recognizer.record(source, duration=duration)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        start_t = offset
                        end_t = offset + duration
                        start_str = self._srt_time(start_t)
                        end_str = self._srt_time(end_t)
                        srt_entries.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
                        idx += 1
                    except (sr.UnknownValueError, sr.RequestError):
                        pass
                    offset += duration

            # Write SRT file
            temp_srt_fd, temp_srt = tempfile.mkstemp(suffix=".srt", dir=temp_dir)
            os.close(temp_srt_fd)
            try:
                with open(temp_srt, 'w', encoding='utf-8') as f:
                    f.write("\n".join(srt_entries))

                # Burn subtitles into video
                # Escape path for ffmpeg subtitles filter
                srt_escaped = temp_srt.replace('\\', '/').replace(':', '\\:')
                cmd = [
                    ffmpeg_path, "-y",
                    "-i", input_path,
                    "-vf", f"subtitles='{srt_escaped}'",
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac",
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    raise ValueError(f"Subtitle burn failed: {result.stderr}")
            finally:
                if os.path.exists(temp_srt):
                    os.remove(temp_srt)
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

    def _srt_time(self, seconds):
        """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def generate_thumbnail(self, input_path, output_jpg_path):
        """Scan video frames, find sharpest frame via Laplacian variance, save as JPG."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file for thumbnail generation")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample every N frames (at least every 30 frames)
        sample_interval = max(1, total_frames // 100)
        best_score = -1
        best_frame = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
            frame_idx += 1

        cap.release()

        if best_frame is not None:
            cv2.imwrite(output_jpg_path, best_frame)
        else:
            raise ValueError("Could not find a suitable frame for thumbnail")

    def apply_video_trim(self, input_path, output_path, start_time, end_time):
        """Trim video using ffmpeg -ss and -to parameters."""
        ffmpeg_path = self._get_ffmpeg_path()
        cmd = [
            ffmpeg_path, "-y",
            "-i", input_path,
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Video trim failed: {result.stderr}")

    def apply_face_detect_zoom(self, frame):
        """Detect face and auto-zoom/crop to face region.

        Uses frame-skip to only run detection every 5 frames, holding the
        last detected crop region between detections for performance.
        """
        self._face_detect_frame_count += 1

        # Only run detection every 5 frames; reuse last region otherwise
        if self._face_detect_frame_count % 5 == 1 or self._face_detect_last_region is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                # Use the largest face
                areas = [w * h for (x, y, w, h) in faces]
                largest_idx = areas.index(max(areas))
                x, y, w, h = faces[largest_idx]

                frame_h, frame_w = frame.shape[:2]
                # Expand the face region to include more context
                cx, cy = x + w // 2, y + h // 2
                zoom_size = max(w, h) * 2
                x1 = max(0, cx - zoom_size // 2)
                y1 = max(0, cy - zoom_size // 2)
                x2 = min(frame_w, cx + zoom_size // 2)
                y2 = min(frame_h, cy + zoom_size // 2)

                self._face_detect_last_region = (x1, y1, x2, y2)
            else:
                # No face found; clear the cached region
                self._face_detect_last_region = None

        # Apply the cached crop region if available
        if self._face_detect_last_region is not None:
            x1, y1, x2, y2 = self._face_detect_last_region
            frame_h, frame_w = frame.shape[:2]
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                result = cv2.resize(cropped, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                return result

        return frame

    def apply_preset_profile(self, input_path, output_path, preset):
        """Apply a preset profile: YouTube, Instagram, or Cinematic."""
        ffmpeg_path = self._get_ffmpeg_path()

        if preset == "YouTube":
            # 1080p + copy audio
            cmd = [
                ffmpeg_path, "-y",
                "-i", input_path,
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                output_path
            ]
        elif preset == "Instagram":
            # 1080x1080 center crop
            cmd = [
                ffmpeg_path, "-y",
                "-i", input_path,
                "-vf", "crop=min(iw\\,ih):min(iw\\,ih),scale=1080:1080",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                output_path
            ]
        elif preset == "Cinematic":
            # Letterbox bars + color grading (warm tone)
            cmd = [
                ffmpeg_path, "-y",
                "-i", input_path,
                "-vf", "pad=iw:iw*9/16:(ow-iw)/2:(oh-ih)/2:black,colorbalance=rs=0.1:gs=0.05:bs=-0.1",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                output_path
            ]
        else:
            # No preset, just copy
            cmd = [ffmpeg_path, "-y", "-i", input_path, "-c", "copy", output_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ValueError(f"Preset profile '{preset}' failed: {result.stderr}")

    def apply_cinematic_bars(self, frame):
        """Add black letterbox bars (top and bottom, ~12% each) to create cinematic look."""
        h, w = frame.shape[:2]
        bar_height = int(h * 0.12)
        result = frame.copy()
        # Top bar
        result[0:bar_height, :] = 0
        # Bottom bar
        result[h - bar_height:h, :] = 0
        return result

    def apply_color_preset(self, frame, preset):
        """Apply color preset transformation per frame."""
        if preset == "Warm":
            # Increase red/yellow, decrease blue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] - 5, 0, 179)  # Shift hue toward warm
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # Boost saturation
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Add warmth via color balance
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) * 1.1, 0, 255).astype(np.uint8)  # Red up
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) * 0.9, 0, 255).astype(np.uint8)  # Blue down
            return result

        elif preset == "Cool":
            # Increase blue tones, decrease warm
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + 10, 0, 179)  # Shift hue toward cool
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) * 1.15, 0, 255).astype(np.uint8)  # Blue up
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) * 0.9, 0, 255).astype(np.uint8)  # Red down
            return result

        elif preset == "Vintage":
            # Desaturate slightly, add warm yellow tint, reduce contrast
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)  # Reduce saturation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)  # Slightly darker
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Add sepia-like warm tint
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float32) + 20, 0, 255).astype(np.uint8)
            result[:, :, 1] = np.clip(result[:, :, 1].astype(np.float32) + 10, 0, 255).astype(np.uint8)
            return result

        elif preset == "Moody Dark":
            # Dark, desaturated, high contrast with cool shadows
            result = cv2.convertScaleAbs(frame, alpha=1.2, beta=-30)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.8, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255)
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Cool shadow tint
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float32) + 10, 0, 255).astype(np.uint8)
            return result

        elif preset == "Bright Pop":
            # High saturation, bright, vibrant
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # Boost saturation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)  # Brighten
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return result

        return frame


class VideoEnhancerGUI:
    """GUI Dashboard for the Video Enhancer."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Enhancer - Super HDR & Color Enhancement | made by @codex_here")
        self.root.geometry("720x680")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)
        self.root.configure(bg="#1a1a2e")

        self.enhancer = VideoEnhancer()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.yt_url = tk.StringVar()
        self.yt_format = tk.StringVar(value="16:9")
        self.processing = False
        self.downloading = False

        # New feature variables
        self.slow_motion_var = tk.BooleanVar(value=False)
        self.voice_enhance_var = tk.BooleanVar(value=False)
        self.remove_noise_var = tk.BooleanVar(value=False)
        self.auto_subtitles_var = tk.BooleanVar(value=False)
        self.thumbnail_var = tk.BooleanVar(value=False)
        self.cinematic_bars_var = tk.BooleanVar(value=False)
        self.face_zoom_var = tk.BooleanVar(value=False)
        self.preset_profile_var = tk.StringVar(value="None")
        self.color_preset_var = tk.StringVar(value="None")
        self.trim_start_var = tk.StringVar(value="")
        self.trim_end_var = tk.StringVar(value="")

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
        """Create all GUI widgets in a compact layout without scrolling."""
        # Simple main frame - no canvas, no scrollbar
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame,
                                text="Video Enhancer",
                                style='Title.TLabel')
        title_label.pack(pady=(8, 1), padx=15)

        subtitle = ttk.Label(main_frame,
                             text="Super HDR + Color Enhancement",
                             style='Status.TLabel')
        subtitle.pack(pady=(0, 5), padx=15)

        # YouTube / Instagram Download Section
        yt_frame = ttk.Frame(main_frame, style='Card.TFrame')
        yt_frame.pack(fill=tk.X, pady=2, ipadx=8, padx=15)

        ttk.Label(yt_frame, text="YouTube / Instagram Download:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=8, pady=(4, 1))

        yt_url_row = ttk.Frame(yt_frame, style='Card.TFrame')
        yt_url_row.pack(fill=tk.X, padx=8, pady=(0, 2))

        self.yt_url_entry = tk.Entry(yt_url_row, textvariable=self.yt_url,
                                     font=('Helvetica', 9),
                                     bg='#0f3460', fg='#a0a0a0',
                                     insertbackground='#ffffff',
                                     relief='flat')
        self.yt_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        self.yt_url_entry.insert(0, "Paste YouTube or Instagram URL here")

        yt_options_row = ttk.Frame(yt_frame, style='Card.TFrame')
        yt_options_row.pack(fill=tk.X, padx=8, pady=(0, 2))

        yt_format_16_9 = tk.Radiobutton(yt_options_row, text="16:9 (Landscape)",
                                        variable=self.yt_format, value="16:9",
                                        bg='#16213e', fg='#ffffff',
                                        selectcolor='#0f3460',
                                        activebackground='#16213e',
                                        activeforeground='#ffffff',
                                        font=('Helvetica', 9))
        yt_format_16_9.pack(side=tk.LEFT, padx=(0, 15))

        yt_format_9_16 = tk.Radiobutton(yt_options_row, text="Short (9:16)",
                                        variable=self.yt_format, value="9:16",
                                        bg='#16213e', fg='#ffffff',
                                        selectcolor='#0f3460',
                                        activebackground='#16213e',
                                        activeforeground='#ffffff',
                                        font=('Helvetica', 9))
        yt_format_9_16.pack(side=tk.LEFT, padx=(0, 15))

        self.yt_download_btn = ttk.Button(yt_options_row, text="Download",
                                          style='Custom.TButton',
                                          command=self._download_youtube)
        self.yt_download_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.yt_status_label = ttk.Label(yt_options_row, text="",
                                         style='Info.TLabel')
        self.yt_status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Input File Section (compact)
        input_frame = ttk.Frame(main_frame, style='Card.TFrame')
        input_frame.pack(fill=tk.X, pady=2, ipadx=8, padx=15)

        input_row = ttk.Frame(input_frame, style='Card.TFrame')
        input_row.pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(input_row, text="Input:",
                  style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 5))

        self.input_entry = tk.Entry(input_row, textvariable=self.input_path,
                                    font=('Helvetica', 9),
                                    bg='#0f3460', fg='#ffffff',
                                    insertbackground='#ffffff',
                                    relief='flat')
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)

        browse_btn = ttk.Button(input_row, text="Browse",
                                style='Custom.TButton',
                                command=self._browse_input)
        browse_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Output File Section (compact)
        output_frame = ttk.Frame(main_frame, style='Card.TFrame')
        output_frame.pack(fill=tk.X, pady=2, ipadx=8, padx=15)

        output_row = ttk.Frame(output_frame, style='Card.TFrame')
        output_row.pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(output_row, text="Output:",
                  style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 5))

        self.output_entry = tk.Entry(output_row, textvariable=self.output_path,
                                     font=('Helvetica', 9),
                                     bg='#0f3460', fg='#ffffff',
                                     insertbackground='#ffffff',
                                     relief='flat')
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)

        browse_out_btn = ttk.Button(output_row, text="Browse",
                                    style='Custom.TButton',
                                    command=self._browse_output)
        browse_out_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Enhancement Options (2 rows of 3 checkboxes)
        options_frame = ttk.Frame(main_frame, style='Card.TFrame')
        options_frame.pack(fill=tk.X, pady=2, ipadx=8, padx=15)

        ttk.Label(options_frame, text="Enhancement Options:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=8, pady=(4, 2))

        self.hdr_var = tk.BooleanVar(value=True)
        self.color_var = tk.BooleanVar(value=True)
        self.minor_zoom_var = tk.BooleanVar(value=False)
        self.random_zoom_var = tk.BooleanVar(value=False)
        self.strong_zoom_var = tk.BooleanVar(value=False)
        self.copyright_free_var = tk.BooleanVar(value=False)

        # Row 1: Super HDR, Color, Minor Zoom
        checks_row = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row.pack(fill=tk.X, padx=8, pady=(0, 1))

        hdr_check = tk.Checkbutton(checks_row, text="Super HDR",
                                   variable=self.hdr_var,
                                   bg='#16213e', fg='#ffffff',
                                   selectcolor='#0f3460',
                                   activebackground='#16213e',
                                   activeforeground='#ffffff',
                                   font=('Helvetica', 9))
        hdr_check.pack(side=tk.LEFT, padx=(0, 20))

        color_check = tk.Checkbutton(checks_row, text="Color",
                                     variable=self.color_var,
                                     bg='#16213e', fg='#ffffff',
                                     selectcolor='#0f3460',
                                     activebackground='#16213e',
                                     activeforeground='#ffffff',
                                     font=('Helvetica', 9))
        color_check.pack(side=tk.LEFT, padx=(0, 20))

        minor_zoom_check = tk.Checkbutton(checks_row, text="Minor Zoom",
                                          variable=self.minor_zoom_var,
                                          bg='#16213e', fg='#ffffff',
                                          selectcolor='#0f3460',
                                          activebackground='#16213e',
                                          activeforeground='#ffffff',
                                          font=('Helvetica', 9))
        minor_zoom_check.pack(side=tk.LEFT)

        # Row 2: Random Zoom, Strong Zoom, Copyright Free
        checks_row2 = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row2.pack(fill=tk.X, padx=8, pady=(0, 4))

        random_zoom_check = tk.Checkbutton(checks_row2, text="Random Zoom",
                                           variable=self.random_zoom_var,
                                           bg='#16213e', fg='#ffffff',
                                           selectcolor='#0f3460',
                                           activebackground='#16213e',
                                           activeforeground='#ffffff',
                                           font=('Helvetica', 9))
        random_zoom_check.pack(side=tk.LEFT, padx=(0, 20))

        strong_zoom_check = tk.Checkbutton(checks_row2, text="Strong Zoom",
                                           variable=self.strong_zoom_var,
                                           bg='#16213e', fg='#ffffff',
                                           selectcolor='#0f3460',
                                           activebackground='#16213e',
                                           activeforeground='#ffffff',
                                           font=('Helvetica', 9))
        strong_zoom_check.pack(side=tk.LEFT, padx=(0, 20))

        copyright_free_check = tk.Checkbutton(checks_row2, text="Copyright Free",
                                              variable=self.copyright_free_var,
                                              bg='#16213e', fg='#ffffff',
                                              selectcolor='#0f3460',
                                              activebackground='#16213e',
                                              activeforeground='#ffffff',
                                              font=('Helvetica', 9))
        copyright_free_check.pack(side=tk.LEFT)

        # Row 3: Audio Features - Slow Motion, Voice Enhance, Remove Noise
        checks_row3 = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row3.pack(fill=tk.X, padx=8, pady=(0, 1))

        slow_motion_check = tk.Checkbutton(checks_row3, text="Slow Motion",
                                           variable=self.slow_motion_var,
                                           bg='#16213e', fg='#ffffff',
                                           selectcolor='#0f3460',
                                           activebackground='#16213e',
                                           activeforeground='#ffffff',
                                           font=('Helvetica', 9))
        slow_motion_check.pack(side=tk.LEFT, padx=(0, 20))

        voice_enhance_check = tk.Checkbutton(checks_row3, text="Voice Enhance",
                                             variable=self.voice_enhance_var,
                                             bg='#16213e', fg='#ffffff',
                                             selectcolor='#0f3460',
                                             activebackground='#16213e',
                                             activeforeground='#ffffff',
                                             font=('Helvetica', 9))
        voice_enhance_check.pack(side=tk.LEFT, padx=(0, 20))

        remove_noise_check = tk.Checkbutton(checks_row3, text="Remove Noise",
                                            variable=self.remove_noise_var,
                                            bg='#16213e', fg='#ffffff',
                                            selectcolor='#0f3460',
                                            activebackground='#16213e',
                                            activeforeground='#ffffff',
                                            font=('Helvetica', 9))
        remove_noise_check.pack(side=tk.LEFT)

        # Row 4: Cinematic Bars, Face Zoom, Auto Subtitles, Thumbnail
        checks_row4 = ttk.Frame(options_frame, style='Card.TFrame')
        checks_row4.pack(fill=tk.X, padx=8, pady=(0, 1))

        cinematic_bars_check = tk.Checkbutton(checks_row4, text="Cinematic Bars",
                                              variable=self.cinematic_bars_var,
                                              bg='#16213e', fg='#ffffff',
                                              selectcolor='#0f3460',
                                              activebackground='#16213e',
                                              activeforeground='#ffffff',
                                              font=('Helvetica', 9))
        cinematic_bars_check.pack(side=tk.LEFT, padx=(0, 20))

        face_zoom_check = tk.Checkbutton(checks_row4, text="Face Zoom",
                                         variable=self.face_zoom_var,
                                         bg='#16213e', fg='#ffffff',
                                         selectcolor='#0f3460',
                                         activebackground='#16213e',
                                         activeforeground='#ffffff',
                                         font=('Helvetica', 9))
        face_zoom_check.pack(side=tk.LEFT, padx=(0, 20))

        auto_subtitles_check = tk.Checkbutton(checks_row4, text="Auto Subtitles",
                                              variable=self.auto_subtitles_var,
                                              bg='#16213e', fg='#ffffff',
                                              selectcolor='#0f3460',
                                              activebackground='#16213e',
                                              activeforeground='#ffffff',
                                              font=('Helvetica', 9))
        auto_subtitles_check.pack(side=tk.LEFT, padx=(0, 20))

        thumbnail_check = tk.Checkbutton(checks_row4, text="Thumbnail",
                                         variable=self.thumbnail_var,
                                         bg='#16213e', fg='#ffffff',
                                         selectcolor='#0f3460',
                                         activebackground='#16213e',
                                         activeforeground='#ffffff',
                                         font=('Helvetica', 9))
        thumbnail_check.pack(side=tk.LEFT)

        # Row 5: Preset Profiles and Color Presets dropdowns
        presets_row = ttk.Frame(options_frame, style='Card.TFrame')
        presets_row.pack(fill=tk.X, padx=8, pady=(0, 1))

        tk.Label(presets_row, text="Preset:", bg='#16213e', fg='#ffffff',
                 font=('Helvetica', 9)).pack(side=tk.LEFT, padx=(0, 4))

        preset_menu = ttk.Combobox(presets_row, textvariable=self.preset_profile_var,
                                   values=["None", "YouTube", "Instagram", "Cinematic"],
                                   state="readonly", width=12,
                                   font=('Helvetica', 9))
        preset_menu.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(presets_row, text="Color:", bg='#16213e', fg='#ffffff',
                 font=('Helvetica', 9)).pack(side=tk.LEFT, padx=(0, 4))

        color_menu = ttk.Combobox(presets_row, textvariable=self.color_preset_var,
                                  values=["None", "Warm", "Cool", "Vintage", "Moody Dark", "Bright Pop"],
                                  state="readonly", width=12,
                                  font=('Helvetica', 9))
        color_menu.pack(side=tk.LEFT)

        # Row 6: Video Trim controls
        trim_row = ttk.Frame(options_frame, style='Card.TFrame')
        trim_row.pack(fill=tk.X, padx=8, pady=(0, 4))

        tk.Label(trim_row, text="Start:", bg='#16213e', fg='#ffffff',
                 font=('Helvetica', 9)).pack(side=tk.LEFT, padx=(0, 4))

        self.trim_start_entry = tk.Entry(trim_row, textvariable=self.trim_start_var,
                                         font=('Helvetica', 9), width=10,
                                         bg='#0f3460', fg='#ffffff',
                                         insertbackground='#ffffff', relief='flat')
        self.trim_start_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=1)
        self.trim_start_entry.insert(0, "00:00:00")

        tk.Label(trim_row, text="End:", bg='#16213e', fg='#ffffff',
                 font=('Helvetica', 9)).pack(side=tk.LEFT, padx=(0, 4))

        self.trim_end_entry = tk.Entry(trim_row, textvariable=self.trim_end_var,
                                       font=('Helvetica', 9), width=10,
                                       bg='#0f3460', fg='#ffffff',
                                       insertbackground='#ffffff', relief='flat')
        self.trim_end_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=1)
        self.trim_end_entry.insert(0, "00:00:00")

        self.trim_btn = tk.Button(trim_row, text="Trim",
                                  command=self._trim_video,
                                  bg='#0f3460', fg='#ffffff',
                                  activebackground='#1a5276',
                                  activeforeground='#ffffff',
                                  font=('Helvetica', 9, 'bold'),
                                  relief='raised', bd=2,
                                  padx=8, pady=1)
        self.trim_btn.pack(side=tk.LEFT)

        # Info Dashboard (single compact row)
        dashboard_frame = ttk.Frame(main_frame, style='Card.TFrame')
        dashboard_frame.pack(fill=tk.X, pady=2, ipadx=8, padx=15)

        dash_row = ttk.Frame(dashboard_frame, style='Card.TFrame')
        dash_row.pack(fill=tk.X, padx=8, pady=3)

        ttk.Label(dash_row, text="Dashboard:",
                  style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 10))

        self.size_label = ttk.Label(dash_row, text="-- MB",
                                    style='Info.TLabel')
        self.size_label.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(dash_row, text="|",
                  style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 8))

        self.resolution_label = ttk.Label(dash_row, text="-- x --",
                                          style='Info.TLabel')
        self.resolution_label.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(dash_row, text="|",
                  style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 8))

        self.fps_label = ttk.Label(dash_row, text="-- fps",
                                   style='Info.TLabel')
        self.fps_label.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(dash_row, text="|",
                  style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 8))

        self.duration_label = ttk.Label(dash_row, text="--:--",
                                        style='Info.TLabel')
        self.duration_label.pack(side=tk.LEFT)

        # Progress Section (compact - all in one row)
        progress_frame = ttk.Frame(main_frame, style='Card.TFrame')
        progress_frame.pack(fill=tk.X, pady=2, ipadx=8, padx=15)

        ttk.Label(progress_frame, text="Progress:",
                  style='Header.TLabel').pack(anchor=tk.W, padx=8, pady=(4, 2))

        # Progress bar + Cancel + Enter all in one row
        progress_bar_row = ttk.Frame(progress_frame, style='Card.TFrame')
        progress_bar_row.pack(fill=tk.X, padx=8, pady=(0, 2))

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
                                    font=('Helvetica', 9, 'bold'),
                                    relief='raised', bd=2,
                                    state=tk.DISABLED,
                                    padx=8, pady=2)
        self.cancel_btn.pack(side=tk.LEFT, padx=(8, 4))

        enter_label = tk.Label(progress_bar_row, text="Enter",
                               bg='#0f3460', fg='#00ff88',
                               font=('Helvetica', 9, 'bold'),
                               padx=8, pady=2,
                               relief='raised', bd=2)
        enter_label.pack(side=tk.LEFT, padx=(4, 0))

        progress_info = ttk.Frame(progress_frame, style='Card.TFrame')
        progress_info.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.progress_label = ttk.Label(progress_info,
                                        text="Ready - Press Enter to Start",
                                        style='Info.TLabel')
        self.progress_label.pack(side=tk.LEFT)

        self.time_label = ttk.Label(progress_info,
                                    text="Time Remaining: --:--",
                                    style='Info.TLabel')
        self.time_label.pack(side=tk.RIGHT)

        # Credit Label at the bottom
        credit_label = tk.Label(main_frame, text="made by @codex_here",
                                bg='#1a1a2e', fg='#00d4ff',
                                font=('Helvetica', 8, 'italic'))
        credit_label.pack(side=tk.BOTTOM, pady=(5, 5))

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
                self.fps_label.config(text=f"{fps:.1f} fps")
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
                and not self.strong_zoom_var.get() and not self.copyright_free_var.get() \
                and not self.slow_motion_var.get() and not self.voice_enhance_var.get() \
                and not self.remove_noise_var.get() and not self.auto_subtitles_var.get() \
                and not self.thumbnail_var.get() and not self.cinematic_bars_var.get() \
                and not self.face_zoom_var.get() \
                and self.preset_profile_var.get() == "None" \
                and self.color_preset_var.get() == "None":
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
                cinematic_bars_enabled=self.cinematic_bars_var.get(),
                face_zoom_enabled=self.face_zoom_var.get(),
                color_preset=self.color_preset_var.get(),
                progress_callback=self._on_progress
            )

            if not success:
                self.root.after(0, self._on_complete, False)
                return

            # Apply post-processing ffmpeg-based features
            current_output = output_path

            # Slow Motion
            if self.slow_motion_var.get():
                temp_path = output_path + ".slow.mp4"
                self.enhancer.apply_slow_motion(current_output, temp_path)
                os.replace(temp_path, current_output)

            # Voice Enhancement + Noise Removal (combined into one pass when both enabled)
            if self.voice_enhance_var.get() and self.remove_noise_var.get():
                temp_path = output_path + ".audio.mp4"
                self.enhancer.apply_voice_enhance_and_denoise(current_output, temp_path)
                os.replace(temp_path, current_output)
            elif self.voice_enhance_var.get():
                temp_path = output_path + ".voice.mp4"
                self.enhancer.apply_voice_enhancement(current_output, temp_path)
                os.replace(temp_path, current_output)
            elif self.remove_noise_var.get():
                temp_path = output_path + ".denoise.mp4"
                self.enhancer.apply_remove_background_noise(current_output, temp_path)
                os.replace(temp_path, current_output)

            # Auto Subtitles
            if self.auto_subtitles_var.get():
                temp_path = output_path + ".subs.mp4"
                try:
                    self.enhancer.apply_auto_subtitles(current_output, temp_path)
                    os.replace(temp_path, current_output)
                except ValueError as e:
                    self.root.after(0, lambda msg=str(e): messagebox.showwarning(
                        "Auto Subtitles", msg))

            # Preset Profile (applied via ffmpeg after frame processing)
            preset = self.preset_profile_var.get()
            if preset and preset != "None":
                temp_path = output_path + ".preset.mp4"
                self.enhancer.apply_preset_profile(current_output, temp_path, preset)
                os.replace(temp_path, current_output)

            # Thumbnail Generator
            if self.thumbnail_var.get():
                thumb_path = os.path.splitext(output_path)[0] + "_thumbnail.jpg"
                try:
                    self.enhancer.generate_thumbnail(input_path, thumb_path)
                except ValueError:
                    pass

            self.root.after(0, self._on_complete, True)

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

    def _validate_time_format(self, time_str):
        """Validate that a time string is in HH:MM:SS format."""
        import re
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        if not re.match(pattern, time_str):
            return False
        parts = time_str.split(':')
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        if minutes >= 60 or seconds >= 60:
            return False
        return True

    def _trim_video(self):
        """Trim the input video using start and end time."""
        if self.processing:
            return

        input_path = self.input_path.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return

        start_time = self.trim_start_var.get().strip()
        end_time = self.trim_end_var.get().strip()

        if not start_time or not end_time:
            messagebox.showerror("Error", "Please enter Start and End times (HH:MM:SS).")
            return

        if start_time == "00:00:00" and end_time == "00:00:00":
            messagebox.showerror("Error", "Please set valid Start and End times.")
            return

        # Validate HH:MM:SS format
        if not self._validate_time_format(start_time):
            messagebox.showerror("Error",
                                 f"Invalid start time format: '{start_time}'.\n"
                                 "Please use HH:MM:SS format (e.g., 00:01:30).")
            return

        if not self._validate_time_format(end_time):
            messagebox.showerror("Error",
                                 f"Invalid end time format: '{end_time}'.\n"
                                 "Please use HH:MM:SS format (e.g., 00:05:00).")
            return

        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_trimmed.mp4"

        self.processing = True
        self.progress_label.config(text="Trimming video...")

        def _trim_thread():
            try:
                self.enhancer.apply_video_trim(input_path, output_path, start_time, end_time)
                self.root.after(0, lambda: self._on_trim_complete(output_path))
            except Exception as e:
                self.root.after(0, self._on_error, str(e))

        thread = threading.Thread(target=_trim_thread, daemon=True)
        thread.start()

    def _on_trim_complete(self, output_path):
        """Called when trimming completes."""
        self.processing = False
        self.progress_label.config(text="Trim Complete!")
        messagebox.showinfo("Success", f"Video trimmed:\n{output_path}")

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
