"""Physiological rPPG / BVP signal extraction using FaceMesh and POS algorithm."""
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.signal as sps

class RPPGPreprocessor:
    """
    Extracts facial skin-region RGB traces, applies the POS algorithm to derive BVP/rPPG signals,
    filters the physiological band (0.5-4.0 Hz), and computes frequency tokens, masks, and SNR.
    """
    def __init__(
        self,
        duration_sec: float = 5.0,
        bandpass_hz: Tuple[float, float] = (0.5, 4.0),
        filter_order: int = 3,
        window_sec: float = 3.0,
        hop_sec: float = 0.5,
        num_windows: int = 5,
        num_freq_bins: int = 48,
        min_face_detect_rate: float = 0.65,
        landmarks: Tuple[int, int, int] = (205, 425, 151),  # Left cheek, right cheek, forehead
        roi_scale: float = 0.06,
    ):
        self.duration_sec = duration_sec
        self.bandpass_hz = bandpass_hz
        self.filter_order = filter_order
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.num_windows = num_windows
        self.num_freq_bins = num_freq_bins
        self.min_face_detect_rate = min_face_detect_rate
        self.landmarks = landmarks
        self.roi_scale = roi_scale
        self._face_mesh = None

    def _extract_rois_from_frame(self, rgb: np.ndarray) -> Tuple[np.ndarray, bool, float, Optional[Tuple[int, int, int, int]]]:
        """
        Extracts 3 facial skin ROIs (left cheek, right cheek, forehead) from frame.
        Supports MediaPipe FaceMesh when available, with automatic fallback to skin-segmented
        anatomical face ROIs.
        Returns: (roi_means: (3, 3), is_detected: bool, laplacian_var: float, face_box: (x0, y0, x1, y1))
        """
        h, w = rgb.shape[:2]
        roi_means = np.full((3, 3), np.nan, dtype=np.float32)
        laplacian_var = 0.0

        # Attempt MediaPipe if mp.solutions exists
        if self._face_mesh is not None:
            try:
                res = self._face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    pts = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)
                    face_w = np.linalg.norm(pts[454] - pts[234])
                    half_sz = max(3, int(self.roi_scale * face_w))
                    for k, lm_idx in enumerate(self.landmarks):
                        x_c, y_c = pts[lm_idx].astype(int)
                        patch = rgb[max(y_c - half_sz, 0):y_c + half_sz, max(x_c - half_sz, 0):x_c + half_sz]
                        if patch.size > 0:
                            roi_means[k] = patch.reshape(-1, 3).mean(axis=0)
                            if k == 0:
                                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
                    margin = 0.2 * max(x1 - x0, y1 - y0)
                    box = (int(max(x0 - margin, 0)), int(max(y0 - margin, 0)), int(min(x1 + margin, w)), int(min(y1 + margin, h)))
                    return roi_means, not np.isnan(roi_means).any(), laplacian_var, box
            except Exception:
                pass

        # Robust Fallback: Chrominance Skin Segmentation + Calibrated Facial Geometry
        # FakeAVCeleb frames are already centered face crops (224x224)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        skin_mask = (cr >= 125) & (cr <= 185) & (cb >= 65) & (cb <= 140)
        skin_ratio = float(skin_mask.mean())

        # If insufficient skin is present in the frame, flag Case A (no face)
        if skin_ratio < 0.12:
            return roi_means, False, 0.0, None

        # ROI 0: Left Cheek (mid-left)
        # ROI 1: Right Cheek (mid-right)
        # ROI 2: Forehead (upper-middle)
        regions = [
            (int(0.48 * h), int(0.68 * h), int(0.20 * w), int(0.42 * w)),  # Left cheek
            (int(0.48 * h), int(0.68 * h), int(0.58 * w), int(0.80 * w)),  # Right cheek
            (int(0.18 * h), int(0.36 * h), int(0.35 * w), int(0.65 * w)),  # Forehead
        ]

        for k, (y0, y1, x0, x1) in enumerate(regions):
            patch = rgb[y0:y1, x0:x1]
            p_mask = skin_mask[y0:y1, x0:x1]
            if np.any(p_mask):
                roi_means[k] = patch[p_mask].mean(axis=0)
            elif patch.size > 0:
                roi_means[k] = patch.reshape(-1, 3).mean(axis=0)

        # Blur metric on face center
        center_patch = rgb[int(0.25 * h):int(0.75 * h), int(0.25 * w):int(0.75 * w)]
        if center_patch.size > 0:
            gray = cv2.cvtColor(center_patch, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        box = (0, 0, w, h)
        is_detected = not np.isnan(roi_means).any()
        return roi_means, is_detected, laplacian_var, box

    @staticmethod
    def pos_algorithm(rgb_trace: np.ndarray, fps: float) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., IEEE TBME 2017).
        Args:
            rgb_trace: (T, 3) mean RGB trace over time
            fps: sampling rate / video frame rate
        Returns:
            (T,) extracted BVP pulse signal
        """
        t_len = len(rgb_trace)
        win_len = int(round(1.6 * fps))
        if win_len < 4 or t_len < win_len:
            # Fallback for ultra-short clips
            return np.zeros(t_len, dtype=np.float32)

        p_mat = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
        h_pulse = np.zeros(t_len, dtype=np.float32)

        for n in range(win_len, t_len + 1):
            c_win = rgb_trace[n - win_len:n].T  # (3, win_len)
            c_norm = c_win / (c_win.mean(axis=1, keepdims=True) + 1e-8)
            s_proj = p_mat @ c_norm
            std_0 = s_proj[0].std()
            std_1 = s_proj[1].std() + 1e-8
            h_segment = s_proj[0] + (std_0 / std_1) * s_proj[1]
            h_pulse[n - win_len:n] += (h_segment - h_segment.mean())

        return h_pulse

    def compute_snr_db(self, psd_power: np.ndarray, freqs: np.ndarray) -> float:
        """
        Computes pulse Signal-to-Noise Ratio (SNR) in dB around peak cardiac frequency.
        """
        band_mask = (freqs >= self.bandpass_hz[0]) & (freqs <= self.bandpass_hz[1])
        if not np.any(band_mask) or np.all(psd_power[band_mask] == 0):
            return 0.0

        sub_freqs = freqs[band_mask]
        sub_power = psd_power[band_mask]
        f0 = sub_freqs[np.argmax(sub_power)]

        # Cardiac signal energy (fundamental + 2nd harmonic)
        signal_mask = (np.abs(freqs - f0) <= 0.35) | (np.abs(freqs - 2.0 * f0) <= 0.5)
        p_signal = np.sum(psd_power[band_mask & signal_mask]) + 1e-8
        p_noise = np.sum(psd_power[band_mask & ~signal_mask]) + 1e-8

        return float(10.0 * np.log10(p_signal / p_noise))

    def process(self, video_path: str) -> Dict[str, np.ndarray]:
        """
        Processes video frames to extract BVP signals, windowed PSD tokens, and reliability metrics.
        Handles Case A (no face) vs Case B (face found).
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps and fps > 5.0) else 25.0

        max_frames = int(self.duration_sec * fps)

        rois_list: List[np.ndarray] = []
        det_flags: List[bool] = []
        laplacian_vars: List[float] = []
        face_boxes: List[Optional[Tuple[int, int, int, int]]] = []
        sample_crops: Dict[int, Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]] = {}

        # 16 evenly spaced frame indices for visual tokens
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        total_frames = min(total_frames, max_frames)
        sample_indices = set(np.linspace(0, max(total_frames - 1, 0), 16).astype(int).tolist())

        t = 0
        while t < total_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            roi_means, is_valid_frame, lap_var, box = self._extract_rois_from_frame(rgb)

            det_flags.append(is_valid_frame)
            rois_list.append(roi_means)
            face_boxes.append(box)
            if lap_var > 0.0:
                laplacian_vars.append(lap_var)

            if t in sample_indices:
                sample_crops[t] = (rgb, box)
            t += 1

        cap.release()

        # Build 16 face crops
        crops = []
        last_box = None
        for t_idx in sorted(sample_crops.keys()):
            frame_rgb, box = sample_crops[t_idx]
            curr_box = box or last_box
            last_box = curr_box or last_box
            if curr_box:
                c = frame_rgb[curr_box[1]:curr_box[3], curr_box[0]:curr_box[2]]
                if c.size == 0:
                    c = frame_rgb
            else:
                c = frame_rgb
            crops.append(cv2.resize(c, (128, 128)))

        while len(crops) < 16:
            crops.append(crops[-1] if crops else np.zeros((128, 128, 3), dtype=np.uint8))
        crops_arr = np.stack(crops[:16]).astype(np.uint8)

        # Output buffers
        rppg_tokens = np.zeros((self.num_windows, self.num_freq_bins * 3 + 3), dtype=np.float32)
        snrs = np.zeros(self.num_windows, dtype=np.float32)
        rppg_mask = np.zeros(self.num_windows, dtype=bool)

        detect_rate = float(np.mean(det_flags)) if det_flags else 0.0
        n_frames = len(det_flags)
        min_frames_needed = int(self.window_sec * fps) + 5

        # Decision: Case A vs Case B
        if detect_rate >= self.min_face_detect_rate and n_frames >= min_frames_needed:
            # Case B: Human face found and stable
            valid_indices = np.where(det_flags)[0]
            rois_arr = np.array(rois_list)
            # Interpolate any dropped frames
            bvp_signals = []
            nyq = fps / 2.0
            low = max(0.01, self.bandpass_hz[0] / nyq)
            high = min(0.99, self.bandpass_hz[1] / nyq)
            b_filt, a_filt = sps.butter(self.filter_order, [low, high], btype="band")

            for r_idx in range(3):
                interp_trace = np.stack([
                    np.interp(np.arange(n_frames), valid_indices, rois_arr[valid_indices, r_idx, c])
                    for c in range(3)
                ], axis=1)
                raw_pulse = self.pos_algorithm(interp_trace, fps)
                filt_pulse = sps.filtfilt(b_filt, a_filt, raw_pulse)
                bvp_signals.append(filt_pulse)

            bvp_arr = np.stack(bvp_signals, axis=0)  # (3, n_frames)

            # Windowing
            win_pts = int(self.window_sec * fps)
            hop_pts = int(self.hop_sec * fps)
            target_freqs = np.linspace(self.bandpass_hz[0], self.bandpass_hz[1], self.num_freq_bins)
            fft_freqs = np.fft.rfftfreq(512, d=1.0 / fps)

            tok_list, snr_list = [], []
            for start_pt in range(0, n_frames - win_pts + 1, hop_pts):
                segment = bvp_arr[:, start_pt:start_pt + win_pts] * np.hanning(win_pts)
                p_spec = np.abs(np.fft.rfft(segment, n=512, axis=1)) ** 2
                psd_interp = np.stack([np.interp(target_freqs, fft_freqs, p_spec[i]) for i in range(3)], axis=0)
                psd_interp /= (psd_interp.sum(axis=1, keepdims=True) + 1e-8)

                corr_mat = np.nan_to_num(np.corrcoef(segment))
                corr_feats = np.array([corr_mat[0, 1], corr_mat[0, 2], corr_mat[1, 2]], dtype=np.float32)

                tok_vec = np.concatenate([psd_interp.ravel(), corr_feats])
                tok_list.append(tok_vec)

                window_snr = np.mean([self.compute_snr_db(p_spec[i], fft_freqs) for i in range(3)])
                snr_list.append(window_snr)

                if len(tok_list) == self.num_windows:
                    break

            k = len(tok_list)
            if k > 0:
                rppg_tokens[:k] = np.array(tok_list, dtype=np.float32)
                snrs[:k] = np.array(snr_list, dtype=np.float32)
                rppg_mask[:k] = True

        # Quality metrics
        file_size_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else 1000
        duration_approx = max(n_frames / fps, 0.1)
        bitrate_kbps = (file_size_bytes * 8.0) / duration_approx / 1000.0
        avg_laplacian = float(np.mean(laplacian_vars)) if laplacian_vars else 0.0

        quality = np.array([
            np.log(bitrate_kbps + 1.0),
            np.log(avg_laplacian + 1.0),
            detect_rate
        ], dtype=np.float32)

        return {
            "crops": crops_arr,
            "rppg_tok": rppg_tokens,
            "rppg_mask": rppg_mask,
            "snr": snrs,
            "quality": quality,
        }
