# Attribution, Licensing & Upstream Code Provenance

This research prototype integrates, adapts, and extends concepts and methodologies from two upstream research repositories.

---

## 1. Upstream Repositories

### Repository 1: NetherlandsForensicInstitute/rPPG-Deepfake-Detection
- **Source**: https://github.com/NetherlandsForensicInstitute/rPPG-Deepfake-Detection
- **Exact Commit**: `955b497052632f33fec318c822987c460a31c590`
- **Paper**: *"Do deepfakes have a heartbeat? Detecting deepfake videos using remote photoplethysmography"*, Forensic Science International: Reports (2026). https://doi.org/10.1016/j.fsir.2026.100509
- **Attribution & Components Adapted**:
  - Physiological bandpass filtering centered on human cardiac frequencies ($0.5 - 4.0\text{ Hz}$).
  - Plane-Orthogonal-to-Skin (POS) color projection algorithm for remote photoplethysmography (Wang et al., 2017).
  - Scientific framing and forensic guidance: explicit warning against treating pulse absence as automatic proof of deepfake manipulation; documentation of compression and motion degradation modes.
- **Modifications / Differences**:
  - The upstream NFI pipeline used TensorFlow/Keras and GPU pyVHR/CuPy environments. Our project reimplemented the POS algorithm and signal processing pipeline in clean, pure Python with NumPy and SciPy, fully integrated into a unified PyTorch framework compatible with Windows, CUDA 13, and CPU environments.

### Repository 2: yagmurtncr/multimodal-deepfake-detection
- **Source**: https://github.com/yagmurtncr/multimodal-deepfake-detection
- **Exact Commit**: `ff5200e149dafd91c5a0a74ee65a9e8c52d556f8`
- **License**: MIT License
- **Attribution & Components Adapted**:
  - Multi-task formulation for FakeAVCeleb v1.2 (`video_fake`, `audio_fake`, `any_fake`).
  - Speaker-disjoint dataset splitting paradigm to guarantee zero biometric identity leakage.
  - Cross-attention fusion architecture (v4 design) where modalities query inter-modal contexts.
- **Modifications / Differences**:
  - The upstream model focused on Xception visual frames and Wav2Vec/Lip-sync CNNs. Our project replaced the lip-sync CNN stream with a physiological rPPG/BVP stream, engineered a dynamic reliability gating module based on pulse SNR and visual quality metadata, and integrated logical consistency regularization into the loss function.

---

## 2. Dataset Attribution
- **Dataset**: FakeAVCeleb v1.2 (Khalid et al., 2021).
- **Location on Host**: `E:\rPPG\FakeAVCeleb_v1.2`
- **Ethics & Privacy Notice**: Video files, subject biometric features, and raw dataset frames are neither redistributed nor committed to version control.
