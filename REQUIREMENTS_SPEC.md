# Complete Requirement & System Specification: Multimodal Deepfake Detector (Tri-Modal Cross-Attention with Dynamic Gating & Physiological rPPG)

## 1. Project Overview & Objective
This project builds an end-to-end, reproducible, multimodal deepfake detection research prototype designed to detect manipulated video, synthetic audio, and multimodal face-swap/lip-sync deepfakes. It operates on the **FakeAVCeleb v1.2** dataset and fuses three distinct information streams:
1. **Video Stream**: Spatial-temporal facial features and manipulation artifacts.
2. **Audio Stream**: Acoustic representations capturing synthetic vocoders, cloning artifacts, and phoneme acoustics.
3. **Physiological (rPPG/BVP) Stream**: Remote photoplethysmography blood volume pulse (BVP) signals extracted from facial skin regions using the Plane-Orthogonal-to-Skin (POS) algorithm and bandpass filtering (0.5–4.0 Hz).

The fusion is executed via a **Tri-Modal Cross-Attention Transformer** coupled with a **Reliability-Aware Dynamic Gating Module**. The model predicts three multi-task outputs:
- `video_fake`: Probability that the video track is manipulated.
- `audio_fake`: Probability that the audio track is synthetic/cloned.
- `any_fake`: Probability that either or both modalities are manipulated.

---

## 2. Hardware & Environment Specifications
- **Operating System**: Microsoft Windows 11 (x64)
- **GPU**: NVIDIA GeForce RTX 5060 (Blackwell architecture, 8GB VRAM)
- **CUDA Driver**: 610.74, CUDA UMD 13.3
- **Python**: 3.13.14 (with `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, CUDA enabled: `True`)
- **Key Libraries**: `mediapipe 1.1.0`, `opencv-python 5.0.0.93`, `scipy 1.18.1`, `pandas 3.1.0`, `scikit-learn 1.9.1`, `transformers 4.57.6`
- **System Tooling**: FFmpeg (available on system PATH), Git 2.55.0.2

---

## 3. Upstream Repositories & Attribution
The prototype synthesizes and adapts core concepts from two reference repositories:
1. **NetherlandsForensicInstitute/rPPG-Deepfake-Detection**
   - *Commit*: `955b497052632f33fec318c822987c460a31c590`
   - *License / Ethics*: Research use notice. Evaluated physiological BVP extraction using POS algorithm, 0.5–4.0 Hz Butterworth bandpass filtering, and pulse analysis across deepfakes.
   - *Forensic Nuance Preserved*: An undetectable pulse is **not** automatic proof of a fake. Motion, compression, low lighting, and skin reflectance can degrade BVP extraction. The model must explicitly represent extraction validity and signal confidence rather than treating rPPG failure as fake evidence.
2. **yagmurtncr/multimodal-deepfake-detection**
   - *Commit*: `ff5200e149dafd91c5a0a74ee65a9e8c52d556f8`
   - *License*: MIT License
   - *Architecture*: Multi-task supervision on FakeAVCeleb v1.2 (`video_fake`, `audio_fake`, `any_fake`), cross-modal attention, and speaker-disjoint splitting to prevent biometric identity leakage.

---

## 4. Architectural Details (Based on System Design & User Blueprint)

### Pipeline Stages
```
[Video Input (.mp4)]
       │
       ▼
[Pre-processing (MediaPipe FaceMesh & FFmpeg Audio)]
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Case A: No Face Detected]               [Case B: Face Detected]
       │                                         │
       │ rPPG Mask = False                       ├─ ROI Extraction (Cheeks & Forehead)
       │ rPPG Weight = 0                         ├─ POS Algorithm (Light Noise Stripping)
       │ SNR = 0 dB                              ├─ 0.5 - 4.0 Hz Butterworth Bandpass
       │ (Proceeds to Step 4)                    └─ Windowed PSD + Inter-ROI Correlation
       │                                                 │
       └────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    [Step 4: Tokenization Layer]  [Step 5: Dynamic Gating Module]
    - Audio: Log-Mel / CNN proj.  - Inputs: SNR, Bitrate, Sharpness,
    - Video: CLIP ViT-B / linear    Face Coverage, Skin tone indicator
    - rPPG: Window PSDs + proj.   - Outputs: Per-window & modality
    - Temporal Positional Embed.    reliability weights [0, 1]
              │                           │
              └─────────────┬─────────────┘
                            │ (Soft gating & attention biasing)
                            ▼
     [Step 6: Tri-Modal Cross-Attention Transformer]
     - Visual queries Audio & rPPG
     - Audio queries Visual & rPPG
     - rPPG queries Visual & Audio
     - Residuals, LayerNorm, FFN, Attention Masks
                            │
                            ▼
     [Step 7: Multi-Task Classification Heads]
     - Sigmoids: P(video_fake), P(audio_fake), P(any_fake)
     - Decision thresholding & consistency check
```

---

## 5. Dataset Structure & Splitting Integrity
- **Dataset Path**: `E:\rPPG\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`
- **Core Authenticity Classes**:
  1. `RealVideo-RealAudio`: `video_fake=0`, `audio_fake=0`, `any_fake=0`
  2. `FakeVideo-RealAudio`: `video_fake=1`, `audio_fake=0`, `any_fake=1`
  3. `RealVideo-FakeAudio`: `video_fake=0`, `audio_fake=1`, `any_fake=1`
  4. `FakeVideo-FakeAudio`: `video_fake=1`, `audio_fake=1`, `any_fake=1`
- **Identity Leakage Prevention**:
  - Speaker-disjoint GroupShuffleSplit (Train: 70%, Validation: 15%, Test: 15%).
  - Guarantees that neither original identities nor their deepfake derivatives cross split boundaries.
- **Fairness & Demographic Tracking**:
  - Preserves demographic metadata (`race`, `gender`) from `meta_data.csv` for post-hoc subgroup audit.
  - Ensures the model does not exploit skin tone or makeup as superficial shortcuts for deepfake classification.

---

## 6. Baselines & Evaluation Matrix
To scientifically assess the value of rPPG and dynamic gating, the project implements and compares:
1. **rPPG-Only Baseline**: Evaluates what physiological signals alone can achieve.
2. **Audio-Only Baseline**: Unimodal audio benchmark.
3. **Video-Only Baseline**: Unimodal visual benchmark.
4. **Late Fusion Baseline**: Concatenation of pooled features without cross-attention.
5. **Tri-Modal Cross-Attention (Ungated)**: Evaluates attention without dynamic reliability weights.
6. **Proposed Model (Tri-Modal Cross-Attention + Dynamic Gating)**: Full model.

### Key Metrics
- ROC-AUC and PR-AUC for each head (`video_fake`, `audio_fake`, `any_fake`).
- Accuracy, Balanced Accuracy, Precision, Recall, and F1 Score at optimal validation threshold.
- Confusion matrices for all four authenticity combinations.
- Subgroup performance audit across demographic groups (`race`, `gender`).

---

## 7. Deliverables Roadmap
1. `src/`: Clean, modular, fully typed Python implementation.
2. `tests/`: Automated unit & smoke tests runnable under CPU and CUDA.
3. `scripts/`:
   - `build_manifest.py`: Scans dataset and writes speaker-disjoint manifest.
   - `extract_features.py`: Caches audio, video, and rPPG tokens with quality metrics.
   - `train_rppg_baseline.py`: Standalone rPPG model training and evaluation.
   - `train_trimodal.py`: Stage-1 unimodal warmup and Stage-2 tri-modal cross-attention training.
   - `evaluate_models.py`: Comprehensive benchmarking and ablation comparisons.
   - `run_inference.py`: CLI tool accepting any video file and outputting structured JSON.
4. `docs/`: In-depth documentation including research gaps, architectural decisions, and setup instructions.
5. GitHub readiness: Git initialized, clean `.gitignore`, zero committed data/weights.
