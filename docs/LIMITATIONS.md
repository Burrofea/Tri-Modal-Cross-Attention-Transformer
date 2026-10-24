# Operational Limitations & Forensic Disclaimers

> **RESEARCH PROTOTYPE NOTICE**  
> This software is an experimental academic and engineering research prototype developed for investigating multimodal cross-attention fusion and physiological signal processing on FakeAVCeleb v1.2. It is **NOT** a court-admissible forensic tool, nor should it be used for stand-alone commercial or legal decision-making.

---

## 1. Domain & Dataset Specific Limitations

1. **FakeAVCeleb Bias & Domain Shift**:
   - The model was developed and validated on FakeAVCeleb v1.2. While FakeAVCeleb represents a diverse benchmark with real-world identities, models trained on it often learn dataset-specific compression signatures, background lighting patterns, and artifact characteristics that do not generalize directly to in-the-wild video platforms (TikTok, YouTube, WhatsApp, CCTV).
2. **Pre-cropped Portrait Assumption**:
   - FakeAVCeleb videos are centered $224 \times 224$ portrait headshots. For arbitrary unconstrained videos featuring multiple people, distant crowds, or rapid profile turns, an external face detection/tracking module (e.g., RetinaFace or YOLO-Face) must be prepended.

---

## 2. Physiological (rPPG) Limitations

1. **Low Lighting & Motion Blur**:
   - Optical BVP extraction depends on diffuse subsurface light scattering. Under extreme low-light conditions, strong backlighting, or rapid head motion, the pulse signal becomes submerged in optical noise ($SNR < 0\text{ dB}$).
2. **Video Compression Artifacts**:
   - Heavy spatial downsampling or high quantization parameter (QP) video compression removes the subtle chromatic fluctuations ($< 1\%$ pixel intensity change) necessary for reliable rPPG estimation.
3. **Absence of Cardiac Pulse is NOT Proof of Deepfake**:
   - As emphasized by the Netherlands Forensic Institute (NFI, 2026), genuine videos frequently produce undetectable or noisy rPPG signals due to environmental factors. The system's dynamic gating module is designed specifically to mitigate this failure mode by attenuating the rPPG modality weight to 0 rather than falsely classifying genuine videos as deepfakes.

---

## 3. Audio & Acoustic Limitations

1. **Missing Audio / Silent Clips**:
   - Videos without speech or acoustic tracks trigger an automatic warning and modality dropout, disabling the audio head.
2. **Studio Audio & Post-Dubbing**:
   - Genuine videos that have been re-dubbed in post-production or recorded with low-bitrate microphone setups may produce acoustic desynchronies that do not constitute malicious synthetic cloning.
