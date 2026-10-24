# Research Gap & Scientific Analysis: Multimodal Fusion & Physiological rPPG

## 1. Context & Motivation

Deepfake generation methods (e.g., FaceSwap, DeepFaceLab, Wav2Lip, neural vocoders) have evolved from crude frame-by-frame blending to temporal consistency models. While unimodal visual detectors achieve high test performance on benchmark datasets, they suffer from severe vulnerabilities:
1. **Audio-Only Blindness**: In a *Real-Video / Fake-Audio* deepfake, the visual stream is completely authentic; an image-only model fails completely.
2. **Visual Compression Sensitivity**: Standard CNN/ViT detectors often overfit to high-frequency sensor noise and compression grid artifacts. Transcoding (e.g., social media re-encoding) degrades their accuracy.
3. **Physiological Independence**: Generative visual models synthesize pixel appearance, but typically lack temporal physiological dynamics—specifically the subtle cardiac pulse reflectance caused by blood volume pulses (BVP).

---

## 2. The Role of Remote Photoplethysmography (rPPG)

Remote Photoplethysmography estimates the blood-volume-pulse (BVP) signal from diffuse facial skin reflectance. In genuine human recordings, cardiac ventricular contractions pump blood through facial microvasculature, inducing minute periodic color fluctuations (predominantly in the green channel).

### Why rPPG Alone is Not Enough (The Research Gap)
A naive hypothesis often cited in early literature is: *"Real videos have a heartbeat; deepfakes do not."*
As rigorously demonstrated by the Netherlands Forensic Institute (NFI, 2026):
1. **Source Heartbeat Transfer**: In face-swap or reenactment deepfakes, if the driving or target video possesses a strong real pulse, the generative model may partially transfer the underlying illumination or periodic motion.
2. **False Negatives via Environmental Attenuation**: Video compression (H.264/H.265 macroblocking), low framerates (<15 fps), motion blur, and uneven lighting destroy or mask subtle BVP signals in **genuine** videos.
3. **Measurement Bias & Fairness Concerns**: Skin reflectance properties and cosmetics alter signal-to-noise ratios. Treating pulse absence as an automatic fake flag results in elevated false-positive rates on darker skin tones or subjects wearing foundation.

---

## 3. Our Scientific Contribution

To address this research gap, this prototype establishes:
1. **Tri-Modal Cross-Attention**: Rather than late-fusing or concatenating isolated features, cross-attention enables each modality to query context from the other two streams. The visual stream can resolve whether face movement accounts for an apparent audio desynchrony, while the audio stream can confirm speech boundaries.
2. **Reliability-Aware Dynamic Gating**: The model explicitly evaluates signal reliability metadata ($SNR$, video bitrate, frame sharpness, face coverage). If rPPG is degraded (Case A: missing face, or noisy capture), its contribution is softly attenuated towards zero, preventing false manipulation verdicts.
3. **Multi-Task Supervised Consistency**: Multi-task heads allow independent scoring of $P(\text{video\_fake})$ and $P(\text{audio\_fake})$, unified under a logical OR constraint for $P(\text{any\_fake})$.
