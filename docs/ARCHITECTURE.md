# Tri-Modal Deepfake Detector Architecture & Design Specification

## 1. System High-Level Architecture

The system processes portrait videos to detect digital manipulations across video, audio, and physiological signals.

```
                                  ┌───────────────────────────┐
                                  │      Input Video .mp4     │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
      ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
      │   A. Visual Stream   │      │   B. Audio Stream    │      │   C. rPPG Stream     │
      │  (16 Face Crops)     │      │ (5s Mono 16 kHz PCM) │      │  (Skin ROI Traces)   │
      └──────────┬───────────┘      └──────────┬───────────┘      └──────────┬───────────┘
                 │                             │                             │
                 ▼                             ▼                             ▼
      ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
      │ ResNet-18 Backbone   │      │ 64-Band Log-Mel Spec │      │ POS Algorithm (BVP)  │
      │ (16 x 512-dim feats) │      │ (64 x 500 STFT Spec) │      │ + 0.5-4 Hz Bandpass  │
      └──────────┬───────────┘      └──────────┬───────────┘      └──────────┬───────────┘
                 │                             │                             │
                 ▼                             ▼                             ▼
      ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
      │ Video Tokenizer      │      │ Audio Tokenizer      │      │ rPPG Tokenizer       │
      │ 16 tokens in R^192   │      │ 25 tokens in R^192   │      │ 5 tokens in R^192    │
      └──────────┬───────────┘      └──────────┬───────────┘      └──────────┬───────────┘
                 │                             │                             │
                 │                             │                ┌────────────┴───────────┐
                 │                             │                │ Step 5: Dynamic Gating │
                 │                             │                │  (SNR + Quality Cues)  │
                 │                             │                └────────────┬───────────┘
                 │                             │                             │
                 └───────────────────────┬─────┴─────────────────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │        Step 6: Tri-Modal Cross-Attention Transformer      │
                 │       - 3 Layers of Bidirectional Pairwise Attention      │
                 │       - Soft Attention Biasing from Reliability Gate      │
                 │       - Modality Dropout & Missing-Stream Tolerance       │
                 └─────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │           Step 7: Multi-Task Classification Heads         │
                 │       P(video_fake), P(audio_fake), P(any_fake)           │
                 │       Enforcing: any_fake = video_fake OR audio_fake      │
                 └───────────────────────────────────────────────────────────┘
```

---

## 2. Stream Details

### A. Video Stream
- Samples 16 evenly spaced frames across the 5.0-second video duration.
- Face crops are resized to $128 \times 128 \times 3$.
- Passed through a torchvision ResNet-18 feature extractor (flattened penultimate layer) producing a 512-dimensional vector per frame.
- Linearly projected into the shared latent space $R^D$ ($D=192$) with learned temporal positional embeddings.

### B. Audio Stream
- Extracted via FFmpeg as a 16 kHz mono float32 PCM stream.
- Windowed via Short-Time Fourier Transform (STFT): Hann window, $N_{FFT}=400$, hop size $= 160$ samples (10 ms).
- Projected through a 64-channel triangular Mel filterbank matrix spanning 0 to 8000 Hz.
- Logarithmically compressed: $M = \log(\text{Mel} + 10^{-6})$, yielding a $(64, 500)$ spectrogram.
- Tokenized through a 3-layer 2D CNN with max-pooling, producing 25 acoustic tokens in $R^D$.

### C. Physiological (rPPG) Stream
- Extracts mean RGB traces from 3 facial regions of interest (Left Cheek, Right Cheek, Forehead).
- Implements the **Plane-Orthogonal-to-Skin (POS)** algorithm (Wang et al., 2017) to eliminate illumination changes and motion noise:
  $$P = \begin{bmatrix} 0 & 1 & -1 \\ -2 & 1 & 1 \end{bmatrix}$$
  $$S = P \cdot C_n$$
  $$h = S_1 + \frac{\sigma(S_1)}{\sigma(S_2)} S_2$$
- Applies a 3rd-order Butterworth bandpass filter centered on the human physiological cardiac range ($0.5 - 4.0\text{ Hz}$, equivalent to 30–240 BPM).
- Evaluates $N_W=5$ overlapping windows (3.0s window, 0.5s hop).
- Computes 48-bin power spectral density (PSD) across each of the 3 ROIs + 3 inter-ROI Pearson correlation coefficients ($3 \times 48 + 3 = 147$ features per window).
- Computes pulse SNR in dB around the dominant cardiac frequency:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{\int_{f_0 \pm \delta} P(f) df}{\int_{\text{band} \setminus \{f_0 \pm \delta\}} P(f) df}\right)$$

---

## 3. Reliability-Aware Dynamic Gating

Extracted rPPG signals can be degraded by extreme compression, low lighting, or absence of a human face. Rather than treating missing rPPG as evidence of manipulation, the model measures signal reliability:

$$q = \left[ \frac{\text{SNR}}{10},\ \log(\text{bitrate} + 1),\ \log(\text{sharpness} + 1),\ \text{face\_detect\_rate} \right]^T$$
$$g = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot q + b_1) + b_2) \odot m_{\text{valid}}$$

- When no face is detected (**Case A**), $m_{\text{valid}} = 0$, forcing $g = 0$.
- Reliability weight $g$ attenuates physiological tokens:
  $$\tilde{R} = R \odot g$$
- $g$ also acts as a soft attention bias in cross-attention:
  $$\text{Bias}_r = \log(\text{clamp}(g, 10^{-4}))$$

---

## 4. Tri-Modal Cross-Attention Transformer

For each of the 3 layers, each modality queries the concatenated representations of the other two:
- **Audio queries Visual and rPPG**:
  $$A^{(l+1)} = \text{LayerNorm}(A^{(l)} + \text{MHA}(A^{(l)}, [V^{(l)}, R^{(l)}]))$$
- **Visual queries Audio and rPPG**:
  $$V^{(l+1)} = \text{LayerNorm}(V^{(l)} + \text{MHA}(V^{(l)}, [A^{(l)}, R^{(l)}]))$$
- **rPPG queries Audio and Visual**:
  $$R^{(l+1)} = \text{LayerNorm}(R^{(l)} + \text{MHA}(R^{(l)}, [A^{(l)}, V^{(l)}]))$$

Followed by standard Pre-LN feed-forward networks (FFN) with GELU activations and residual connections.

---

## 5. Multi-Task Supervision & Consistency Loss

The classification head outputs logits for three binary targets:
1. $y_{\text{video}}$: Video authenticity
2. $y_{\text{audio}}$: Audio authenticity
3. $y_{\text{any}}$: Multimodal authenticity

The total loss combines multi-task BCE and a logical consistency regularizer:
$$\mathcal{L} = \text{BCE}(p_v, y_v) + \text{BCE}(p_a, y_a) + \text{BCE}(p_{\text{any}}, y_{\text{any}}) + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
$$\mathcal{L}_{\text{cons}} = \| p_{\text{any}} - (1 - (1 - p_v)(1 - p_a)) \|_2^2$$
