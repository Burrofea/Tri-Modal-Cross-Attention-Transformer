# Setup & Execution Guide: Multimodal Deepfake Detector

This guide provides end-to-end instructions for running the Multimodal Deepfake Detection research prototype on Windows (with CUDA acceleration on NVIDIA RTX 5060) and Linux/Google Colab.

---

## 1. Environment & Hardware Prerequisites

### Windows 11 with NVIDIA RTX 5060
1. **Python**: Python 3.10 to 3.13 (Python 3.13.14 with CUDA 13.0 PyTorch tested and verified).
2. **CUDA & GPU**: NVIDIA Driver (v610+), RTX 5060 Laptop/Desktop GPU.
3. **FFmpeg**: Must be available on system PATH (`ffmpeg.exe`).
4. **Git**: Required for version control.

Verify your environment:
```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
ffmpeg -version
```

---

## 2. Quickstart Execution Workflow

All commands should be executed from the project root directory:

### Step 1: Scan Dataset & Generate Speaker-Disjoint Manifest
Scans `E:\rPPG\FakeAVCeleb_v1.2` and produces `index.csv` with speaker-disjoint 70/15/15 partitions:
```powershell
python scripts/build_manifest.py --root "E:\rPPG\FakeAVCeleb_v1.2" --out "index.csv" --per_type 1200
```

### Step 2: Extract & Cache Multimodal Features
Extracts audio Mel-spectrograms, face crops, POS-derived rPPG tokens, and ResNet-18 visual embeddings into `feats/`:
```powershell
# Full dataset extraction:
python scripts/extract_features.py --index index.csv --out feats

# Fast smoke/debug test (e.g. 100 samples):
python scripts/extract_features.py --index index.csv --out feats --limit 100
```

### Step 3: Train Standalone rPPG-Only Baseline
Trains the unimodal physiological Transformer baseline on CUDA:
```powershell
python scripts/train_rppg_baseline.py --index index.csv --feats feats --epochs 15 --device cuda --out rppg_baseline.pt
```

### Step 4: Train Tri-Modal Cross-Attention Model
Executes Stage-1 unimodal warmup and Stage-2 full tri-modal cross-attention training:
```powershell
# Stage 1: Unimodal warmup
python scripts/train_trimodal.py --index index.csv --feats feats --stage 1 --epochs 10 --device cuda --out stage1_warmup.pt

# Stage 2: Tri-modal cross-attention with dynamic gating
python scripts/train_trimodal.py --index index.csv --feats feats --stage 2 --init_ckpt checkpoints/stage1_warmup.pt --epochs 20 --device cuda --out trimodal_model.pt
```

### Step 5: Benchmark Evaluation & Ablation Suite
Computes test set ROC-AUC, PR-AUC, Accuracy, F1, demographic fairness audits, and modality ablation comparisons:
```powershell
python scripts/evaluate_models.py --index index.csv --feats feats --ckpt checkpoints/trimodal_model.pt --baseline_ckpt checkpoints/rppg_baseline.pt --device cuda --out_dir results
```

### Step 6: Single-Video CLI Inference
Runs real-time inference on any video file and prints calibrated JSON:
```powershell
python scripts/run_inference.py --video "E:\rPPG\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\RealVideo-RealAudio\African\men\id00076\00109.mp4" --ckpt checkpoints/trimodal_model.pt --device cuda
```

---

## 3. Automated Unit Testing
Run the comprehensive test suite (16 tests covering tokenizers, POS extraction, bandpass filters, dynamic gating, attention masks, split leakage, and CLI inference):
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 4. Linux / Google Colab Setup Path

```bash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd multimodal_deepfake_detector

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run inference
python scripts/run_inference.py --video sample.mp4 --device cuda
```
