"""Interactive Web Application for Video Deepfake Detection.
Upload any video and inspect real-time predictions from both the Developed Tri-Modal Model and the rPPG Baseline.
"""
import os
import sys
import json
import tempfile
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.inference.predictor import DeepfakePredictor
from src.models.rppg_baseline import RPPGBaselineModel

PORT = 7860
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRIMODAL_CKPT = "checkpoints/trimodal_scaled.pt"
if not os.path.exists(TRIMODAL_CKPT):
    TRIMODAL_CKPT = "checkpoints/trimodal_model.pt"

BASELINE_CKPT = "checkpoints/rppg_baseline_scaled.pt"
if not os.path.exists(BASELINE_CKPT):
    BASELINE_CKPT = "checkpoints/rppg_baseline.pt"

print(f"Loading Developed Tri-Modal Model from {TRIMODAL_CKPT} on {DEVICE}...")
predictor = DeepfakePredictor(checkpoint_path=TRIMODAL_CKPT, device=DEVICE)

print(f"Loading rPPG Baseline Model from {BASELINE_CKPT} on {DEVICE}...")
baseline_model = RPPGBaselineModel().to(DEVICE)
if os.path.exists(BASELINE_CKPT):
    b_ckpt = torch.load(BASELINE_CKPT, map_location=DEVICE)
    baseline_model.load_state_dict(b_ckpt.get("state_dict", b_ckpt))
baseline_model.eval()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Tri-Modal Deepfake Detector</title>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8;
      --fake: #ef4444; --real: #22c55e;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text);
      margin: 0; padding: 30px 20px; display: flex; justify-content: center;
    }
    .container { max-width: 860px; width: 100%; }
    .header { text-align: center; margin-bottom: 25px; }
    .header h1 { margin: 0; font-size: 26px; color: var(--text); }
    .header p { color: var(--muted); font-size: 14px; margin-top: 6px; }
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 24px; margin-bottom: 20px;
    }
    .benchmarks {
      display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;
    }
    .bench-box {
      background: #0b1329; border: 1px solid var(--border); border-radius: 8px; padding: 14px;
    }
    .bench-title { font-size: 13px; color: var(--muted); text-transform: uppercase; font-weight: 600; }
    .bench-val { font-size: 20px; font-weight: bold; margin-top: 4px; color: var(--accent); }
    .bench-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .upload-area {
      border: 2px dashed var(--border); border-radius: 10px; padding: 30px 20px;
      text-align: center; cursor: pointer; transition: 0.2s;
    }
    .upload-area:hover { border-color: var(--accent); background: rgba(56, 189, 248, 0.04); }
    input[type="file"] { display: none; }
    .btn {
      background: var(--accent); color: #000; font-weight: 600; padding: 10px 22px;
      border: none; border-radius: 6px; cursor: pointer; margin-top: 15px; font-size: 14px;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    #loader { display: none; margin-top: 15px; text-align: center; color: var(--accent); font-size: 14px; }
    .result-section { display: none; margin-top: 20px; }
    .verdict-badge {
      font-size: 24px; font-weight: 800; text-align: center; padding: 16px; border-radius: 8px;
      text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px;
    }
    .verdict-fake { background: rgba(239, 68, 68, 0.15); color: var(--fake); border: 1px solid var(--fake); }
    .verdict-real { background: rgba(34, 197, 94, 0.15); color: var(--real); border: 1px solid var(--real); }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 14px; }
    .metric-lbl { color: var(--muted); }
    .metric-val { font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Multimodal Deepfake Detector</h1>
      <p>Tri-Modal Cross-Attention (Visual + Audio + Physiological rPPG) with Dynamic Reliability Gating</p>
    </div>

    <div class="benchmarks">
      <div class="bench-box">
        <div class="bench-title">Developed Model (Tri-Modal)</div>
        <div class="bench-val">97.67% Accuracy</div>
        <div class="bench-sub">ROC-AUC: 0.9858 | F1: 0.9882 (Speaker-Disjoint)</div>
      </div>
      <div class="bench-box">
        <div class="bench-title">Base Model (rPPG Only)</div>
        <div class="bench-val">96.28% Accuracy</div>
        <div class="bench-sub">ROC-AUC: 0.7705 | Audio-Fake AUC: 50.0% (Blind to Audio)</div>
      </div>
    </div>

    <div class="card">
      <div class="upload-area" onclick="document.getElementById('fileInput').click()">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin: 0 auto; color: var(--accent);">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <p id="fileLabel" style="margin-top: 10px; font-weight: 500;">Click to select an MP4 video or drag & drop</p>
      </div>
      <input type="file" id="fileInput" accept="video/mp4" onchange="handleFileSelect(this)">
      <div style="text-align: center;">
        <button id="analyzeBtn" class="btn" onclick="uploadAndAnalyze()" disabled>Analyze Video Authenticity</button>
      </div>
      <div id="loader">Extracting Visual, Audio, and Physiological Streams & Running GPU Inference...</div>
    </div>

    <div id="resultCard" class="card result-section">
      <div id="verdictBadge" class="verdict-badge"></div>
      
      <div class="grid-2">
        <div>
          <h3 style="margin-top:0; font-size:15px; color:var(--accent);">Developed Tri-Modal Predictions</h3>
          <div class="metric-row"><span class="metric-lbl">Overall Manipulation Prob:</span><span id="pAny" class="metric-val"></span></div>
          <div class="metric-row"><span class="metric-lbl">Video Track Manipulation:</span><span id="pVid" class="metric-val"></span></div>
          <div class="metric-row"><span class="metric-lbl">Audio Track Manipulation:</span><span id="pAud" class="metric-val"></span></div>
          <div class="metric-row"><span class="metric-lbl">Model Benchmark Accuracy:</span><span class="metric-val" style="color:var(--real)">97.67%</span></div>
        </div>

        <div>
          <h3 style="margin-top:0; font-size:15px; color:var(--accent);">rPPG Base Model Comparison</h3>
          <div class="metric-row"><span class="metric-lbl">Baseline Fake Prob:</span><span id="pBase" class="metric-val"></span></div>
          <div class="metric-row"><span class="metric-lbl">Baseline Benchmark Acc:</span><span class="metric-val">96.28%</span></div>
          <div class="metric-row"><span class="metric-lbl">Cardiac Pulse SNR:</span><span id="qSnr" class="metric-val"></span></div>
          <div class="metric-row"><span class="metric-lbl">Face Visibility Rate:</span><span id="qFace" class="metric-val"></span></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;
    function handleFileSelect(input) {
      if (input.files.length > 0) {
        selectedFile = input.files[0];
        document.getElementById('fileLabel').innerText = "Selected: " + selectedFile.name + " (" + (selectedFile.size / 1024 / 1024).toFixed(2) + " MB)";
        document.getElementById('analyzeBtn').disabled = false;
      }
    }

    async function uploadAndAnalyze() {
      if (!selectedFile) return;
      document.getElementById('analyzeBtn').disabled = true;
      document.getElementById('loader').style.display = 'block';
      document.getElementById('resultCard').style.display = 'none';

      const formData = new FormData();
      formData.append("video", selectedFile);

      try {
        const resp = await fetch("/analyze", { method: "POST", body: formData });
        const data = await resp.json();
        document.getElementById('loader').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;

        if (data.error) {
          alert("Analysis Error: " + data.error);
          return;
        }

        const badge = document.getElementById('verdictBadge');
        if (data.verdict === "DEEPFAKE") {
          badge.className = "verdict-badge verdict-fake";
          badge.innerText = "PREDICTED: DEEPFAKE (" + (data.probabilities.any_fake * 100).toFixed(1) + "% Confidence)";
        } else {
          badge.className = "verdict-badge verdict-real";
          badge.innerText = "PREDICTED: AUTHENTIC / REAL (" + ((1 - data.probabilities.any_fake) * 100).toFixed(1) + "% Confidence)";
        }

        document.getElementById('pAny').innerText = (data.probabilities.any_fake * 100).toFixed(1) + "%";
        document.getElementById('pVid').innerText = (data.probabilities.video_fake * 100).toFixed(1) + "%";
        document.getElementById('pAud').innerText = data.probabilities.audio_fake !== null ? (data.probabilities.audio_fake * 100).toFixed(1) + "%" : "N/A (Silent)";
        document.getElementById('pBase').innerText = (data.probabilities.baseline_fake * 100).toFixed(1) + "%";
        document.getElementById('qSnr').innerText = data.physiological.snr_db + " dB";
        document.getElementById('qFace').innerText = (data.physiological.face_coverage * 100).toFixed(0) + "%";

        document.getElementById('resultCard').style.display = 'block';
      } catch (err) {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        alert("Request failed: " + err);
      }
    }
  </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/analyze":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_error(400, "Expected multipart/form-data")
                return

            boundary = content_type.split("boundary=")[-1].encode("utf-8")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Extract video binary from multipart payload
            parts = body.split(b"--" + boundary)
            video_bytes = None
            for p in parts:
                if b'filename="' in p:
                    header, data = p.split(b"\r\n\r\n", 1)
                    video_bytes = data.rstrip(b"\r\n")
                    break

            if not video_bytes:
                self.send_error(400, "No video file found in payload")
                return

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            try:
                # 1. Run Tri-Modal inference
                pred_out = predictor.predict_video(tmp_path)

                # 2. Run Baseline rPPG inference
                rppg_res = predictor.rppg_prep.process(tmp_path)
                b_batch = {
                    "tok": torch.from_numpy(rppg_res["rppg_tok"]).unsqueeze(0).to(DEVICE),
                    "rm": torch.from_numpy(rppg_res["rppg_mask"]).unsqueeze(0).to(DEVICE),
                }
                with torch.no_grad():
                    base_prob = float(baseline_model(b_batch)["probabilities"][0].cpu().item())

                avg_snr = float(rppg_res["snr"].mean()) if len(rppg_res["snr"]) else 0.0

                resp = {
                    "status": "success",
                    "verdict": pred_out["predictions"]["verdict"],
                    "probabilities": {
                        "any_fake": pred_out["probabilities"]["any_fake_probability"],
                        "video_fake": pred_out["probabilities"]["video_fake_probability"],
                        "audio_fake": pred_out["probabilities"]["audio_fake_probability"],
                        "baseline_fake": round(base_prob, 4),
                    },
                    "benchmarks": {
                        "trimodal_accuracy": "97.67%",
                        "trimodal_auc": 0.9858,
                        "baseline_accuracy": "96.28%",
                        "baseline_auc": 0.7705,
                    },
                    "physiological": {
                        "snr_db": round(avg_snr, 2),
                        "face_coverage": round(float(rppg_res["quality"][2]), 2),
                    }
                }
            except Exception as e:
                resp = {"error": str(e)}
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

def main():
    server = HTTPServer(("0.0.0.0", PORT), RequestHandler)
    print(f"\n=======================================================")
    print(f"  Interactive Deepfake Detection Web Demo running at:")
    print(f"  http://localhost:{PORT}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")

if __name__ == "__main__":
    main()
