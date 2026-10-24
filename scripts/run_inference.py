"""Command-line inference script for single-video deepfake analysis."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import DeepfakePredictor

def main():
    parser = argparse.ArgumentParser(description="Run deepfake detection on a single video file.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/trimodal_model.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save JSON result")
    args = parser.parse_args()

    predictor = DeepfakePredictor(checkpoint_path=args.ckpt, device=args.device)
    result = predictor.predict_video(args.video)

    formatted_json = json.dumps(result, indent=2)
    print(formatted_json)

    if args.out:
        with open(args.out, "w") as f:
            f.write(formatted_json)
        print(f"\nResult saved to: {args.out}")

if __name__ == "__main__":
    main()
