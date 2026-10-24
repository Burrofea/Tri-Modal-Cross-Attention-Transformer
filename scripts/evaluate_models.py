"""Script to run comprehensive evaluation, ablations, and fairness audits."""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import MultimodalDataset
from src.models.trimodal_transformer import TriModalTransformer
from src.models.rppg_baseline import RPPGBaselineModel
from src.evaluation.evaluator import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tri-Modal and Baseline Deepfake Detectors.")
    parser.add_argument("--index", type=str, default="index.csv", help="Path to manifest CSV")
    parser.add_argument("--feats", type=str, default="feats", help="Features directory")
    parser.add_argument("--ckpt", type=str, default="checkpoints/trimodal_model.pt", help="Path to trimodal checkpoint")
    parser.add_argument("--baseline_ckpt", type=str, default="checkpoints/rppg_baseline.pt", help="Path to rppg baseline checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save evaluation results")
    args = parser.parse_args()

    df = pd.read_csv(args.index)
    val_ds = MultimodalDataset(df, args.feats, split="val")
    test_ds = MultimodalDataset(df, args.feats, split="test")

    print(f"Loaded for Evaluation: Val={len(val_ds)}, Test={len(test_ds)}")
    if len(test_ds) == 0:
        print("Error: No test samples found in features directory!")
        sys.exit(1)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Load TriModal model
    trimodal = TriModalTransformer().to(device)
    if Path(args.ckpt).exists():
        print(f"Loading TriModal weights from: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        trimodal.load_state_dict(ckpt.get("state_dict", ckpt))
    else:
        print(f"Warning: TriModal checkpoint {args.ckpt} not found! Evaluating initialized model.")

    # Load Baseline model
    baseline = None
    if Path(args.baseline_ckpt).exists():
        print(f"Loading rPPG Baseline weights from: {args.baseline_ckpt}")
        baseline = RPPGBaselineModel().to(device)
        b_ckpt = torch.load(args.baseline_ckpt, map_location=device)
        baseline.load_state_dict(b_ckpt.get("state_dict", b_ckpt))
    else:
        print(f"Notice: Baseline checkpoint {args.baseline_ckpt} not found. Running TriModal ablations only.")

    evaluator = ModelEvaluator(device=args.device)
    results = evaluator.run_benchmark_suite(
        trimodal_model=trimodal,
        baseline_model=baseline,
        val_loader=val_loader,
        test_loader=test_loader,
        test_df=test_ds.df,
        output_dir=args.out_dir,
    )

    print("\n================ BENCHMARK RESULTS SUMMARY ================")
    for model_name, metric_dict in results.items():
        if model_name == "subgroup_fairness_audit":
            continue
        any_m = metric_dict.get("any_fake", {})
        vid_m = metric_dict.get("video_fake", {})
        aud_m = metric_dict.get("audio_fake", {})
        print(f"\nModel: {model_name}")
        print(f"  [Any Fake]   ROC-AUC: {any_m.get('roc_auc', 0):.4f} | PR-AUC: {any_m.get('pr_auc', 0):.4f} | Acc: {any_m.get('accuracy', 0):.4f} | F1: {any_m.get('f1', 0):.4f}")
        print(f"  [Video Fake] ROC-AUC: {vid_m.get('roc_auc', 0):.4f} | F1: {vid_m.get('f1', 0):.4f}")
        print(f"  [Audio Fake] ROC-AUC: {aud_m.get('roc_auc', 0):.4f} | F1: {aud_m.get('f1', 0):.4f}")

if __name__ == "__main__":
    main()
