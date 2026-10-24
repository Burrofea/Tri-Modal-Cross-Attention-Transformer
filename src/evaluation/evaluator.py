"""Evaluation pipeline for benchmarking multimodal and baseline models."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import EvaluatorMetrics

class ModelEvaluator:
    """
    Executes comprehensive test set evaluations, ablations, and fairness audits across demographic groups.
    """
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    @torch.no_grad()
    def predict_multimodal(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        disable_modalities: tuple = (),
        use_dynamic_gating: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        model.eval()
        p_v, p_a, p_any = [], [], []
        targets = []
        uids = []

        for batch in data_loader:
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(b, stage=2, disable_modalities=disable_modalities, use_dynamic_gating=use_dynamic_gating)

            p_v.extend(torch.sigmoid(out["logits_video"]).cpu().numpy())
            p_a.extend(torch.sigmoid(out["logits_audio"]).cpu().numpy())
            p_any.extend(torch.sigmoid(out["logits_any"]).cpu().numpy())
            targets.append(b["y"].cpu().numpy())
            uids.extend(batch["uid"])

        return {
            "p_video": np.array(p_v),
            "p_audio": np.array(p_a),
            "p_any": np.array(p_any),
            "targets": np.concatenate(targets, axis=0),
            "uids": uids,
        }

    @torch.no_grad()
    def predict_baseline(self, baseline_model: nn.Module, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        baseline_model.eval()
        p_v = []
        targets = []
        uids = []

        for batch in data_loader:
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = baseline_model(b)
            p_v.extend(out["probabilities"].cpu().numpy())
            targets.append(b["y"].cpu().numpy())
            uids.extend(batch["uid"])

        p_v_arr = np.array(p_v)
        return {
            "p_video": p_v_arr,
            "p_audio": np.zeros_like(p_v_arr),
            "p_any": p_v_arr,
            "targets": np.concatenate(targets, axis=0),
            "uids": uids,
        }

    def run_benchmark_suite(
        self,
        trimodal_model: nn.Module,
        baseline_model: Optional[nn.Module],
        val_loader: DataLoader,
        test_loader: DataLoader,
        test_df: pd.DataFrame,
        output_dir: str = "results",
    ) -> Dict[str, Dict]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Determine optimal threshold on validation set
        val_preds = self.predict_multimodal(trimodal_model, val_loader)
        th_any = EvaluatorMetrics.find_optimal_threshold(val_preds["targets"][:, 2], val_preds["p_any"])
        th_v = EvaluatorMetrics.find_optimal_threshold(val_preds["targets"][:, 0], val_preds["p_video"])
        th_a = EvaluatorMetrics.find_optimal_threshold(val_preds["targets"][:, 1], val_preds["p_audio"])

        print(f"Optimal Validation Thresholds -> Any: {th_any:.3f}, Video: {th_v:.3f}, Audio: {th_a:.3f}")

        # 2. Configurations for ablations
        experiments = {
            "Proposed (Tri-Modal + Dynamic Gating)": {"disable": (), "gating": True},
            "Ablation: Tri-Modal without Gating": {"disable": (), "gating": False},
            "Ablation: Audio + Video (No rPPG)": {"disable": ("rppg",), "gating": False},
            "Ablation: Audio-Only": {"disable": ("video", "rppg"), "gating": False},
            "Ablation: Video-Only": {"disable": ("audio", "rppg"), "gating": False},
        }

        results: Dict[str, Dict] = {}

        for exp_name, cfg in experiments.items():
            test_preds = self.predict_multimodal(
                trimodal_model, test_loader,
                disable_modalities=cfg["disable"],
                use_dynamic_gating=cfg["gating"]
            )
            y_true_any = test_preds["targets"][:, 2]
            metrics_any = EvaluatorMetrics.compute_binary_metrics(y_true_any, test_preds["p_any"], threshold=th_any)
            metrics_video = EvaluatorMetrics.compute_binary_metrics(test_preds["targets"][:, 0], test_preds["p_video"], threshold=th_v)
            metrics_audio = EvaluatorMetrics.compute_binary_metrics(test_preds["targets"][:, 1], test_preds["p_audio"], threshold=th_a)

            results[exp_name] = {
                "any_fake": metrics_any,
                "video_fake": metrics_video,
                "audio_fake": metrics_audio,
            }

        # 3. Baseline: rPPG-only
        if baseline_model is not None:
            base_preds = self.predict_baseline(baseline_model, test_loader)
            th_base = EvaluatorMetrics.find_optimal_threshold(base_preds["targets"][:, 0], base_preds["p_video"])
            base_metrics = EvaluatorMetrics.compute_binary_metrics(base_preds["targets"][:, 0], base_preds["p_video"], threshold=th_base)
            results["Baseline: rPPG-Only Transformer"] = {
                "any_fake": base_metrics,
                "video_fake": base_metrics,
                "audio_fake": {"roc_auc": 0.5, "accuracy": 0.5, "f1": 0.0},
            }

        # 4. Demographic audit (Race & Gender) for Proposed vs No-rPPG
        full_preds = self.predict_multimodal(trimodal_model, test_loader, disable_modalities=(), use_dynamic_gating=True)
        no_rppg_preds = self.predict_multimodal(trimodal_model, test_loader, disable_modalities=("rppg",), use_dynamic_gating=False)

        subgroup_audits = {}
        for col in ["race", "gender", "category"]:
            if col in test_df.columns:
                subgroup_audits[col] = {}
                for grp_name, grp_indices in test_df.groupby(col).indices.items():
                    idx = np.array(list(grp_indices))
                    if len(idx) < 5:
                        continue
                    y_sub = full_preds["targets"][idx, 2]
                    # Only compute AUC if both classes exist in subgroup
                    if len(np.unique(y_sub)) > 1:
                        auc_full = EvaluatorMetrics.compute_binary_metrics(y_sub, full_preds["p_any"][idx], threshold=th_any)["roc_auc"]
                        auc_no_rppg = EvaluatorMetrics.compute_binary_metrics(y_sub, no_rppg_preds["p_any"][idx], threshold=th_any)["roc_auc"]
                    else:
                        auc_full = float(np.mean(y_sub == (full_preds["p_any"][idx] >= th_any)))
                        auc_no_rppg = float(np.mean(y_sub == (no_rppg_preds["p_any"][idx] >= th_any)))

                    subgroup_audits[col][str(grp_name)] = {
                        "count": int(len(idx)),
                        "auc_full_trimodal": round(float(auc_full), 4),
                        "auc_without_rppg": round(float(auc_no_rppg), 4),
                    }

        results["subgroup_fairness_audit"] = subgroup_audits

        # Save JSON report
        report_file = out_path / "benchmark_evaluation_results.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark results successfully saved to: {report_file}")
        return results
