"""Training loops, validation, and checkpoint management for multimodal and baseline models."""
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .losses import MultiTaskConsistencyLoss

class MultimodalTrainer:
    """
    Manages two-stage training for the Tri-Modal Cross-Attention Transformer.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = MultiTaskConsistencyLoss()

    def train_epoch(self, stage: int = 2) -> Dict[str, float]:
        self.model.train()
        total_losses = {"loss": 0.0, "loss_video": 0.0, "loss_audio": 0.0, "loss_any": 0.0}
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.optimizer.zero_grad()

            outputs = self.model(b, stage=stage)
            loss_dict = self.criterion(outputs, b["y"], stage=stage)

            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in total_losses:
                if k in loss_dict:
                    total_losses[k] += float(loss_dict[k].item())
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, stage: int = 2) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_targets = []
        all_p_v, all_p_a, all_p_any = [], [], []

        for batch in self.val_loader:
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(b, stage=stage)
            loss_dict = self.criterion(outputs, b["y"], stage=stage)
            total_loss += float(loss_dict["loss"].item())
            n_batches += 1

            targets = b["y"].cpu().numpy()
            all_targets.append(targets)

            if stage == 1:
                p_v = torch.sigmoid(outputs["aux_v"]).cpu().numpy()
                p_a = torch.sigmoid(outputs["aux_a"]).cpu().numpy()
                all_p_v.extend(p_v)
                all_p_a.extend(p_a)
                all_p_any.extend(np.maximum(p_v, p_a))
            else:
                all_p_v.extend(torch.sigmoid(outputs["logits_video"]).cpu().numpy())
                all_p_a.extend(torch.sigmoid(outputs["logits_audio"]).cpu().numpy())
                all_p_any.extend(torch.sigmoid(outputs["logits_any"]).cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0)
        avg_loss = total_loss / max(1, n_batches)

        metrics = {"val_loss": avg_loss}
        try:
            metrics["auc_video"] = float(roc_auc_score(all_targets[:, 0], all_p_v))
        except Exception:
            metrics["auc_video"] = 0.5
        try:
            metrics["auc_audio"] = float(roc_auc_score(all_targets[:, 1], all_p_a))
        except Exception:
            metrics["auc_audio"] = 0.5
        try:
            metrics["auc_any"] = float(roc_auc_score(all_targets[:, 2], all_p_any))
        except Exception:
            metrics["auc_any"] = 0.5

        return metrics

    def fit(self, epochs: int = 15, stage: int = 2, patience: int = 4, save_name: str = "best_model.pt") -> str:
        best_auc = -1.0
        patience_counter = 0
        best_path = str(self.checkpoint_dir / save_name)

        print(f"--- Starting Training (Stage {stage}) on {self.device} for {epochs} epochs ---")
        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(stage=stage)
            val_metrics = self.evaluate(stage=stage)
            elapsed = time.time() - t0

            primary_auc = val_metrics.get("auc_any", 0.5)
            print(f"Epoch {ep:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val AUC (any): {primary_auc:.4f} (v: {val_metrics.get('auc_video',0.5):.3f}, a: {val_metrics.get('auc_audio',0.5):.3f})")

            if primary_auc > best_auc:
                best_auc = primary_auc
                patience_counter = 0
                torch.save({
                    "stage": stage,
                    "epoch": ep,
                    "state_dict": self.model.state_dict(),
                    "val_metrics": val_metrics,
                }, best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {ep} (patience={patience})")
                    break

        print(f"Training Stage {stage} completed. Best checkpoint saved to: {best_path}")
        return best_path


class BaselineTrainer:
    """
    Trains and evaluates the standalone rPPG-only baseline model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 2e-4,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, epochs: int = 15, patience: int = 4, save_name: str = "rppg_baseline.pt") -> str:
        best_auc = -1.0
        patience_counter = 0
        best_path = str(self.checkpoint_dir / save_name)

        print(f"--- Starting rPPG Baseline Training on {self.device} for {epochs} epochs ---")
        for ep in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                self.optimizer.zero_grad()
                out = self.model(b)
                y_v = b["y"][:, 0]  # Video fake target
                loss = self.criterion(out["logits_video"], y_v)
                loss.backward()
                self.optimizer.step()
                train_loss += float(loss.item())
                n_batches += 1

            train_loss /= max(1, n_batches)

            # Validation
            self.model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in self.val_loader:
                    b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = self.model(b)
                    all_preds.extend(out["probabilities"].cpu().numpy())
                    all_targets.extend(b["y"][:, 0].cpu().numpy())

            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            try:
                val_auc = float(roc_auc_score(all_targets, all_preds))
            except Exception:
                val_auc = 0.5

            print(f"Baseline Epoch {ep:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | Val Video AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save({"epoch": ep, "state_dict": self.model.state_dict(), "auc": val_auc}, best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {ep}")
                    break

        print(f"rPPG Baseline training finished. Checkpoint saved: {best_path}")
        return best_path
