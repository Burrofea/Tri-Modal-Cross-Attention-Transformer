"""Script to train the standalone physiological rPPG-only baseline model."""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import MultimodalDataset
from src.models.rppg_baseline import RPPGBaselineModel
from src.training.trainer import BaselineTrainer

def main():
    parser = argparse.ArgumentParser(description="Train rPPG-only baseline model.")
    parser.add_argument("--index", type=str, default="index.csv", help="Path to manifest CSV")
    parser.add_argument("--feats", type=str, default="feats", help="Features directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--out", type=str, default="rppg_baseline.pt", help="Checkpoint filename")
    args = parser.parse_args()

    df = pd.read_csv(args.index)
    train_ds = MultimodalDataset(df, args.feats, split="train")
    val_ds = MultimodalDataset(df, args.feats, split="val")

    print(f"Loaded rPPG dataset: Train={len(train_ds)}, Val={len(val_ds)}")
    if len(train_ds) == 0:
        print("Error: No training features found! Please run feature extraction first.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RPPGBaselineModel()
    trainer = BaselineTrainer(model, train_loader, val_loader, device=args.device, lr=args.lr)
    trainer.fit(epochs=args.epochs, save_name=args.out)

if __name__ == "__main__":
    main()
