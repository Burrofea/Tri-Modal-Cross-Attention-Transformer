"""Script to train the Tri-Modal Cross-Attention Deepfake Detector."""
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
from src.training.trainer import MultimodalTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Tri-Modal Cross-Attention Deepfake Detector.")
    parser.add_argument("--index", type=str, default="index.csv", help="Path to manifest CSV")
    parser.add_argument("--feats", type=str, default="feats", help="Features directory")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="Stage 1 (warmup) or Stage 2 (cross-attention)")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Initial checkpoint path to load")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--out", type=str, default="trimodal_model.pt", help="Checkpoint output filename")
    args = parser.parse_args()

    df = pd.read_csv(args.index)
    train_ds = MultimodalDataset(df, args.feats, split="train")
    val_ds = MultimodalDataset(df, args.feats, split="val")

    print(f"Loaded Multimodal dataset: Train={len(train_ds)}, Val={len(val_ds)}")
    if len(train_ds) == 0:
        print("Error: No training features found! Please run feature extraction first.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TriModalTransformer()
    if args.init_ckpt and Path(args.init_ckpt).exists():
        print(f"Loading initial weights from {args.init_ckpt}...")
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    trainer = MultimodalTrainer(model, train_loader, val_loader, device=args.device, lr=args.lr)
    trainer.fit(epochs=args.epochs, stage=args.stage, save_name=args.out)

if __name__ == "__main__":
    main()
