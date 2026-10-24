"""Script to scan FakeAVCeleb directory, generate labels, and create speaker-disjoint splits."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.scanner import FakeAVCelebScanner
from src.data.splitter import GroupedSplitter

def main():
    parser = argparse.ArgumentParser(description="Scan FakeAVCeleb and build speaker-disjoint dataset manifest.")
    parser.add_argument("--root", type=str, default=r"E:\rPPG\FakeAVCeleb_v1.2", help="FakeAVCeleb root directory")
    parser.add_argument("--out", type=str, default="index.csv", help="Output CSV path")
    parser.add_argument("--per_type", type=int, default=1200, help="Cap per manipulation category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Scanning FakeAVCeleb dataset at: {args.root}")
    scanner = FakeAVCelebScanner(args.root)
    df = scanner.scan(per_type_limit=args.per_type, random_seed=args.seed)

    print(f"Total discovered samples: {len(df)}")
    print("Class distribution:")
    print(df["category"].value_counts())

    print("\nApplying speaker-disjoint 70/15/15 split...")
    splitter = GroupedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    df_split = splitter.split(df, group_col="speaker")

    df_split.to_csv(args.out, index=False)
    print(f"\nManifest saved to: {args.out}")
    print("\nSamples per Split and Category:")
    print(df_split.groupby(["split", "category"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()
