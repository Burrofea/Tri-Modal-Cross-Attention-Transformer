"""Multimodal feature extraction and disk caching script."""
import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.rppg import RPPGPreprocessor
from src.preprocessing.video import VideoPreprocessor

def process_single_video(
    uid: str,
    path: str,
    out_dir: str,
    audio_prep: AudioPreprocessor,
    rppg_prep: RPPGPreprocessor,
    video_prep: VideoPreprocessor
) -> Tuple[str, Optional[str]]:
    npz_path = os.path.join(out_dir, f"{uid}.npz")
    emb_path = os.path.join(out_dir, "emb", f"{uid}.npy")

    if os.path.exists(npz_path) and os.path.exists(emb_path):
        return uid, None

    try:
        # Audio extraction
        mel, _ = audio_prep.process(path)

        # rPPG and Face extraction
        rppg_res = rppg_prep.process(path)
        crops = rppg_res["crops"]
        rppg_tok = rppg_res["rppg_tok"]
        rppg_mask = rppg_res["rppg_mask"]
        snr = rppg_res["snr"]
        quality = rppg_res["quality"]

        # Video embedding
        emb = video_prep.embed_crops(crops)

        # Save files
        np.savez(
            npz_path,
            mel=mel,
            crops=crops,
            rppg_tok=rppg_tok,
            rppg_mask=rppg_mask,
            snr=snr,
            quality=quality,
        )
        np.save(emb_path, emb)
        return uid, None
    except Exception:
        return uid, traceback.format_exc()

def main():
    parser = argparse.ArgumentParser(description="Extract and cache multimodal features.")
    parser.add_argument("--index", type=str, default="index.csv", help="Path to manifest CSV")
    parser.add_argument("--out", type=str, default="feats", help="Output features directory")
    parser.add_argument("--limit", type=int, default=0, help="Debug: process only first N samples")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "emb"), exist_ok=True)

    df = pd.read_csv(args.index)
    if args.limit > 0:
        df = df.head(args.limit)

    print(f"Extracting features for {len(df)} samples into '{args.out}'...")

    audio_prep = AudioPreprocessor()
    rppg_prep = RPPGPreprocessor()
    video_prep = VideoPreprocessor(device="cuda" if torch.cuda.is_available() else "cpu")

    failures = 0
    for row in tqdm(df.itertuples(), total=len(df), desc="Extracting"):
        uid, err = process_single_video(
            row.uid, row.path, args.out, audio_prep, rppg_prep, video_prep
        )
        if err:
            failures += 1
            if failures <= 3:
                print(f"\n[Error on {uid} ({row.path})]:\n{err}")

    print(f"\nFeature extraction finished. Total samples processed: {len(df) - failures}, Failures: {failures}")

if __name__ == "__main__":
    main()
