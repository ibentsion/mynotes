"""experiment.py — run train_ctc then evaluate in one command."""

import argparse
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run train_ctc then evaluate end-to-end. Accepts all train hyperparams."
    )
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/model"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--min_labeled", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    train_rc = subprocess.run(
        [
            sys.executable, "-m", "src.train_ctc",
            "--manifest", str(args.manifest),
            "--output_dir", str(args.output_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--val_frac", str(args.val_frac),
            "--min_labeled", str(args.min_labeled),
            "--num_workers", str(args.num_workers),
        ]
    ).returncode
    if train_rc != 0:
        print(f"ERROR: train_ctc failed (exit {train_rc})", file=sys.stderr)
        return train_rc

    return subprocess.run(
        [
            sys.executable, "-m", "src.evaluate",
            "--manifest", str(args.manifest),
            "--output_dir", str(args.output_dir),
            "--val_frac", str(args.val_frac),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
