"""train_ctc.py — train CRNN+CTC on labeled Hebrew crops, log to ClearML."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from clearml import Task  # noqa: F401  # module-level for test patchability — RESEARCH.md Pattern 6
from torch.utils.data import DataLoader, Dataset

from src.clearml_utils import init_task, upload_file_artifact
from src.ctc_utils import (
    CRNN,
    build_charset,
    build_half_page_units,
    cer,
    crnn_collate,
    encode_label,
    greedy_decode,
    load_crop,
    resolve_device,
    save_charset,
    split_units,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CRNN+CTC on labeled Hebrew crops; log to ClearML."
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


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, charset: list[str]) -> None:
        self._df = df
        self._charset = charset

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[int]]:
        row = self._df.iloc[index]
        image = load_crop(str(row["crop_path"]))
        label_ids = encode_label(str(row["label"]), self._charset)
        return image, label_ids


def main() -> int:
    # Pitfall 2: Task.init BEFORE argparse.parse_args()
    task = init_task("handwriting-hebrew-ocr", "train_baseline_ctc", tags=["phase-3"])

    parser = _build_parser()
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: --manifest does not exist: {args.manifest}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.manifest)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)  # TRAN-01

    if len(labeled) < args.min_labeled:
        print(
            f"ERROR: only {len(labeled)} labeled crops; need at least {args.min_labeled}.",
            file=sys.stderr,
        )
        return 3

    if labeled["label"].fillna("").eq("").any():
        print("ERROR: at least one labeled row has an empty label.", file=sys.stderr)
        return 4

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TRAN-07: connect ALL hyperparameters
    task.connect(vars(args), name="hyperparams")

    # TRAN-02: charset built from labeled labels (NFC inside build_charset)
    charset = build_charset(labeled["label"].tolist())
    save_charset(args.output_dir / "charset.json", charset)

    # TRAN-03: half-page split (D-03 + D-04)
    units = build_half_page_units(labeled)
    train_keys, val_keys = split_units(units, val_frac=args.val_frac)
    train_idx = [i for k in train_keys for i in units[k]]
    val_idx = [i for k in val_keys for i in units[k]]

    if not train_idx or not val_idx:
        print(
            f"ERROR: split produced empty set (train={len(train_idx)}, val={len(val_idx)}).",
            file=sys.stderr,
        )
        return 5

    device = resolve_device()  # TRAN-05

    train_ds = CropDataset(labeled.iloc[train_idx].reset_index(drop=True), charset)
    val_ds = CropDataset(labeled.iloc[val_idx].reset_index(drop=True), charset)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=crnn_collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=crnn_collate,
        num_workers=args.num_workers,
    )

    model = CRNN(num_classes=len(charset) + 1).to(device)  # TRAN-04 (+1 for blank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")

    logger = task.get_logger()
    best_val_cer = float("inf")
    checkpoint_path = args.output_dir / "checkpoint.pt"

    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        train_loss_sum, train_steps = 0.0, 0
        for images, labels, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            logits = model(images)  # (T, N, C)
            log_probs = logits.log_softmax(2)  # Pitfall 4
            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1
        train_loss = train_loss_sum / max(train_steps, 1)

        # --- val ---
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        cer_total, cer_count = 0.0, 0
        with torch.no_grad():
            for images, labels, input_lengths, target_lengths in val_loader:
                images = images.to(device)
                logits = model(images)
                log_probs = logits.log_softmax(2)
                val_loss_sum += ctc_loss(
                    log_probs, labels, input_lengths, target_lengths
                ).item()
                val_steps += 1
                # Per-sample CER
                offset = 0
                for n in range(log_probs.size(1)):
                    sample_log_probs = log_probs[:, n, :]  # (T, C)
                    pred_indices = greedy_decode(sample_log_probs)  # list[int]
                    pred_text = "".join(charset[i - 1] for i in pred_indices)
                    tgt_len = int(target_lengths[n].item())
                    tgt_indices = labels[offset : offset + tgt_len].tolist()
                    tgt_text = "".join(charset[i - 1] for i in tgt_indices)
                    offset += tgt_len
                    cer_total += cer(tgt_text, pred_text)
                    cer_count += 1
        val_loss = val_loss_sum / max(val_steps, 1)
        val_cer = cer_total / max(cer_count, 1)

        # TRAN-08: report scalars
        logger.report_scalar(title="loss", series="train", iteration=epoch, value=train_loss)
        logger.report_scalar(title="loss", series="val", iteration=epoch, value=val_loss)
        logger.report_scalar(title="cer", series="val", iteration=epoch, value=val_cer)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), checkpoint_path)  # TRAN-06

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_cer={val_cer:.4f} best_val_cer={best_val_cer:.4f}"
        )

    # TRAN-08: artifact uploads at end of training
    if checkpoint_path.exists():
        upload_file_artifact(task, "checkpoint", checkpoint_path)
    upload_file_artifact(task, "charset", args.output_dir / "charset.json")

    print(
        f"Done. best_val_cer={best_val_cer:.4f} "
        f"checkpoint={checkpoint_path} charset={args.output_dir / 'charset.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
