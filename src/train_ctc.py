"""train_ctc.py — train CRNN+CTC on labeled Hebrew crops, log to ClearML."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from clearml import Task  # noqa: F401  # module-level for test patchability — RESEARCH.md Pattern 6
from torch.utils.data import DataLoader

from src.clearml_utils import init_task, remap_dataset_paths, upload_file_artifact
from src.ctc_utils import (
    CRNN,
    AugmentTransform,
    CropDataset,
    build_charset,
    build_half_page_units,
    cer,
    crnn_collate,
    greedy_decode,
    predict_single,
    resolve_device,
    save_charset,
    split_units,
)

DEBUG_SAMPLES = 5


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
    p.add_argument("--min_labeled", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--aug_copies", type=int, default=0,
                   help="Augmented copies per original crop (0 = disabled, per D-05)")
    p.add_argument("--rotation_max", type=float, default=7.0,
                   help="Max rotation in degrees for augmentation (D-02)")
    p.add_argument("--brightness_delta", type=float, default=0.10,
                   help="Max brightness multiplicative delta for augmentation (D-02)")
    p.add_argument("--noise_sigma", type=float, default=0.02,
                   help="Gaussian noise sigma for augmentation (D-02)")
    p.add_argument("--enqueue", action="store_true", default=False,
                   help="Enqueue task to ClearML queue instead of running locally (D-07)")
    p.add_argument("--queue_name", type=str, default="gpu",
                   help="ClearML queue name when --enqueue is set (D-07)")
    p.add_argument("--dataset_id", type=str, default=None,
                   help="ClearML dataset ID; downloads and remaps paths in-memory (D-09)")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    tags = ["phase-4", "gpu"] if args.enqueue else ["phase-4"]
    task = init_task("handwriting-hebrew-ocr", "train_baseline_ctc", tags=tags)

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

    if args.dataset_id is not None:
        labeled = remap_dataset_paths(labeled, args.dataset_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # TRAN-07: connect ALL hyperparameters; MUST come before execute_remotely
    task.connect(vars(args), name="hyperparams")

    if args.enqueue:
        task.execute_remotely(queue_name=args.queue_name)
        # local process exits here; agent re-runs from top and skips this call

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

    augment: AugmentTransform | None = None
    if args.aug_copies > 0:
        augment = AugmentTransform(
            rotation_max=args.rotation_max,
            brightness_delta=args.brightness_delta,
            noise_sigma=args.noise_sigma,
        )
        effective_n = len(train_idx) * (1 + args.aug_copies)
        print(
            f"augmentation: aug_copies={args.aug_copies}, "
            f"effective dataset size: {effective_n} "
            f"(was {len(train_idx)})"
        )
        task.connect({"effective_train_size": effective_n}, name="hyperparams")

    train_ds = CropDataset(
        labeled.iloc[train_idx].reset_index(drop=True),
        charset,
        augment=augment,        # None when aug_copies=0 (D-05)
        aug_copies=args.aug_copies,
    )
    val_df = labeled.iloc[val_idx].reset_index(drop=True)
    val_ds = CropDataset(val_df, charset)  # no augment — D-04 val always clean
    n_debug = min(DEBUG_SAMPLES, len(val_df))
    debug_samples = [
        (str(val_df.iloc[i]["crop_path"]), str(val_df.iloc[i]["label"])) for i in range(n_debug)
    ]
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
                val_loss_sum += ctc_loss(log_probs, labels, input_lengths, target_lengths).item()
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

        with torch.no_grad():
            lines = [f"=== debug samples epoch={epoch} ==="]
            for i, (crop_path, gt) in enumerate(debug_samples):
                pred = predict_single(model, charset, device, crop_path)
                lines.append(f"[{i}] {crop_path} | gt={gt} | pred={pred}")
            text_block = "\n".join(lines)
        logger.report_text(
            title="debug_samples",
            series="val",
            iteration=epoch,
            print_console=False,
            msg=text_block,
        )

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
