"""train_ctc.py — train CRNN+CTC on labeled Hebrew crops, log to ClearML."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from clearml import Task  # noqa: F401  # module-level for test patchability — RESEARCH.md Pattern 6

from src.clearml_utils import init_task, remap_dataset_paths, upload_file_artifact

DEBUG_SAMPLES = 5


def _apply_params_file(args: argparse.Namespace) -> None:
    """Mutate args from a JSON file, casting each value to the existing arg's type.

    Unknown keys are ignored. Per RESEARCH.md Pitfall 4: cast via the destination
    attr's existing type so e.g. `8.0` in JSON -> int 8 if args.batch_size is int.
    """
    if args.params is None:
        return
    if not args.params.exists():
        raise FileNotFoundError(f"--params file not found: {args.params}")
    loaded = json.loads(args.params.read_text())
    for k, v in loaded.items():
        if hasattr(args, k):
            existing = getattr(args, k)
            if existing is None:
                setattr(args, k, v)
            else:
                setattr(args, k, type(existing)(v))


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
    p.add_argument(
        "--aug_copies",
        type=int,
        default=4,
        help="Augmented copies per original crop (0 = disabled, per D-05)",
    )
    p.add_argument(
        "--rotation_max",
        type=float,
        default=7.0,
        help="Max rotation in degrees for augmentation (D-02)",
    )
    p.add_argument(
        "--brightness_delta",
        type=float,
        default=0.10,
        help="Max brightness multiplicative delta for augmentation (D-02)",
    )
    p.add_argument(
        "--noise_sigma",
        type=float,
        default=0.02,
        help="Gaussian noise sigma for augmentation (D-02)",
    )
    p.add_argument(
        "--rnn_hidden",
        type=int,
        default=256,
        choices=[128, 256, 512],
        help="CRNN BiLSTM hidden size (D-02 Phase 5)",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=2,
        choices=[1, 2],
        help="CRNN BiLSTM layer count (D-02 Phase 5)",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (regularization C from tune analysis)",
    )
    p.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Load hyperparams from JSON (best_params.json) — D-10 Phase 5",
    )
    p.add_argument(
        "--enqueue",
        action="store_true",
        default=False,
        help="Enqueue task to ClearML queue instead of running locally (D-07)",
    )
    p.add_argument(
        "--queue_name",
        type=str,
        default="gpu",
        help="ClearML queue name when --enqueue is set (D-07)",
    )
    p.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="ClearML dataset ID; downloads and remaps paths in-memory (D-09)",
    )
    return p


def _report_prob_heatmap(
    logger: Any,
    probs_np: Any,
    charset: list[str],
    gt: str,
    pred: str,
    epoch: int,
    sample_idx: int,
) -> None:
    """Log CTC probability heatmap (T × top-N classes) to ClearML Debug Samples tab.

    Rows are classes (top-30 by peak prob, blank always included and shown in red).
    Columns are timesteps. Bright = high probability. Blank dominating all columns
    is the visual signature of blank collapse.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ["<blank>"] + list(charset)
    max_by_class = probs_np.max(axis=0)
    n_show = min(30, len(labels))
    top_idx = sorted(range(len(labels)), key=lambda i: -max_by_class[i])[:n_show]
    if 0 not in top_idx:
        top_idx = [0] + top_idx[: n_show - 1]
    top_idx = sorted(top_idx)

    sub = probs_np[:, top_idx].T  # (n_show, T)
    sub_labels = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, sub.shape[1] // 3), max(4, len(sub_labels) // 2)))
    im = ax.imshow(sub, aspect="auto", cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(sub_labels)))
    ax.set_yticklabels(sub_labels, fontsize=7)
    ax.set_xlabel("Timestep")
    ax.set_title(f"sample_{sample_idx} epoch={epoch} | gt={gt!r} | pred={pred!r}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    if "<blank>" in sub_labels:
        ax.get_yticklabels()[sub_labels.index("<blank>")].set_color("red")
    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="prob_heatmap",
        series=f"sample_{sample_idx}",
        iteration=epoch,
        figure=fig,
        report_image=True,
        report_interactive=False,
    )
    plt.close(fig)


def _report_annotated_crop(
    logger: Any,
    crop_hw: Any,
    gt: str,
    pred: str,
    epoch: int,
    sample_idx: int,
    series_prefix: str = "sample",
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(max(4, crop_hw.shape[1] / 40), 2.0))
    ax.imshow(crop_hw, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    color = "green" if gt == pred else "red"
    ax.set_title(
        f"epoch={epoch}  gt={gt!r}  pred={pred!r}",
        fontsize=10,
        color=color,
    )
    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="debug_samples",
        series=f"{series_prefix}_{sample_idx}",
        iteration=epoch,
        figure=fig,
        report_image=True,
        report_interactive=False,
    )
    plt.close(fig)


def _report_saliency_panel(
    logger: Any,
    model: Any,
    charset: list[str],
    device: Any,
    picks: list[tuple[str, str, float]],
    epoch: int,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from src.ctc_utils import compute_char_saliency  # noqa: PLC0415

    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 2.2 * len(picks)))
    if len(picks) == 1:
        axes = [axes]
    for ax, (crop_path, gt, sample_cer) in zip(axes, picks, strict=True):
        crop_hw, pred, sal = compute_char_saliency(model, charset, device, crop_path)
        ax.imshow(crop_hw, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
        ax.imshow(sal, cmap="jet", aspect="auto", alpha=0.5, vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"cer={sample_cer:.2f}  gt={gt!r}  pred={pred!r}", fontsize=9)
    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="saliency_chars",
        series="percentiles",
        iteration=epoch,
        figure=fig,
        report_image=True,
        report_interactive=False,
    )
    plt.close(fig)


def run_training(
    args: argparse.Namespace,
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> float:
    """Run train+val for args.epochs. Returns best_val_cer.

    Pre-conditions: args.manifest exists; args.output_dir is writable; the caller
    (main() or tune.py) has already initialised a ClearML task and called
    task.connect(vars(args)). This function reads task = Task.current_task() to
    get the logger.

    on_epoch_end(epoch, val_cer) is called after each validation pass when provided.
    Used by tune.py for Optuna pruning. If the callback raises, the exception
    propagates and no further epochs run. The most recent best checkpoint (if any
    was saved earlier in the loop) remains on disk.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.ctc_utils import (  # noqa: PLC0415
        CRNN,
        AugmentTransform,
        CropDataset,
        build_charset,
        build_half_page_units,
        cer,
        crnn_collate,
        greedy_decode,
        load_crop,
        predict_single_with_probs,
        resolve_device,
        save_charset,
        split_units,
    )

    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(args.manifest)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)  # TRAN-01

    if args.dataset_id is not None:
        labeled = remap_dataset_paths(labeled, args.dataset_id)

    # TRAN-02: charset built from labeled labels (NFC inside build_charset)
    charset = build_charset(labeled["label"].tolist())
    save_charset(args.output_dir / "charset.json", charset)

    # TRAN-03: half-page split (D-03 + D-04)
    units = build_half_page_units(labeled)
    train_keys, val_keys = split_units(units, val_frac=args.val_frac)
    train_idx = [i for k in train_keys for i in units[k]]
    val_idx = [i for k in val_keys for i in units[k]]

    if not train_idx or not val_idx:
        raise ValueError(f"split produced empty set (train={len(train_idx)}, val={len(val_idx)}).")

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
        augment=augment,  # None when aug_copies=0 (D-05)
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

    model = CRNN(  # TRAN-04 (+1 for blank)
        num_classes=len(charset) + 1,
        rnn_hidden=args.rnn_hidden,
        num_layers=args.num_layers,
    ).to(device)
    # Blank index 0 starts with a negative bias so the model doesn't immediately
    # collapse to predicting all-blank. The optimizer can move this freely after
    # the first few epochs once non-blank representations have formed.
    model.fc.bias.data[0] = -2.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    ctc_loss_per_sample = torch.nn.CTCLoss(blank=0, zero_infinity=False, reduction="none")

    best_val_cer = float("inf")
    checkpoint_path = args.output_dir / "checkpoint.pt"

    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        train_loss_sum, train_steps, inf_count = 0.0, 0, 0
        for images, labels, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            logits = model(images)  # (T, N, C)
            log_probs = logits.log_softmax(2)  # Pitfall 4
            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
            with torch.no_grad():
                per_sample = ctc_loss_per_sample(log_probs, labels, input_lengths, target_lengths)
                inf_count += int(per_sample.isinf().sum().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1
        train_loss = train_loss_sum / max(train_steps, 1)

        # --- val ---
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        cer_total, cer_count = 0.0, 0
        blank_frac_sum, empty_preds = 0.0, 0
        per_sample_cer: list[tuple[str, str, str, float]] = []
        dataset_idx = 0
        with torch.no_grad():
            for images, labels, input_lengths, target_lengths in val_loader:
                images = images.to(device)
                logits = model(images)
                log_probs = logits.log_softmax(2)
                val_loss_sum += ctc_loss(log_probs, labels, input_lengths, target_lengths).item()
                val_steps += 1
                blank_frac_sum += (logits.argmax(dim=2) == 0).float().mean().item()
                # Per-sample CER
                offset = 0
                for n in range(log_probs.size(1)):
                    sample_log_probs = log_probs[:, n, :]  # (T, C)
                    pred_indices = greedy_decode(sample_log_probs)  # list[int]
                    pred_text = "".join(charset[i - 1] for i in pred_indices)
                    if not pred_text:
                        empty_preds += 1
                    tgt_len = int(target_lengths[n].item())
                    tgt_indices = labels[offset : offset + tgt_len].tolist()
                    tgt_text = "".join(charset[i - 1] for i in tgt_indices)
                    offset += tgt_len
                    sample_cer = cer(tgt_text, pred_text)
                    cer_total += sample_cer
                    cer_count += 1
                    crop_path_n = str(val_df.iloc[dataset_idx + n]["crop_path"])
                    per_sample_cer.append((crop_path_n, tgt_text, pred_text, sample_cer))
                dataset_idx += images.size(0)
        val_loss = val_loss_sum / max(val_steps, 1)
        val_cer = cer_total / max(cer_count, 1)
        blank_frac = blank_frac_sum / max(val_steps, 1)
        empty_frac = empty_preds / max(cer_count, 1)

        scheduler.step(val_cer)

        # TRAN-08: report scalars
        logger.report_scalar(title="loss", series="train", iteration=epoch, value=train_loss)
        logger.report_scalar(title="loss", series="val", iteration=epoch, value=val_loss)
        logger.report_scalar(title="cer", series="val", iteration=epoch, value=val_cer)
        logger.report_scalar(title="blank_frac", series="val", iteration=epoch, value=blank_frac)
        logger.report_scalar(title="empty_frac", series="val", iteration=epoch, value=empty_frac)
        logger.report_scalar(
            title="inf_loss_count", series="train", iteration=epoch, value=inf_count
        )
        logger.report_scalar(
            title="lr", series="train", iteration=epoch,
            value=optimizer.param_groups[0]["lr"],
        )

        with torch.no_grad():
            for i, (crop_path, gt) in enumerate(debug_samples):
                pred, probs = predict_single_with_probs(model, charset, device, crop_path)
                print(f"[{i}] gt={gt!r} pred={pred!r}")
                raw = load_crop(crop_path).squeeze(0).numpy()  # (H, W) float [0, 1]
                _report_annotated_crop(logger, raw, gt, pred, epoch, i)
                _report_prob_heatmap(
                    logger, probs.cpu().numpy(), charset, gt, pred, epoch, i
                )

        if per_sample_cer:
            sorted_by_cer = sorted(per_sample_cer, key=lambda r: r[3])
            n = len(sorted_by_cer)
            pct_indices = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
            picks = [
                (sorted_by_cer[idx][0], sorted_by_cer[idx][1], sorted_by_cer[idx][3])
                for idx in pct_indices
            ]
            _report_saliency_panel(logger, model, charset, device, picks, epoch)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), checkpoint_path)  # TRAN-06

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_cer={val_cer:.4f} best_val_cer={best_val_cer:.4f} "
            f"blank_frac={blank_frac:.3f} empty_frac={empty_frac:.3f} inf_loss={inf_count}"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_cer)

    # TRAN-08: artifact uploads at end of training
    if checkpoint_path.exists():
        upload_file_artifact(task, "checkpoint", checkpoint_path)
    upload_file_artifact(task, "charset", args.output_dir / "charset.json")

    return best_val_cer


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _apply_params_file(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 6

    tags = ["phase-4", "gpu"] if args.enqueue else ["phase-4"]
    if args.params is not None:
        tags.append("phase-5")
    task = init_task("handwriting-hebrew-ocr", "train_baseline_ctc", tags=tags)

    # TRAN-07: connect ALL hyperparameters; MUST come before execute_remotely
    task.connect(vars(args), name="hyperparams")

    if args.enqueue:
        task.execute_remotely(queue_name=args.queue_name)
        # local process exits here via os._exit(); code below only runs on agent

    # Resolve manifest from dataset when not available locally (agent path)
    if args.dataset_id is not None and not args.manifest.exists():
        from clearml import Dataset  # noqa: PLC0415

        args.manifest = (
            Path(Dataset.get(dataset_id=args.dataset_id).get_local_copy()) / "manifest.csv"
        )

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

    try:
        best_val_cer = run_training(args)
    except ValueError as e:
        # split produced empty set (preserved exit code 5)
        print(f"ERROR: {e}", file=sys.stderr)
        return 5

    print(f"Done. best_val_cer={best_val_cer:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
