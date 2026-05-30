"""train_ctc.py — train CRNN+CTC on labeled Hebrew crops, log to ClearML."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from clearml import Task  # noqa: F401  # module-level for test patchability — RESEARCH.md Pattern 6

from src.clearml_utils import (
    init_task,
    remap_dataset_paths,
    remap_synthetic_paths,
    upload_file_artifact,
)
from src.run_config import load_config, peek_mode

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
        "--elastic_alpha",
        type=float,
        default=0.0,
        help=(
            "Elastic deformation alpha (displacement magnitude). "
            "0 = disabled (D-02). Try 30-80 for visible warping."
        ),
    )
    p.add_argument(
        "--elastic_sigma",
        type=float,
        default=5.0,
        help="Elastic deformation sigma (smoothness). Used only when --elastic_alpha > 0.",
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
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience epochs on val_cer (0 = disabled)",
    )
    p.add_argument(
        "--blank_bias_init",
        type=float,
        default=-2.0,
        help=(
            "Initial fc.bias for CTC blank token (index 0). "
            "Try -3.0 or -4.0 if blank_frac stays > 0.9 after training stabilizes. "
            "Space and other real chars don't need suppression — blank is synthetic."
        ),
    )
    p.add_argument(
        "--words_file",
        type=Path,
        default=None,
        help="Word list (one word per line) whose characters are merged into the charset "
        "so it stays stable even when labeled data is sparse (e.g. data/words.txt)",
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
        help="ClearML dataset ID. pretrain=synthetic crops, finetune=real labeled data.",
    )
    p.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="Number of pre-training epochs on synthetic data (0 = skip pre-training, D-05).",
    )
    p.add_argument(
        "--pretrain_lr",
        type=float,
        default=1e-3,
        help="Learning rate for pre-training (independent of --lr used for fine-tuning, D-07).",
    )
    p.add_argument(
        "--pretrain_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Checkpoint to load before fine-tuning. Local path (ends in .pt) or "
            "ClearML task ID (fetches checkpoint_pretrain artifact). "
            "Can also be set via config.yaml."
        ),
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
        help="'pretrain' trains on synthetic data; 'finetune' trains on real labeled data.",
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


def _report_char_distribution(
    logger: Any,
    charset: list[str],
    labels: list[str],
) -> None:
    """Log character frequency bar chart + table to ClearML once at training start.

    Shows which characters are rare vs dominant so bias and weighting decisions
    are grounded in data rather than guesswork.
    """
    import unicodedata  # noqa: PLC0415

    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    counts: dict[str, int] = {}
    for lab in labels:
        for ch in unicodedata.normalize("NFC", lab):
            counts[ch] = counts.get(ch, 0) + 1

    sorted_chars = sorted(charset, key=lambda c: -counts.get(c, 0))
    freqs = [counts.get(c, 0) for c in sorted_chars]
    total = max(sum(freqs), 1)
    pcts = [100.0 * f / total for f in freqs]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_chars) * 0.3)))
    y_pos = list(range(len(sorted_chars)))
    ax.barh(y_pos, pcts, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_chars, fontsize=8)
    ax.set_xlabel("% of all characters")
    ax.set_title(f"Character distribution ({total} chars, {len(charset)} unique)")
    ax.invert_yaxis()
    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="char_distribution",
        series="labeled_set",
        iteration=0,
        figure=fig,
        report_image=False,
        report_interactive=False,
    )
    plt.close(fig)

    table = pd.DataFrame({
        "char": sorted_chars,
        "count": freqs,
        "pct": [f"{p:.1f}%" for p in pcts],
    })
    logger.report_table(
        title="char_distribution",
        series="table",
        iteration=0,
        table_plot=table,
    )


def _eval_val_epoch(
    model: Any,
    val_loader: Any,
    val_df: Any,
    *,
    ctc_loss: Any,
    charset: list[str],
    device: Any,
) -> tuple[float, float, float, float, list[tuple[str, str, str, float]]]:
    """Run one val epoch; return (val_loss, val_cer, blank_frac, empty_frac, per_sample_cer)."""
    import torch  # noqa: PLC0415

    from src.ctc_utils import cer, greedy_decode  # noqa: PLC0415

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
            offset = 0
            for n in range(log_probs.size(1)):
                sample_log_probs = log_probs[:, n, :]
                pred_indices = greedy_decode(sample_log_probs)
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
    return val_loss, val_cer, blank_frac, empty_frac, per_sample_cer


def _report_epoch_debug(
    model: Any,
    logger: Any,
    charset: list[str],
    device: Any,
    debug_samples: list[tuple[str, str]],
    per_sample_cer: list[tuple[str, str, str, float]],
    epoch: int,
) -> None:
    """Log per-epoch debug crops, heatmaps, and saliency panel to ClearML."""
    import torch  # noqa: PLC0415

    from src.ctc_utils import load_crop, predict_single_with_probs  # noqa: PLC0415

    with torch.no_grad():
        for i, (crop_path, gt) in enumerate(debug_samples):
            pred, probs = predict_single_with_probs(model, charset, device, crop_path)
            print(f"[{i}] gt={gt!r} pred={pred!r}")
            raw = load_crop(crop_path).squeeze(0).numpy()
            _report_annotated_crop(logger, raw, gt, pred, epoch, i)
            _report_prob_heatmap(logger, probs.cpu().numpy(), charset, gt, pred, epoch, i)
    if per_sample_cer:
        sorted_by_cer = sorted(per_sample_cer, key=lambda r: r[3])
        n = len(sorted_by_cer)
        pct_indices = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
        picks = [
            (sorted_by_cer[idx][0], sorted_by_cer[idx][1], sorted_by_cer[idx][3])
            for idx in pct_indices
        ]
        _report_saliency_panel(logger, model, charset, device, picks, epoch)


def _run_loop(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    val_df: Any,
    optimizer: Any,
    *,
    scheduler: Any,
    ctc_loss: Any,
    ctc_loss_per_sample: Any,
    device: Any,
    args: argparse.Namespace,
    logger: Any,
    charset: list[str],
    checkpoint_path: Path,
    debug_samples: list[tuple[str, str]],
    series_prefix: str = "",
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> float:
    import torch  # noqa: PLC0415

    best_val_cer = float("inf")
    patience_left = args.patience if args.patience > 0 else None
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, train_steps, inf_count = 0.0, 0, 0
        for images, labels, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            logits = model(images)
            log_probs = logits.log_softmax(2)
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
        model.eval()
        val_loss, val_cer, blank_frac, empty_frac, per_sample_cer = _eval_val_epoch(
            model, val_loader, val_df,
            ctc_loss=ctc_loss,
            charset=charset, device=device,
        )
        scheduler.step(val_cer)
        sp = series_prefix
        logger.report_scalar(title="loss", series=f"{sp}train", iteration=epoch, value=train_loss)
        logger.report_scalar(title="loss", series=f"{sp}val", iteration=epoch, value=val_loss)
        logger.report_scalar(title="cer", series=f"{sp}val", iteration=epoch, value=val_cer)
        logger.report_scalar(
            title="blank_frac", series=f"{sp}val", iteration=epoch, value=blank_frac,
        )
        logger.report_scalar(
            title="empty_frac", series=f"{sp}val", iteration=epoch, value=empty_frac,
        )
        logger.report_scalar(
            title="inf_loss_count", series=f"{sp}train", iteration=epoch, value=inf_count,
        )
        logger.report_scalar(
            title="lr", series=f"{sp}train", iteration=epoch,
            value=optimizer.param_groups[0]["lr"],
        )
        _report_epoch_debug(model, logger, charset, device, debug_samples, per_sample_cer, epoch)
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), checkpoint_path)
            if patience_left is not None:
                patience_left = args.patience
        else:
            if patience_left is not None:
                patience_left -= 1
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_cer={val_cer:.4f} best_val_cer={best_val_cer:.4f} "
            f"blank_frac={blank_frac:.3f} empty_frac={empty_frac:.3f} inf_loss={inf_count}"
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, val_cer)
        if patience_left is not None and patience_left == 0:
            print(f"Early stop at epoch={epoch}: patience={args.patience} exhausted.")
            break
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    return best_val_cer


def _run_pretrain(
    model: Any,
    args: argparse.Namespace,
    device: Any,
    logger: Any,
    task: Any,
    charset: list[str],
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> float:
    import math  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.ctc_utils import CropDataset, crnn_collate  # noqa: PLC0415

    synth_df = pd.read_csv(args.manifest)
    if getattr(args, "dataset_id", None) is not None:
        synth_df = remap_synthetic_paths(synth_df, args.dataset_id)
    n = len(synth_df)
    n_val = max(1, math.ceil(n * args.val_frac))
    val_df_pre = synth_df.iloc[:n_val].reset_index(drop=True)
    train_df = synth_df.iloc[n_val:].reset_index(drop=True)
    train_ds = CropDataset(train_df, charset)
    val_ds = CropDataset(val_df_pre, charset)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=crnn_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=crnn_collate, num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6,
    )
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    ctc_loss_per_sample = torch.nn.CTCLoss(blank=0, zero_infinity=False, reduction="none")
    pretrain_args = argparse.Namespace(**vars(args))
    pretrain_args.epochs = args.pretrain_epochs
    checkpoint_pretrain_path = args.output_dir / "checkpoint_pretrain.pt"
    best_pretrain_cer = _run_loop(
        model, train_loader, val_loader, val_df_pre, optimizer,
        scheduler=scheduler,
        ctc_loss=ctc_loss,
        ctc_loss_per_sample=ctc_loss_per_sample,
        device=device,
        args=pretrain_args,
        logger=logger,
        charset=charset,
        checkpoint_path=checkpoint_pretrain_path,
        debug_samples=[],
        series_prefix="pretrain/",
        on_epoch_end=on_epoch_end,
    )
    upload_file_artifact(task, "checkpoint_pretrain", checkpoint_pretrain_path)
    return best_pretrain_cer


def _setup_finetune_loaders(
    labeled: Any,
    args: argparse.Namespace,
    charset: list[str],
    task: Any,
) -> tuple[Any, Any, Any, Any, list[tuple[str, str]]]:
    """Build train/val DataLoaders for fine-tuning. Returns (loader, loader, df, aug, samples)."""
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from src.ctc_utils import (  # noqa: PLC0415
        AugmentTransform,
        CropDataset,
        build_half_page_units,
        crnn_collate,
        split_units,
    )

    units = build_half_page_units(labeled)
    train_keys, val_keys = split_units(units, val_frac=args.val_frac)
    train_idx = [i for k in train_keys for i in units[k]]
    val_idx = [i for k in val_keys for i in units[k]]
    if not train_idx or not val_idx:
        raise ValueError(f"split produced empty set (train={len(train_idx)}, val={len(val_idx)}).")

    augment: AugmentTransform | None = None
    if args.aug_copies > 0:
        augment = AugmentTransform(
            rotation_max=args.rotation_max,
            brightness_delta=args.brightness_delta,
            noise_sigma=args.noise_sigma,
            elastic_alpha=args.elastic_alpha,
            elastic_sigma=args.elastic_sigma,
        )

    train_base_df = labeled.iloc[train_idx].reset_index(drop=True)

    if augment is not None:
        train_base_len = len(train_base_df)
        effective_n = train_base_len * (1 + args.aug_copies)
        print(
            f"augmentation: aug_copies={args.aug_copies}, "
            f"effective dataset size: {effective_n} (was {len(train_idx)})"
        )
        task.connect({"effective_train_size": effective_n}, name="hyperparams")
    train_ds = CropDataset(train_base_df, charset, augment=augment, aug_copies=args.aug_copies)
    val_df = labeled.iloc[val_idx].reset_index(drop=True)
    val_ds = CropDataset(val_df, charset)
    n_debug = min(DEBUG_SAMPLES, len(val_df))
    debug_samples = [
        (str(val_df.iloc[i]["crop_path"]), str(val_df.iloc[i]["label"])) for i in range(n_debug)
    ]
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=crnn_collate, num_workers=args.num_workers,
    )
    # shuffle=False and drop_last=False are required: _eval_val_epoch maps batches back to
    # val_df by sequential index; any reordering or row drop would silently corrupt attribution.
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=crnn_collate, num_workers=args.num_workers,
    )
    return train_loader, val_loader, val_df, augment, debug_samples


def run_training(
    args: argparse.Namespace,
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> float:
    """Run fine-tuning on labeled crops. Returns best_val_cer.

    If args.pretrain_manifest is set, runs synthetic pre-training only and returns
    without fine-tuning (two-call interface — D-05).
    on_epoch_end(epoch, val_cer) is called after each val pass when provided.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import torch  # noqa: PLC0415

    from src.ctc_utils import CRNN, build_charset, resolve_device, save_charset  # noqa: PLC0415

    task = Task.current_task()
    logger = task.get_logger()
    device = resolve_device()

    # Pre-train path: build charset from synthetic manifest, train, return (D-05)
    if getattr(args, "mode", "finetune") == "pretrain":
        synth_df = pd.read_csv(args.manifest)
        charset = build_charset(synth_df["label"].tolist())
        save_charset(args.output_dir / "charset.json", charset)
        model = CRNN(
            num_classes=len(charset) + 1, rnn_hidden=args.rnn_hidden, num_layers=args.num_layers,
        ).to(device)
        model.fc.bias.data[0] = args.blank_bias_init
        return _run_pretrain(model, args, device, logger, task, charset, on_epoch_end=on_epoch_end)

    # Fine-tune path
    df = pd.read_csv(args.manifest)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)  # TRAN-01
    if args.dataset_id is not None:
        labeled = remap_dataset_paths(labeled, args.dataset_id)

    extra_words: list[str] | None = None
    if args.words_file is not None:
        if not args.words_file.exists():
            raise FileNotFoundError(f"--words_file not found: {args.words_file}")
        extra_words = args.words_file.read_text(encoding="utf-8").splitlines()
    charset = build_charset(labeled["label"].tolist(), extra_words=extra_words)
    save_charset(args.output_dir / "charset.json", charset)
    _report_char_distribution(logger, charset, labeled["label"].tolist())

    train_loader, val_loader, val_df, _, debug_samples = _setup_finetune_loaders(
        labeled, args, charset, task,
    )

    model = CRNN(
        num_classes=len(charset) + 1, rnn_hidden=args.rnn_hidden, num_layers=args.num_layers,
    ).to(device)
    if getattr(args, "pretrain_checkpoint_path", None) is not None:
        checkpoint_ref = str(args.pretrain_checkpoint_path)
        if not checkpoint_ref.endswith(".pt"):
            from clearml import Task as _ClearMLTask  # noqa: PLC0415
            remote_task = _ClearMLTask.get_task(task_id=checkpoint_ref)
            checkpoint_ref = remote_task.artifacts["checkpoint_pretrain"].get_local_copy()
            print(f"Resolved checkpoint from ClearML task {args.pretrain_checkpoint_path}")
        state = torch.load(checkpoint_ref, weights_only=True, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded pre-trained weights from {checkpoint_ref}")
    model.fc.bias.data[0] = args.blank_bias_init
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6,
    )
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    ctc_loss_per_sample = torch.nn.CTCLoss(blank=0, zero_infinity=False, reduction="none")

    checkpoint_path = args.output_dir / "checkpoint.pt"
    best_val_cer = _run_loop(
        model, train_loader, val_loader, val_df, optimizer,
        scheduler=scheduler, ctc_loss=ctc_loss, ctc_loss_per_sample=ctc_loss_per_sample,
        device=device, args=args, logger=logger, charset=charset,
        checkpoint_path=checkpoint_path, debug_samples=debug_samples,
        series_prefix="", on_epoch_end=on_epoch_end,
    )
    if checkpoint_path.exists():
        upload_file_artifact(task, "checkpoint", checkpoint_path)
    upload_file_artifact(task, "charset", args.output_dir / "charset.json")
    return best_val_cer


def main() -> int:
    _config = load_config(mode=peek_mode())
    parser = _build_parser()
    if _config.get("datasets"):
        parser.set_defaults(dataset_id=_config["datasets"].get("id"))  # ty: ignore[unresolved-attribute]
    if _config.get("hyperparams"):
        parser.set_defaults(**_config["hyperparams"])  # ty: ignore[invalid-argument-type]
    if _config.get("queue_name"):
        parser.set_defaults(queue_name=_config["queue_name"])
    if _config.get("manifest"):
        parser.set_defaults(manifest=Path(str(_config["manifest"])))
    if _config.get("pretrain_checkpoint_path"):
        parser.set_defaults(pretrain_checkpoint_path=str(_config["pretrain_checkpoint_path"]))
    args = parser.parse_args()

    try:
        _apply_params_file(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 6

    print(f"[train_ctc] mode={args.mode}")

    tags = ["phase-4", "gpu"] if args.enqueue else ["phase-4"]
    if args.params is not None:
        tags.append("phase-5")
    task_name = "train_pretrain" if args.mode == "pretrain" else "train_finetune"
    task = init_task("handwriting-hebrew-ocr", task_name, tags=tags)

    # TRAN-07: connect ALL hyperparameters; MUST come before execute_remotely
    task.connect(vars(args), name="hyperparams")

    if args.enqueue:
        task.execute_remotely(queue_name=args.queue_name)
        # local process exits here via os._exit(); code below only runs on agent

    # Resolve manifest from ClearML dataset when not available locally (agent path)
    if args.dataset_id is not None and not args.manifest.exists():
        from clearml import Dataset  # noqa: PLC0415

        alias = "real" if args.mode == "finetune" else "synthetic"
        dataset_root = Path(Dataset.get(dataset_id=args.dataset_id, alias=alias).get_local_copy())
        args.manifest = dataset_root / "manifest.csv"

    if not args.manifest.exists():
        print(f"ERROR: --manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    if args.mode == "finetune":
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
