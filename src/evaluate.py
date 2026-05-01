"""evaluate.py — load CRNN+CTC checkpoint, run greedy decode on val split,
write eval_report.csv, log to ClearML."""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from clearml import (
    Task,  # noqa: F401 — module-level import required for @patch("src.evaluate.Task")
)

from src.clearml_utils import init_task, upload_file_artifact
from src.ctc_utils import (
    CRNN,
    build_half_page_units,
    cer,
    greedy_decode,
    load_charset,
    load_crop,
    resolve_device,
    split_units,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained CRNN+CTC checkpoint on the val split."
    )
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/model"))
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    return p


def main() -> int:
    # Pitfall 2: Task.init BEFORE argparse.parse_args()
    task = init_task("handwriting-hebrew-ocr", "evaluate_model", tags=["phase-3"])

    parser = _build_parser()
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: --manifest does not exist: {args.manifest}", file=sys.stderr)
        task.close()
        return 2

    checkpoint_path = args.output_dir / "checkpoint.pt"
    charset_path = args.output_dir / "charset.json"
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        task.close()
        return 3
    if not charset_path.exists():
        print(f"ERROR: charset not found: {charset_path}", file=sys.stderr)
        task.close()
        return 4

    # EVAL-04: hyperparameters tracked in ClearML
    task.connect(vars(args), name="hyperparams")

    df = pd.read_csv(args.manifest)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)

    charset = load_charset(charset_path)

    # Reproduce training split — MUST match train_ctc.py order/logic exactly
    units = build_half_page_units(labeled)
    _, val_keys = split_units(units, val_frac=args.val_frac)
    val_idx = [i for k in val_keys for i in units[k]]
    if not val_idx:
        print("ERROR: validation split is empty.", file=sys.stderr)
        task.close()
        return 5

    val_df = labeled.iloc[val_idx].reset_index(drop=True)

    device = resolve_device()
    model = CRNN(num_classes=len(charset) + 1).to(device)
    state = torch.load(checkpoint_path, weights_only=True)  # Pitfall 7
    model.load_state_dict(state)
    model.eval()

    image_paths: list[str] = []
    targets: list[str] = []
    predictions: list[str] = []

    with torch.no_grad():
        for idx in range(len(val_df)):
            row = val_df.iloc[idx]
            crop_path = str(row["crop_path"])
            target_text = str(row["label"])
            image = load_crop(crop_path).unsqueeze(0).to(device)  # (1, 1, 64, W)
            # Pad width to multiple of 4 (matches collate Pitfall 1 fix)
            w = image.size(3)
            padded_w = math.ceil(w / 4) * 4
            if padded_w != w:
                pad = torch.zeros(1, 1, image.size(2), padded_w, device=device)
                pad[:, :, :, :w] = image
                image = pad
            logits = model(image)  # (T, 1, C)
            log_probs = logits.log_softmax(2)
            pred_indices = greedy_decode(log_probs[:, 0, :])
            pred_text = "".join(charset[i - 1] for i in pred_indices)

            image_paths.append(crop_path)
            targets.append(target_text)
            predictions.append(pred_text)

    # EVAL-03: write eval_report.csv with the four required columns IN ORDER
    report = pd.DataFrame(
        {
            "image_path": image_paths,
            "target": targets,
            "prediction": predictions,
            "is_exact": [t == p for t, p in zip(targets, predictions, strict=True)],
        },
        columns=["image_path", "target", "prediction", "is_exact"],
    )
    report_path = args.output_dir / "eval_report.csv"
    report.to_csv(report_path, index=False)

    # EVAL-02: aggregate CER (mean of per-crop CER)
    cer_values = [cer(t, p) for t, p in zip(targets, predictions, strict=True)]
    avg_cer = sum(cer_values) / max(len(cer_values), 1)
    exact_match_rate = sum(report["is_exact"]) / max(len(report), 1)

    # EVAL-04: log + upload
    logger = task.get_logger()
    logger.report_scalar(title="cer", series="val", iteration=0, value=avg_cer)
    logger.report_scalar(title="exact_match", series="val", iteration=0, value=exact_match_rate)
    upload_file_artifact(task, "eval_report", report_path)

    print(
        f"Done. n_val={len(report)} cer={avg_cer:.4f} "
        f"exact_match_rate={exact_match_rate:.4f} report={report_path}"
    )
    task.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
