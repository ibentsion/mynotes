#!/usr/bin/env python3
"""Compare OCR accuracy of multiple OpenAI models on known-correct crops.

Usage:
    OPENAI_API_KEY=sk-... uv run python eval_ocr_models.py

Reads crops whose notes field starts with "auto:" — these have been manually
corrected by the user after auto-labeling, so their labels are ground truth.
"""

import base64
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5.4",
]

PROMPT = (
    "Transcribe the Hebrew handwriting in this image exactly as written. "
    "Return only the transcribed text, no explanation or punctuation additions."
)


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def transcribe(client: OpenAI, model: str, crop_path: Path) -> str:
    b64 = encode(crop_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                }},
            ],
        }],
        max_tokens=100,
    )
    return (resp.choices[0].message.content or "").strip()


def _levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for ch_a in a:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, ch_b in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ch_a != ch_b))
    return dp[len(b)]


def cer(pred: str, ref: str) -> float:
    return _levenshtein(pred, ref) / max(len(ref), 1)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    client = OpenAI()
    df = pd.read_csv(
        "data/manifest.csv",
        dtype={"label": object, "notes": object, "flag_reasons": object},
    )
    crops = df[df["notes"].astype(str).str.startswith("auto:")].copy()

    if crops.empty:
        print("No crops with auto: notes found in manifest.", file=sys.stderr)
        return 1

    print(f"Ground-truth crops: {len(crops)}")
    print(f"Models to test: {', '.join(MODELS)}\n")

    ground_truth = crops["label"].tolist()
    crop_paths = [Path(p) for p in crops["crop_path"]]
    short_names = [p.name.split("_p")[1] if "_p" in p.name else p.name for p in crop_paths]

    results: dict[str, list[str]] = {}
    for model in MODELS:
        print(f"Testing {model}...", flush=True)
        preds = []
        for path in crop_paths:
            try:
                preds.append(transcribe(client, model, path))
            except Exception as exc:
                preds.append(f"[ERR: {exc}]")
        results[model] = preds

    # --- Comparison table ---
    col = 16
    header = f"{'crop':<20} {'correct':<{col}}" + "".join(f" {m[:col]:<{col}}" for m in MODELS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for i, (name, correct) in enumerate(zip(short_names, ground_truth, strict=True)):
        row = f"{name:<20} {str(correct):<{col}}"
        for model in MODELS:
            pred = results[model][i]
            mark = "✓" if pred == correct else "✗"
            row += f" {mark}{pred[: col - 1]:<{col - 1}}"
        print(row)

    # --- Per-model summary ---
    print("\n" + "=" * len(header))
    print("Model accuracy (exact match) and avg character error rate:")
    print("-" * 60)
    scored = []
    for model in MODELS:
        exact = sum(results[model][i] == ground_truth[i] for i in range(len(crops)))
        avg_cer = sum(cer(results[model][i], str(ground_truth[i])) for i in range(len(crops)))
        avg_cer /= len(crops)
        scored.append((model, exact, avg_cer))
        print(f"  {model:<22} {exact}/{len(crops)} exact  CER: {avg_cer:.2f}")

    best = min(scored, key=lambda x: x[2])
    print(f"\nBest model: {best[0]}  ({best[1]}/{len(crops)} exact, CER {best[2]:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
