import os
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path("config")
CONFIG_PATH = Path("config.yaml")


def peek_mode(argv: list[str] | None = None) -> str | None:
    """Read --mode from argv (default: sys.argv) before full argparse runs.

    Used by train_ctc and tune to select the right config file before set_defaults.
    """
    tokens = argv if argv is not None else sys.argv
    for i, token in enumerate(tokens):
        if token.startswith("--mode="):
            val = token[7:]
            return val if val in ("pretrain", "finetune") else None
        if token == "--mode" and i + 1 < len(tokens):
            val = tokens[i + 1]
            return val if val in ("pretrain", "finetune") else None
    return None


def load_config(path: Path | None = None, mode: str | None = None) -> dict[str, object]:
    if path is None:
        env_path = os.environ.get("CONFIG_PATH")
        if env_path:
            path = Path(env_path)
        elif mode is not None:
            mode_path = CONFIG_DIR / f"{mode}.yaml"
            path = mode_path if mode_path.exists() else CONFIG_PATH
        else:
            path = CONFIG_PATH
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def update_config(path: Path | None = None, mode: str | None = None, **kwargs: object) -> None:
    """Patch config at dotted key paths, e.g. update_config(**{"datasets.synthetic_id": "abc"}).

    With mode=, writes to config/{mode}.yaml if it exists; otherwise config.yaml.
    """
    if path is None:
        if mode is not None:
            mode_path = CONFIG_DIR / f"{mode}.yaml"
            path = mode_path if mode_path.exists() else CONFIG_PATH
        else:
            path = CONFIG_PATH
    cfg: dict[str, object] = load_config(path) or {}
    for dotted_key, value in kwargs.items():
        parts = dotted_key.split(".")
        node: dict[str, object] = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]  # ty: ignore[invalid-assignment]
        node[parts[-1]] = value
    path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
