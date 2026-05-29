import os
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | None = None) -> dict[str, object]:
    if path is None:
        env_path = os.environ.get("CONFIG_PATH")
        path = Path(env_path) if env_path else CONFIG_PATH
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def update_config(path: Path = CONFIG_PATH, **kwargs: object) -> None:
    """Patch config at dotted key paths, e.g. update_config(**{"datasets.synthetic_id": "abc"})."""
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
