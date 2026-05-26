from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
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
