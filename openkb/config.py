from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "sonnet",
    "provider": "claude",
    "effort": "medium",
    "language": "en",
    "pageindex_threshold": 20,
    "insights_cooldown": 3600,
    "background_insights_cooldown_seconds": 3600,
}

GLOBAL_CONFIG_DIR = Path.home() / ".config" / "openkb"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "global.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from config_path, merged with DEFAULT_CONFIG.

    If the file does not exist, returns a copy of the defaults.
    """
    config = dict(DEFAULT_CONFIG)
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        config.update(data)
    if "insights_cooldown" not in data and "background_insights_cooldown_seconds" in data:
        config["insights_cooldown"] = config["background_insights_cooldown_seconds"]
    elif "background_insights_cooldown_seconds" not in data and "insights_cooldown" in data:
        config["background_insights_cooldown_seconds"] = config["insights_cooldown"]
    return config


def save_config(config_path: Path, config: dict) -> None:
    """Persist config dict to YAML, creating parent directories as needed."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=True)


def load_global_config() -> dict[str, Any]:
    """Load the global config from ~/.config/openkb/global.yaml."""
    if GLOBAL_CONFIG_PATH.exists():
        with GLOBAL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def save_global_config(config: dict[str, Any]) -> None:
    """Save the global config to ~/.config/openkb/global.yaml."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with GLOBAL_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=True)


def register_kb(kb_path: Path) -> None:
    """Register a KB path in the global config's known_kbs list."""
    gc = load_global_config()
    known = gc.get("known_kbs", [])
    resolved = str(kb_path.resolve())
    if resolved not in known:
        known.append(resolved)
        gc["known_kbs"] = known
    gc["default_kb"] = resolved
    save_global_config(gc)
