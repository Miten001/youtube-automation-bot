"""Configuration loading and validation.

Settings come from three layers, later layers override earlier ones:
  1. Built-in defaults (DEFAULT_CONFIG)
  2. A YAML config file (config.yaml)
  3. Environment variables (.env) for secrets
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG: Dict[str, Any] = {
    "output_dir": "output",
    "enhance": {
        "enabled": True,
        "target_height": 1080,
        "crf": 20,
        "preset": "slow",
        "audio_bitrate": "192k",
        "apply_filters": True,
        "max_bitrate": None,
    },
    "tags": {
        "enabled": True,
        "max_tags": 15,
        "always_include": [],
        "stopwords_extra": [],
        # USA-targeted hashtags merged into every video's tag list. Set to []
        # to disable, or replace with your own region/niche hashtags.
        "region_hashtags": [
            "usa",
            "america",
            "trendingusa",
            "viralusa",
            "fypusa",
        ],
    },
    "title_hashtags": {
        # Append hashtags to the title (e.g. "My Video #usa #coffee").
        "enabled": True,
        # How many hashtags to append to the title.
        "count": 3,
    },
    "auto_title": {
        # When no --title is given, generate a catchy one automatically.
        "enabled": True,
        # catchy | question | clean
        "style": "catchy",
        # Max length of the base title (hashtags are added after this).
        "max_length": 70,
        # Prepend a relevant emoji.
        "emoji": True,
    },
    "youtube": {
        "enabled": True,
        "privacy_status": "private",
        "category_id": "22",
        "made_for_kids": False,
    },
}


@dataclass
class Secrets:
    """Credentials loaded from environment variables."""

    yt_client_secret_file: str = "client_secret.json"
    yt_token_file: str = "token.json"

    @classmethod
    def from_env(cls) -> "Secrets":
        load_dotenv()
        return cls(
            yt_client_secret_file=os.getenv("YT_CLIENT_SECRET_FILE", "client_secret.json"),
            yt_token_file=os.getenv("YT_TOKEN_FILE", "token.json"),
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    secrets: Secrets = field(default_factory=Secrets)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        data = copy.deepcopy(DEFAULT_CONFIG)
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with path.open("r", encoding="utf-8") as fh:
                file_data = yaml.safe_load(fh) or {}
            data = _deep_merge(data, file_data)
        return cls(data=data, secrets=Secrets.from_env())

    # Convenience accessors -------------------------------------------------
    @property
    def output_dir(self) -> str:
        return self.data["output_dir"]

    @property
    def enhance(self) -> Dict[str, Any]:
        return self.data["enhance"]

    @property
    def tags(self) -> Dict[str, Any]:
        return self.data["tags"]

    @property
    def title_hashtags(self) -> Dict[str, Any]:
        return self.data["title_hashtags"]

    @property
    def auto_title(self) -> Dict[str, Any]:
        return self.data["auto_title"]

    @property
    def youtube(self) -> Dict[str, Any]:
        return self.data["youtube"]

    def validate_for_targets(self) -> None:
        """Raise a helpful error if required YouTube credentials are missing."""
        problems = []
        if not Path(self.secrets.yt_client_secret_file).exists():
            problems.append(
                f"YouTube: client secret file '{self.secrets.yt_client_secret_file}' "
                "not found (set YT_CLIENT_SECRET_FILE in .env)."
            )
        if problems:
            raise ValueError("Configuration problems:\n  - " + "\n  - ".join(problems))
