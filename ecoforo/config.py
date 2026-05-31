"""Application configuration loaded from environment and file."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _parse_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


class Config:
    DATABASE_URL: str = os.getenv(
        "ECOFORO_DATABASE_URL",
        f"sqlite:///{PROJECT_ROOT / 'ecoforo.db'}",
    )
    _DATA_DIR: Path | None = None
    PROXY_URL: str = os.getenv("ECOFORO_PROXY_URL", "socks5h://127.0.0.1:10808")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    LOG_LEVEL: str = os.getenv("ECOFORO_LOG_LEVEL", "INFO")
    MAX_EVENTS_PER_RUN: int = _parse_int_env("ECOFORO_MAX_EVENTS_PER_RUN", 5000)

    @property
    def DATA_DIR(self) -> Path:
        if self._DATA_DIR is None:
            d = Path(os.getenv("ECOFORO_DATA_DIR", str(PROJECT_ROOT / "data")))
            d.mkdir(parents=True, exist_ok=True)
            self._DATA_DIR = d
        return self._DATA_DIR


config = Config()
