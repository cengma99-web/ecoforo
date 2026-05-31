"""Application configuration loaded from environment and file."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _data_dir() -> Path:
    d = os.getenv("ECOFORO_DATA_DIR", str(PROJECT_ROOT / "data"))
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


class Config:
    DATABASE_URL: str = os.getenv(
        "ECOFORO_DATABASE_URL",
        f"sqlite:///{PROJECT_ROOT / 'ecoforo.db'}",
    )
    DATA_DIR: Path = _data_dir()
    PROXY_URL: str = os.getenv("ECOFORO_PROXY_URL", "socks5h://127.0.0.1:10808")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    LOG_LEVEL: str = os.getenv("ECOFORO_LOG_LEVEL", "INFO")
    MAX_EVENTS_PER_RUN: int = int(os.getenv("ECOFORO_MAX_EVENTS_PER_RUN", "5000"))


config = Config()
