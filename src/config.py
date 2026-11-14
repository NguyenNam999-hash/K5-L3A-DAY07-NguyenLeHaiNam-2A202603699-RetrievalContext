"""Central configuration and path helpers for the Day 7 lab."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6_ROOT = ROOT.parent / "K5-L3-Day06-NguyenLeHaiNam-DataIngestionEngineering"


@dataclass(frozen=True)
class Settings:
    root: Path
    database_url: str
    embedding_model: str
    strategy: str
    openai_api_key: str
    openai_base_url: str
    model: str
    session_id: str
    day6_root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def day6_data_dir(self) -> Path:
        return self.data_dir / "day6"

    @property
    def benchmark_path(self) -> Path:
        return self.root / "benchmark" / "questions.json"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def advanced_dir(self) -> Path:
        return self.root / "advanced"


def _read_day6_database_url(day6_root: Path) -> str:
    env_path = day6_root / "starter" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return os.getenv("DATABASE_URL", "")


def load_settings(*, require_llm: bool = True) -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    day6_root = Path(os.getenv("DAY6_ROOT", str(DEFAULT_DAY6_ROOT))).expanduser()
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        database_url = _read_day6_database_url(day6_root)
    missing = []
    if not database_url:
        missing.append("DATABASE_URL")
    if require_llm:
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL"):
            if not os.getenv(key):
                missing.append(key)
    if missing:
        raise SystemExit(
            "Missing configuration: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill it in."
        )
    return Settings(
        root=ROOT,
        database_url=database_url,
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        strategy=os.getenv("DAY6_STRATEGY", "C"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        model=os.getenv("MODEL", ""),
        session_id=os.getenv("OPENCODE_SESSION_ID", ""),
        day6_root=day6_root,
    )
