import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("MORPHEU_DATA_DIR", PROJECT_ROOT / "data")))
    frontend_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "frontend" / "dist")
    # Shortest reference clip accepted for cloning, in seconds.
    min_reference_seconds: float = 1.0
    max_reference_seconds: float = 60.0

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "morpheu.sqlite3"
