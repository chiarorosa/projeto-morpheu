"""Generation history in SQLite; audio files live in data/outputs/."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    mode TEXT NOT NULL,
    model_id TEXT NOT NULL,
    text TEXT NOT NULL,
    language TEXT NOT NULL,
    voice_id TEXT,
    speaker TEXT,
    instruct TEXT,
    params TEXT NOT NULL,
    audio_file TEXT NOT NULL,
    duration REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    elapsed REAL NOT NULL
)
"""


class History:
    def __init__(self, db_path: Path, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(SCHEMA)
            self._conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"])
        d.pop("audio_file")
        return d

    def new_audio_path(self) -> tuple[str, Path]:
        gen_id = uuid.uuid4().hex
        return gen_id, self.outputs_dir / f"{gen_id}.wav"

    def add(self, gen_id: str, *, mode, model_id, text, language, voice_id, speaker, instruct,
            params, audio_path: Path, duration, sample_rate, elapsed) -> dict:
        with self._lock:
            self._conn.execute(
                "INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (gen_id, time.time(), mode, model_id, text, language, voice_id, speaker, instruct,
                 json.dumps(params), audio_path.name, duration, sample_rate, elapsed),
            )
            self._conn.commit()
        return self.get(gen_id)

    def get(self, gen_id: str) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT * FROM generations WHERE id=?", (gen_id,)).fetchone()
        if row is None:
            raise KeyError(gen_id)
        return self._row(row)

    def audio_path(self, gen_id: str) -> Path:
        with self._lock:
            row = self._conn.execute("SELECT audio_file FROM generations WHERE id=?", (gen_id,)).fetchone()
        if row is None or not (self.outputs_dir / row[0]).exists():
            raise KeyError(gen_id)
        return self.outputs_dir / row[0]

    def list(self, limit: int = 200, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM generations ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, gen_id: str) -> None:
        path = self.audio_path(gen_id) if self._exists(gen_id) else None
        with self._lock:
            cur = self._conn.execute("DELETE FROM generations WHERE id=?", (gen_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(gen_id)
        if path:
            path.unlink(missing_ok=True)

    def _exists(self, gen_id: str) -> bool:
        try:
            self.audio_path(gen_id)
            return True
        except KeyError:
            return False

    def close(self) -> None:
        self._conn.close()
