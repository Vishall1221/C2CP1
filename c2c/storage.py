from contextlib import closing, contextmanager
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile
from typing import TypeVar

from filelock import FileLock, Timeout
from pydantic import BaseModel

from .config import ROOT, UserError, VideoSettings


RecordType = TypeVar("RecordType", bound=BaseModel)
PREPARE_VERSION = 2


def fingerprint(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(content)
            temporary.close()
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def timestamp(ms: int) -> str:
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class Job:
    def __init__(self, source: Path, settings: VideoSettings, output_root: Path | None = None):
        self.source = source
        with source.open("rb") as handle:
            self.asset_id = file_digest(handle, "sha256").hexdigest()
        self.prepare_key = fingerprint([PREPARE_VERSION, settings.model_dump()])
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", source.stem).strip("-_")[:45] or "video"
        self.path = (output_root or ROOT / "output") / f"{slug}-{self.asset_id[:12]}-{self.prepare_key[:8]}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = self.path / "work.sqlite3"
        with closing(sqlite3.connect(self.db)) as db, db:
            db.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
            """)

    @contextmanager
    def lock(self):
        try:
            with FileLock(self.path / ".job.lock", timeout=0):
                yield
        except Timeout as exc:
            raise UserError("This clip is being processed in another window. Try again when it finishes.") from exc

    def read(self, key: str, model: type[RecordType]) -> RecordType | None:
        with closing(sqlite3.connect(self.db)) as db:
            row = db.execute("SELECT payload FROM state WHERE key = ?", (key,)).fetchone()
        return None if row is None else model.model_validate_json(row[0])

    def save(self, key: str, record: BaseModel):
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute(
                "INSERT INTO state VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET payload = excluded.payload",
                (key, record.model_dump_json()),
            )
