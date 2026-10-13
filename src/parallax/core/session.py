from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parallax.core.types import Message


def get_default_sessions_dir() -> Path:
    """Return default session storage directory ~/.parallax/sessions"""
    sessions_dir = Path.home() / ".parallax" / "sessions"
    sessions_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return sessions_dir


def _current_timestamp() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    """
    First-class persistent conversation session in the Parallax runtime.
    Preserves multi-turn message history, active model, and workspace context.
    """
    id: str
    workspace: str
    model: str
    title: str = "New session"
    created_at: str = field(default_factory=_current_timestamp)
    updated_at: str = field(default_factory=_current_timestamp)
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        """Count of conversational user turns."""
        return sum(1 for m in self.messages if m.role == "user")

    def update(self, messages: list[Message], model: str | None = None) -> None:
        """Update session messages and timestamp, deriving a title if unset."""
        self.messages = list(messages)
        self.updated_at = _current_timestamp()
        if model:
            self.model = model

        # Derive title from first user message if title is default
        if self.title in ("New session", ""):
            for m in self.messages:
                if m.role == "user" and m.content.strip():
                    first_line = m.content.strip().splitlines()[0]
                    self.title = first_line[:60] + "..." if len(first_line) > 60 else first_line
                    break

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "id": self.id,
            "workspace": self.workspace,
            "model": self.model,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize session from dictionary."""
        raw_msgs = data.get("messages", [])
        messages = [Message.from_dict(m) for m in raw_msgs]
        return cls(
            id=data.get("id", ""),
            workspace=data.get("workspace", ""),
            model=data.get("model", ""),
            title=data.get("title", "New session"),
            created_at=data.get("created_at", _current_timestamp()),
            updated_at=data.get("updated_at", _current_timestamp()),
            messages=messages,
            metadata=data.get("metadata", {}),
        )


class SessionStore:
    """
    Filesystem-backed persistent storage for Parallax sessions.
    Stores session JSON files in ~/.parallax/sessions/ with atomic writes.
    """

    def __init__(self, directory: Path | None = None, base_dir: Path | None = None):
        self.directory = directory or base_dir or get_default_sessions_dir()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _file_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def create_session(
        self,
        workspace: str,
        model: str,
        title: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create and persist a new Session."""
        if not session_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            short_id = os.urandom(3).hex()
            session_id = f"sess_{ts}_{short_id}"

        session = Session(
            id=session_id,
            workspace=str(workspace),
            model=model,
            title=title or "New session",
        )
        self.save_session(session)
        return session

    def save_session(self, session: Session) -> Path:
        """Save session to file atomically with restrictive file permissions."""
        path = self._file_path(session.id)
        temp_file = path.with_suffix(".tmp")
        data = session.to_dict()

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.chmod(temp_file, 0o600)
        temp_file.replace(path)
        return path

    def get_session(self, session_id: str) -> Session | None:
        """
        Retrieve a session by exact ID or prefix match.
        Returns None if not found.
        """
        # Exact match
        path = self._file_path(session_id)
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return Session.from_dict(json.load(f))
            except Exception:
                return None

        # Prefix or substring match (e.g. user typed a1b2 or sess_a1b2)
        norm_id = session_id.lower().strip()
        candidates: list[Path] = []
        for p in self.directory.glob("*.json"):
            stem = p.stem.lower()
            if stem == norm_id or stem.startswith(norm_id) or norm_id in stem:
                candidates.append(p)

        if len(candidates) == 1:
            try:
                with open(candidates[0], "r", encoding="utf-8") as f:
                    return Session.from_dict(json.load(f))
            except Exception:
                return None

        return None

    def list_sessions(
        self, workspace: str | None = None, limit: int = 50
    ) -> list[Session]:
        """
        List all saved sessions ordered by updated_at descending.
        Optionally filter by workspace.
        """
        sessions: list[Session] = []
        for p in self.directory.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sess = Session.from_dict(data)
                    if workspace and Path(sess.workspace).resolve() != Path(workspace).resolve():
                        continue
                    sessions.append(sess)
            except Exception:
                continue

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session file. Returns True if deleted, False otherwise."""
        session = self.get_session(session_id)
        if session:
            path = self._file_path(session.id)
            if path.is_file():
                try:
                    path.unlink()
                    return True
                except OSError:
                    return False
        return False
