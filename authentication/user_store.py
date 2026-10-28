"""
user_store.py — local SQLite backing store for THFSA users.

Database: authentication/data/thfsa_users.db (override with THFSA_DB_PATH).

Never stores plaintext passwords — they are hashed with bcrypt. Never stores
a raw passkey — the per-user TOTP secret (a derived secret) is stored so the
server can verify OTP codes. Biometric rows hold only the derived JSON
template from user_biometric, never raw fingerprint images.

All functions accept an optional `db_path` so tests can point at a throwaway
database instead of the real one.
"""

import contextlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import bcrypt

_DEFAULT_DB = Path(__file__).resolve().parent / "data" / "thfsa_users.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    passkey_secret TEXT NOT NULL,
    v_hex          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS biometric_templates (
    user_id            TEXT PRIMARY KEY REFERENCES users(user_id),
    template           TEXT NOT NULL,
    template_metadata  TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scanner_bindings (
    user_id    TEXT PRIMARY KEY REFERENCES users(user_id),
    scanner_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_db_path() -> Path:
    return Path(os.environ.get("THFSA_DB_PATH", _DEFAULT_DB))


@contextlib.contextmanager
def connect(db_path=None):
    """Yield a committed connection. Creates the schema on first touch."""
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
        if "v_hex" not in cols:  # v2 upgrade for DBs created before auth_flow
            conn.execute(
                "ALTER TABLE users ADD COLUMN v_hex TEXT NOT NULL DEFAULT ''")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row) -> dict | None:
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """bcrypt hash, authenticated form ($2b$ prefix)."""
    return bcrypt.hashpw(password.encode("utf-8"),
                         bcrypt.gensalt()).decode("ascii")


def create_user(user_id: str, password_hash: str, passkey_secret: str,
                v_hex: str = "", db_path=None) -> dict:
    if user_exists(user_id, db_path=db_path):
        raise ValueError(f"user already exists: {user_id}")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO users(user_id, password_hash, passkey_secret, v_hex, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, password_hash, passkey_secret, v_hex, _now()),
        )
    user = get_user(user_id, db_path=db_path)
    assert user is not None  # just inserted, see create_user contract
    return user


def get_user(user_id: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT user_id, password_hash, passkey_secret, v_hex, created_at"
            " FROM users WHERE user_id = ?", (user_id,),
        ).fetchone()
    return _row_to_dict(row)


def user_exists(user_id: str, db_path=None) -> bool:
    return get_user(user_id, db_path=db_path) is not None


def verify_password(user_id: str, password: str, db_path=None) -> bool:
    """bcrypt constant-time check. False on unknown user or bad password."""
    user = get_user(user_id, db_path=db_path)
    if user is None:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"),
                              user["password_hash"].encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# biometric templates
# ---------------------------------------------------------------------------

def store_biometric_template(user_id: str, template: dict, db_path=None) -> None:
    metadata = {k: template[k] for k in template if k != "vector_b64"}
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO biometric_templates"
            "(user_id, template, template_metadata, created_at) VALUES (?, ?, ?, ?)",
            (user_id, json.dumps(template), json.dumps(metadata), _now()),
        )


def get_biometric_template(user_id: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT template FROM biometric_templates WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["template"])


# ---------------------------------------------------------------------------
# scanner bindings
# ---------------------------------------------------------------------------

def bind_scanner_to_user(user_id: str, scanner_id: int, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scanner_bindings"
            "(user_id, scanner_id, created_at) VALUES (?, ?, ?)",
            (user_id, int(scanner_id), _now()),
        )


def get_bound_scanner(user_id: str, db_path=None) -> int | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT scanner_id FROM scanner_bindings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["scanner_id"]) if row is not None else None