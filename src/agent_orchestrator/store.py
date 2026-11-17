"""SQLite storage: append-only event log plus a derived checkpoint per run."""

import json
import sqlite3
import time

from .state import RunState, initial, rebuild, reduce

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  type TEXT NOT NULL,
  data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq);
CREATE TABLE IF NOT EXISTS checkpoints (run_id TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'event log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'event log is append-only'); END;
"""


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def append(self, run_id, type_, data) -> dict:
        """Append an event and update the checkpoint in one transaction."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            ts = time.time()
            cur = self.db.execute(
                "INSERT INTO events(run_id, ts, type, data) VALUES (?,?,?,?)",
                (run_id, ts, type_, json.dumps(data, sort_keys=True)),
            )
            event = {
                "seq": cur.lastrowid,
                "run_id": run_id,
                "ts": ts,
                "type": type_,
                "data": data,
            }
            state = self._load(run_id) or initial(run_id)
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoints(run_id, state) VALUES (?,?)",
                (run_id, json.dumps(reduce(state, event).to_dict())),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return event

    def _load(self, run_id):
        row = self.db.execute(
            "SELECT state FROM checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()
        return RunState.from_dict(json.loads(row[0])) if row else None

    def load(self, run_id) -> RunState:
        s = self._load(run_id)
        if s is None:
            raise KeyError(run_id)
        return s

    def events(self, run_id) -> list[dict]:
        rows = self.db.execute(
            "SELECT seq, run_id, ts, type, data FROM events WHERE run_id=? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [
            {
                "seq": r[0],
                "run_id": r[1],
                "ts": r[2],
                "type": r[3],
                "data": json.loads(r[4]),
            }
            for r in rows
        ]

    def rebuild(self, run_id) -> RunState:
        return rebuild(run_id, self.events(run_id))

    def run_ids(self) -> list[str]:
        rows = self.db.execute(
            "SELECT run_id FROM events GROUP BY run_id ORDER BY MIN(seq)"
        ).fetchall()
        return [r[0] for r in rows]
