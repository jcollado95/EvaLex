"""
Usage logging for the EvaLex demo.

Every evaluation request is stored in a local SQLite database. Run this module
to print usage statistics:

    python demo/usage.py [--db path/to/usage.sqlite3]
"""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = os.getenv(
    "EVALEX_USAGE_DB",
    str(Path(__file__).resolve().parent / "usage" / "usage.sqlite3"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,       -- UTC, ISO 8601
    session TEXT,                  -- anonymous browser session id
    status TEXT NOT NULL,          -- ok | rejected (invalid input) | error (server failure)
    message TEXT,                  -- validation or error message
    language TEXT,
    model TEXT,
    num_definitions INTEGER,
    max_tokens_words INTEGER,
    num_words INTEGER,
    words TEXT,                    -- JSON list
    prompts_edited INTEGER,        -- 1 if the prompts differ from the language defaults
    prompts TEXT,                  -- JSON: the prompts that were sent
    known_count INTEGER,
    duration_s REAL,
    results TEXT,                  -- JSON: {word: known}
    details TEXT                   -- JSON: per definition (word, definition, candidates, match)
)
"""

JSON_FIELDS = ("words", "prompts", "results", "details")


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open the usage database, creating it if needed."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    return conn


def log_query(record: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """Store one evaluation request. Logging errors never break the demo."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    for field in JSON_FIELDS:
        if field in row:
            row[field] = json.dumps(row[field], ensure_ascii=False)
    if "prompts_edited" in row:
        row["prompts_edited"] = int(bool(row["prompts_edited"]))

    columns = ", ".join(row)
    placeholders = ", ".join(f":{c}" for c in row)
    try:
        with closing(connect(db_path)) as conn, conn:
            conn.execute(f"INSERT INTO queries ({columns}) VALUES ({placeholders})", row)
    except Exception as e:
        print(f"Warning: could not log usage: {e}")


def print_stats(db_path: str = DB_PATH) -> None:
    """Print a summary of the logged usage."""
    if not Path(db_path).exists():
        print(f"No usage database at {db_path}")
        return

    def table(title: str, query: str, params: Optional[tuple] = ()) -> None:
        rows = conn.execute(query, params).fetchall()
        print(f"\n{title}")
        for row in rows:
            print("  " + " | ".join("" if v is None else str(v) for v in row))

    with closing(connect(db_path)) as conn:
        total, sessions, first, last = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT session), MIN(timestamp), MAX(timestamp) FROM queries"
        ).fetchone()
        print(f"Queries: {total} from {sessions} sessions ({first} to {last})")

        table("By status", "SELECT status, COUNT(*) FROM queries GROUP BY status ORDER BY 2 DESC")
        table(
            "By model (successful): queries | words | known % | avg seconds",
            """SELECT model, COUNT(*), SUM(num_words),
                      ROUND(100.0 * SUM(known_count) / SUM(num_words), 1), ROUND(AVG(duration_s), 1)
               FROM queries WHERE status = 'ok' GROUP BY model ORDER BY 2 DESC""",
        )
        table(
            "By language (successful): queries | words | known %",
            """SELECT language, COUNT(*), SUM(num_words),
                      ROUND(100.0 * SUM(known_count) / SUM(num_words), 1)
               FROM queries WHERE status = 'ok' GROUP BY language ORDER BY 2 DESC""",
        )
        table(
            "Settings (successful): definitions | max tokens | queries",
            """SELECT num_definitions, max_tokens_words, COUNT(*) FROM queries
               WHERE status = 'ok' GROUP BY 1, 2 ORDER BY 3 DESC""",
        )
        table(
            "Edited prompts (successful): edited | queries",
            "SELECT prompts_edited, COUNT(*) FROM queries WHERE status = 'ok' GROUP BY 1",
        )
        table(
            "By day: date | queries",
            "SELECT substr(timestamp, 1, 10), COUNT(*) FROM queries GROUP BY 1 ORDER BY 1 DESC LIMIT 14",
        )
        table(
            "Most evaluated words: word | language | times",
            """SELECT value, language, COUNT(*) FROM queries, json_each(queries.words)
               WHERE status = 'ok' GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 15""",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print EvaLex demo usage statistics")
    parser.add_argument("--db", default=DB_PATH, help="Path to the usage database")
    print_stats(parser.parse_args().db)
