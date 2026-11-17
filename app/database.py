
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "database"
DB_PATH = DB_DIR / "predictions.db"


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_database():
    with closing(get_connection()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                inference_time_ms REAL NOT NULL,
                device TEXT NOT NULL,
                model_epoch INTEGER NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_created
            ON prediction_history(created_at DESC)
        """)

        conn.commit()


def save_prediction(filename, image_sha256, result):
    created_at = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                filename,
                image_sha256,
                predicted_class,
                confidence,
                probabilities,
                inference_time_ms,
                device,
                model_epoch
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                filename,
                image_sha256,
                result["predicted_class"],
                result["confidence"],
                json.dumps(result["probabilities"]),
                result["inference_time_ms"],
                result["device"],
                result["model_epoch"],
            ),
        )

        record_id = cursor.lastrowid
        conn.commit()

        return record_id


def get_history(limit=100, offset=0):
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    records = []

    for row in rows:
        item = dict(row)
        item["probabilities"] = json.loads(
            item["probabilities"]
        )
        records.append(item)

    return records


def get_statistics():
    with closing(get_connection()) as conn:

        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            """
        ).fetchone()[0]

        rows = conn.execute(
            """
            SELECT
                predicted_class,
                COUNT(*) AS count
            FROM prediction_history
            GROUP BY predicted_class
            ORDER BY count DESC
            """
        ).fetchall()

        avg_time = conn.execute(
            """
            SELECT AVG(inference_time_ms)
            FROM prediction_history
            """
        ).fetchone()[0]

    return {
        "total_predictions": total,
        "class_distribution": [
            dict(row) for row in rows
        ],
        "average_inference_ms": (
            round(avg_time, 2)
            if avg_time is not None
            else None
        ),
    }
