import sqlite3, json, os, time

class ExperienceBuffer:
    def __init__(self, db_path="meta_learner/data/experience.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _ensure_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def add(self, event: dict):
        conn = sqlite3.connect(self.db_path)
        ts = event.get('ts', time.time())
        conn.execute(
            "INSERT INTO experiences (payload, ts) VALUES (?, ?)",
            (json.dumps(event), ts),
        )
        conn.commit()
        conn.close()

    def sample(self, n: int):
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT payload FROM experiences ORDER BY RANDOM() LIMIT ?", (n,)
        )
        rows = [json.loads(row[0]) for row in cur.fetchall()]
        conn.close()
        return rows
