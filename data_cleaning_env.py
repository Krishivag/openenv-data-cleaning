import json
import os
import sqlite3
import tempfile
from typing import Any, Optional

def clamp_score(score: float) -> float:
    # Robust clamping so rounding errors cannot make it 0.0 or 1.0
    epsilon = 0.01
    return float(max(epsilon, min(1.0 - epsilon, float(score))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State


class DataCleanerAction(Action):
    sql_command: str


class DataCleanerObservation(Observation):
    query_result: str
    error: Optional[str] = None


class DataCleanerState(State):
    task_name: str
    db_path: str


class DataCleaningEnv(Environment[DataCleanerAction, DataCleanerObservation, DataCleanerState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = ""
        self.conn = None
        self.task_name = ""
        self.step_cnt = 0
        self.episode_id = ""

    def _normalize_score(self, raw):
        """Clamp raw score to strict (0, 1) — maps [0,1] -> [0.01, 0.99]."""
        clamped = max(0.0, min(1.0, raw))
        return 0.01 + clamped * 0.98

    def _setup_db(self):
        if self.conn:
            self.conn.close()
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        self.db_path = path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    # --- task: easy (dedup users) ---

    def _populate_easy_task(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        c.executemany("INSERT INTO users VALUES (?, ?, ?)", [
            (1, "Alice", "alice@test.com"),
            (2, "Bob", "bob@test.com"),
            (3, "Alice", "alice@test.com"),
            (4, "Charlie", "charlie@test.com"),
            (5, "Bob", "bob@test.com"),
            (6, "David", "david@test.com"),
            (7, "Charlie", "charlie@test.com"),
        ])
        self.conn.commit()

    def _eval_easy_task(self):
        c = self.conn.cursor()
        c.execute("SELECT id, name, email FROM users ORDER BY id")
        rows = c.fetchall()

        expected = {
            ("Alice", "alice@test.com"),
            ("Bob", "bob@test.com"),
            ("Charlie", "charlie@test.com"),
            ("David", "david@test.com"),
        }
        current = set((r["name"], r["email"]) for r in rows)

        if current != expected:
            return clamp_score(0.0)
        if len(rows) == 4:
            return clamp_score(1.0)
        if len(rows) < 7:
            return clamp_score((7 - len(rows)) / 3.0)
        return clamp_score(0.0)

    # --- task: medium (normalize emails) ---

    def _populate_medium_task(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE contacts (id INTEGER PRIMARY KEY, email TEXT)")
        c.executemany("INSERT INTO contacts VALUES (?, ?)", [
            (1, " ALICE@test.com "),
            (2, "Bob@TEST.COM"),
            (3, "charlie@test.com  "),
            (4, "  DaViD@TeSt.CoM  "),
            (5, "eve@test.com"),
        ])
        self.conn.commit()

    def _eval_medium_task(self):
        c = self.conn.cursor()
        c.execute("SELECT email FROM contacts ORDER BY id")
        rows = c.fetchall()
        if len(rows) != 5:
            return clamp_score(0.0)

        expected = [
            "alice@test.com", "bob@test.com", "charlie@test.com",
            "david@test.com", "eve@test.com",
        ]
        correct = sum(1 for i, r in enumerate(rows) if r["email"] == expected[i])
        return clamp_score(correct / 5.0)

    # --- task: hard (impute null salaries with dept avg) ---

    def _populate_hard_task(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
        c.executemany("INSERT INTO departments VALUES (?, ?)", [
            (1, "Engineering"), (2, "Sales"),
        ])

        c.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary INTEGER)")
        c.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", [
            (1, "John", 1, 110000),
            (2, "Jane", 1, 100000),
            (3, "Bob",  1, None),      # should get 105000
            (4, "Sara", 2, 60000),
            (5, "Jim",  2, 70000),
            (6, "Rick", 2, None),      # should get 70000
            (7, "Lisa", 2, 80000),
            (8, "Tom",  1, None),      # should get 105000
        ])
        self.conn.commit()

    def _eval_hard_task(self):
        c = self.conn.cursor()
        c.execute("SELECT id, salary FROM employees ORDER BY id")
        rows = c.fetchall()

        expected = {
            1: 110000, 2: 100000, 3: 105000,
            4: 60000,  5: 70000,  6: 70000,
            7: 80000,  8: 105000,
        }
        correct = sum(1 for r in rows if r["salary"] == expected.get(r["id"]))
        return clamp_score(correct / 8.0)

    # --- OpenEnv interface ---

    def reset(self, seed=None, episode_id=None, **kwargs):
        self.episode_id = episode_id or "easy"
        self.step_cnt = 0
        self._reset_rubric()

        eid = self.episode_id.lower()
        if "medium" in eid:
            self.task_name = "medium"
        elif "hard" in eid:
            self.task_name = "hard"
        else:
            self.task_name = "easy"

        self._setup_db()

        if self.task_name == "easy":
            self._populate_easy_task()
            intro = ("Task: Deduplicate the 'users' table. Remove duplicate "
                     "combinations of (name, email) keeping the minimum id. "
                     "Schema: users(id, name, email).")
        elif self.task_name == "medium":
            self._populate_medium_task()
            intro = ("Task: Standardize the 'contacts' table emails. Trim "
                     "leading/trailing whitespaces and convert to lowercase. "
                     "Schema: contacts(id, email).")
        else:
            self._populate_hard_task()
            intro = ("Task: Impute missing salaries in 'employees'. Update "
                     "employees with NULL salary to the average salary of their "
                     "respective department. Schema: departments(id, name), "
                     "employees(id, name, dept_id, salary).")

        return DataCleanerObservation(query_result=intro, reward=clamp_score(0.01))

    def step(self, action, timeout_s=None, **kwargs):
        self.step_cnt += 1
        query = action.sql_command.strip()
        error = None
        result_str = ""

        try:
            c = self.conn.cursor()
            c.execute(query)
            if query.upper().startswith(("SELECT", "PRAGMA")):
                rows = c.fetchmany(50)
                if not rows:
                    result_str = "No rows returned."
                else:
                    result_str = json.dumps([dict(r) for r in rows], indent=2)
            else:
                self.conn.commit()
                result_str = f"Query executed successfully. Rows affected: {c.rowcount}"
        except Exception as e:
            error = str(e)
            result_str = "Error executing query."

        if self.task_name == "easy":
            raw = self._eval_easy_task()
        elif self.task_name == "medium":
            raw = self._eval_medium_task()
        else:
            raw = self._eval_hard_task()

        score = clamp_score(self._normalize_score(raw))
        done = raw >= 0.99 or self.step_cnt >= 10

        return DataCleanerObservation(
            query_result=result_str, error=error, done=done, reward=score,
        )

    @property
    def state(self):
        return DataCleanerState(
            step_count=self.step_cnt,
            episode_id=self.episode_id,
            task_name=self.task_name,
            db_path=self.db_path,
        )

    def close(self):
        if self.conn:
            self.conn.close()
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        super().close()
