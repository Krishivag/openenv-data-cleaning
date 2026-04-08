import json
import sqlite3
import tempfile
import traceback
from typing import Any, Dict, List, Optional

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
        self.db_path: str = ""
        self.conn: Optional[sqlite3.Connection] = None
        self.task_name: str = ""
        self.step_cnt: int = 0
        self.episode_id: str = ""

    def _setup_db(self):
        if self.conn:
            self.conn.close()
        # Use tempfile to allow concurrent sessions safely
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        self.db_path = path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _populate_easy_task(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        users = [
            (1, "Alice", "alice@test.com"),
            (2, "Bob", "bob@test.com"),
            (3, "Alice", "alice@test.com"),
            (4, "Charlie", "charlie@test.com"),
            (5, "Bob", "bob@test.com"),
            (6, "David", "david@test.com"),
            (7, "Charlie", "charlie@test.com"),
        ]
        c.executemany("INSERT INTO users VALUES (?, ?, ?)", users)
        self.conn.commit()

    def _eval_easy_task(self) -> float:
        c = self.conn.cursor()
        c.execute("SELECT id, name, email FROM users ORDER BY id")
        rows = c.fetchall()
        
        # Expected distinct combinations
        expected_combinations = {
            ("Alice", "alice@test.com"),
            ("Bob", "bob@test.com"),
            ("Charlie", "charlie@test.com"),
            ("David", "david@test.com")
        }
        
        # Current combinations
        current_combinations = set((row['name'], row['email']) for row in rows)
        
        score = 0.0
        # If they deleted wrong people, score is 0
        if current_combinations != expected_combinations:
            return 0.0
        
        # If they kept only 4 rows, they perfectly deduplicated
        if len(rows) == 4:
            score = 1.0
        elif len(rows) < 7:
            # Partial score if they deleted some but not all duplicates
            score = (7 - len(rows)) / 3.0
            
        return max(0.0, min(1.0, score))

    def _populate_medium_task(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE contacts (id INTEGER PRIMARY KEY, email TEXT)")
        contacts = [
            (1, " ALICE@test.com "),
            (2, "Bob@TEST.COM"),
            (3, "charlie@test.com  "),
            (4, "  DaViD@TeSt.CoM  "),
            (5, "eve@test.com")
        ]
        c.executemany("INSERT INTO contacts VALUES (?, ?)", contacts)
        self.conn.commit()

    def _eval_medium_task(self) -> float:
        c = self.conn.cursor()
        c.execute("SELECT email FROM contacts ORDER BY id")
        rows = c.fetchall()
        if len(rows) != 5:
            return 0.0
        
        expected = [
            "alice@test.com",
            "bob@test.com",
            "charlie@test.com",
            "david@test.com",
            "eve@test.com"
        ]
        correct = sum(1 for i, row in enumerate(rows) if row['email'] == expected[i])
        return correct / 5.0

    def _populate_hard_task(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
        depts = [(1, "Engineering"), (2, "Sales")]
        c.executemany("INSERT INTO departments VALUES (?, ?)", depts)
        
        c.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary INTEGER)")
        # Engineering Avg without NULL: (110k + 100k) / 2 = 105000
        # Sales Avg without NULL: (60k + 70k + 80k) / 3 = 70000
        emps = [
            (1, "John", 1, 110000),
            (2, "Jane", 1, 100000),
            (3, "Bob", 1, None),
            (4, "Sara", 2, 60000),
            (5, "Jim", 2, 70000),
            (6, "Rick", 2, None),
            (7, "Lisa", 2, 80000),
            (8, "Tom", 1, None)
        ]
        c.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", emps)
        self.conn.commit()

    def _eval_hard_task(self) -> float:
        c = self.conn.cursor()
        c.execute("SELECT id, salary FROM employees ORDER BY id")
        rows = c.fetchall()
        
        expected_salaries = {
            1: 110000, 2: 100000, 3: 105000,  # Bob gets 105000
            4: 60000, 5: 70000, 6: 70000,     # Rick gets 70000
            7: 80000, 8: 105000               # Tom gets 105000
        }
        
        correct = 0
        total = 8
        for row in rows:
            emp_id = row['id']
            salary = row['salary']
            if emp_id in expected_salaries and salary == expected_salaries[emp_id]:
                correct += 1
                
        return correct / total

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataCleanerObservation:
        self.episode_id = episode_id or "easy"
        self.step_cnt = 0
        self._reset_rubric()
        
        # Determine task
        if "easy" in self.episode_id.lower():
            self.task_name = "easy"
        elif "medium" in self.episode_id.lower():
            self.task_name = "medium"
        elif "hard" in self.episode_id.lower():
            self.task_name = "hard"
        else:
            self.task_name = "easy"
            
        self._setup_db()
        
        if self.task_name == "easy":
            self._populate_easy_task()
            intro = "Task: Deduplicate the 'users' table. Remove duplicate combinations of (name, email) keeping the minimum id. Schema: users(id, name, email)."
        elif self.task_name == "medium":
            self._populate_medium_task()
            intro = "Task: Standardize the 'contacts' table emails. Trim leading/trailing whitespaces and convert to lowercase. Schema: contacts(id, email)."
        else:
            self._populate_hard_task()
            intro = "Task: Impute missing salaries in 'employees'. Update employees with NULL salary to the average salary of their respective department. Schema: departments(id, name), employees(id, name, dept_id, salary)."
            
        return DataCleanerObservation(query_result=intro)

    def step(
        self,
        action: DataCleanerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataCleanerObservation:
        self.step_cnt += 1
        query = action.sql_command.strip()
        
        error = None
        result_str = ""
        
        # Execute query
        try:
            c = self.conn.cursor()
            c.execute(query)
            if query.upper().startswith("SELECT") or query.upper().startswith("PRAGMA"):
                rows = c.fetchmany(50)  # limit to 50 rows to avoid blowing up context
                if not rows:
                    result_str = "No rows returned."
                else:
                    # Convert to list of dicts
                    columns = rows[0].keys()
                    rows_dicts = [dict(r) for r in rows]
                    result_str = json.dumps(rows_dicts, indent=2)
            else:
                self.conn.commit()
                result_str = f"Query executed successfully. Rows affected: {c.rowcount}"
        except Exception as e:
            error = str(e)
            result_str = "Error executing query."

        # Evaluate score
        if self.task_name == "easy":
            score = self._eval_easy_task()
        elif self.task_name == "medium":
            score = self._eval_medium_task()
        else:
            score = self._eval_hard_task()
            
        done = score >= 0.99 or self.step_cnt >= 10
        obs = DataCleanerObservation(
            query_result=result_str,
            error=error,
            done=done,
            reward=score
        )
        return obs

    @property
    def state(self) -> DataCleanerState:
        return DataCleanerState(
            step_count=self.step_cnt,
            episode_id=self.episode_id,
            task_name=self.task_name,
            db_path=self.db_path
        )

    def close(self) -> None:
        if self.conn:
            self.conn.close()
        # Note: Temp file gets leaked without os.remove, adding cleanup.
        import os
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        super().close()
