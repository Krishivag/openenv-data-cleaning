---
title: OpenEnv Data Cleaning
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags: [openenv]
---

# SQLite Data Cleaning Environment

This is an [OpenEnv](https://github.com/openenv-org/openenv) environment simulating a Data Engineer/Data Analyst task. The agent is given an in-memory SQLite database containing dirty, duplicate, or missing data records, and it must issue precise SQL `UPDATE`, `DELETE`, or `INSERT` statements to clean the data.

## Motivation & Domain
Data cleaning logic is messy, context-dependent, and requires iterative diagnosis—making it a perfect task for evaluating reasoning agents. In real-world data pipelines, messy records are the norm, and an LLM that can successfully navigate schema discovery, spot inconsistencies, and execute fixing queries without deleting correct data is highly valuable.

## Action & Observation Spaces
- **Action**: `sql_command: str`. The agent issues a single SQL DML or DDL query string.
- **Observation**: `query_result: str` and `error: Optional[str]`. The environment executes the query. If it's a `SELECT` or `PRAGMA`, the result set is returned as a JSON string (up to 50 rows). Otherwise, a success string indicates affected rows. If the query fails (e.g., syntax error), `error` contains the SQLite traceback.
- **Reward**: `score: float`. 0.0 to 1.0 indicating completion percentage. Every intermediate step calculates the completion heuristic (e.g., percentage of formatted emails correct).

## Tasks
The environment features 3 increasing difficulty graded tasks:
1. **Easy**: *Deduplicate Users*. The agent must identify duplicate rows based on combination logic and delete the redundant instances while keeping identical primary items intact.
2. **Medium**: *Standardize Emails*. A table contains malformed strings (cased, trailing spaces). The agent must use SQL functions (`TRIM`, `LOWER`) to standardize.
3. **Hard**: *Impute missing salaries*. The agent deals with relational integrity. Employees have `NULL` salaries but contain a `dept_id`. The agent must compute the average salary of each distinct department and assign it respectively using `UPDATE ... FROM ...` or subqueries.

## Setup & Running the Baseline
1. Clone the repository.
2. Install `uv` and `openenv-core` via `pip install -e .`
3. Export your API keys:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export MODEL_NAME="gpt-4o"
   ```
4. Run the deterministic baseline:
   ```bash
   python inference.py
   ```
5. Spin up the server manually via Docker:
   ```bash
   docker build -t data_cleaning_env .
   docker run -p 7860:7860 data_cleaning_env
   ```

## Scores
The provided `inference.py` script validates standard format outputs required by the hackathon evaluation parameters.
