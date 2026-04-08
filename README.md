---
title: OpenEnv Data Cleaning
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags: [openenv, reinforcement-learning, data-engineering]
---

<div align="center">
  # 🧹 OpenEnv Data Cleaning Environment

  [![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blue.svg)](https://github.com/openenv-org/openenv)
  [![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

  **A real-world RL environment designed to train and evaluate AI agents on complex SQL-based data engineering tasks.**
</div>

---

## 🏗 Overview

The **Data Cleaning Environment** is a specialized [OpenEnv](https://github.com/openenv-org/openenv) environment that simulates a professional Data Engineer/Analyst workflow. The agent is tasked with diagnosing and repairing an "in-memory" SQLite database containing common real-world data issues: duplicates, inconsistent formatting, and missing relational data.

### 🎯 Objective
Issue precise SQL `UPDATE`, `DELETE`, or `INSERT` statements to achieve a target clean state, verified by deterministic grading scripts.

---

## ⚡ Key Features

- **Standardized API**: Full compliance with the `step()`, `reset()`, and `state()` OpenEnv protocol.
- **Graded Difficulty**: Three distinct task levels ranging from simple deduplication to complex relational imputation.
- **Deterministic Evaluation**: Rich reward signals (0.0 - 1.0) based on actual data integrity metrics.
- **Secure Sandbox**: Each session operates on an isolated, in-memory SQLite instance via `tempfile`.

---

## 🧠 Action & Observation Spaces

### **Action Space**
The agent interacts via the `sql_command: str` field.
- **Supported**: Any valid SQLite DML/DDL (SELECT, UPDATE, DELETE, INSERT, PRAGMA).

### **Observation Space**
- **`query_result`**: The output of the executed SQL command.
  - `SELECT` queries return up to 50 rows formatted as a JSON string.
  - `UPDATE/DELETE` returns "Rows affected: N".
- **`error`**: Full SQLite traceback if a query fails (syntax errors, constraint violations).
- **`reward`**: A float from `0.0` to `1.0` representing current task progress.

---

## 📋 Task Suite

| Difficulty | Task Name | Description | Key Skills |
|:---:|:---|:---|:---|
| 🟢 | **Easy** | User Deduplication | Identifying duplicates and keeping the base record intact. |
| 🟡 | **Medium** | Email Standardization | SQL string functions (`LOWER`, `TRIM`) for uniformity. |
| 🔴 | **Hard** | Salary Imputation | Multi-table joins and subqueries to calculate and fill averages. |

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Krishivag/openenv-data-cleaning.git
cd openenv-data-cleaning
pip install -e .
```

### 2. Configure Environment
```bash
export OPENAI_API_KEY="your-key-here"
export MODEL_NAME="gpt-4o"
```

### 3. Run the Baseline
Evaluate the environment's health and grading using the provided inference script:
```bash
python inference.py
```

### 4. Deploy with Docker
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

---

## 🛡 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
  Built for the <b>OpenEnv Hackathon</b> • 2026
</div>
