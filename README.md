---
title: OpenEnv Data Cleaning
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
tags: [openenv, reinforcement-learning, data-engineering]
---

# OpenEnv Data Cleaning Environment

An [OpenEnv](https://github.com/openenv-org/openenv) environment for training RL agents on SQL data cleaning tasks. The agent works with a SQLite database and issues queries to fix data quality issues like duplicates, inconsistent formatting, and missing values.

## Tasks

| Level | Task | What the agent needs to do |
|-------|------|---------------------------|
| Easy | Deduplication | Remove duplicate rows from `users` table, keep lowest id |
| Medium | Standardization | Normalize emails in `contacts` — trim whitespace, lowercase |
| Hard | Imputation | Fill NULL salaries in `employees` with department averages |

## Setup

```bash
pip install -e .

# required
export HF_TOKEN="your-hf-token"

# optional (have defaults)
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"

python inference.py
```

## Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

## Structure

```
├── data_cleaning_env.py    # environment (reset/step/state)
├── inference.py            # agent loop
├── server/app.py           # FastAPI server
├── openenv.yaml            # OpenEnv config
├── pyproject.toml
├── Dockerfile
└── requirements.txt
```
