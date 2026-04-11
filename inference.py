import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from data_cleaning_env import DataCleanerAction, DataCleaningEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "data_cleaning"
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 150
SUCCESS_THRESHOLD = 0.97

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Data Engineer working with a SQLite database.
    Your task is to fix and clean the data.
    You will be provided with the current context.
    Reply with exactly one SQL query (no markdown, no quotes, just the SQL text).
    Only run data modification queries (UPDATE, DELETE, INSERT) when you are ready."""
).strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    err = str(error).replace("\n", " ") if error else "null"
    act = action.replace("\n", " ").strip()
    print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success, steps, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={r}", flush=True)


def build_prompt(step, last_result, last_error, history):
    hist = "\n".join(history[-4:]) if history else "None"
    err_line = f"Last error: {last_error}" if last_error else ""
    return textwrap.dedent(f"""\
        Step: {step}
        Last observation: {last_result}
        {err_line}
        Previous steps history:
        {hist}

        Send your next SQL statement. ONLY reply with the raw SQL code. No markdown backticks."""
    ).strip()


def ask_model(step, last_result, last_error, history):
    prompt = build_prompt(step, last_result, last_error, history)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
        return text if text else "SELECT 1;"
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "SELECT 1;"


def main():
    env = DataCleaningEnv()

    for task_name in ["easy", "medium", "hard"]:
        history = []
        rewards = []
        steps_taken = 0
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(episode_id=task_name)
            last_result = obs.query_result
            last_error = obs.error

            for step in range(1, MAX_STEPS + 1):
                if getattr(obs, "done", False):
                    break

                sql = ask_model(step, last_result, last_error, history)
                obs = env.step(DataCleanerAction(sql_command=sql))

                reward = obs.reward if obs.reward is not None else 0.01
                if isinstance(reward, bool):
                    reward = float(reward)
                if reward <= 0.0: reward = 0.01
                if reward >= 1.0: reward = 0.99

                rewards.append(reward)
                steps_taken = step
                last_result = obs.query_result
                last_error = obs.error

                log_step(step, sql, reward, obs.done, obs.error)
                history.append(f"Step {step}: {sql!r} -> error: {obs.error}")

                if obs.done:
                    break

            final = rewards[-1] if rewards else 0.01
            success = final >= SUCCESS_THRESHOLD

        except Exception as e:
            print(f"[DEBUG] Episode error: {e}", flush=True)
        finally:
            try:
                env.close()
            except Exception:
                pass
            log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    main()
