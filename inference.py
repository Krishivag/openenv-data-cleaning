import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from data_cleaning_env import DataCleanerAction, DataCleaningEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "easy")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "data_cleaning")
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.97

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Data Engineer working with a SQLite database.
    Your task is to fix and clean the data. 
    You will be provided with the current context.
    Reply with exactly one SQL query (no markdown, no quotes, just the SQL text).
    Only run data modification queries (UPDATE, DELETE, INSERT) when you are ready.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error).replace('\n', ' ') if error else "null"
    done_val = str(done).lower()
    # Normalize action string avoiding newlines
    action_log = action.replace('\n', ' ').strip()
    print(
        f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, last_result: str, last_error: Optional[str], history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    error_msg = f"Last error: {last_error}" if last_error else ""
    return textwrap.dedent(
        f"""
        Step: {step}
        Last observation: {last_result}
        {error_msg}
        Previous steps history:
        {history_block}
        
        Send your next SQL statement. ONLY reply with the raw SQL code. No markdown backticks.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, last_result: str, last_error: Optional[str], history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_result, last_error, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Remove markdown if present
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
        return text if text else "SELECT 1;"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SELECT 1;"

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Instantiate direct environment
    env = DataCleaningEnv()

    for task_name in ["easy", "medium", "hard"]:
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.01
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = await env.reset_async(episode_id=task_name)
            last_result = obs.query_result
            last_error = obs.error

            for step in range(1, MAX_STEPS + 1):
                if getattr(obs, 'done', False):
                    break

                message = get_model_message(client, step, last_result, last_error, history)

                obs = await env.step_async(DataCleanerAction(sql_command=message))

                reward = obs.reward if obs.reward is not None else 0.01
                if isinstance(reward, bool):
                    reward = float(reward)
                done = obs.done
                error = obs.error

                rewards.append(reward)
                steps_taken = step
                last_result = obs.query_result
                last_error = error

                log_step(step=step, action=message, reward=reward, done=done, error=error)

                history.append(f"Step {step}: {message!r} -> error: {error}")

                if done:
                    # Final score is the last reward in this env design
                    score = reward
                    break

            score = min(max(score, 0.01), 0.99)  # clamp to strict (0, 1)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            try:
                env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
