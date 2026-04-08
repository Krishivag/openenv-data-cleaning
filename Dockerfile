FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Upgrade pip and install curl
RUN pip install --upgrade pip

# Copy project files
COPY --chown=user:user . /app/

# Install the application
RUN pip install --user -e .

# Run openenv serve on HF standard port
ENV HOST="0.0.0.0"
ENV PORT="7860"

# OpenEnv serves by default using fastapi
# You can also run using uv_run or uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
