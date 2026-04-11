FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
RUN pip install --upgrade pip

COPY --chown=user:user . /app/
RUN pip install --user -e .

ENV HOST="0.0.0.0"
ENV PORT="7860"
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
