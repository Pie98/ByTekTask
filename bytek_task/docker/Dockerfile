FROM python:3.11-slim

WORKDIR /app

RUN python -m pip install poetry>1 poetry-plugin-export>1

COPY pyproject.toml poetry.lock /app/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls

RUN python -m pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY /bytek_task /app/bytek_task

WORKDIR = app/bytek_task

ENV PYTHONPATH = "/app/bytek_task"

# Calcolo dinamico dei worker nel CMD usando shell
CMD exec gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers $((2 * $(nproc) + 1)) \
  --bind 0.0.0.0:8000 \
  --limit-concurrency 100 \
  --timeout 60 \
  --access-logfile - \
  --log-level info