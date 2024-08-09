# Base
FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y cron curl zip unzip sqlite3 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip
RUN pip install pip poetry setuptools wheel -U --no-cache-dir
COPY alembic.ini pyproject.toml poetry.lock .
RUN poetry install --without=dev --no-cache
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY app /app
EXPOSE 80

# Development
FROM base AS development
RUN poetry install --no-cache
ENTRYPOINT ["/entrypoint.sh"]

# Production
FROM base AS production
RUN poetry install --without=dev --no-cache
ENTRYPOINT ["/entrypoint.sh"]
