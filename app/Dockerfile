# FROM python:3.10-slim
FROM python:3.12-slim

RUN apt-get update && apt-get install -y cron curl zip unzip sqlite3 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

RUN pip install pip poetry setuptools wheel -U --no-cache-dir
COPY pyproject.toml poetry.lock .
RUN poetry install --no-cache

RUN crontab /asdf/crontab

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY app /app
RUN poetry install --no-cache

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]

