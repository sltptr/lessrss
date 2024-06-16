FROM python:3.12-slim

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN apt-get update && apt-get install -y cron && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN mkdir /config
COPY config /config

COPY src/crontab /etc/cron.d/app-crontab
RUN chmod 0644 /etc/cron.d/app-crontab && crontab /etc/cron.d/app-crontab && touch /var/log/cron.log

COPY src/requirements.txt .
RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY src .

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]

