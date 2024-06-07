FROM python:3.12-slim
RUN apt-get update && apt-get install -y cron

WORKDIR /app

COPY crontab /etc/cron.d/app-crontab
RUN chmod 0644 /etc/cron.d/app-crontab && crontab /etc/cron.d/app-crontab && touch /var/log/cron.log

COPY requirements.txt .

RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD source .venv/bin/activate && python workflows/tfidf.py && cron && tail -f /var/log/cron.log

