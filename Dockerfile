FROM python:3.10-slim

WORKDIR /src

RUN apt-get update && apt-get install -y cron curl zip unzip sqlite3 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

COPY src/requirements.txt .
RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/crontab /etc/cron.d/app-crontab
RUN chmod 0644 /etc/cron.d/app-crontab && crontab /etc/cron.d/app-crontab && \
mkdir -p /var/log/cron/generate /var/log/cron/tfidf /var/log/cron/distilbert

COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY src .
RUN chmod +x /src/loadenv.sh


EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]

