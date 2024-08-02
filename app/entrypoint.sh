#!/bin/bash

if [ ! -f /config/config.yml ]; then
  echo "Error: Missing config/config.yml, check out config/examples for references"
  exit 1
fi

if [ ! -f /config/crontab ]; then
  echo "Error: Missing config/crontab, check out config/examples for references"
  exit 1
fi

if [ "$1" = "migrate" ]; then
  poetry run alembic --config /alembic.ini upgrade head
  exit 0
fi

printenv >/env.txt
crontab /config/crontab
cron
exec poetry run start
