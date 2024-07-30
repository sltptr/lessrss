#!/bin/bash

if [ ! -f /config/config.yml ]; then
  cp /config/default-config.yml /config/config.yml
fi

if [ "$1" = "migrate" ]; then
  poetry run alembic --config /app/alembic.ini upgrade head
  exit 0
fi

printenv >/env.txt
cron
exec poetry run app
