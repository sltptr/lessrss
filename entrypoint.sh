#!/bin/bash

if [ ! -f /config/config.yml ]; then
  cp /config/default-config.yml /config/config.yml
fi

cron
poetry run alembic upgrade head
exec poetry run python -m lss
