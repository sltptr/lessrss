#!/bin/bash

if [ ! -f /config/config.yml ]; then
	cp /config/default-config.yml /config/config.yml
fi

cron
source /src/.venv/bin/activate
exec python -u -m app
