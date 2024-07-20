#!/bin/bash

if [ ! -f /config/config.yml ]; then
  cp /config/default-config.yml /config/config.yml
fi

printenv >/etc/environment
cron
exec poetry run python -m lss
