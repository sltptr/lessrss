#!/bin/bash

cron
source /app/.venv/bin/activate
exec python -u run.py
