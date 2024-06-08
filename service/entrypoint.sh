#!/bin/bash

cron
source /app/.venv/bin/activate
exec python run.py
