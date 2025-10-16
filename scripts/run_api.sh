#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export USE_SYNTHETIC=${USE_SYNTHETIC:-1}
export SYMBOLS=${SYMBOLS:-AAPL,MSFT,NVDA}
uvicorn services.api.main:app --host 0.0.0.0 --port 8000