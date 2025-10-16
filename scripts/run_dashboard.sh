#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export API_URL=${API_URL:-http://localhost:8000}
export SYMBOLS=${SYMBOLS:-AAPL,MSFT,NVDA}
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0