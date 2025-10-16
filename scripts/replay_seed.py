from __future__ import annotations
import argparse
import os
import time
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
parser.add_argument("--symbol", default="AAPL")
parser.add_argument("--text", default="AAPL beats earnings with strong guidance")
args = parser.parse_args()

r = requests.post(f"{args.api}/inject", json={"symbol": args.symbol, "text": args.text}, timeout=5)
print(r.status_code, r.text)
