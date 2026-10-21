#!/usr/bin/env bash
# Offline end-to-end demo: index the bundled wiki, search it three ways, answer with citations.
set -euo pipefail
cd "$(dirname "$0")/.."
DB="${DB:-/tmp/ragkit-demo.db}"
rm -f "$DB"

uv run ragkit ingest src/ragkit/data/eval/docs --db "$DB"
uv run ragkit query "how many days of paid time off do I get" --db "$DB" --mode bm25 -k 3
uv run ragkit query "how many days of paid time off do I get" --db "$DB" --mode hybrid -k 3
uv run ragkit ask "what is the Beacon API rate limit" --db "$DB"
uv run ragkit eval
