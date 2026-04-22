#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[athena] Creating virtual environment with Python 3.13..."
python3.13 -m venv .venv

echo "[athena] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[athena] Upgrading pip..."
pip install --quiet --upgrade pip

echo "[athena] Installing dependencies..."
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    echo "[athena] Copying .env.example -> .env  (fill in your keys!)"
    cp .env.example .env
else
    echo "[athena] .env already exists, skipping copy."
fi

echo ""
echo "[athena] Setup complete."
echo "  Activate env : source project5/.venv/bin/activate"
echo "  Seed data    : python scripts/seed_data.py"
echo "  Sync all     : python scripts/sync_all.py"
echo "  Run bot      : python interface/bot.py"
