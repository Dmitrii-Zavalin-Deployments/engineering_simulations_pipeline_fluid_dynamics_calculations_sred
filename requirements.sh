#!/bin/bash
set -euo pipefail

echo "🚀 Upgrading core Python packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "📦 Installing solver dependencies from requirements.txt..."
python -m pip install -r requirements.txt
