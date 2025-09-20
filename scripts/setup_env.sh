#!/usr/bin/env bash
set -e
python3 -m pip install --upgrade pip
pip install docker-compose
# Hint: install huggingface-cli and login manually: `pip install huggingface_hub` then `huggingface-cli login`.
# This script does NOT download heavy model weights.
echo "Environment setup done. Please set HF_TOKEN in .env and login to Hugging Face if required."
